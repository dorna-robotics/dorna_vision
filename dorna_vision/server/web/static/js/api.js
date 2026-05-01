// JS port of dorna_vision_client.VisionClient.
// Same JSON envelope, same commands. Browser-only — uses native WebSocket.
//
// Usage:
//   import { VisionClient } from "/static/js/api.js";
//   const vc = new VisionClient();
//   await vc.connect();              // defaults: same host as the page, /ws path
//   const hello = await vc.hello();
//   const devs  = await vc.cameraList();
//   const det   = vc.detection("d1");
//   const res   = await det.run();

const DEFAULT_TIMEOUT = 10_000; // ms

export class VisionServerError extends Error {
  constructor(code, msg) {
    super(`${code}: ${msg}`);
    this.code = code;
    this.msg = msg;
  }
}

export class VisionClient {
  constructor() {
    this._ws = null;
    this._idCounter = 1;
    this._pending = new Map();        // id -> {resolve, reject, needsBinary, json}
    this._lastBinaryHolder = null;    // pending entry waiting for its binary frame
    this._defaultTimeout = DEFAULT_TIMEOUT;
    this._listeners = new Set();      // ('open'|'close'|'error', payload)
  }

  on(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
  _emit(ev, p) { for (const fn of this._listeners) try { fn(ev, p); } catch {} }

  isConnected() { return !!this._ws && this._ws.readyState === WebSocket.OPEN; }

  connect({ url, timeout = 5000, defaultTimeout = DEFAULT_TIMEOUT } = {}) {
    this._defaultTimeout = defaultTimeout;
    const wsUrl = url || `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;

    return new Promise((resolve, reject) => {
      let settled = false;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      const fail = (err) => {
        if (settled) return;
        settled = true;
        try { ws.close(); } catch {}
        reject(err instanceof Error ? err : new Error(String(err)));
      };

      const handshakeTimer = setTimeout(() => fail(new Error(`websocket connect timed out after ${timeout}ms`)), timeout);

      ws.addEventListener("open", () => {
        clearTimeout(handshakeTimer);
        this._ws = ws;
        this._emit("open");
        // Verify with hello so we don't pretend we're "up" before the server replies.
        this.hello({ timeout })
          .then(info => { if (!settled) { settled = true; resolve(info); } })
          .catch(err => fail(err));
      });

      ws.addEventListener("message", (ev) => this._onMessage(ev));

      ws.addEventListener("error", (ev) => {
        this._emit("error", ev);
        fail(new Error("websocket error"));
      });

      ws.addEventListener("close", () => {
        const wasUp = !!this._ws;
        this._ws = null;
        this._emit("close");
        if (!wasUp) fail(new Error("websocket closed before handshake completed"));
        // Reject all pending
        for (const [id, p] of this._pending) p.reject(new Error("connection closed"));
        this._pending.clear();
        this._lastBinaryHolder = null;
      });
    });
  }

  close() {
    try { this._ws && this._ws.close(); } catch {}
    this._ws = null;
  }

  _onMessage(ev) {
    if (typeof ev.data === "string") {
      let payload; try { payload = JSON.parse(ev.data); } catch { return; }
      // Route to a pending request reply FIRST. Server replies always
      // carry a numeric `id` that matches a request we sent — even when
      // they also carry a `type` field (e.g. camera_get_img's binary
      // envelope echoes the image type). Falling back to event dispatch
      // only when no pending request matches keeps the two channels
      // (request/reply vs server-initiated events) properly separated.
      const id = payload.id;
      if (id != null && this._pending.has(id)) {
        const p = this._pending.get(id);
        if (payload.binary_follows) {
          p.json = payload;
          p.needsBinary = true;
          this._lastBinaryHolder = p;
          return;
        }
        this._pending.delete(id);
        p.resolve(payload);
        return;
      }
      // Server-initiated event — dispatch by type.
      if (payload.type && this._eventListeners) {
        const subs = this._eventListeners[payload.type];
        if (subs) for (const cb of subs.slice()) {
          try { cb(payload); } catch (e) { console.error("event listener", e); }
        }
      }
      return;
    } else {
      // binary frame — pair with last pending that asked for one
      const p = this._lastBinaryHolder;
      this._lastBinaryHolder = null;
      if (!p) return;
      const id = p.json && p.json.id;
      if (id != null) this._pending.delete(id);
      p.resolve({ json: p.json, binary: ev.data });
    }
  }

  _send(cmd, args = {}, { timeout } = {}, binary = null) {
    if (!this.isConnected()) return Promise.reject(new Error("not connected"));
    const id = this._idCounter++;
    // When the caller supplies a binary follow-frame, flag it in the JSON
    // envelope so the server knows to wait for the next ws message and
    // pair them. Used for shipping ML weights inline with detection_add
    // and image bytes inline with detection.run.
    const envelope = JSON.stringify(
      binary ? { cmd, id, args, binary_follows: true } : { cmd, id, args }
    );

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this._pending.delete(id);
        reject(new Error(`timeout waiting for reply to ${cmd} (id=${id})`));
      }, timeout || (binary ? Math.max(60000, this._defaultTimeout) : this._defaultTimeout));

      const wrap = {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject:  (e) => { clearTimeout(timer); reject(e); },
        needsBinary: false,
        json: null,
      };
      this._pending.set(id, wrap);

      try {
        this._ws.send(envelope);
        if (binary) {
          this._ws.send(binary instanceof ArrayBuffer ? binary : (binary.buffer || binary));
        }
      } catch (ex) {
        clearTimeout(timer);
        this._pending.delete(id);
        reject(new Error(`send failed: ${ex && ex.message ? ex.message : ex}`));
      }
    }).then(reply => {
      if (reply && reply.binary) return reply;     // binary path
      if (reply && reply.error) {
        throw new VisionServerError(reply.error.code || "INTERNAL", reply.error.msg || "");
      }
      return reply;
    });
  }

  // ---------------- typed commands (lifecycle / discovery / binary) -------

  hello(opts)              { return this._send("hello", {}, opts); }
  cameraList(opts)         { return this._send("camera_list", {}, opts).then(r => r.devices || []); }
  cameraAdd(serial_number, connectKwargs = {}, opts) { return this._send("camera_add", { serial_number, ...connectKwargs }, opts); }
  cameraRemove(serial_number, opts) { return this._send("camera_remove", { serial_number }, opts); }
  cameraRecover(serial_number, opts) { return this._send("camera_recover", { serial_number }, opts); }

  /** Register a listener for server-initiated events (JSON frames with a
   * "type" field and no "id"). Distinct from on(fn), which is for
   * connection lifecycle events ("close", etc.).
   * Returns an unsubscribe function. */
  onEvent(type, callback) {
    if (!this._eventListeners) this._eventListeners = {};
    (this._eventListeners[type] ||= []).push(callback);
    return () => {
      const subs = this._eventListeners[type];
      if (!subs) return;
      const i = subs.indexOf(callback);
      if (i >= 0) subs.splice(i, 1);
    };
  }
  cameraGetImg(serial_number, type = "color_img", quality = 75, opts) {
    return this._send("camera_get_img", { serial_number, type, quality }, opts);  // {json, binary}
  }
  robotAdd(host, kw = {}, opts) { return this._send("robot_add", { host, ...kw }, opts); }
  robotRemove(host, opts)  { return this._send("robot_remove", { host }, opts); }
  detectionList(opts)      { return this._send("detection_list", {}, opts).then(r => r.detections || []); }
  // `body` carries the JSON kwargs for Detection.__init__.
  // `binary` (optional ArrayBuffer) is the ML model file content — sent
  // as a follow-up binary frame; the server writes it to a per-call
  // temp file just long enough for Detection's loader to consume it.
  detectionAdd(name, body, binary = null, opts) {
    return this._send("detection_add", { name, ...body }, opts, binary);
  }
  detectionRemove(name, opts)    { return this._send("detection_remove", { name }, opts); }
  detectionGetImg(name, type = "img", quality = 85, opts) {
    return this._send("detection_get_img", { name, type, quality }, opts);  // resolves to {json, binary}
  }

  // ---------------- proxy-style RPC ---------------------------------------

  detection(name) { return new _ObjectProxy(this, "detection", name); }
  camera(serial_number) { return new _ObjectProxy(this, "camera", serial_number); }
  robot(host) { return new _ObjectProxy(this, "robot", host); }

  _call(target, name, method, args = [], kwargs = {}, opts, binary = null) {
    return this._send("call", { target, name, method, args: [...args], kwargs: { ...kwargs } }, opts, binary)
      .then(reply => reply.result);
  }
}

// A lightweight Proxy: every property access becomes a remote call.
function _ObjectProxy(client, target, name) {
  const obj = {
    _client: client,
    _target: target,
    _name: name,
    get_img(type = "img", quality = 85, opts) {
      if (target !== "detection") throw new Error("get_img is only valid on a detection proxy");
      return client.detectionGetImg(name, type, quality, opts);
    },
  };
  return new Proxy(obj, {
    get(t, prop) {
      if (prop in t) return t[prop];
      if (typeof prop !== "string" || prop.startsWith("_")) return undefined;
      return (...args) => {
        // optional last arg: { _timeout, _kwargs, _binary }
        let kwargs = {};
        let opts;
        let binary = null;
        const last = args.length ? args[args.length - 1] : null;
        if (last && typeof last === "object"
            && (last._timeout !== undefined || last._kwargs || last._binary)) {
          args.pop();
          if (last._timeout !== undefined) opts = { timeout: last._timeout };
          if (last._kwargs) kwargs = last._kwargs;
          if (last._binary) binary = last._binary;
        }
        return client._call(target, name, prop, args, kwargs, opts, binary);
      };
    },
  });
}
