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

const DEFAULT_TIMEOUT = 10_000;      // ms, per request
const RECONNECT_DELAY = 1_000;       // ms, first retry; doubles each attempt
const RECONNECT_MAX_DELAY = 10_000;  // ms, backoff ceiling
const HEARTBEAT = 30_000;            // ms, idle liveness probe
const HEARTBEAT_TIMEOUT = 8_000;     // ms, a probe unanswered this long = dead link

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
    this._listeners = new Set();      // ('open'|'ready'|'reconnecting'|'close'|'error', payload)
    // Connection-keeping state. `_wanted` is the app's intent — true from
    // connect() until close() — and is what reconnects key on. `_detached`
    // holds sockets already given up on: a dead TCP link can deliver its
    // close event minutes after we abandoned it, and that must not touch
    // the socket that replaced it.
    this._opts = {};
    this._wanted = false;
    this._connecting = false;
    this._attempt = 0;
    this._reconnectTimer = null;
    this._hbTimer = null;
    this._probing = false;
    this._detached = new WeakSet();
  }

  on(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
  _emit(ev, p) { for (const fn of this._listeners) try { fn(ev, p); } catch {} }

  isConnected() { return !!this._ws && this._ws.readyState === WebSocket.OPEN; }

  /** Open the connection and verify it with `hello`. Resolves with the
   *  hello reply; rejects if the first attempt fails.
   *
   *  autoReconnect keeps the link alive for the life of the page: when it
   *  drops (server restart, NAT idle timeout, laptop sleep) it reconnects
   *  with backoff, forever. An idle link is also probed every `heartbeat`
   *  ms, because a socket the browser still reports OPEN can be dead
   *  underneath (a NAT that forgot it drops packets in both directions
   *  and nobody sends a FIN) — an unanswered probe abandons the socket
   *  and reconnects, instead of the GUI claiming "connected" until a
   *  request times out. Lifecycle events via on(): "open" (socket up),
   *  "ready" (hello answered — usable; fires after every reconnect too),
   *  "reconnecting" ({attempt, delay}), "close", "error". */
  connect({ url, timeout = 5000, defaultTimeout = DEFAULT_TIMEOUT,
            autoReconnect = false, reconnectDelay = RECONNECT_DELAY,
            reconnectMaxDelay = RECONNECT_MAX_DELAY,
            heartbeat = HEARTBEAT, heartbeatTimeout = HEARTBEAT_TIMEOUT } = {}) {
    this._defaultTimeout = defaultTimeout;
    this._opts = { url, timeout, autoReconnect, reconnectDelay, reconnectMaxDelay,
                   heartbeat, heartbeatTimeout };
    this._wanted = true;
    this._attempt = 0;
    this._clearReconnectTimer();
    return this._open();
  }

  /** Make sure the link is alive right now. Call when the tab returns to
   *  the foreground or the OS says the network is back: a connected
   *  socket is probed (it may have died while the tab slept), a
   *  disconnected one is reconnected immediately, skipping the backoff. */
  reconnectNow() {
    if (!this._wanted) return;
    if (this.isConnected()) { this._probe(); return; }
    if (this._connecting) return;
    this._clearReconnectTimer();
    this._attempt = 0;
    this._open().catch(() => {});
  }

  close() {
    this._wanted = false;
    this._clearReconnectTimer();
    this._stopHeartbeat();
    this._drop("closed");
  }

  _open() {
    const { url, timeout } = this._opts;
    const wsUrl = url || `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
    this._connecting = true;

    return new Promise((resolve, reject) => {
      let settled = false;
      let ws = null;
      const fail = (err) => {
        if (settled) return;
        settled = true;
        this._connecting = false;
        if (ws) { this._detached.add(ws); try { ws.close(); } catch {} }
        reject(err instanceof Error ? err : new Error(String(err)));
        this._scheduleReconnect();
      };

      try { ws = new WebSocket(wsUrl); } catch (ex) { fail(ex); return; }
      ws.binaryType = "arraybuffer";
      const handshakeTimer = setTimeout(
        () => fail(new Error(`websocket connect timed out after ${timeout}ms`)), timeout);

      ws.addEventListener("open", () => {
        clearTimeout(handshakeTimer);
        if (this._detached.has(ws)) return;
        this._ws = ws;
        this._attempt = 0;
        this._emit("open");
        // Verify with hello so we don't pretend we're "up" before the server replies.
        this.hello({ timeout })
          .then(info => {
            if (settled || this._ws !== ws) return;
            settled = true;
            this._connecting = false;
            this._startHeartbeat();
            this._emit("ready", info);
            resolve(info);
          })
          .catch(err => {
            if (this._ws === ws) this._drop("hello failed");
            fail(err);
          });
      });

      ws.addEventListener("message", (ev) => { if (this._ws === ws) this._onMessage(ev); });

      ws.addEventListener("error", (ev) => {
        if (this._detached.has(ws)) return;
        this._emit("error", ev);
        if (!settled) fail(new Error("websocket error"));
      });

      ws.addEventListener("close", () => {
        clearTimeout(handshakeTimer);
        if (this._detached.has(ws)) return;          // already replaced or abandoned
        if (this._ws === ws) this._drop("connection closed");
        else fail(new Error("websocket closed before handshake completed"));
      });
    });
  }

  /** Abandon the current socket — it closed, or it stopped answering —
   *  fail everything in flight, and let the reconnect policy take over.
   *  Idempotent per socket. */
  _drop(reason) {
    const ws = this._ws;
    if (!ws) return;
    this._ws = null;
    this._detached.add(ws);
    try { ws.close(); } catch {}
    this._stopHeartbeat();
    for (const [, p] of this._pending) p.reject(new Error(reason));
    this._pending.clear();
    this._lastBinaryHolder = null;
    this._emit("close");
    this._scheduleReconnect();
  }

  _scheduleReconnect() {
    if (!this._opts.autoReconnect || !this._wanted) return;
    if (this._reconnectTimer || this._connecting || this.isConnected()) return;
    const { reconnectDelay, reconnectMaxDelay } = this._opts;
    const delay = Math.min(reconnectMaxDelay, reconnectDelay * 2 ** this._attempt);
    this._attempt += 1;
    this._emit("reconnecting", { attempt: this._attempt, delay });
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._wanted || this._connecting || this.isConnected()) return;
      this._open().catch(() => {});
    }, delay);
  }

  _clearReconnectTimer() {
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    const iv = this._opts.heartbeat;
    if (!iv) return;
    // Probe only an IDLE link: in-flight requests are their own proof of
    // life, and their timeouts run a probe themselves (see _send).
    this._hbTimer = setInterval(() => {
      if (this.isConnected() && this._pending.size === 0) this._probe();
    }, iv);
  }

  _stopHeartbeat() {
    if (this._hbTimer) { clearInterval(this._hbTimer); this._hbTimer = null; }
  }

  _probe() {
    if (!this.isConnected() || this._probing) return;
    const ws = this._ws;
    this._probing = true;
    this.hello({ timeout: this._opts.heartbeatTimeout || HEARTBEAT_TIMEOUT })
      .then(() => { this._probing = false; })
      .catch(() => {
        this._probing = false;
        if (this._ws === ws) this._drop("connection stopped answering");
      });
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
        this._probe();   // a reply that never came may mean a dead link — find out now
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
  cameraInfo(serial_number, opts)    { return this._send("camera_info", { serial_number }, opts); }
  /** Focus control (uEye XS). args: {mode, position} or {region:[x0,y0,x1,y1]}.
      Region sweeps take ~15-30 s — give them a generous reply timeout. */
  cameraFocus(serial_number, args = {}, opts) {
    const o = (args.region && !(opts && opts.timeout)) ? { ...(opts || {}), timeout: 120000 } : opts;
    return this._send("camera_focus", { serial_number, ...args }, o);
  }

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
  // maxSide caps the PREVIEW's long edge (server-side downscale, after
  // detection has already run at full resolution). Reply meta carries
  // source_shape + scale. Never use it where the pixels are coordinates.
  detectionGetImg(name, type = "img", quality = 85, opts, maxSide = null) {
    const args = { name, type, quality };
    if (maxSide) args.max_side = maxSide;
    return this._send("detection_get_img", args, opts);  // resolves to {json, binary}
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
