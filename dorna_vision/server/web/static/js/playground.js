// Playground — schema-driven Detection editor with Raw JSON escape hatch.
//
// Server-side: holds one detection named PG_NAME for this session. Each Run
// removes + re-adds it under that name so any kwarg change applies cleanly.
// Promote re-registers the same config under a user-supplied name so it
// persists in the pool and is reachable as vc.detection("name").

const PG_NAME = "_playground";

// ── ArUco dictionaries (matches cv.aruco.DICT_*) ────────────────────
const ARUCO_DICTS = [
  "DICT_4X4_50","DICT_4X4_100","DICT_4X4_250","DICT_4X4_1000",
  "DICT_5X5_50","DICT_5X5_100","DICT_5X5_250","DICT_5X5_1000",
  "DICT_6X6_50","DICT_6X6_100","DICT_6X6_250","DICT_6X6_1000",
  "DICT_7X7_50","DICT_7X7_100","DICT_7X7_250","DICT_7X7_1000",
  "DICT_ARUCO_ORIGINAL",
  "DICT_APRILTAG_16h5","DICT_APRILTAG_25h9","DICT_APRILTAG_36h10","DICT_APRILTAG_36h11",
];

const REFINE_OPTS = ["CORNER_REFINE_NONE","CORNER_REFINE_SUBPIX","CORNER_REFINE_CONTOUR","CORNER_REFINE_APRILTAG"];

// ── Per-cmd schemas ────────────────────────────────────────────────
// Each schema = ordered list of fields. Defaults match the underlying
// function/class signatures from dorna_vision.find / util / board / ai.
const CMD_SCHEMAS = {
  cnt: {
    label: "Contour",
    fields: [
      { key: "inv",      label: "Inverse",          kind: "bool",   default: true },
      // type=0/1 are global (Otsu/Binary). type=2/3 are adaptive — local
      // thresholds that handle uneven lighting/vignetting where global
      // Otsu fails. In adaptive modes `thr` doubles as half-blockSize
      // and `mean_sub` is the C constant; same overload as before.
      { key: "type",     label: "Type",             kind: "select",
        options: [
          {v:0,t:"Otsu (auto)"},
          {v:1,t:"Binary"},
          {v:2,t:"Adaptive (Gaussian)"},
          {v:3,t:"Adaptive (Mean)"},
        ], default: 0 },
      { key: "thr",      label: "Threshold value",  kind: "slider", min: 0, max: 255, step: 1, default: 127 },
      { key: "blur",     label: "Smoothing blur",   kind: "slider", min: 1, max: 20, step: 1, default: 3 },
      { key: "mean_sub", label: "Mean subtract",    kind: "slider", min: -200, max: 200, step: 1, default: 0 },
    ],
  },
  poly: {
    label: "Polygon",
    fields: [
      { key: "inv",      label: "Inverse",          kind: "bool",   default: true },
      { key: "type",     label: "Type",             kind: "select",
        options: [
          {v:0,t:"Otsu (auto)"},
          {v:1,t:"Binary"},
          {v:2,t:"Adaptive (Gaussian)"},
          {v:3,t:"Adaptive (Mean)"},
        ], default: 0 },
      { key: "thr",      label: "Threshold value",  kind: "slider", min: 0, max: 255, step: 1, default: 127 },
      { key: "blur",     label: "Smoothing blur",   kind: "slider", min: 1, max: 20, step: 1, default: 3 },
      { key: "mean_sub", label: "Mean subtract",    kind: "slider", min: -200, max: 200, step: 1, default: 0 },
      { key: "side",     label: "Sides",            kind: "slider", min: 3, max: 20, step: 1, default: 3 },
    ],
  },
  // Unified "Blob" entry covering two backend cmds: SimpleBlobDetector
  // (cmd="blob", needs threshold tuning) and MSER (cmd="mser", picks
  // regions that stay stable across a threshold sweep — robust to
  // lighting variation). The `method` field is transient: it doesn't
  // get sent as a kwarg; the field-change handler rewrites
  // _cfg.detection.cmd between "blob" and "mser" via methodMap.
  blob: {
    label: "Blob",
    methodMap: { standard: "blob", lighting_tolerant: "mser" },
    fields: [
      { key: "method", label: "Method", kind: "select", transient: true,
        options: [
          { v: "standard",          t: "Standard" },
          { v: "lighting_tolerant", t: "Robust to lighting" },
        ], default: "standard" },

      // Standard (cmd="blob") — SimpleBlobDetector
      // `threshold` and `area` are range pairs (lo, hi). detect.py's blob
      // branch splits them into the find.blob() {min,max}Threshold /
      // {min,max}Area kwargs the OpenCV detector expects. Using a range
      // widget makes lo > hi impossible — the detector throws on that.
      { key: "threshold", label: "Threshold (min..max)", kind: "range", min: 0, max: 255, step: 1, default: [10, 220],
        showWhen: (v) => v.method === "standard" },
      // OpenCV's blob detector enforces `0 < minArea` — start at 1
      // instead of 0 so the lo handle can't produce an invalid config.
      { key: "area",      label: "Area (px², min..max)",  kind: "range", min: 1, max: 50000, step: 10, default: [100, 5000],
        showWhen: (v) => v.method === "standard" },
      // OpenCV's blob detector enforces `0 < minCircularity` — start at
      // 0.01 instead of 0 so the slider can't produce an invalid config.
      { key: "minCircularity", label: "Min circularity", kind: "slider", min: 0.01, max: 1, step: 0.01, default: 0.30,
        showWhen: (v) => v.method === "standard" },
      { key: "blob_is_dark", label: "Dark blobs (vs light)", kind: "bool", default: true,
        showWhen: (v) => v.method === "standard" },
      { key: "use_clahe",    label: "CLAHE preprocessing",   kind: "bool", default: false,
        showWhen: (v) => v.method === "standard" },

      // Robust to lighting (cmd="mser") — Maximally Stable Extremal Regions
      { key: "delta",         label: "Stability range",     kind: "slider", min: 1, max: 20, step: 1, default: 5,
        showWhen: (v) => v.method === "lighting_tolerant" },
      // Note: the "area" key is shared between methods (different
      // defaults). showWhen disambiguates which one is active.
      { key: "area",          label: "Area (px², min..max)", kind: "range", min: 1, max: 100000, step: 10, default: [60, 14400],
        showWhen: (v) => v.method === "lighting_tolerant" },
      { key: "max_variation", label: "Max area variation",  kind: "slider", min: 0.05, max: 1, step: 0.01, default: 0.25,
        showWhen: (v) => v.method === "lighting_tolerant" },
      { key: "min_diversity", label: "Min region separation", kind: "slider", min: 0, max: 1, step: 0.01, default: 0.2,
        showWhen: (v) => v.method === "lighting_tolerant" },
      { key: "nms_iou",       label: "Overlap threshold",    kind: "slider", min: 0.05, max: 1, step: 0.01, default: 0.5,
        showWhen: (v) => v.method === "lighting_tolerant" },
      { key: "blob_is_dark",  label: "Dark blobs (vs light)", kind: "bool", default: true,
        showWhen: (v) => v.method === "lighting_tolerant" },
    ],
  },
  // `elp` is a UI-only union of two backend cmds: edge-based EDLines
  // (cmd="elp") and contour-fit (cmd="elp_fit"). The `method` field is
  // transient — it's not sent as a kwarg; instead the field-change
  // handler rewrites _cfg.detection.cmd between "elp" and "elp_fit"
  // via methodMap. showWhen hides parameters that don't apply.
  elp: {
    label: "Ellipse",
    methodMap: { edge: "elp", fit: "elp_fit" },
    fields: [
      { key: "method", label: "Method", kind: "select", transient: true,
        options: [
          { v: "edge", t: "Edge-based" },
          { v: "fit",  t: "Contour fit" },
        ], default: "edge" },

      // Edge-based (cmd="elp")
      { key: "pf_mode",                 label: "Auto detection",         kind: "bool",   default: false,
        showWhen: (v) => v.method === "edge" },
      { key: "nfa_validation",          label: "False alarm validation", kind: "bool",   default: true,
        showWhen: (v) => v.method === "edge" },
      { key: "min_path_length",         label: "Min path length",        kind: "slider", min: 1,  max: 1000, step: 1, default: 50,
        showWhen: (v) => v.method === "edge" },
      { key: "min_line_length",         label: "Min line length",        kind: "slider", min: 1,  max: 1000, step: 1, default: 10,
        showWhen: (v) => v.method === "edge" },
      { key: "sigma",                   label: "Blur",                   kind: "slider", min: 0,  max: 20,   step: 0.1, default: 1,
        showWhen: (v) => v.method === "edge" },
      { key: "gradient_threshold_value",label: "Gradient",               kind: "slider", min: 1,  max: 100,  step: 1, default: 20,
        showWhen: (v) => v.method === "edge" },

      // Contour fit (cmd="elp_fit")
      { key: "use_otsu",         label: "Otsu thresholding",       kind: "bool",   default: true,
        showWhen: (v) => v.method === "fit" },
      { key: "area_range",       label: "Area (px², min..max)",    kind: "range",  min: 1, max: 1000000, step: 100, default: [300, 500000],
        showWhen: (v) => v.method === "fit" },
      { key: "aspect_ratio_tol", label: "Aspect ratio tolerance",  kind: "slider", min: 0, max: 1, step: 0.01, default: 0.1,
        showWhen: (v) => v.method === "fit" },
      { key: "circularity_min",  label: "Min circularity",         kind: "slider", min: 0, max: 1, step: 0.01, default: 0.7,
        showWhen: (v) => v.method === "fit" },
      { key: "convexity_min",    label: "Min convexity",           kind: "slider", min: 0, max: 1, step: 0.01, default: 0.9,
        showWhen: (v) => v.method === "fit" },
      // RANSAC-wrapped fitEllipse — more robust to noisy contour points
      // than plain least-squares. Off = original cv.fitEllipse behavior.
      { key: "use_ransac",       label: "RANSAC robust fit",       kind: "bool",   default: true,
        showWhen: (v) => v.method === "fit" },
      { key: "ransac_iters",     label: "RANSAC iterations",       kind: "slider", min: 5, max: 200, step: 1, default: 40,
        showWhen: (v) => v.method === "fit" && v.use_ransac },
      { key: "ransac_tol",       label: "RANSAC inlier tol (px)",  kind: "slider", min: 0.5, max: 10, step: 0.1, default: 2.0,
        showWhen: (v) => v.method === "fit" && v.use_ransac },
    ],
  },
  aruco: {
    label: "ArUco",
    fields: [
      { key: "dictionary",    label: "Dictionary",         kind: "select", options: ARUCO_DICTS.map(v => ({v,t:v})), default: "DICT_4X4_100" },
      { key: "marker_length", label: "Marker length (mm)", kind: "slider", min: 1, max: 100, step: 0.1, default: 20 },
      { key: "refine",        label: "Refinement",         kind: "select", options: REFINE_OPTS.map(v => ({v,t:v})), default: "CORNER_REFINE_APRILTAG" },
      { key: "subpix",        label: "Sub pixel",          kind: "bool",   default: false },
    ],
  },
  charuco: {
    label: "ChArUco",
    fields: [
      { key: "sqr_x",         label: "Squares X", kind: "number", default: 7 },
      { key: "sqr_y",         label: "Squares Y", kind: "number", default: 7 },
      { key: "sqr_length",    label: "Square length (mm)",  kind: "number", default: 30 },
      { key: "marker_length", label: "Marker length (mm)",  kind: "number", default: 24 },
      { key: "dictionary",    label: "Dictionary", kind: "select", options: ARUCO_DICTS.map(v => ({v,t:v})), default: "DICT_5X5_1000" },
    ],
  },
  barcode: {
    label: "Barcode",
    fields: [
      { key: "format", label: "Format", kind: "select",
        options: ["Any","Aztec","Codabar","Code39","Code93","Code128","DataMatrix","DataBar","DataBarExpanded","DataBarLimited","DXFilmEdge","EAN8","EAN13","ITF","PDF417","QRCode","MicroQRCode","RMQRCode","UPCA","UPCE","LinearCodes","MaxiCode","MatrixCodes"].map(v => ({v, t: v})),
        default: "Any" },
    ],
  },
  ocr: {
    label: "OCR",
    fields: [
      { key: "conf", label: "Confidence", kind: "slider", min: 0.01, max: 1, step: 0.01, default: 0.1 },
    ],
  },
  od: {
    label: "Object Detection (ML)",
    fields: [
      { key: "conf", label: "Confidence",  kind: "slider", min: 0.01, max: 1, step: 0.01, default: 0.3 },
      { key: "cls",  label: "Detection classes", kind: "json", default: [], keepEmpty: true },
    ],
  },
  rod: {
    label: "Rotated Object Detection (ML)",
    fields: [
      { key: "conf", label: "Confidence",  kind: "slider", min: 0.01, max: 1, step: 0.01, default: 0.15 },
      { key: "cls",  label: "Detection classes", kind: "json", default: [], keepEmpty: true },
    ],
  },
  anom: {
    label: "Anomaly Detection (ML)",
    fields: [
      { key: "threshold", label: "Score threshold (0 = mark all fail, 1 = mark all pass)",
        kind: "slider", min: 0, max: 1, step: 0.001, default: 0.5 },
    ],
  },
  cls: {
    label: "Classification (ML)",
    fields: [
      { key: "conf", label: "Confidence",  kind: "slider", min: 0.01, max: 1, step: 0.01, default: 0.5 },
    ],
  },
  kp: {
    label: "Keypoints (ML)",
    fields: [
      // Per-keypoint confidence threshold — keypoints below this are
      // dropped from the result. There's no object-class filter
      // anymore: the new top-down kp flow runs once over the whole
      // image and returns a single keypoint set, not per-object lists.
      { key: "conf",    label: "Confidence",       kind: "slider", min: 0.01, max: 1, step: 0.01, default: 0.2 },
      // Object bbox stored as 4 corner points (same format as ROI). An
      // empty list means "use whole image" — the backend converts the
      // corners to an axis-aligned (x, y, w, h) rect before passing to
      // KP. Picking a tight bbox + leaving padding at 1.25 reproduces
      // the training-time framing for best accuracy.
      { key: "bbox",    label: "Bounding box (4 corners)", kind: "json", picker: "bbox", default: [], keepEmpty: true },
      // Affine-crop margin around the bbox. 1.0 = no padding (tight
      // crop), 1.25 = training-time default (12.5% extra on each side).
      // Usually fixed at 1.25 once you have a real bbox; tune only if
      // your training conventions differ.
      { key: "padding", label: "Crop padding (×)", kind: "slider", min: 1.0,  max: 2.0, step: 0.05, default: 1.25 },
    ],
  },
};

// ── Detection-wide section schemas ────────────────────────────────
// `tab` chooses the builder tab. `key` is a dotted path into _cfg
// (e.g. "limit.bb" → _cfg.limit.bb.{field}). `null` means top-level.
const SECTION_SCHEMAS = [
  // Initialization
  {
    tab: "init", key: null, label: "Frame",
    desc: "Specify the reference frame relative to the robot's base (eye-in-hand) or the camera (eye-to-hand). All measurements are reported with respect to this frame.",
    fields: [
      { key: "frame", label: "x · y · z · a · b · c", kind: "vec6", default: [0,0,0,0,0,0] },
    ],
  },

  // Image
  {
    tab: "image", key: null, label: "Orientation", enable: true,
    desc: "Rotate the source image clockwise as needed.",
    fields: [
      { key: "rot",  label: "Clockwise rotation", kind: "select",
        options: [{v:0,t:"No rotation"},{v:90,t:"90°"},{v:180,t:"180°"},{v:270,t:"270°"}], default: 0 },
    ],
  },
  {
    tab: "image", key: "roi", label: "Region of Interest", enable: true,
    desc: "Restrict detection to a sub-area of the image. Use the polygon selector on the output image to define the area.",
    fields: [
      { key: "corners", label: "Corners (polygon)", kind: "json", picker: "polygon", default: [] },
      { key: "offset",  label: "Offset (px)", kind: "slider", min: -200, max: 200, step: 1, default: 0 },
      { key: "inv",     label: "Invert region", kind: "bool", default: false, asInt: true },
      { key: "crop",    label: "Crop region",   kind: "bool", default: false, asInt: true },
    ],
  },
  {
    tab: "image", key: "intensity", label: "Intensity", enable: true,
    desc: "Adjust brightness and contrast to enhance image details before detection runs.",
    fields: [
      { key: "a", label: "Contrast (a)",  kind: "slider", min: 0, max: 4,   step: 0.01, default: 1 },
      { key: "b", label: "Brightness (b)", kind: "slider", min: -255, max: 255, step: 1, default: 0 },
    ],
  },
  {
    tab: "image", key: "color", label: "Color Mask", enable: true,
    desc: "Filter the image to a hue/saturation/value range. Use this to isolate a specific colour before detection.",
    fields: [
      { key: "low_hsv",  label: "Low  H · S · V",  kind: "vec3", min: 0, max: 255, default: [0, 0, 0] },
      { key: "high_hsv", label: "High H · S · V",  kind: "vec3", min: 0, max: 255, default: [255, 255, 255] },
      { key: "inv",      label: "Invert mask", kind: "bool", default: false, asInt: true },
    ],
  },

  // Setting
  {
    tab: "setting", key: "sort", label: "Sorting Result", enable: true,
    desc: "Arrange detected objects by confidence, pixel distance, bounding-box area, or 3D distance.",
    fields: [
      { key: "cmd", label: "Sort by", kind: "select",
        options: [
          {v:"",t:"No sorting"},{v:"shuffle",t:"Random shuffle"},{v:"conf",t:"Confidence"},
          {v:"area",t:"Bounding box area"},{v:"pxl",t:"Pixel distance"},{v:"xyz",t:"xyz distance"},
        ], default: "" },
      { key: "max_det",   label: "Max detections per run", kind: "slider", min: 1, max: 200, step: 1, default: 100,
        showWhen: (v) => v.cmd !== "" },
      { key: "ascending", label: "Ascending order",        kind: "bool",   default: false,
        showWhen: (v) => ["conf","area","pxl","xyz"].includes(v.cmd) },
      { key: "pxl",       label: "Pixel target [width, height]", kind: "json", default: [],
        showWhen: (v) => v.cmd === "pxl" },
      { key: "xyz",       label: "xyz target (mm) [x, y, z]",    kind: "json", default: [],
        showWhen: (v) => v.cmd === "xyz" },
    ],
  },
  {
    tab: "setting", key: "limit.bb", label: "Bounding Box Limits", enable: true,
    desc: "Filter detected objects by their bounding-box shape (aspect ratio) and size (area in pixels²).",
    fields: [
      { key: "area",         label: "Area (px²)",   kind: "range", min: 0, max: 100000, step: 100, default: [0, 100000] },
      { key: "aspect_ratio", label: "Aspect ratio", kind: "range", min: 0, max: 1,     step: 0.01, default: [0, 1] },
    ],
  },
  {
    tab: "setting", key: "limit.center", label: "Center Pixel Limits", enable: true,
    desc: "Restrict detections to those whose bounding-box center falls inside the given pixel area.",
    fields: [
      { key: "width",  label: "Width (px)",  kind: "range", min: 0, max: 2000, step: 1, default: [0, 2000] },
      { key: "height", label: "Height (px)", kind: "range", min: 0, max: 2000, step: 1, default: [0, 2000] },
      { key: "inv",    label: "Invert range", kind: "bool", default: false, asInt: true },
    ],
  },
  {
    tab: "setting", key: "limit.xyz", label: "XYZ Limits", enable: true,
    desc: "Apply 3D constraints to remove detections outside the specified x, y, z range relative to the frame.",
    fields: [
      { key: "x", label: "x (mm)", kind: "range", min: -1000, max: 1000, step: 1, default: [-1000, 1000] },
      { key: "y", label: "y (mm)", kind: "range", min: -1000, max: 1000, step: 1, default: [-1000, 1000] },
      { key: "z", label: "z (mm)", kind: "range", min: -1000, max: 1000, step: 1, default: [-1000, 1000] },
      { key: "inv", label: "Invert range", kind: "bool", default: false, asInt: true },
    ],
  },
  // 6D Pose, Rotation Vector Limits, and Translation Vector Limits
  // sections used to live here. They moved out of the run() pipeline
  // and into opt-in helper methods on the Detection object:
  //   detection.pose_plane(result, plane_geometry)
  //   detection.pose_kp(kp_list, kp_geometry)
  //   detection.filter_rvec(results, rvec_base=..., x_angle=..., ...)
  //   detection.filter_tvec(results, x=..., y=..., z=..., inv=...)
  // Call them on the result list after run() — same pattern as
  // detection.grasp(...). ArUco / ChArUco still emit rvec/tvec
  // automatically because pose is intrinsic to those detectors.
  {
    tab: "setting", key: "display", label: "Display", enable: true,
    desc: "Configure how inference results are presented and saved.",
    fields: [
      { key: "label", label: "Display style", kind: "select",
        options: [
          { v: -1, t: "None" },
          { v:  0, t: "Bounding box only" },
          { v:  1, t: "Bounding box and labels" },
        ], default: 0 },
      { key: "save_img",     label: "Save annotated image",       kind: "save_path", default: 0, placeholder: "output/<timestamp>.jpg" },
      { key: "save_img_roi", label: "Save unannotated ROI image", kind: "save_path", default: 0, placeholder: "output/roi_<timestamp>.jpg" },
    ],
  },
];

// ── Helpers ────────────────────────────────────────────────────────

const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

function escHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
// Compact-but-readable JSON: same as JSON.stringify (no newlines), but
// with a space after each `,` and `:`. Used for the json textareas so a
// list like ["a","b","c"] reads as ["a", "b", "c"].
function _jsonCompact(v) {
  if (Array.isArray(v)) return "[" + v.map(_jsonCompact).join(", ") + "]";
  if (v && typeof v === "object") {
    const keys = Object.keys(v);
    return "{" + keys.map(k => `${JSON.stringify(k)}: ${_jsonCompact(v[k])}`).join(", ") + "}";
  }
  return JSON.stringify(v);
}
function toast(msg, kind = "ok") {
  const area = $("#toastArea");
  if (!area) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  area.appendChild(el);
  setTimeout(() => el.remove(), 3500);
  el.addEventListener("click", () => el.remove());
}
function isPageActive() {
  const sec = document.querySelector('section[data-page="playground"]');
  return !!sec && sec.classList.contains("active");
}
function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

// ── Dotted-path helpers (for nested config like limit.bb) ──────────
function getPath(obj, path) {
  if (!path) return obj;
  return path.split(".").reduce((o, k) => (o && typeof o === "object") ? o[k] : undefined, obj);
}
function ensurePath(obj, path) {
  if (!path) return obj;
  const parts = path.split(".");
  let o = obj;
  for (const k of parts) {
    if (!o[k] || typeof o[k] !== "object" || Array.isArray(o[k])) o[k] = {};
    o = o[k];
  }
  return o;
}
function deletePath(obj, path) {
  if (!path) return;
  const parts = path.split(".");
  const trail = [];
  let o = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!o[parts[i]] || typeof o[parts[i]] !== "object") return;
    trail.push([o, parts[i]]);
    o = o[parts[i]];
  }
  delete o[parts[parts.length - 1]];
  while (trail.length) {
    const [parent, key] = trail.pop();
    if (parent[key] && typeof parent[key] === "object" && !Array.isArray(parent[key]) && Object.keys(parent[key]).length === 0) {
      delete parent[key];
    } else break;
  }
}

// ── State ──────────────────────────────────────────────────────────

let _vc = null;

// _cfg = canonical Detection kwargs. Form view writes to it; Raw JSON view
// shows JSON.stringify(_cfg). On view switch from Raw → Form, the JSON is
// parsed back into _cfg.
// Default: no detection method selected (matches gui.py "No detection").
let _cfg = {};

let _live = false;
let _liveTickRunning = false;
let _currentImgTab = "img";
let _currentView = "method";   // active builder tab — "method" | "pre" | "post" | "json"
let _imgObjUrl = null;
let _fpsTimes = [];
let _hasRun = false;       // becomes true after first successful Run; gates image fetches

// File pickers hold a reference to the picked File. Bytes are read fresh
// on each Run/Initialize and shipped inline with the call — no separate
// upload step, no server-side staging dir. Refs cleared on Re-init or
// when the user picks a different file.
let _pickedImageFile = null;
let _pickedModelFile = null;

// Class names known to whichever ML model the user just initialized.
// Populated from Detection.classes() on Initialize; used to seed the
// per-method `cls` filter field in CMD_SCHEMAS so the user sees what's
// available rather than guessing the names.
let _modelClasses = [];
// ML method that's actually loaded into the Detection. Set at Initialize
// from AI Models. Used to filter the runtime method dropdown so users
// can't pick an ML method that wasn't loaded — that would crash the
// server with "'Detection' object has no attribute '<x>'".
let _loadedMlType = "";
// Trained anomaly threshold pulled from the ANOM pickle on Initialize.
// Used to seed the threshold slider so users start at the value the
// training pipeline picked, instead of the schema's static 0.5.
let _trainedThreshold = null;

const ML_CMDS = ["od", "rod", "cls", "kp", "anom"];

const SOURCE_KEY      = "dorna_playground_source";
// SOURCE_PATH_KEY removed — we no longer persist the file path; uploads
// land in a session-scoped server temp dir and don't survive reconnects.

// UX state
let _lastValid = [];                  // latest detection_run results (for highlight + sparkline)
let _selectedDetId = null;            // id of clicked-on detection row
const _zoom = { s: 1, tx: 0, ty: 0 }; // image stage transform
const _eyedrop = { active: false, target: null };  // 'low' | 'high'

// ── Renderers ─────────────────────────────────────────────────────

function fieldId(section, key) { return `pgF_${section}_${key}`.replace(/[^a-zA-Z0-9_]/g, "_"); }

function renderField(sectionKey, field, value) {
  const id = fieldId(sectionKey, field.key);
  // Compound widgets put the id on a wrapper div, not on a single form
  // element. Don't bind <label for=…> to those — browser warns otherwise.
  const isCompound = field.kind === "vec3" || field.kind === "vec6" || field.kind === "range";
  const labelFor = isCompound ? "" : ` for="${id}"`;
  const lbl = `<label${labelFor} class="pg-flabel">${escHtml(field.label)}</label>`;
  let ctrl = "";
  switch (field.kind) {
    case "bool": {
      const checked = !!value ? "checked" : "";
      ctrl = `<label class="pg-switch"><input id="${id}" type="checkbox" ${checked}/><span></span></label>`;
      break;
    }
    case "select": {
      // Fall back to the schema default when no value is set, so a select
      // visually reflects its default (e.g. display.label = 0 → "Bounding
      // box only") instead of the browser's first-option default.
      const effective = (value === undefined || value === null) ? field.default : value;
      const opts = field.options.map(o => {
        const v = (o.v === null || o.v === undefined) ? "" : String(o.v);
        const sel = String(effective ?? "") === v ? "selected" : "";
        return `<option value="${escHtml(v)}" ${sel}>${escHtml(o.t)}</option>`;
      }).join("");
      ctrl = `<select class="input pg-input" id="${id}">${opts}</select>`;
      break;
    }
    case "number": {
      const v = (value === null || value === undefined) ? "" : value;
      ctrl = `<input class="input pg-input" type="number" id="${id}" value="${escHtml(v)}"/>`;
      break;
    }
    case "text": {
      ctrl = `<input class="input pg-input" type="text" id="${id}" value="${escHtml(value ?? "")}"/>`;
      break;
    }
    case "slider": {
      const v = (value === null || value === undefined) ? field.default : value;
      ctrl = `<div class="pg-slider-row">
        <input class="pg-slider" id="${id}" name="${id}" type="range" min="${field.min}" max="${field.max}" step="${field.step}" value="${v}"/>
        <input class="input pg-slider-num" name="${id}_n" type="number" min="${field.min}" max="${field.max}" step="${field.step}" value="${v}" data-for="${id}"/>
      </div>`;
      break;
    }
    case "vec3": {
      const v = Array.isArray(value) && value.length === 3 ? value : (Array.isArray(field.default) && field.default.length === 3 ? field.default : [0,0,0]);
      const mk = (i) => `<input class="input pg-input pg-vec-cell" name="${id}_${i}" type="number" min="${field.min ?? ""}" max="${field.max ?? ""}" value="${v[i]}" data-vec-cell="${i}" data-vec-id="${id}"/>`;
      ctrl = `<div class="pg-vec-row pg-vec3" id="${id}">${mk(0)}${mk(1)}${mk(2)}</div>`;
      break;
    }
    case "vec6": {
      const v = Array.isArray(value) && value.length === 6 ? value : (Array.isArray(field.default) && field.default.length === 6 ? field.default : [0,0,0,0,0,0]);
      const mk = (i) => `<input class="input pg-input pg-vec-cell" name="${id}_${i}" type="number" value="${v[i]}" data-vec-cell="${i}" data-vec-id="${id}"/>`;
      ctrl = `<div class="pg-vec-row pg-vec6" id="${id}">${mk(0)}${mk(1)}${mk(2)}${mk(3)}${mk(4)}${mk(5)}</div>`;
      break;
    }
    case "range": {
      // [ number ] [ ─── dual-handle slider ─── ] [ number ]   on one line.
      const v = Array.isArray(value) && value.length === 2 ? value : [field.min, field.max];
      const lo = v[0], hi = v[1];
      const min = field.min, max = field.max, step = field.step ?? 1;
      ctrl = `<div class="pg-rs-wrap">
        <input type="number" name="${id}_lo_n" class="input pg-rs-num pg-rs-nlow"  min="${min}" max="${max}" step="${step}" value="${lo}"/>
        <div class="pg-rs" id="${id}" data-min="${min}" data-max="${max}" data-step="${step}">
          <div class="pg-rs-track"></div>
          <div class="pg-rs-fill"></div>
          <input type="range" name="${id}_lo" min="${min}" max="${max}" step="${step}" value="${lo}" class="pg-rs-low"/>
          <input type="range" name="${id}_hi" min="${min}" max="${max}" step="${step}" value="${hi}" class="pg-rs-high"/>
        </div>
        <input type="number" name="${id}_hi_n" class="input pg-rs-num pg-rs-nhigh" min="${min}" max="${max}" step="${step}" value="${hi}"/>
      </div>`;
      break;
    }
    case "json": {
      const v = value === undefined ? field.default : value;
      let extra = "";
      if (field.picker === "polygon" || field.picker === "bbox") {
        // The action row is rendered by _renderRoiActions(targetId) and
        // re-rendered whenever state changes (no polygon / has polygon /
        // editing). Single delegated click handler in init(). The
        // picker mode rides on the action div so handlers can switch
        // between free-form polygon and axis-aligned rectangle UX.
        extra = `<div class="pg-roi-actions" data-roi-actions="${id}" data-pick-mode="${field.picker}"></div>`;
      }
      ctrl = `<textarea class="input pg-input pg-json-mini" id="${id}" rows="3" spellcheck="false">${escHtml(_jsonCompact(v))}</textarea>${extra}`;
      break;
    }
    case "save_path": {
      // Switch + optional custom path. Empty path → server uses the default
      // output/<timestamp>.jpg name; non-empty path → that path is used.
      const enabled = !!value;
      const pathStr = typeof value === "string" ? value : "";
      const checked = enabled ? "checked" : "";
      const ph = field.placeholder || "output/<timestamp>.jpg";
      ctrl = `<div class="pg-savepath-row">
        <label class="pg-switch"><input id="${id}" type="checkbox" ${checked}/><span></span></label>
        <input class="input pg-input pg-savepath-in" id="${id}_path" type="text" placeholder="${escHtml(ph)}" value="${escHtml(pathStr)}" ${enabled ? "" : "disabled"}/>
      </div>`;
      break;
    }
    default:
      ctrl = `<input class="input pg-input" type="text" id="${id}" value="${escHtml(value ?? "")}"/>`;
  }
  return `<div class="pg-field">${lbl}${ctrl}</div>`;
}

function readField(sectionKey, field) {
  const id = fieldId(sectionKey, field.key);
  const el = document.getElementById(id);
  if (!el) return field.default;
  switch (field.kind) {
    case "bool": {
      const v = el.checked;
      return field.asInt ? (v ? 1 : 0) : v;
    }
    case "select": {
      let v = el.value;
      // Coerce numeric-looking option values back to numbers when defaults are numeric
      if (typeof field.default === "number" && v !== "") return Number(v);
      return v;
    }
    case "number": {
      if (el.value === "") return null;
      const n = Number(el.value);
      return Number.isNaN(n) ? null : n;
    }
    case "text":   return el.value;
    case "slider": return Number(el.value);
    case "vec3":
    case "vec6": {
      const cells = $$(`[data-vec-id="${id}"]`).sort((a,b) => Number(a.dataset.vecCell) - Number(b.dataset.vecCell));
      return cells.map(c => Number(c.value));
    }
    case "range": {
      const wrap = document.getElementById(id);
      if (!wrap) return field.default;
      const lo = Number(wrap.querySelector(".pg-rs-low").value);
      const hi = Number(wrap.querySelector(".pg-rs-high").value);
      return [Math.min(lo, hi), Math.max(lo, hi)];
    }
    case "json": {
      // Blank textarea ≠ empty array. Blank means "not provided"
      // (returns undefined, which the field-write logic drops from
      // _cfg). An explicitly typed `[]` stays as an empty list.
      const raw = (el.value || "").trim();
      if (!raw) return undefined;
      try { return JSON.parse(raw); } catch { return field.default; }
    }
    case "save_path": {
      if (!el.checked) return null;     // off → drop from cfg
      const path = (document.getElementById(id + "_path")?.value ?? "").trim();
      return path ? path : 1;           // 1 = use server default name
    }
  }
  return el.value;
}

function wireFieldEvents(sectionKey, field, onChange) {
  const id = fieldId(sectionKey, field.key);
  const el = document.getElementById(id);
  if (!el) return;
  if (field.kind === "slider") {
    const num = document.querySelector(`[data-for="${id}"]`);
    el.addEventListener("input", () => { if (num) num.value = el.value; onChange(); });
    if (num) num.addEventListener("input", () => { el.value = num.value; onChange(); });
  } else if (field.kind === "vec3" || field.kind === "vec6") {
    $$(`[data-vec-id="${id}"]`).forEach(c => c.addEventListener("input", onChange));
  } else if (field.kind === "range") {
    const wrap = document.getElementById(id);
    if (wrap) {
      const lo   = wrap.querySelector(".pg-rs-low");
      const hi   = wrap.querySelector(".pg-rs-high");
      const fill = wrap.querySelector(".pg-rs-fill");
      const outer = wrap.parentElement;       // .pg-rs-wrap (slider + two number inputs)
      const nlo  = outer?.querySelector(".pg-rs-nlow");
      const nhi  = outer?.querySelector(".pg-rs-nhigh");
      const min  = Number(wrap.dataset.min);
      const max  = Number(wrap.dataset.max);
      const clamp = n => Math.max(min, Math.min(max, n));
      const update = (src) => {
        let l, h;
        if (src === "num") { l = Number(nlo.value); h = Number(nhi.value); }
        else               { l = Number(lo.value);  h = Number(hi.value);  }
        if (Number.isNaN(l)) l = min;
        if (Number.isNaN(h)) h = max;
        l = clamp(l); h = clamp(h);
        if (l > h) { if (src === "num") l = h; else h = l; }
        lo.value  = String(l);  hi.value  = String(h);
        if (nlo)  nlo.value = String(l);
        if (nhi)  nhi.value = String(h);
        const range = max - min || 1;
        if (fill) {
          fill.style.left  = ((l - min) / range * 100) + "%";
          fill.style.width = ((h - l) / range * 100) + "%";
        }
      };
      lo.addEventListener("input",  () => { update("slider"); onChange(); });
      hi.addEventListener("input",  () => { update("slider"); onChange(); });
      nlo?.addEventListener("input", () => { update("num");    onChange(); });
      nhi?.addEventListener("input", () => { update("num");    onChange(); });
      update("slider");
    }
  } else if (field.kind === "json") {
    el.addEventListener("change", onChange);
    el.addEventListener("blur",   onChange);
  } else if (field.kind === "save_path") {
    const pathEl = document.getElementById(id + "_path");
    el.addEventListener("change", () => {
      if (pathEl) pathEl.disabled = !el.checked;
      onChange();
    });
    pathEl?.addEventListener("input", onChange);
  } else {
    el.addEventListener("change", onChange);
    if (field.kind === "text" || field.kind === "number") el.addEventListener("input", onChange);
  }
}

// ── Method (cmd) selector + per-cmd fields ────────────────────────

// Some schemas unify multiple backend cmds under a single dropdown
// entry — e.g. Blob covers both `blob` and `mser`, Ellipse covers
// `elp` and `elp_fit`. They declare a `methodMap` mapping a UI method
// value → backend cmd. These helpers resolve in either direction.
//
// schemaForCmd("mser")     → "blob"   (parent schema key)
// methodForCmd("elp_fit")  → "fit"    (which `method` option matches)
function schemaForCmd(cmd) {
  if (CMD_SCHEMAS[cmd]) return cmd;
  for (const [k, s] of Object.entries(CMD_SCHEMAS)) {
    if (s.methodMap && Object.values(s.methodMap).includes(cmd)) return k;
  }
  return cmd;
}
function methodForCmd(cmd) {
  for (const s of Object.values(CMD_SCHEMAS)) {
    if (!s.methodMap) continue;
    for (const [m, c] of Object.entries(s.methodMap)) {
      if (c === cmd) return m;
    }
  }
  return null;
}
// Set of cmds that are folded into a parent schema's methodMap and
// therefore should NOT appear as separate dropdown entries.
function _foldedCmds() {
  const out = new Set();
  for (const [k, s] of Object.entries(CMD_SCHEMAS)) {
    if (!s.methodMap) continue;
    for (const c of Object.values(s.methodMap)) {
      if (c !== k) out.add(c);
    }
  }
  return out;
}

function renderMethodPicker() {
  const sel = $("#pgMethod");
  if (!sel) return;
  // ML methods only show up in the picker for the type that was actually
  // loaded into the Detection at Initialize time — picking a different
  // ML method would crash with "'Detection' object has no attribute '<x>'".
  // Same idea as gui.py:610-616 appending the matching ML option only
  // after init_od / init_cls / init_kp. Pre-init we hide ALL ML methods.
  const folded = _foldedCmds();
  let html = `<option value="">No detection</option>`;
  for (const [k, s] of Object.entries(CMD_SCHEMAS)) {
    if (ML_CMDS.includes(k) && k !== _loadedMlType) continue;
    // Skip cmds that live under a parent schema's method selector.
    if (folded.has(k)) continue;
    html += `<option value="${escHtml(k)}">${escHtml(s.label)}</option>`;
  }
  sel.innerHTML = html;
  // If the previously-saved cmd is no longer offered (e.g. Re-init with
  // a different ML type), drop the stale value rather than letting the
  // browser silently fall back to the first option.
  // For folded cmds (e.g. elp_fit, mser), the dropdown shows the parent
  // schema entry selected; the underlying cmd in _cfg.detection stays
  // as the folded value so the backend dispatches correctly.
  const rawCur = _cfg.detection?.cmd ?? "";
  const cur = schemaForCmd(rawCur);
  const valid = Array.from(sel.options).some(o => o.value === cur);
  sel.value = valid ? cur : "";
  if (!valid && _cfg.detection) {
    delete _cfg.detection.cmd;
    if (Object.keys(_cfg.detection).length === 0) delete _cfg.detection;
  }
}

function renderMethodFields() {
  const cmd = _cfg.detection?.cmd ?? "";
  const wrap = $("#pgMethodFields");
  if (!cmd) {
    wrap.innerHTML = `<div class="muted" style="padding:14px 4px;font-size:12px;text-align:center">Select a detection method above to configure its parameters.</div>`;
    return;
  }
  // Folded cmds (mser, elp_fit) share their parent's schema. Normalize
  // for lookup so the same form (with the method selector switched
  // appropriately) renders for both backend cmds.
  const schemaCmd = schemaForCmd(cmd);
  const schema = CMD_SCHEMAS[schemaCmd];
  if (!schema || !schema.fields.length) {
    wrap.innerHTML = `<div class="muted" style="padding:14px 4px;font-size:12px;text-align:center">This method has no parameters.</div>`;
    return;
  }
  const det = _cfg.detection || {};
  // Seed `cls` with the loaded model's classes when (a) the model
  // exposes some, (b) this method has a cls field, and (c) the user
  // hasn't filled it yet. Empty list = all classes (same as API
  // default) but pre-filling makes the available names discoverable.
  // Field kind is "json", so we pass the actual array — renderField
  // handles the stringification.
  const seeded = { ...det };
  const hasCls = Array.isArray(det.cls) ? det.cls.length > 0 : !!det.cls;
  if (_modelClasses.length && schema.fields.some(f => f.key === "cls") && !hasCls) {
    seeded.cls = _modelClasses.slice();
  }
  // Transient `method` field for unified schemas: UI-only, derived from
  // the current cmd via the schema's methodMap (reverse lookup).
  if (schema.methodMap) {
    const m = methodForCmd(cmd);
    if (m) seeded.method = m;
  }
  wrap.innerHTML = schema.fields.map(f => renderField("detection", f, seeded[f.key])).join("");

  // Populate action rows for any polygon / bbox picker fields the
  // schema declared. The renderField output leaves these divs empty
  // — _renderRoiActions fills them with the right Edit / Save / Pick
  // buttons based on current state. (Sections do the same dance in
  // renderSections; this is the parallel call site for cmd fields.)
  wrap.querySelectorAll("[data-roi-actions]").forEach(el => _renderRoiActions(el.dataset.roiActions));

  // Tack a little "All" button next to the cls textarea so users can
  // restore the full class list after accidentally erasing items.
  // Only visible when the loaded model actually has classes to restore.
  const clsField = schema.fields.find(f => f.key === "cls");
  if (clsField && _modelClasses.length) {
    const ta = document.getElementById(fieldId("detection", "cls"));
    const fieldEl = ta?.closest(".pg-field");
    const label = fieldEl?.querySelector(".pg-flabel");
    if (label) {
      const row = document.createElement("div");
      row.className = "pg-flabel-row";
      label.replaceWith(row);
      row.appendChild(label);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-ghost btn-sm";
      btn.textContent = "All";
      btn.title = "Restore the full class list from the loaded model";
      btn.addEventListener("click", () => {
        if (!ta) return;
        ta.value = _jsonCompact(_modelClasses.slice());
        ta.dispatchEvent(new Event("change", { bubbles: true }));
      });
      row.appendChild(btn);
    }
  }

  // Same idea for the ANOM threshold slider — a "Trained" button that
  // resets it to the value the training pipeline picked. Only shown
  // when the loaded model actually exposes a trained threshold.
  const thrField = schema.fields.find(f => f.key === "threshold");
  if (thrField && _trainedThreshold != null) {
    const sl = document.getElementById(fieldId("detection", "threshold"));
    const fieldEl = sl?.closest(".pg-field");
    const label = fieldEl?.querySelector(".pg-flabel");
    if (label) {
      const row = document.createElement("div");
      row.className = "pg-flabel-row";
      label.replaceWith(row);
      row.appendChild(label);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-ghost btn-sm";
      btn.textContent = "Trained";
      btn.title = `Reset to the trained threshold (${_trainedThreshold.toFixed(3)})`;
      btn.addEventListener("click", () => {
        if (!sl) return;
        sl.value = String(_trainedThreshold);
        sl.dispatchEvent(new Event("input", { bubbles: true }));
      });
      row.appendChild(btn);
    }
  }

  // Conditional field visibility — fields with showWhen get hidden when
  // their predicate fails. Re-checked after every edit so toggling the
  // method selector immediately swaps which parameters show.
  const _readMethodVals = () => {
    const out = {};
    for (const f of schema.fields) out[f.key] = readField("detection", f);
    return out;
  };
  const _applyShowWhen = () => {
    if (!schema.fields.some(f => typeof f.showWhen === "function")) return;
    const vals = _readMethodVals();
    for (const f of schema.fields) {
      if (typeof f.showWhen !== "function") continue;
      const fid = fieldId("detection", f.key);
      const fieldEl = document.getElementById(fid)?.closest(".pg-field");
      fieldEl?.toggleAttribute("hidden", !f.showWhen(vals, _cfg));
    }
  };
  _applyShowWhen();

  schema.fields.forEach(f => wireFieldEvents("detection", f, () => {
    const v = readField("detection", f);
    if (!_cfg.detection) _cfg.detection = {};
    // Transient fields don't live in _cfg.detection. For the `method`
    // selector on a unified schema, instead of writing detection.method
    // we rewrite detection.cmd to the backend cmd for the chosen method
    // (looked up via the schema's methodMap). Picker selection is set
    // to the parent schema key to keep the dropdown in sync — we don't
    // re-render the form, which would discard in-progress edits.
    if (f.transient) {
      if (f.key === "method" && schema.methodMap) {
        const newCmd = schema.methodMap[v];
        if (newCmd) {
          // Switching method = different backend cmd with different
          // expected kwargs. Rebuild the detection branch from scratch
          // so stale keys from the previous method (e.g. `threshold`
          // when leaving standard for lighting-tolerant) don't leak
          // through. Defaults are evaluated under the *new* method so
          // showWhen filters correctly.
          const newDet = { cmd: newCmd };
          const _vals = {};
          for (const f2 of schema.fields) {
            if (f2.default !== undefined) _vals[f2.key] = f2.default;
          }
          _vals.method = v;
          for (const f2 of schema.fields) {
            if (f2.keepEmpty) continue;
            if (f2.transient) continue;
            if (typeof f2.showWhen === "function" && !f2.showWhen(_vals, _cfg)) continue;
            if (f2.default !== null && f2.default !== undefined) newDet[f2.key] = f2.default;
          }
          _cfg.detection = newDet;
          const sel = $("#pgMethod");
          if (sel) sel.value = schemaCmd;
          // Re-render so the new method's fields appear with fresh
          // values (the previous render only had the OTHER method's
          // fields wired up).
          renderMethodFields();
          syncJson();
          return;
        }
      }
      _applyShowWhen();
      syncJson();
      return;
    }
    // Same three-state semantics as section fields: undefined/null/blank
    // drops the key. Empty arrays are dropped too unless the field opts
    // in via keepEmpty (cls=[] is meaningfully different from cls absent).
    const isEmpty = v === null || v === undefined || v === "" ||
      (!f.keepEmpty && Array.isArray(v) && v.length === 0);
    if (isEmpty) delete _cfg.detection[f.key];
    else _cfg.detection[f.key] = v;
    _applyShowWhen();
    syncJson();
  }));
}

// ── Sections ──────────────────────────────────────────────────────

function sectionId(sec) {
  return `pgS_${(sec.key || sec.label).replace(/[^a-zA-Z0-9_]/g, "_")}`;
}

// Stable identifier for sections — uses the dotted key when present
// (e.g. "limit.bb"), otherwise falls back to the label. Required because
// some sections (Orientation, Camera Mounting, …) have key:null and write
// directly into _cfg root, but still need an Apply/Reset handle.
function sectionSlug(sec) { return sec.key || sec.label; }
function _findSection(slug) { return SECTION_SCHEMAS.find(s => sectionSlug(s) === slug); }
function _isSectionApplied(sec) {
  if (sec.key) return !!getPath(_cfg, sec.key);
  return sec.fields.some(f => Object.prototype.hasOwnProperty.call(_cfg, f.key));
}

function renderSections() {
  const sectionsByTab = { init: [], image: [], setting: [] };
  for (const sec of SECTION_SCHEMAS) sectionsByTab[sec.tab]?.push(sec);

  for (const [tab, secs] of Object.entries(sectionsByTab)) {
    const host = document.getElementById(`pgSections${tab[0].toUpperCase() + tab.slice(1)}`);
    if (!host) continue;
    host.innerHTML = secs.map(sec => {
      const slug = sectionSlug(sec);
      const enabled = sec.enable ? _isSectionApplied(sec) : true;
      const enableHtml = sec.enable ? `
        <label class="pg-enable" title="Apply this section">
          <span class="pg-enable-text">Apply</span>
          <span class="pg-switch">
            <input type="checkbox" data-sec-enable="${escHtml(slug)}" ${enabled ? "checked" : ""}/>
            <span></span>
          </span>
        </label>` : "";
      const resetHtml = (sec.key || sec.enable) ? `
        <button type="button" class="pg-section-reset" data-section-reset="${escHtml(slug)}" title="Reset this section to defaults">Reset</button>` : "";
      return `
      <div class="pg-group is-collapsed" data-section-key="${escHtml(slug)}">
        <div class="pg-group-head" data-toggle role="button" tabindex="0" aria-expanded="false">
          <span class="pg-chevron">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
          </span>
          <span class="pg-group-title">${escHtml(sec.label)}</span>
          <span class="pg-section-acts">
            ${resetHtml}
            ${enableHtml}
          </span>
        </div>
        <div class="pg-fields" id="${sectionId(sec)}" hidden></div>
      </div>`;
    }).join("");

    // Wire collapse toggles for this tab
    host.querySelectorAll(".pg-group-head[data-toggle]").forEach(btn => {
      const toggle = () => {
        const group = btn.closest(".pg-group");
        const fields = group.querySelector(".pg-fields");
        const willExpand = group.classList.contains("is-collapsed");
        group.classList.toggle("is-collapsed", !willExpand);
        btn.setAttribute("aria-expanded", String(willExpand));
        if (fields) {
          if (willExpand) fields.removeAttribute("hidden");
          else            fields.setAttribute("hidden", "");
        }
      };
      btn.addEventListener("click", toggle);
      btn.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });
    });

    // Stop the Apply checkbox from also toggling the collapse
    host.querySelectorAll(".pg-enable").forEach(lbl => {
      lbl.addEventListener("click", e => e.stopPropagation());
    });

    // Wire Apply checkboxes per section (gates whether the section lands in _cfg)
    secs.filter(s => s.enable).forEach(sec => {
      const cb = host.querySelector(`input[data-sec-enable="${sectionSlug(sec)}"]`);
      if (!cb) return;
      const ns = sec.key || "_root";
      cb.addEventListener("change", () => {
        if (cb.checked) {
          // Re-materialize from current form values
          if (sec.key) {
            const target = ensurePath(_cfg, sec.key);
            for (const f of sec.fields) {
              const v = readField(ns, f);
              const isEmpty = v === null || v === "" || (Array.isArray(v) && v.length === 0);
              if (!isEmpty) target[f.key] = v;
            }
          } else {
            for (const f of sec.fields) {
              const v = readField(ns, f);
              const isEmpty = v === null || v === "" || (Array.isArray(v) && v.length === 0);
              if (!isEmpty) _cfg[f.key] = v;
            }
          }
        } else {
          if (sec.key) deletePath(_cfg, sec.key);
          else for (const f of sec.fields) delete _cfg[f.key];
        }
        syncJson();
      });
    });

    // (Pick-polygon buttons are wired once in init() via event delegation
    //  so they work regardless of when the field HTML lands.)

    secs.forEach(sec => {
      const wrap = document.getElementById(sectionId(sec));
      const branch = (sec.key ? (getPath(_cfg, sec.key) || {}) : _cfg);
      const ns = sec.key || "_root";
      const descHtml = sec.desc ? `<p class="pg-desc">${escHtml(sec.desc)}</p>` : "";
      wrap.innerHTML = descHtml + sec.fields.map(f => renderField(ns, f, branch[f.key])).join("");
      // For Color Mask, attach a small eyedrop icon button next to each HSV
      // field's own label so the action lives on the same row as its target.
      if (sec.key === "color") {
        for (const which of ["low", "high"]) {
          const fid = fieldId(ns, which + "_hsv");
          const fieldEl = document.getElementById(fid)?.closest(".pg-field");
          const label   = fieldEl?.querySelector(".pg-flabel");
          if (!label) continue;
          const row = document.createElement("div");
          row.className = "pg-flabel-row";
          label.replaceWith(row);
          row.appendChild(label);
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "btn btn-ghost btn-sm btn-icon";
          btn.dataset.eyedrop = which;
          btn.title = `Eyedrop ${which} HSV from image`;
          btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/><circle cx="12" cy="12" r="4"/></svg>`;
          row.appendChild(btn);
        }
      }
      // After field HTML is in the DOM, paint any ROI action rows for the
      // polygon-picker fields. They depend on current edit state + textarea value.
      wrap.querySelectorAll("[data-roi-actions]").forEach(el => _renderRoiActions(el.dataset.roiActions));
      // Conditional field visibility within a section — fields with showWhen
      // get hidden when their predicate fails. Re-checked after every edit.
      const _readSectionVals = () => {
        const out = {};
        for (const f of sec.fields) out[f.key] = readField(ns, f);
        return out;
      };
      const _applyShowWhen = () => {
        if (!sec.fields.some(f => typeof f.showWhen === "function")) return;
        const vals = _readSectionVals();
        for (const f of sec.fields) {
          if (typeof f.showWhen !== "function") continue;
          const fid = fieldId(ns, f.key);
          const fieldEl = document.getElementById(fid)?.closest(".pg-field");
          fieldEl?.toggleAttribute("hidden", !f.showWhen(vals, _cfg));
        }
      };
      _applyShowWhen();
      sec.fields.forEach(f => wireFieldEvents(ns, f, () => {
        // Auto-apply: if the user edits a field in a section that has an
        // Apply toggle but isn't applied yet, flip Apply on. Matches user
        // expectation that "I changed it → it takes effect" — without this,
        // editing rot/feed/etc. silently does nothing while Apply is off.
        if (sec.enable && !_isSectionApplied(sec)) {
          const cb = host.querySelector(`input[data-sec-enable="${sectionSlug(sec)}"]`);
          if (cb && !cb.checked) cb.checked = true;
        }
        const v = readField(ns, f);
        // Treat blank/undefined/null as "not provided" — drop the key.
        // Empty arrays / objects are normally also dropped (keeps the
        // JSON view tidy), unless the field opts in with keepEmpty:true,
        // which is meaningful for filter fields like cls=[] (= keep
        // nothing) vs cls absent (= keep all).
        const isEmpty = v === null || v === undefined || v === "" ||
          (!f.keepEmpty && Array.isArray(v) && v.length === 0) ||
          (!f.keepEmpty && v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length === 0);
        if (isEmpty) {
          if (sec.key) {
            const parent = getPath(_cfg, sec.key);
            if (parent && typeof parent === "object") {
              delete parent[f.key];
              if (Object.keys(parent).length === 0) deletePath(_cfg, sec.key);
            }
          } else {
            delete _cfg[f.key];
          }
        } else {
          if (sec.key) {
            const target = ensurePath(_cfg, sec.key);
            target[f.key] = v;
          } else {
            _cfg[f.key] = v;
          }
        }
        _applyShowWhen();   // re-evaluate per-field visibility after edits
        syncJson();
      }));
    });
  }
}

// ── Two-way binding with raw JSON ─────────────────────────────────

function syncJson() {
  $("#pgConfig").textContent    = JSON.stringify(_cfg, null, 2);
  $("#pgPySnippet").textContent = _buildPySnippet();
  _persistCfg();
  _refreshTabDots();
  _refreshImgTabs();
  refreshRoiOverlay();   // dim region depends on roi.inv — repaint on any change
  _applySourceUI();      // mount type toggles Robot card visibility
  _scheduleAutoRerun();  // live re-tune: re-run on the cached frame
}

// Render a JS value as a Python literal. Pretty-prints nested dicts/lists,
// inlining short ones. Indent is the column the value will start at (used
// to align continuations).
function _toPyLiteral(v, indent = 0) {
  if (v === null || v === undefined)     return "None";
  if (v === true)                        return "True";
  if (v === false)                       return "False";
  if (typeof v === "number")             return String(v);
  if (typeof v === "string")             return JSON.stringify(v);
  const pad = " ".repeat(indent);
  if (Array.isArray(v)) {
    if (v.length === 0) return "[]";
    const items = v.map(x => _toPyLiteral(x, indent + 2));
    if (items.every(s => !s.includes("\n")) && items.join(", ").length < 60) {
      return "[" + items.join(", ") + "]";
    }
    return "[\n" + items.map(s => pad + "  " + s).join(",\n") + "\n" + pad + "]";
  }
  if (typeof v === "object") {
    const keys = Object.keys(v);
    if (keys.length === 0) return "{}";
    const items = keys.map(k => `${JSON.stringify(k)}: ${_toPyLiteral(v[k], indent + 2)}`);
    if (items.every(s => !s.includes("\n")) && items.join(", ").length < 60) {
      return "{" + items.join(", ") + "}";
    }
    return "{\n" + items.map(s => pad + "  " + s).join(",\n") + "\n" + pad + "}";
  }
  return JSON.stringify(v);
}

function _buildPySnippet() {
  const source = _currentSource();
  const camSn = $("#pgCamera")?.value || "";
  const imgName  = _pickedImageFile?.name || "img/test.jpg";
  const lines = [];
  lines.push(`from dorna_vision_client import VisionClient`);
  lines.push(``);
  lines.push(`vc = VisionClient()`);
  lines.push(`vc.connect()  # default host="127.0.0.1", port=8765`);
  lines.push(``);
  lines.push(`vc.detection_add(`);
  lines.push(`    name="my_detection",`);
  if (source === "camera" && camSn) lines.push(`    camera_serial_number=${JSON.stringify(camSn)},`);
  for (const [k, v] of Object.entries(_cfg)) {
    lines.push(`    ${k}=${_toPyLiteral(v, 4)},`);
  }
  lines.push(`)`);
  lines.push(``);
  if (source === "file") {
    lines.push(`# A local file path on YOUR machine — the client reads the bytes`);
    lines.push(`# and ships them inline with the call; nothing is staged on the server.`);
    lines.push(`result = vc.detection("my_detection").run(data=${JSON.stringify(imgName)})`);
  } else {
    lines.push(`result = vc.detection("my_detection").run()`);
  }
  return lines.join("\n");
}

// Orchestrator copy pattern — checkmark on success, fallback for non-HTTPS.
function _copyToClipboard(text, btn) {
  const orig = btn.innerHTML;
  const ok = () => {
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--green)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
    setTimeout(() => { btn.innerHTML = orig; }, 1500);
  };
  const fallback = () => {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;left:-9999px;top:-9999px";
      document.body.appendChild(ta);
      ta.select();
      const success = document.execCommand("copy");
      document.body.removeChild(ta);
      if (success) ok(); else toast("Copy failed", "bad");
    } catch { toast("Copy failed", "bad"); }
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(ok).catch(fallback);
  } else {
    fallback();
  }
}

// ── Persistence + tab indicators ─────────────────────────────────

const STORAGE_KEY = "dorna_playground_cfg";
// Debounce localStorage writes — they're synchronous and slow, and a
// slider drag during Live mode can fire dozens of input events per
// second. Coalesce to one write per 250ms.
let _persistTimer = null;
function _persistCfg() {
  if (_persistTimer) return;
  _persistTimer = setTimeout(() => {
    _persistTimer = null;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(_cfg)); } catch {}
  }, 250);
}
function _loadStoredCfg() {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    return s ? JSON.parse(s) : null;
  } catch { return null; }
}
function _tabHasConfig(tab) {
  if (tab === "detection") return !!_cfg.detection?.cmd;
  if (tab === "json")      return false;
  for (const sec of SECTION_SCHEMAS) {
    if (sec.tab !== tab) continue;
    if (sec.key) {
      if (getPath(_cfg, sec.key) !== undefined) return true;
    } else {
      for (const f of sec.fields) if (_cfg[f.key] !== undefined) return true;
    }
  }
  return false;
}
// ── Reconnect banner ─────────────────────────────────────────────
function _setReconnectBanner(visible) {
  const el = $("#pgReconnect");
  if (!el) return;
  if (visible) el.removeAttribute("hidden");
  else         el.setAttribute("hidden", "");
}

// ── Clear all + per-section reset ─────────────────────────────────
function clearAll() {
  _cfg = {};
  _selectedDetId = null;
  syncForm();
  syncJson();
  refreshRoiOverlay();
  _renderHighlight();
  toast("Config cleared", "ok");
}
// Capture which section bodies are currently expanded, so callers that
// trigger a renderSections() rebuild (Reset, eyedropper, etc.) can restore
// them afterward instead of visually collapsing the user's open sections.
function _snapshotOpenSections() {
  return $$(".pg-group:not(.is-collapsed)[data-section-key]")
    .map(g => g.dataset.sectionKey)
    .filter(Boolean);
}
function _restoreOpenSections(keys) {
  for (const k of keys) {
    const g = document.querySelector(`.pg-group[data-section-key="${CSS.escape(k)}"]`);
    if (!g) continue;
    g.classList.remove("is-collapsed");
    g.querySelector(".pg-group-head")?.setAttribute("aria-expanded", "true");
    g.querySelector(".pg-fields")?.removeAttribute("hidden");
  }
}

function resetSection(slug) {
  if (!slug) return;
  const sec = _findSection(slug);
  if (!sec) return;
  const openKeys = _snapshotOpenSections();
  if (!openKeys.includes(slug)) openKeys.push(slug);
  if (sec.key) deletePath(_cfg, sec.key);
  else         for (const f of sec.fields) delete _cfg[f.key];
  syncForm();
  _restoreOpenSections(openKeys);
  syncJson();
  refreshRoiOverlay();
}

// ── Detection highlight (click row → outline on image) ───────────
function _renderHighlight() {
  const svg = $("#pgHiSvg");
  const img = $("#pgImg");
  if (!svg || !img || !img.naturalWidth) { if (svg) svg.innerHTML = ""; return; }
  // Position the SVG over the image (same trick as ROI)
  const wrap = $("#pgImgStage");
  const wrapR = wrap.getBoundingClientRect();
  const imgR  = img.getBoundingClientRect();
  svg.style.position = "absolute";
  svg.style.left   = (imgR.left - wrapR.left) + "px";
  svg.style.top    = (imgR.top  - wrapR.top)  + "px";
  svg.style.width  = imgR.width  + "px";
  svg.style.height = imgR.height + "px";
  svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);

  const det = _lastValid.find(d => d.id === _selectedDetId);
  if (!det || !Array.isArray(det.corners) || det.corners.length < 3) {
    svg.innerHTML = "";
    return;
  }
  const pts = det.corners.map(c => `${c[0]},${c[1]}`).join(" ");
  svg.innerHTML = `<polygon points="${pts}" class="pg-hi-poly"/>`;
}
function _selectDetection(id) {
  _selectedDetId = (_selectedDetId === id) ? null : id;
  // re-style result rows
  $$(".pg-table tr[data-det-id]").forEach(r => {
    r.classList.toggle("is-selected", String(r.dataset.detId) === String(_selectedDetId));
  });
  _renderHighlight();
}

// ── Image zoom & pan ─────────────────────────────────────────────
function _applyImgTransform() {
  const stage = $("#pgImgStage");
  if (!stage) return;
  stage.style.transform = `translate(${_zoom.tx}px, ${_zoom.ty}px) scale(${_zoom.s})`;
  stage.style.transformOrigin = "0 0";
  $("#pgZoomReset")?.toggleAttribute("hidden", _zoom.s === 1 && _zoom.tx === 0 && _zoom.ty === 0);
  // Re-position overlays after transform changes
  refreshRoiOverlay();
  _renderHighlight();
}
function _resetZoom() {
  _zoom.s = 1; _zoom.tx = 0; _zoom.ty = 0;
  _applyImgTransform();
}
function _zoomAt(deltaY, clientX, clientY) {
  const wrap = $("#pgImgWrap");
  if (!wrap) return;
  const r = wrap.getBoundingClientRect();
  const x = clientX - r.left, y = clientY - r.top;
  const scaleFactor = deltaY < 0 ? 1.15 : 1 / 1.15;
  const newS = Math.max(1, Math.min(8, _zoom.s * scaleFactor));
  // Keep cursor position stable in image coords
  _zoom.tx = x - (x - _zoom.tx) * (newS / _zoom.s);
  _zoom.ty = y - (y - _zoom.ty) * (newS / _zoom.s);
  _zoom.s = newS;
  if (_zoom.s === 1) { _zoom.tx = 0; _zoom.ty = 0; }
  _applyImgTransform();
}

// ── HSV eyedropper ───────────────────────────────────────────────
function _rgbToHsv(r, g, b) {
  // Returns OpenCV-style H (0–179), S/V (0–255)
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const v = max;
  const s = max === 0 ? 0 : (max - min) / max;
  let h = 0;
  const d = max - min;
  if (d !== 0) {
    if (max === r)      h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else                h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return [Math.round(h / 2), Math.round(s * 255), Math.round(v * 255)];
}
function _startEyedrop(target) {
  const img = $("#pgImg");
  if (!img || !img.naturalWidth) { toast("Capture an image first (click Run once)", "warn"); return; }
  _eyedrop.active = true;
  _eyedrop.target = target;
  $("#pgImgWrap")?.classList.add("is-eyedrop");
  toast(`Click a pixel on the image to set "${target}"`, "ok");
}
function _onImgEyedrop(ev) {
  if (!_eyedrop.active) return;
  const img = $("#pgImg");
  const r = img.getBoundingClientRect();
  const x = (ev.clientX - r.left) * img.naturalWidth  / r.width;
  const y = (ev.clientY - r.top)  * img.naturalHeight / r.height;
  if (x < 0 || y < 0 || x > img.naturalWidth || y > img.naturalHeight) return;
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  c.getContext("2d").drawImage(img, 0, 0);
  let px;
  try { px = c.getContext("2d").getImageData(Math.floor(x), Math.floor(y), 1, 1).data; }
  catch (e) { toast("Browser blocked pixel read (CORS): " + e.message, "bad"); _endEyedrop(); return; }
  const [h, s, v] = _rgbToHsv(px[0], px[1], px[2]);
  // Write into the color section's low_hsv or high_hsv
  const target = _eyedrop.target;
  ensurePath(_cfg, "color")[target === "low" ? "low_hsv" : "high_hsv"] = [h, s, v];
  // Also enable the section if it isn't
  if (!_cfg.color.cmd) {} // no-op; section is keyed by "color" so just having entries enables it
  // Preserve which sections were open so the rebuild doesn't slam them shut.
  const openKeys = _snapshotOpenSections();
  if (!openKeys.includes("color")) openKeys.push("color");
  syncForm();
  _restoreOpenSections(openKeys);
  syncJson();
  toast(`Set ${target} HSV to [${h}, ${s}, ${v}]`, "ok");
  _endEyedrop();
}
function _endEyedrop() {
  _eyedrop.active = false;
  _eyedrop.target = null;
  $("#pgImgWrap")?.classList.remove("is-eyedrop");
}

function _refreshTabDots() {
  $$(".pg-build-tab").forEach(b => {
    const tab = b.dataset.tab;
    const has = _tabHasConfig(tab);
    let dot = b.querySelector(".pg-tab-dot");
    if (has && !dot) {
      dot = document.createElement("span");
      dot.className = "pg-tab-dot";
      b.appendChild(dot);
    } else if (!has && dot) {
      dot.remove();
    }
  });
}

function syncForm() {
  renderMethodPicker();
  renderMethodFields();
  renderSections();
  _applySourceUI();   // re-hide camera/robot sections if source = file
}

function setBuilderTab(tab) {
  // Entering the Code view → re-render snippets from current _cfg
  if (_currentView !== "json" && tab === "json") syncJson();
  _currentView = tab;
  $$(".pg-build-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".pg-build-pane").forEach(p => p.classList.toggle("active", p.dataset.pane === tab));
}

// ── Camera picker ─────────────────────────────────────────────────

async function refreshCameraPicker() {
  const sel = $("#pgCamera");
  if (!_vc?.isConnected() || !sel) return;
  let added = [];
  try {
    const list = await _vc.cameraList();
    added = list.filter(d => d.added && d.attached);
  } catch {}
  const prev = sel.value;
  if (!added.length) {
    sel.innerHTML = `<option value="">(no cameras added — go to Cameras tab)</option>`;
    return;
  }
  sel.innerHTML = added.map(d => `<option value="${escHtml(d.serial_number)}">${escHtml(d.serial_number)} — ${escHtml(d.name || "RealSense")}</option>`).join("");
  if (prev && added.find(d => d.serial_number === prev)) sel.value = prev;
}

// ── Server-side playground detection ───────────────────────────────

// "camera" → frames come from the connected RealSense (server-side capture).
// "file"   → server reads cv.imread(<path>); the path is resolved on the
//             *server* (Pi), not the browser.
function _currentSource() { return $("#pgSource")?.value || "file"; }
// File path text is no longer kept in DOM — the picker stores the File
// reference directly in _pickedImageFile and bytes ship inline on Run.

// Mounting setup is UI-only state (NOT in detection cfg). The detection's
// camera_mount kwarg is only set when the user supplies custom T+ej via
// "Use custom calibration" — same as gui.py, which only writes
// _prm["camera_mount"] = {type, T, ej} when camera_clb_apply is checked.
const MOUNT_SETUP_KEY  = "dorna_playground_mount_setup";
const MOUNT_USE_CAL_KEY = "dorna_playground_mount_use_cal";
// AI Models state is session-scoped — the model file picker is just a
// File handle, the cmd type is read from the pickle's meta.type at
// Initialize time. Nothing here persists across reloads.
const INITIALIZED_KEY  = "dorna_playground_initialized";
function _mountSetup() {
  // gui.py default = Eye-to-hand (camera_setup_type value=1).
  return $("#pgMountSetup")?.value || localStorage.getItem(MOUNT_SETUP_KEY) || "to_hand";
}
function _isEyeInHand() { return _mountSetup() === "in_hand"; }
function _useCustomCal() {
  return !!$("#pgMountUseCal")?.checked;
}
function _applySourceUI() {
  const src = _currentSource();
  $("#pgCameraField")?.toggleAttribute("hidden",  src !== "camera");
  $("#pgChannelField")?.toggleAttribute("hidden", src !== "camera");
  $("#pgFileField")?.toggleAttribute("hidden",    src !== "file");
  const snippet = $("#pgPySnippet");
  if (snippet) snippet.textContent = _buildPySnippet();
}

// ── Init/Run lifecycle (mirrors the public API) ───────────────────
//
// The Detection object lives on the server between Runs. Initialize
// builds it once (loading any ML model). Each Run just calls .run()
// with runtime kwargs — Detection.run does setattr() on whatever is
// passed, so ROI / color / detection method / etc. update live without
// a rebuild. This avoids re-loading the OpenVINO model on every click.
//
// Re-initialize tears down the server-side detection so a fresh
// Initialize will rebuild from the current init params.

// Keys that belong to Detection.__init__ — sent at detection_add time
// and frozen until Re-initialize. Everything else in _cfg flows into
// .run() each tick.
const INIT_PARAM_KEYS = ["robot_host", "camera_mount", "frame"];

function _buildInitBody() {
  const body = {};
  // Always bind a camera at init time when one is picked — even when
  // Source is currently "file". Detection.run with data=bytes/path
  // overrides the camera capture, so a camera-bound Detection works
  // for both modes. A None-camera Detection, by contrast, crashes on
  // the next camera-mode Run because get_camera_data calls
  // self.camera.get_all(). This makes Source a pure runtime concern.
  const camSn = $("#pgCamera")?.value || "";
  if (camSn) body.camera_serial_number = camSn;
  for (const k of INIT_PARAM_KEYS) {
    if (k in _cfg) body[k] = _cfg[k];
  }
  // AI Models — when a model file is picked, send the bytes inline.
  // The cmd (od / rod / cls / kp / anom) is read from the pickle's
  // `meta.type` server-side, so the user doesn't have to pick the
  // detection type up front. We only include the filename here as a
  // hint so the server picks the right tempfile suffix.
  if (_pickedModelFile) {
    body.detection = { path: _pickedModelFile.name };
  }
  return body;
}

function _buildRunArgs() {
  // Everything in _cfg that isn't an init kwarg is fair game at runtime.
  const args = {};
  for (const [k, v] of Object.entries(_cfg)) {
    if (INIT_PARAM_KEYS.includes(k)) continue;
    args[k] = v;
  }
  return args;
}

async function initializePlayground() {
  if (!_vc?.isConnected()) { toast("Not connected", "warn"); return false; }
  if (_currentSource() === "camera" && !$("#pgCamera")?.value) {
    toast("Pick a camera first", "warn"); return false;
  }
  // Model file is optional — non-ML detection (cnt, poly, aruco …)
  // doesn't need one. When picked, ship bytes inline; the server reads
  // meta.type from the pickle and configures the right ML cmd.
  let modelBytes = null;
  if (_pickedModelFile) {
    try { modelBytes = await _pickedModelFile.arrayBuffer(); }
    catch (e) { toast(`Could not read model file: ${e.message || e}`, "bad"); return false; }
  }
  try {
    try { await _vc.detectionRemove(PG_NAME); } catch {}
    const reply = await _vc.detectionAdd(PG_NAME, _buildInitBody(), modelBytes);
    // Server echoes the detected cmd back so we can filter the runtime
    // method picker without asking the user to pre-select a type.
    _loadedMlType = (reply && reply.cmd) || "";
    return true;
  } catch (e) {
    toast(`Initialize failed: ${e.message || e}`, "bad");
    return false;
  }
}

async function teardownPlayground() {
  if (!_vc?.isConnected()) return;
  try { await _vc.detectionRemove(PG_NAME); } catch {}
}

async function runOnce() {
  if (!_vc?.isConnected()) { toast("Not connected", "warn"); return; }
  if (!_initializedFlag()) {
    toast("Click Initialize Parameters first", "warn"); return;
  }
  const source = _currentSource();
  const args = _buildRunArgs();
  let runOpts = { _kwargs: args };
  if (source === "file") {
    if (!_pickedImageFile) { toast("Pick an image file first", "warn"); return; }
    try { runOpts._binary = await _pickedImageFile.arrayBuffer(); }
    catch (e) { toast(`Could not read image: ${e.message || e}`, "bad"); return; }
  }
  let valid;
  try {
    valid = (await _vc.detection(PG_NAME).run(runOpts)) || [];
  } catch (e) {
    const msg = String(e?.message || e);
    // Server lost the detection (session restart, manual remove, server
    // restart). Flip back to un-initialized so the user sees the gate
    // and stop Live mode so we don't spam the server log.
    if (/detection not found/i.test(msg)) {
      stopLive();
      _setInitialized(false);
      _resetOutputUI();
      toast("Detection was lost — click Initialize Parameters again", "warn");
    } else {
      toast(`Run failed: ${msg}`, "bad");
    }
    return;
  }
  _hasRun = true;
  await refreshOutput();
  renderResults(valid);
  refreshRoiOverlay();      // re-paint polygon over the new frame
  return valid;
}

function _initializedFlag() {
  return !document.body.classList.contains("is-pre-init");
}

// Re-run the detection on the last cached frame whenever the user tweaks
// runtime settings — same UX as gui.py. Uses the typed detection_run cmd
// with use_last=true so the server skips capture and just re-applies the
// detection pipeline to the camera_data already in memory. Debounced so
// dragging a slider doesn't hammer the server.
let _autoRerunTimer = null;
function _scheduleAutoRerun() {
  if (!_initializedFlag()) return;        // no detection on the server yet
  if (!_hasRun) return;                   // nothing cached to re-run on
  if (_live) return;                      // Live already runs continuously
  if (_autoRerunTimer) clearTimeout(_autoRerunTimer);
  _autoRerunTimer = setTimeout(() => {
    _autoRerunTimer = null;
    _autoRerun();
  }, 150);
}
async function _autoRerun() {
  if (!_initializedFlag() || !_hasRun || _live) return;
  if (!_vc?.isConnected()) return;
  const args = { name: PG_NAME, use_last: true, ..._buildRunArgs() };
  let reply;
  try {
    reply = await _vc._send("detection_run", args);
  } catch (e) {
    // Silent — user can hit Run to retry. We don't want to spam toasts
    // for every flaky tick during a slider drag.
    return;
  }
  await refreshOutput();
  renderResults(reply?.valid || []);
  refreshRoiOverlay();
}

// UI state toggle: locks/unlocks the init cards, shows/hides Init+Re-init
// buttons, snaps the user back to the Init tab if they were on a runtime
// tab. Module-scoped because runOnce calls it on "detection not found"
// recovery (server lost the detection out from under us).
function _setInitialized(on) {
  document.body.classList.toggle("is-pre-init", !on);
  $("#pgInitBtn")?.toggleAttribute("hidden",  on);
  $("#pgReinitBtn")?.toggleAttribute("hidden", !on);
  document.querySelectorAll('.pg-build-pane[data-pane="init"] .pg-group').forEach(g => {
    g.classList.toggle("is-locked", on);
  });
  // Source MODE is runtime — user can flip Camera ⇄ File any time, no
  // re-init needed (file-mode runs send bytes that override the camera).
  // But the Camera dropdown picks WHICH camera the Detection binds to,
  // so it locks after init. Re-initialize to switch cameras.
  $("#pgCamera")?.toggleAttribute("disabled", on);
  if (!on) setBuilderTab("init");
}
// Wire a "Choose file…" picker button to a hidden file input. The picked
// File is held in memory by the caller (via onPicked) and its bytes are
// read fresh on every Run / Initialize that needs them — they ship
// inline with the call as a binary follow-frame. The server never holds
// the file outside the lifetime of the Detection it feeds.
function _wireFilePicker({ pickBtn, fileInput, nameEl, onPicked }) {
  const btn = $(pickBtn);
  const inp = $(fileInput);
  const out = $(nameEl);
  if (!btn || !inp || !out) return;
  btn.addEventListener("click", () => inp.click());
  inp.addEventListener("change", () => {
    const file = inp.files?.[0];
    if (!file) return;
    out.textContent = file.name;
    out.parentElement?.classList.add("is-uploaded");
    onPicked?.(file);
  });
}

function _resetOutputUI() {
  _lastValid = [];
  _hasRun = false;
  if (_imgObjUrl) { URL.revokeObjectURL(_imgObjUrl); _imgObjUrl = null; }
  const img = $("#pgImg");
  if (img) { img.src = ""; img.style.display = "none"; }
  const empty = $("#pgImgEmpty");
  if (empty) empty.style.display = "block";
  renderResults([]);
}

async function refreshOutput() {
  // Bail before hitting the server when there's nothing to fetch.
  // Pre-init: detection_get_img would throw "detection not found".
  // Post-init but pre-Run: it would throw "detection has no 'img' yet".
  // Both are noisy in the server log even though we silently catch.
  if (!_initializedFlag() || !_hasRun) return;
  const tab = _currentImgTab;
  try {
    const { binary } = await _vc.detectionGetImg(PG_NAME, tab, 80);
    if (!binary) return;
    if (_imgObjUrl) URL.revokeObjectURL(_imgObjUrl);
    _imgObjUrl = URL.createObjectURL(new Blob([binary], { type: "image/jpeg" }));
    const img = $("#pgImg");
    img.src = _imgObjUrl;
    img.style.display = "block";
    $("#pgImgEmpty").style.display = "none";
    if (!img.complete) await new Promise(r => img.addEventListener("load", r, { once: true }));
    refreshRoiOverlay();
  } catch (e) {
    if (tab !== "img") {
      try {
        const { binary } = await _vc.detectionGetImg(PG_NAME, "img", 80);
        if (binary) {
          if (_imgObjUrl) URL.revokeObjectURL(_imgObjUrl);
          _imgObjUrl = URL.createObjectURL(new Blob([binary], { type: "image/jpeg" }));
          $("#pgImg").src = _imgObjUrl;
        }
      } catch {}
    }
  }
}

function renderResults(valid) {
  _lastValid = Array.isArray(valid) ? valid : [];
  $("#pgValidCount").textContent = String(_lastValid.length);
  $("#pgRawResult").textContent = JSON.stringify(_lastValid, null, 2);

  const root = $("#pgResults");
  if (!_lastValid.length) {
    root.innerHTML = `<div class="muted" style="padding:14px 0">No detections.</div>`;
    _selectedDetId = null;
    _renderHighlight();
    return;
  }
  const cols = ["id", "cls", "conf", "center", "xyz"];
  const fmt = (v) => {
    if (v == null) return "—";
    if (Array.isArray(v)) return "[" + v.map(x => typeof x === "number" ? (Number.isInteger(x) ? x : x.toFixed(2)) : fmt(x)).join(", ") + "]";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
    return String(v);
  };
  const head = `<tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
  const rows = _lastValid.map(r => `<tr data-det-id="${escHtml(r.id)}" class="${String(r.id) === String(_selectedDetId) ? "is-selected" : ""}">${cols.map(c => `<td>${escHtml(fmt(r[c]))}</td>`).join("")}</tr>`).join("");
  root.innerHTML = `<table class="pg-table">${head}${rows}</table>`;
  // Click-to-highlight
  root.querySelectorAll("tr[data-det-id]").forEach(tr => {
    tr.addEventListener("click", () => _selectDetection(tr.dataset.detId));
  });
  _renderHighlight();
}

// ── Live mode ──────────────────────────────────────────────────────

// Cap Live to ~10fps. Higher rates queue JPEG decodes faster than the
// browser can paint them, and the backlog eventually OOMs the tab. The
// cap applies a minimum frame interval; if the server is slower than
// that, the loop runs at server speed.
const LIVE_MIN_INTERVAL_MS = 100;

async function liveLoop() {
  while (_live && isPageActive() && _vc?.isConnected()) {
    if (_liveTickRunning) { await new Promise(r => setTimeout(r, 10)); continue; }
    _liveTickRunning = true;
    const t0 = performance.now();
    try {
      await runOnce();
    } catch (e) {
      console.error("live tick failed", e);
      _live = false; _setLiveButtons(false);
      toast(`Live stopped: ${e.message || e}`, "bad");
      break;
    } finally { _liveTickRunning = false; }
    const dt = performance.now() - t0;
    _fpsTimes.push(dt);
    if (_fpsTimes.length > 10) _fpsTimes.shift();
    const avg = _fpsTimes.reduce((a, b) => a + b, 0) / _fpsTimes.length;
    if (avg > 0) $("#pgFps").textContent = `${(1000 / avg).toFixed(1)} fps`;
    // Yield to the browser so it can paint + GC; also enforce the FPS cap.
    const wait = Math.max(LIVE_MIN_INTERVAL_MS - dt, 16);
    await new Promise(r => setTimeout(r, wait));
  }
  if (!_live) $("#pgFps").textContent = "";
}
function _setLiveButtons(running) {
  $("#pgRunBtn").disabled = running;
  $("#pgLiveBtn").style.display = running ? "none" : "";
  $("#pgStopBtn").style.display = running ? "" : "none";
}
function startLive() {
  if (_live) return;
  _live = true; _fpsTimes = [];
  _setLiveButtons(true);
  liveLoop();
}
function stopLive() { _live = false; _setLiveButtons(false); }

// ── Promote ───────────────────────────────────────────────────────

function openPromote() {
  const inp = $("#pgPromoteName");
  inp.value = "";
  $("#pgPromoteOverlay").classList.add("show");
  setTimeout(() => inp.focus(), 0);
}
function closePromote() { $("#pgPromoteOverlay").classList.remove("show"); }
async function submitPromote() {
  const name = $("#pgPromoteName").value.trim();
  if (!name) { toast("Name is required", "warn"); return; }
  if (name === PG_NAME) { toast("Pick a different name", "warn"); return; }
  const camSn = $("#pgCamera").value;
  if (!camSn) { toast("Pick a camera first", "warn"); return; }
  closePromote();
  try {
    await _vc.detectionAdd(name, { camera_serial_number: camSn, ..._cfg });
    toast(`Promoted as "${name}"`, "ok");
  } catch (e) { toast(`Promote failed: ${e.message || e}`, "bad"); }
}

// ── Save / load ────────────────────────────────────────────────────

function saveConfig() {
  if (!Object.keys(_cfg).length) { toast("Nothing to save", "warn"); return; }
  const blob = new Blob([JSON.stringify(_cfg, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "detection_config.json"; a.click();
  URL.revokeObjectURL(url);
}
function loadConfig() { $("#pgLoadFile").click(); }
async function onLoadFile(ev) {
  const file = ev.target.files?.[0];
  ev.target.value = "";
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("must be a JSON object");
    _cfg = parsed;
    syncJson();
    syncForm();
  } catch (e) {
    toast(`Load failed: ${e.message || e}`, "bad");
  }
}

// ── Tabs (image) ──────────────────────────────────────────────────

function setImgTab(tab) {
  _currentImgTab = tab;
  $$(".pg-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  refreshOutput();
}

// Threshold image is only produced by the polygon and contour detectors
// (matching gui.py — only methods 3/4 expose `img_thr`). Hide the tab
// otherwise, and snap back to Annotated if the user was viewing it.
const _IMG_TAB_REQUIREMENTS = {
  img_thr: (cfg) => ["poly", "cnt"].includes(cfg?.detection?.cmd),
};
function _refreshImgTabs() {
  let activeStillVisible = true;
  $$(".pg-tab").forEach(b => {
    const req = _IMG_TAB_REQUIREMENTS[b.dataset.tab];
    if (!req) return;
    const ok = !!req(_cfg);
    b.toggleAttribute("hidden", !ok);
    if (!ok && _currentImgTab === b.dataset.tab) activeStillVisible = false;
  });
  if (!activeStillVisible) setImgTab("img");
}

// ── ROI polygon editor (clicks on the captured image) ─────────────
//
// State machine — three states drive both the on-image overlay and the
// action buttons inside the form (under the corners textarea):
//
//   no polygon        →  [ Pick polygon on image ]
//   polygon present   →  [ Edit polygon ]  [ Remove polygon ]
//   editing           →  [ Save polygon ]  [ Cancel ]  [ Clear all ]
//
// Editing copies the current corners into _roiEdit.points; Cancel discards;
// Save writes them back to the textarea (and into _cfg.roi.corners if the
// section's Apply checkbox is on).

// `mode` switches between two picker behaviors that share this state +
// SVG overlay + click handlers:
//   "polygon" — free-form polygon (used by ROI corners). Each click adds
//               a vertex; vertices drag freely.
//   "bbox"    — axis-aligned rectangle. First two clicks set opposite
//               corners and expand to 4 corners; vertex drags keep the
//               rectangle axis-aligned by also adjusting neighbors.
const _roiEdit = { active: false, targetId: null, points: [], mode: "polygon" };

function _polyTargetId() {
  // Single ROI polygon picker in the schema; locate its textarea id.
  const el = document.querySelector("[data-roi-actions]");
  return el?.dataset.roiActions || null;
}

function _polygonFromTextarea(targetId) {
  const ta = document.getElementById(targetId);
  if (!ta) return [];
  try {
    const v = JSON.parse(ta.value || "[]");
    if (!Array.isArray(v)) return [];
    // ROI corners are pixel indices — round here so any decimal noise
    // from older saves / external edits is normalized everywhere the
    // polygon is read (display, edit init, results overlay).
    return v
      .filter(p => Array.isArray(p) && p.length === 2)
      .map(([x, y]) => [Math.round(Number(x)), Math.round(Number(y))]);
  } catch { return []; }
}

function _renderRoiActions(targetId) {
  const root = document.querySelector(`[data-roi-actions="${targetId}"]`);
  if (!root) return;
  const editing = _roiEdit.active && _roiEdit.targetId === targetId;
  const points = editing ? _roiEdit.points : _polygonFromTextarea(targetId);
  const has = points.length > 0;
  // Pick mode (polygon vs bbox) lives on the action div as a
  // data-attribute set by the field renderer. Defaults to polygon for
  // any picker that doesn't declare a mode.
  const mode = root.dataset.pickMode === "bbox" ? "bbox" : "polygon";
  const noun = mode === "bbox" ? "bbox" : "polygon";

  let html = "";
  if (editing) {
    const hint = mode === "bbox"
      ? (points.length < 2 ? "click 2 corners" : "drag to resize")
      : `${points.length} point${points.length === 1 ? "" : "s"}`;
    html = `
      <button type="button" class="btn btn-primary btn-sm" data-roi-act="save"   data-roi-target="${targetId}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        Save ${noun}
      </button>
      <button type="button" class="btn btn-sm" data-roi-act="cancel" data-roi-target="${targetId}">Cancel</button>
      <span class="pg-roi-count">${hint}</span>
    `;
  } else if (has) {
    const hint = mode === "bbox" ? "rectangle set" : `${points.length} point${points.length === 1 ? "" : "s"}`;
    html = `
      <button type="button" class="btn btn-primary btn-sm pg-pick-poly" data-pick-target="${targetId}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>
        Edit ${noun}
      </button>
      <button type="button" class="btn btn-danger btn-sm" data-roi-act="remove" data-roi-target="${targetId}">Remove ${noun}</button>
      <span class="pg-roi-count">${hint}</span>
    `;
  } else {
    const verb = mode === "bbox" ? "Pick bbox on image" : "Pick polygon on image";
    html = `
      <button type="button" class="btn btn-primary btn-sm pg-pick-poly" data-pick-target="${targetId}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/></svg>
        ${verb}
      </button>
    `;
  }
  root.innerHTML = html;
}

async function _captureRawForRoi() {
  if (!_vc?.isConnected()) {
    toast("Not connected to the server", "bad");
    return false;
  }
  const sn = $("#pgCamera").value;
  if (!sn) {
    toast("Pick a camera in the dropdown at the top of Playground first (or add one in the Cameras tab)", "warn");
    return false;
  }
  try {
    const { binary } = await _vc.cameraGetImg(sn, "color_img", 80);
    if (!binary) throw new Error("server returned no image bytes");
    if (_imgObjUrl) URL.revokeObjectURL(_imgObjUrl);
    _imgObjUrl = URL.createObjectURL(new Blob([binary], { type: "image/jpeg" }));
    const img = $("#pgImg");
    img.src = _imgObjUrl;
    img.style.display = "block";
    $("#pgImgEmpty").style.display = "none";
    if (!img.complete) await new Promise(r => img.addEventListener("load", r, { once: true }));
    refreshRoiOverlay();   // image just changed — repaint any persisted polygon
    return !!img.naturalWidth;
  } catch (e) {
    toast(`Could not capture image: ${e.message || e}`, "bad");
    return false;
  }
}

async function startRoiEdit(targetId) {
  const img = $("#pgImg");
  if (!img || !img.naturalWidth || img.style.display === "none") {
    const ok = await _captureRawForRoi();
    if (!ok) return;
  }
  // Pick mode rides on the action div's data-pick-mode attribute (set
  // by the field renderer). Default to polygon for backward compat
  // with any picker that doesn't declare a mode.
  const actionEl = document.querySelector(`[data-roi-actions="${targetId}"]`);
  const mode = actionEl?.dataset.pickMode === "bbox" ? "bbox" : "polygon";

  _roiEdit.active = true;
  _roiEdit.targetId = targetId;
  _roiEdit.mode = mode;
  _roiEdit.points = _polygonFromTextarea(targetId).map(p => [Number(p[0]), Number(p[1])]);

  const overlay = $("#pgRoiOverlay");
  overlay.addEventListener("click", _onRoiClick);
  overlay.addEventListener("dblclick", _onRoiDblClick);
  window.addEventListener("resize", refreshRoiOverlay);
  refreshRoiOverlay();
  _renderRoiActions(targetId);
}

function _endRoiEdit(save) {
  const targetId = _roiEdit.targetId;
  if (save) {
    const ta = document.getElementById(targetId);
    if (ta) {
      // ROI corners are pixel indices — strip any fractional noise
      // that may have crept in from older saves or external edits.
      let cleaned = _roiEdit.points.map(
        ([x, y]) => [Math.round(Number(x)), Math.round(Number(y))]
      );
      // Bbox mode: canonicalize to [TL, TR, BR, BL] order in case the
      // rect got "flipped" during drag (corner pulled past its
      // diagonal neighbor). Skipped if the user only placed 1 corner —
      // that intermediate state isn't a real bbox.
      if (_roiEdit.mode === "bbox" && cleaned.length === 4) {
        const xs = cleaned.map(p => p[0]);
        const ys = cleaned.map(p => p[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs);
        const y0 = Math.min(...ys), y1 = Math.max(...ys);
        cleaned = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
      }
      ta.value = JSON.stringify(cleaned);
      ta.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }
  const overlay = $("#pgRoiOverlay");
  overlay.removeEventListener("click", _onRoiClick);
  overlay.removeEventListener("dblclick", _onRoiDblClick);
  window.removeEventListener("resize", refreshRoiOverlay);
  _roiEdit.active = false;
  _roiEdit.targetId = null;
  _roiEdit.points = [];
  _roiEdit.mode = "polygon";
  refreshRoiOverlay();
  if (targetId) _renderRoiActions(targetId);
}

function _removePolygon(targetId) {
  const ta = document.getElementById(targetId);
  if (!ta) return;
  ta.value = "[]";
  ta.dispatchEvent(new Event("change", { bubbles: true }));
  refreshRoiOverlay();
  _renderRoiActions(targetId);
}

let _ptDragIdx = null;
let _justDragged = false;

// Snap `p` so that the segment from `anchor → p` lies on the nearest
// multiple of 45° (0°/45°/90°/135°/…). Length is preserved — only the
// direction is constrained. Matches the Shift-drag behavior in Figma /
// Illustrator / Photoshop. Output is rounded to integer pixels (ROI
// coords are pixel indices, so fractional values are noise).
function _snapToAngle(p, anchor) {
  const dx = p[0] - anchor[0];
  const dy = p[1] - anchor[1];
  const len = Math.hypot(dx, dy);
  if (len === 0) return p;
  const step = Math.PI / 4;
  const snapped = Math.round(Math.atan2(dy, dx) / step) * step;
  return [
    Math.round(anchor[0] + len * Math.cos(snapped)),
    Math.round(anchor[1] + len * Math.sin(snapped)),
  ];
}

function _onPtMouseDown(idx, ev) {
  ev.stopPropagation();
  ev.preventDefault();
  _ptDragIdx = idx;
  _justDragged = false;
  document.addEventListener("mousemove", _onPtDragMove);
  document.addEventListener("mouseup",   _onPtDragEnd);
}
function _onPtDragMove(ev) {
  if (_ptDragIdx === null) return;
  // Allow out-of-bounds during drag: a vertex that's already outside
  // the image (e.g. a polygon saved from a previous larger image)
  // must be draggable everywhere on screen, not clamped at the edge.
  let p = _imgClickToPixel(ev, true);
  if (!p) return;

  if (_roiEdit.mode === "bbox" && _roiEdit.points.length === 4) {
    // Axis-aligned rectangle constraint. Corners are stored in
    // [TL, TR, BR, BL] order; dragging any one moves it AND its two
    // neighbors so the rect stays axis-aligned. The opposite corner
    // (i+2 mod 4) stays fixed. Adjacency:
    //   TL=0 ↔ TR=1 (share y), TL=0 ↔ BL=3 (share x)
    //   TR=1 ↔ BR=2 (share x)
    //   BR=2 ↔ BL=3 (share y)
    const adj = {
      0: { hN: 1, vN: 3 },
      1: { hN: 0, vN: 2 },
      2: { hN: 3, vN: 1 },
      3: { hN: 2, vN: 0 },
    }[_ptDragIdx];
    const pts = _roiEdit.points;
    pts[_ptDragIdx]  = [p[0], p[1]];
    pts[adj.hN] = [pts[adj.hN][0], p[1]];   // horizontal neighbor — shares y
    pts[adj.vN] = [p[0], pts[adj.vN][1]];   // vertical neighbor — shares x
    _justDragged = true;
    _renderRoiSvg();
    return;
  }

  // Shift = constrain the edge from the previous vertex onto the
  // nearest 45° angle (vertical / horizontal / diagonal). Falls back
  // to no-op if the polygon has fewer than 2 points.
  if (ev.shiftKey && _roiEdit.points.length > 1) {
    const n = _roiEdit.points.length;
    const prevIdx = (_ptDragIdx - 1 + n) % n;
    p = _snapToAngle(p, _roiEdit.points[prevIdx]);
  }
  _justDragged = true;
  _roiEdit.points[_ptDragIdx] = p;
  _renderRoiSvg();
}
function _onPtDragEnd() {
  _ptDragIdx = null;
  document.removeEventListener("mousemove", _onPtDragMove);
  document.removeEventListener("mouseup",   _onPtDragEnd);
  setTimeout(() => { _justDragged = false; }, 60);
}
function _onPtContextMenu(idx, ev) {
  ev.preventDefault();
  ev.stopPropagation();
  // Bbox mode: deleting a corner would break the 4-corner rect
  // invariant, so right-click does nothing. User can hit Remove to
  // clear and start over.
  if (_roiEdit.mode === "bbox") return;
  _roiEdit.points.splice(idx, 1);
  _renderRoiSvg();
}

function _onRoiClick(ev) {
  if (_justDragged)                          return;
  if (ev.target.closest(".pg-roi-bar"))      return;
  if (ev.target.closest(".pg-roi-pt"))       return;   // clicking an existing point ≠ adding new
  let p = _imgClickToPixel(ev);
  if (!p) return;

  if (_roiEdit.mode === "bbox") {
    // 2-click flow: first click sets a single anchor; second click
    // expands to a 4-corner axis-aligned rect. After 4 corners exist
    // additional clicks are ignored — the user adjusts via dragging
    // corners, or hits Remove to start over.
    const n = _roiEdit.points.length;
    if (n === 0) {
      _roiEdit.points.push(p);
    } else if (n === 1) {
      const a = _roiEdit.points[0];
      const x0 = Math.min(a[0], p[0]);
      const x1 = Math.max(a[0], p[0]);
      const y0 = Math.min(a[1], p[1]);
      const y1 = Math.max(a[1], p[1]);
      _roiEdit.points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    }
    // n >= 2 (degenerate / 4-corner): ignore the click.
    _renderRoiSvg();
    return;
  }

  // Polygon mode (free-form). Shift = snap the new point so the edge
  // from the last existing vertex lies on a 45° angle.
  if (ev.shiftKey && _roiEdit.points.length > 0) {
    p = _snapToAngle(p, _roiEdit.points[_roiEdit.points.length - 1]);
  }
  _roiEdit.points.push(p);
  _renderRoiSvg();
}
function _onRoiDblClick(ev) {
  if (ev.target.closest(".pg-roi-bar")) return;
  if (ev.target.closest(".pg-roi-pt"))  return;
  _endRoiEdit(true);
}

// Convert a mouse event to image-pixel coordinates.
//
// `allowOutOfBounds` controls what happens when the cursor sits
// outside the image:
//   false (default) — return null. Right for click-to-add: an
//                     accidental click on the page background
//                     shouldn't seed a stray ROI vertex.
//   true            — return the (possibly negative or beyond-image)
//                     coords. Right for vertex DRAG: a user grabbing
//                     a vertex that's already outside the image
//                     (e.g. an ROI saved from a larger previous
//                     image) needs to be able to drag it freely;
//                     rejecting out-of-bounds froze the drag.
function _imgClickToPixel(ev, allowOutOfBounds = false) {
  const img = $("#pgImg");
  if (!img.naturalWidth) return null;
  const r = img.getBoundingClientRect();
  const x = (ev.clientX - r.left) * img.naturalWidth  / r.width;
  const y = (ev.clientY - r.top)  * img.naturalHeight / r.height;
  if (!allowOutOfBounds && (x < 0 || y < 0 || x > img.naturalWidth || y > img.naturalHeight)) {
    return null;
  }
  return [Math.round(x), Math.round(y)];
}

function _syncRoiSvg() {
  // Position the SVG exactly over the displayed image.
  //
  // Subtlety: the SVG's positioned ancestor is `.pg-img-stage`
  // (position:relative), NOT the wrapper. The wrapper is a flex
  // container that may center the stage horizontally — so the stage
  // typically sits some pixels right of the wrapper's left edge.
  // Computing `left = imgR.left - wrapR.left` instead of relative to
  // the stage shifts the SVG by exactly that flex-centering offset,
  // which is why ROI vertices render to the right of where they were
  // clicked. Use the stage's rect to anchor.
  const img = $("#pgImg");
  const stage = $("#pgImgStage");
  const svg = $("#pgRoiSvg");
  if (!img.naturalWidth || !stage || !svg) return;
  const stageR = stage.getBoundingClientRect();
  const imgR   = img.getBoundingClientRect();
  svg.style.position = "absolute";
  svg.style.left   = (imgR.left - stageR.left) + "px";
  svg.style.top    = (imgR.top  - stageR.top)  + "px";
  svg.style.width  = imgR.width  + "px";
  svg.style.height = imgR.height + "px";
  // SVG defaults to overflow:hidden, which clips any vertex/line whose
  // coords fall outside the viewBox (= image bounds). When a saved
  // polygon has vertices outside the new image, that hides the
  // handles. Allow them to render outside so they're still grabbable.
  svg.style.overflow = "visible";
  svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
}

// Single source for the on-image polygon overlay. Displays the polygon read-
// only when one exists in the form, switches to interactive in edit mode.
function refreshRoiOverlay() {
  const overlay = $("#pgRoiOverlay");
  const svg     = $("#pgRoiSvg");
  const wrap    = $("#pgImgWrap");
  const img     = $("#pgImg");
  if (!overlay || !svg || !wrap) return;

  const editing = _roiEdit.active;
  const targetId = _roiEdit.targetId || _polyTargetId();
  const pts = editing
    ? _roiEdit.points
    : (targetId ? _polygonFromTextarea(targetId) : []);

  // No polygon and not editing → hide entirely
  if (!editing && pts.length === 0) {
    overlay.setAttribute("hidden", "");
    wrap.classList.remove("is-roi-editing", "is-roi-shown");
    return;
  }
  // Only paint the ROI polygon over the Annotated image — not on Threshold etc.
  if (!editing && _currentImgTab !== "img") {
    overlay.setAttribute("hidden", "");
    wrap.classList.remove("is-roi-editing", "is-roi-shown");
    return;
  }
  // Need an image to anchor the SVG to
  if (!img || !img.naturalWidth) {
    overlay.setAttribute("hidden", "");
    wrap.classList.remove("is-roi-editing", "is-roi-shown");
    return;
  }

  overlay.removeAttribute("hidden");
  wrap.classList.toggle("is-roi-editing", editing);
  wrap.classList.toggle("is-roi-shown", !editing);
  // Mode flag drives the green vs blue stroke. During edit it's
  // _roiEdit.mode; when just displaying a saved target it's the
  // target's own data-pick-mode (looked up from the action div).
  let mode = "polygon";
  if (editing) {
    mode = _roiEdit.mode;
  } else if (targetId) {
    const actionEl = document.querySelector(`[data-roi-actions="${targetId}"]`);
    if (actionEl?.dataset.pickMode === "bbox") mode = "bbox";
  }
  overlay.classList.toggle("is-bbox", mode === "bbox");
  _syncRoiSvg();

  let inner = "";
  // In display mode, dim the region the ROI is masking out so the user can
  // see what's discarded. The dim side flips with `inv`: by default the
  // outside is masked (so dim outside); with inv=1 the inside is masked.
  // The actual masking is done server-side by the ROI class; this is a
  // purely visual aid. Bbox mode skips the dim — a bbox is just a
  // detection hint, not something that masks pixels.
  if (!editing && pts.length >= 3 && mode !== "bbox") {
    const W = img.naturalWidth, H = img.naturalHeight;
    const polyD = "M" + pts.map(p => `${p[0]},${p[1]}`).join(" L") + " Z";
    const inv = !!Number(_cfg?.roi?.inv);
    inner += inv
      ? `<path d="${polyD}" class="pg-roi-mask"/>`
      : `<path d="M0,0 L${W},0 L${W},${H} L0,${H} Z ${polyD}" fill-rule="evenodd" class="pg-roi-mask"/>`;
  }
  if (pts.length >= 2) {
    const ptStr = pts.map(p => `${p[0]},${p[1]}`).join(" ");
    inner += pts.length >= 3
      ? `<polygon points="${ptStr}" class="pg-roi-poly"/>`
      : `<polyline points="${ptStr}" class="pg-roi-line"/>`;
  }
  // Interactive handles only in edit mode
  if (editing) {
    pts.forEach((p, i) => {
      inner += `<g class="pg-roi-pt" data-idx="${i}"><circle cx="${p[0]}" cy="${p[1]}" r="8"/><text x="${p[0]}" y="${p[1]}" dy="3.5" text-anchor="middle">${i + 1}</text></g>`;
    });
  }
  svg.innerHTML = inner;

  if (editing) {
    svg.querySelectorAll(".pg-roi-pt").forEach(g => {
      const idx = Number(g.dataset.idx);
      g.addEventListener("mousedown",  (e) => _onPtMouseDown(idx, e));
      g.addEventListener("contextmenu",(e) => _onPtContextMenu(idx, e));
    });
  }
}

// Backward-compatible alias for the old name used by the drag handlers.
const _renderRoiSvg = refreshRoiOverlay;

// ── Lifecycle ─────────────────────────────────────────────────────

export function init(vc) {
  _vc = vc;

  // Restore the user's last-used config so a reload doesn't lose work.
  const stored = _loadStoredCfg();
  if (stored && typeof stored === "object" && !Array.isArray(stored)) _cfg = stored;

  // Method picker — first render
  syncForm();
  syncJson();
  renderResults([]);   // seed the empty "Detections 0 / No detections." state

  // Esc cancels an active ROI edit / eyedrop; Cmd/Ctrl+Enter = Run
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (_roiEdit.active) _endRoiEdit(false);
      if (_eyedrop.active) _endEyedrop();
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && isPageActive() && !_roiEdit.active) {
      e.preventDefault();
      runOnce().catch(err => toast(`Run failed: ${err.message || err}`, "bad"));
    }
  });

  // Clear all + presets
  $("#pgClearAll")?.addEventListener("click", clearAll);

  // Reconnect banner — vc emits 'open' / 'close' events.
  vc.on?.((ev) => {
    if (ev === "close") _setReconnectBanner(true);
    if (ev === "open")  _setReconnectBanner(false);
  });

  // Per-section Reset (delegated)
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-section-reset]");
    if (!btn) return;
    e.stopPropagation();   // don't toggle the section's collapse
    resetSection(btn.dataset.sectionReset);
  });

  // Eyedropper buttons in Color Mask section (delegated)
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-eyedrop]");
    if (!btn) return;
    _startEyedrop(btn.dataset.eyedrop);
  });

  // Image: zoom (wheel) + pan (drag) + reset
  const wrap = $("#pgImgWrap");
  if (wrap) {
    wrap.addEventListener("wheel", (e) => {
      if (!$("#pgImg")?.naturalWidth) return;
      e.preventDefault();
      _zoomAt(e.deltaY, e.clientX, e.clientY);
    }, { passive: false });

    let dragging = null;
    wrap.addEventListener("mousedown", (e) => {
      // Don't pan when in ROI edit mode (those clicks add points) or eyedropper
      if (_roiEdit.active || _eyedrop.active) return;
      // Only left-button pan when zoomed in; middle-button always
      if (e.button !== 0 && e.button !== 1) return;
      if (e.button === 0 && _zoom.s <= 1) return;
      dragging = { x: e.clientX, y: e.clientY, tx: _zoom.tx, ty: _zoom.ty };
      wrap.classList.add("is-panning");
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      _zoom.tx = dragging.tx + (e.clientX - dragging.x);
      _zoom.ty = dragging.ty + (e.clientY - dragging.y);
      _applyImgTransform();
    });
    document.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = null;
      wrap.classList.remove("is-panning");
    });
  }
  $("#pgZoomReset")?.addEventListener("click", _resetZoom);

  // Expand the current annotated frame in the global lightbox overlay —
  // mirrors the camera card's expand button.
  $("#pgExpand")?.addEventListener("click", () => {
    const src = $("#pgImg")?.src;
    if (!src) { toast("Run once to capture a frame first", "warn"); return; }
    const overlay = $("#imgLightbox");
    const img     = $("#imgLightboxImg");
    const cap     = $("#imgLightboxCap");
    if (!overlay || !img) return;
    img.src = src;
    if (cap) {
      const tab = _currentImgTab === "img_thr" ? "Threshold" : "Annotated";
      cap.textContent = `Playground · ${tab}`;
    }
    // The overlay's "Capture again" button is for camera-card flows; it
    // looks up a camera SN that the playground never sets. Hide it here.
    delete overlay.dataset.sn;
    $("#imgLightboxCapture")?.setAttribute("hidden", "");
    overlay.classList.add("show");
  });

  // HSV eyedropper — listen for clicks on the image whenever it's active.
  $("#pgImg")?.addEventListener("click", (e) => {
    if (_eyedrop.active) { e.preventDefault(); e.stopPropagation(); _onImgEyedrop(e); }
  });

  // Method changed
  $("#pgMethod")?.addEventListener("change", () => {
    const newCmd = $("#pgMethod").value;
    if (!newCmd) {
      // "No detection" — drop the whole detection branch
      delete _cfg.detection;
    } else {
      const schema = CMD_SCHEMAS[newCmd];
      const detection = { cmd: newCmd };
      // For showWhen evaluation we need a values snapshot reflecting
      // the *default* state (so e.g. method="standard" hides the
      // lighting_tolerant fields). Build it from each field's default.
      const _defaults = {};
      if (schema) for (const f of schema.fields) {
        if (f.default !== undefined) _defaults[f.key] = f.default;
      }
      if (schema) for (const f of schema.fields) {
        // Skip fields that opt into "absent vs empty" distinction (e.g. cls
        // for OD/ROD). Their default is `[]` in the schema as a render
        // hint, but `[]` means "filter everything out" at runtime — we
        // don't want that auto-applied just from picking a method.
        // Leaving the key absent keeps the API default (no filter / all
        // classes) until the user explicitly types something.
        if (f.keepEmpty) continue;
        // Transient fields (e.g. unified Ellipse's `method`) are UI-only
        // — they don't get persisted into _cfg.detection, so don't seed
        // their default either.
        if (f.transient) continue;
        // Skip fields that are hidden under the default method — for
        // unified schemas this prevents the inactive method's keys
        // (which often share names with the active method's, e.g.
        // `area` defined twice with different ranges) from clobbering
        // the visible defaults.
        if (typeof f.showWhen === "function" && !f.showWhen(_defaults, _cfg)) continue;
        if (f.default !== null && f.default !== undefined) detection[f.key] = f.default;
      }
      // ML methods pull their model path from the AI Models init section.
      if (["od","cls","kp"].includes(newCmd)) {
        const p = ($("#pgMlPath")?.value || "").trim();
        if (p) detection.path = p;
      }
      // ANOM: seed the threshold slider with the training-time value
      // so the user starts from a sensible default for THIS model
      // rather than the schema's static 0.5.
      if (newCmd === "anom" && _trainedThreshold != null) {
        detection.threshold = _trainedThreshold;
      }
      _cfg.detection = detection;
    }
    renderMethodFields();
    syncJson();
  });

  // Builder tabs
  $$(".pg-build-tab").forEach(b => b.addEventListener("click", () => setBuilderTab(b.dataset.tab)));
  // Copy buttons in the Code pane
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-copy-target]");
    if (!btn) return;
    const target = document.getElementById(btn.dataset.copyTarget);
    const text = target?.textContent ?? "";
    if (!text.trim()) { toast("Nothing to copy", "warn"); return; }
    _copyToClipboard(text, btn);
  });

  // Camera selection feeds into the Python snippet — re-render on change
  $("#pgCamera")?.addEventListener("change", () => {
    $("#pgPySnippet").textContent = _buildPySnippet();
  });

  // Source picker (Camera vs File path) — also hides camera/robot-dependent
  // sections when source is a file (matching gui.py's hide_show_source +
  // hide_show_ip behavior).
  // Channel select <-> _cfg.feed
  const ch = $("#pgChannel");
  if (ch) {
    if (_cfg.feed) ch.value = _cfg.feed;
    ch.addEventListener("change", () => {
      const v = ch.value;
      if (!v || v === "color_img") delete _cfg.feed;   // default — keep cfg slim
      else _cfg.feed = v;
      syncJson();
    });
  }
  $("#pgSource")?.addEventListener("change", () => {
    try { localStorage.setItem(SOURCE_KEY, $("#pgSource").value); } catch {}
    _applySourceUI();
  });
  // Image file picker — bytes are streamed inline with each Run() via
  // the binary follow-frame; nothing persists on the server.
  _wireFilePicker({
    pickBtn:   "#pgImgFilePick",
    fileInput: "#pgImgFile",
    nameEl:    "#pgImgFileName",
    onPicked: (file) => {
      _pickedImageFile = file;
      $("#pgPySnippet").textContent = _buildPySnippet();
    },
  });
  // Restore last-used source mode (file path itself isn't persisted —
  // uploads are session-scoped temp files, so a stale path would 404).
  try {
    const s = localStorage.getItem(SOURCE_KEY);
    if (s === "camera" || s === "file") $("#pgSource").value = s;
  } catch {}
  _applySourceUI();

  // ── Camera Mounting ────────────────────────────────────────────
  // gui.py model: setup type + robot IP + (optional) custom calibration
  // are all init-time settings. We mirror the same conditional flow:
  //  * Eye-to-hand (default): no robot IP, no custom calibration UI,
  //    no camera_mount or robot_host in _cfg.
  //  * Eye-in-hand: robot IP shown → writes _cfg.robot_host. Custom
  //    calibration toggle reveals T + ej; when on, _cfg.camera_mount
  //    becomes {type, T, ej}. Otherwise no camera_mount key (server
  //    default "dorna_ta_j4_1" applies).
  const _writeMountToCfg = () => {
    const inHand = _isEyeInHand();
    // robot_host: only when Eye-in-hand and non-blank
    const ip = ($("#pgMountIp")?.value || "").trim();
    if (inHand && ip) _cfg.robot_host = ip;
    else delete _cfg.robot_host;
    // camera_mount: only when Eye-in-hand + custom calibration
    if (!inHand || !_useCustomCal()) {
      delete _cfg.camera_mount;
    } else {
      const T = $$("[data-mount-t]").map(c => Number(c.value));
      let ej;
      try { ej = JSON.parse($("#pgMountEj").value); } catch { ej = null; }
      if (!Array.isArray(ej)) ej = [0,0,0,0,0,0,0,0];
      _cfg.camera_mount = { type: "dorna_ta_j4_1", T, ej };
    }
    syncJson();
  };
  const _applyMountUI = () => {
    const inHand = _isEyeInHand();
    const useCal = _useCustomCal();
    $("#pgMountIpField")?.toggleAttribute("hidden",     !inHand);
    $("#pgMountUseCalField")?.toggleAttribute("hidden", !inHand);
    $("#pgMountTField")?.toggleAttribute("hidden",      !(inHand && useCal));
    $("#pgMountEjField")?.toggleAttribute("hidden",     !(inHand && useCal));
    // Switching away from Eye-in-hand cancels custom calibration.
    if (!inHand && useCal) {
      const cb = $("#pgMountUseCal");
      if (cb) cb.checked = false;
    }
    _writeMountToCfg();
  };
  $("#pgMountSetup")?.addEventListener("change", () => {
    try { localStorage.setItem(MOUNT_SETUP_KEY, $("#pgMountSetup").value); } catch {}
    _applyMountUI();
  });
  $("#pgMountIp")?.addEventListener("input",  _writeMountToCfg);
  $("#pgMountUseCal")?.addEventListener("change", () => {
    try { localStorage.setItem(MOUNT_USE_CAL_KEY, $("#pgMountUseCal").checked ? "1" : "0"); } catch {}
    _applyMountUI();
  });
  $$("[data-mount-t]").forEach(c => c.addEventListener("input", _writeMountToCfg));
  $("#pgMountEj")?.addEventListener("input", _writeMountToCfg);
  // Restore mount UI state from localStorage
  try {
    const ms = localStorage.getItem(MOUNT_SETUP_KEY);
    if (ms === "in_hand" || ms === "to_hand") $("#pgMountSetup").value = ms;
    if (localStorage.getItem(MOUNT_USE_CAL_KEY) === "1" && $("#pgMountUseCal")) $("#pgMountUseCal").checked = true;
    // Restore robot IP from cfg if present
    if (_cfg.robot_host && $("#pgMountIp")) $("#pgMountIp").value = _cfg.robot_host;
  } catch {}
  _applyMountUI();

  // ── AI Models ─────────────────────────────────────────────────
  // Model file picker — bytes ship inline with the next detection_add
  // (Initialize). The server reads `meta.type` from the pickle and
  // configures the right ML cmd, so there's nothing to choose up front.
  _wireFilePicker({
    pickBtn:   "#pgMlFilePick",
    fileInput: "#pgMlFile",
    nameEl:    "#pgMlFileName",
    onPicked: (file) => {
      _pickedModelFile = file;
      syncJson();   // refresh JSON view + Python snippet
    },
  });

  // ── Initialize Parameters / Re-initialize ────────────────────
  // Matches gui.py: Initialize creates the Detection on the server
  // (loading any ML model) and reveals the runtime tabs. Re-initialize
  // tears it down and re-locks the UI. Each Run uses the existing
  // Detection without rebuilding — the ML model stays loaded.
  $("#pgInitBtn")?.addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    const ok = await initializePlayground();
    btn.disabled = false;
    if (ok) {
      _setInitialized(true);
      // _loadedMlType is set inside initializePlayground() from the
      // server's reply (it auto-detected the cmd from the pickle).
      toast("Initialized — model loaded, ready to Run", "ok");
      // Pre-fill the per-method `cls` filter with whatever classes the
      // loaded model knows about. Re-render method picker + fields so
      // unavailable ML cmds drop out and any visible cls textarea picks
      // up the freshly-loaded class names.
      try { _modelClasses = (await _vc.detection(PG_NAME).classes()) || []; }
      catch { _modelClasses = []; }
      // For ANOM, also pull the trained threshold so the slider can
      // start at the value the training pipeline picked.
      try { _trainedThreshold = await _vc.detection(PG_NAME).threshold(); }
      catch { _trainedThreshold = null; }
      renderMethodPicker();
      renderMethodFields();
    }
  });
  $("#pgReinitBtn")?.addEventListener("click", async () => {
    await teardownPlayground();
    _setInitialized(false);
    _resetOutputUI();
    _modelClasses = [];
    _loadedMlType = "";
    _trainedThreshold = null;
    renderMethodPicker();
    renderMethodFields();
  });
  // Always start un-initialized. The server-side Detection lives in the
  // WebSocket *session* — every page reload (and every server restart)
  // gets a fresh session with no detections. Restoring "initialized" from
  // localStorage would lie to the user and every Run would fault with
  // "detection not found". Cleaner to make them click Initialize once
  // per session.
  _setInitialized(false);
  try { localStorage.removeItem(INITIALIZED_KEY); } catch {}

  // Wire collapse toggles for any statically-rendered groups (Detection
  // tab, plus the static cards in Init/Image — Camera Mounting,
  // AI Models, and the Source card pinned at the top of Image).
  // Schema-rendered groups already get their handlers wired in
  // renderSections().
  document.querySelectorAll('.pg-build-pane[data-pane="detection"] .pg-group-head[data-toggle], .pg-build-pane[data-pane="image"] [data-section-key="__source"] .pg-group-head[data-toggle], .pg-build-pane[data-pane="init"] [data-section-key="__mounting"] .pg-group-head[data-toggle], .pg-build-pane[data-pane="init"] [data-section-key="__aimodels"] .pg-group-head[data-toggle]').forEach(btn => {
    btn.addEventListener("click", () => {
      const group = btn.closest(".pg-group");
      const fields = group.querySelector(".pg-fields");
      const willExpand = group.classList.contains("is-collapsed");
      group.classList.toggle("is-collapsed", !willExpand);
      btn.setAttribute("aria-expanded", String(willExpand));
      if (fields) {
        if (willExpand) fields.removeAttribute("hidden");
        else            fields.setAttribute("hidden", "");
      }
    });
  });

  // Actions
  $("#pgRunBtn")?.addEventListener("click", () => runOnce().catch(e => toast(`Run failed: ${e.message || e}`, "bad")));
  $("#pgLiveBtn")?.addEventListener("click", startLive);
  $("#pgStopBtn")?.addEventListener("click", stopLive);
  $("#pgPromoteBtn")?.addEventListener("click", openPromote);
  $("#pgSaveBtn")?.addEventListener("click", saveConfig);
  $("#pgLoadBtn")?.addEventListener("click", loadConfig);
  $("#pgLoadFile")?.addEventListener("change", onLoadFile);

  // Promote modal
  $("#pgPromoteClose")?.addEventListener("click", closePromote);
  $("#pgPromoteCancel")?.addEventListener("click", closePromote);
  $("#pgPromoteConfirm")?.addEventListener("click", submitPromote);
  $("#pgPromoteOverlay")?.addEventListener("click", (e) => { if (e.target.id === "pgPromoteOverlay") closePromote(); });

  // Image tabs
  $$(".pg-tab").forEach(b => b.addEventListener("click", () => setImgTab(b.dataset.tab)));

  // Pick-polygon and ROI action buttons — single delegated click handler so
  // it doesn't matter when the buttons get rendered.
  document.addEventListener("click", (e) => {
    const pick = e.target.closest(".pg-pick-poly");
    if (pick) {
      startRoiEdit(pick.dataset.pickTarget).catch(err => {
        toast(`ROI edit failed: ${err.message || err}`, "bad");
      });
      return;
    }
    const act = e.target.closest("[data-roi-act]");
    if (act) {
      const action = act.dataset.roiAct;
      const targetId = act.dataset.roiTarget;
      if      (action === "save")   _endRoiEdit(true);
      else if (action === "cancel") _endRoiEdit(false);
      else if (action === "remove") _removePolygon(targetId);
    }
  });
}

export function onShow() { refreshCameraPicker(); }
export function onHide() {
  stopLive();
  if (_roiEdit.active) _endRoiEdit(false);
}
