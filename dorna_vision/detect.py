from dorna_vision.visual import *
from dorna_vision.find import *
from dorna_vision.ai import *
from dorna_vision.util import *
from dorna_vision.draw import *
from dorna_vision.pose import *
from dorna_vision.limit import *
from dorna_vision.sort import *
from dorna2 import pose as dorna_pose

import time
import cv2 as cv
import os
import threading
import numpy as np

from dorna2 import Kinematic

class Detection(object):
    """docstring for Detect"""
    def __init__(self,
            camera=None,
            robot=None,
            camera_mount="dorna_ta_j4_1",
            frame=[0, 0, 0, 0, 0, 0], 
            feed="color_img",
            rot= 0, 
            intensity={"a":1.0, "b":0},
            color={"low_hsv":[0, 0, 0], "high_hsv":[255, 255, 255], "inv":0},
            roi={"corners": [], "inv": 0, "crop": 0},
            detection={},
            limit = {#"area":[], "aspect_ratio":[], "inv":0,
                     #"xyz":{"x":[], "y":[], "z":[], "inv":0},
                     #"bb":{"area":[], "aspect_ratio":[], "inv":0},
                     #"center":{"width":[], "height":[], "inv":0},
                     # rvec/tvec limits no longer live in `limit`; call
                     # detection.filter_rvec(...) / filter_tvec(...) on
                     # the result list instead.
                     },
            # 6D pose used to be a stage configured here as `pose`.
            # Moved to opt-in helpers: detection.pose_plane(...) /
            # pose_kp(...) called after run() on the result you want.
            sort={"cmd": None, "max_det":100}, # {"cmd":"conf", "ascending":False, "max_det":100}, {"cmd":"pxl", "pxl":[w,h], "ascending":True, "max_det":100}, {"cmd":"xyz", "xyz":[x,y,z], "ascending":True, "max_det":100}
            display={"label":0, "save_img":0, "save_img_roi":0},
            **kwargs
        ):
        super(Detection, self).__init__()
        
        self.camera = camera
        self.robot = robot
        self.camera_mount_label = camera_mount
        self.camera_mount = self.set_camera_mount(camera_mount)
        self.frame = frame
        self.feed = feed
        self.rot = rot
        self.intensity = intensity
        self.color = color
        self.roi = roi
        self.detection = detection
        self.limit = limit
        self.sort = sort
        self.display = display
        self.kwargs = kwargs

        # retval
        self.retval = self.init_retval()

        # thread list
        self.thread_list = []

        # camera data
        self.camera_data = None

        # img
        self.img = np.zeros((10, 10), dtype=np.uint8)

        # kinematic
        self.kinematic = Kinematic()

        # ML detection
        if "cmd" in self.detection and self.detection["cmd"] == "cls":
            self.init_cls(self.detection["path"])
        if "cmd" in self.detection and self.detection["cmd"] == "od":
            self.init_od(self.detection["path"])
        if "cmd" in self.detection and self.detection["cmd"] == "rod":
            self.init_rod(self.detection["path"])
        if "cmd" in self.detection and self.detection["cmd"] == "anom":
            self.init_anom(self.detection["path"])
        if "cmd" in self.detection and self.detection["cmd"] == "kp":
            self.init_kp(self.detection["path"])
        elif "cmd" in self.detection and self.detection["cmd"] == "ocr":
            self.init_ocr()


    def set_camera_mount(self, camera_mount):
        retval = {"ej": [0, 0, 0, 0, 0, 0, 0, 0]}
        if self.robot is None:
            pass
        elif type(camera_mount) == dict and "type" in camera_mount and "T" in camera_mount and "ej" in camera_mount:
            retval = {
                "type": camera_mount["type"],
                "T": dorna_pose.xyzabc_to_T(np.array(camera_mount["T"])),
                "ej": camera_mount["ej"]
            }
        elif type(camera_mount) == str:
            for i in range(len(self.robot.config["camera_mount"])):
                if self.robot.config["camera_mount"][i]["type"] == camera_mount:
                    tmp =dict(self.robot.config["camera_mount"][i])
                    retval = {
                        "type": tmp["type"],
                        "T": dorna_pose.xyzabc_to_T(np.array(tmp["T"])),
                        "ej": tmp["ej"]
                    }
                    break

        return retval

            

    def init_kp(self, path):
        self.kp = KP(path)


    def init_cls(self, path):
        self.cls = CLS(path)


    def init_od(self, path):
        self.od = OD(path)


    def init_rod(self, path):
        self.rod = ROD(path)


    def init_anom(self, path):
        self.anom = ANOM(path)


    def init_ocr(self):
        self.ocr = OCR()


    def threshold(self):
        """
        Trained anomaly threshold for the loaded ANOM model. Used by the
        playground to prefill the threshold slider with a sensible default
        — anomaly scores have no natural [0, 1] bound, so without this the
        user is just guessing. Returns None for non-ANOM cmds.
        """
        cmd = (self.detection or {}).get("cmd")
        if cmd == "anom" and getattr(self, "anom", None):
            v = getattr(self.anom, "threshold", None)
            return float(v) if v is not None else None
        return None


    def classes(self):
        """
        Return the list of class names known to whichever ML model is
        loaded for the current detection cmd. Used by the playground to
        pre-fill the per-method `cls` filter field after Initialize, so
        users see which classes are available rather than guessing them.

        Returns an empty list when no ML model is loaded (or when the
        current cmd is a non-ML detector like cnt / poly / aruco / ocr).

        NOTE: not named `cls()` because `self.cls` is already used as the
        CLS classifier instance set by init_cls — that instance attribute
        shadows class methods in Python, so the proxy call would end up
        invoking the classifier's inference instead.
        """
        cmd = (self.detection or {}).get("cmd")
        if cmd == "od"   and getattr(self, "od",   None):  return list(self.od.cls)
        if cmd == "rod"  and getattr(self, "rod",  None):  return list(self.rod.cls)
        if cmd == "cls"  and getattr(self, "cls",  None) and hasattr(self.cls, "cls"):
            return list(self.cls.cls)
        if cmd == "anom" and getattr(self, "anom", None):  return list(self.anom.cls)
        if cmd == "kp"   and getattr(self, "kp",   None):
            return list(getattr(self.kp, "keypoint_names", []) or [])
        return []


    def get_camera_data(self, data=None):
        self.camera_data = {key:None for key in ["depth_frame", "ir_frame", "color_frame", "depth_img", "ir_img", "color_img", "depth_int", "frames", "joint", "K", "D", "timestamp"]}

        if type(data) == str: # read from file
            data = cv.imread(data)
            self.camera_data[self.feed] = data
            self.camera_data["timestamp"] = time.time()
        
        elif type(data) == dict: # keep the current
            for k in data:
                self.camera_data[k] = data[k]
        
        else: # update
            joint = None
            depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, timestamp = self.camera.get_all()
            K = self.camera.camera_matrix(depth_int)
            D = self.camera.dist_coeffs(depth_int)
            try:
                joint = self.robot.joint()
                if "ej" in self.camera_mount:
                    for i in range(min(len(self.camera_mount["ej"]), len(joint))):
                        joint[i] += self.camera_mount["ej"][i]

            except Exception as ex:
                pass
            self.camera_data = {
                "depth_frame": depth_frame,
                "ir_frame": ir_frame,
                "color_frame": color_frame,
                "depth_img": depth_img,
                "ir_img": ir_img,
                "color_img": color_img,
                "depth_int": depth_int,
                "frames": frames,
                "joint": joint,
                "timestamp": timestamp,
                "K": K,
                "D": D
            }
        return self.camera_data


    def pixel_to_xyz(self, pxl):
        h, w, _ =self.camera_data["depth_img"].shape
        try:
            
            # rot
            if self.rot == 90:
                _pxl = [h-1-pxl[1], pxl[0]]
            elif self.rot == 180:
                _pxl = [w-1-pxl[0], h-1-pxl[1]]
            elif self.rot == 270:
                _pxl = [pxl[1], w-1-pxl[0]]
            else:
                _pxl = [pxl[0], pxl[1]]

            # xyz_target_2_cam
            xyz_target_to_cam = self.camera.xyz(_pxl, self.camera_data["depth_frame"], self.camera_data["depth_int"])[0].tolist()
            T_target_to_cam = np.array(dorna_pose.xyzabc_to_T(np.concatenate((np.array(xyz_target_to_cam), np.array([0, 0, 0])))))
            
            # apply frame
            T_target_to_frame = np.matmul(self.frame_mat_inv, T_target_to_cam)
            xyz_target_to_frame = dorna_pose.T_to_xyzabc(T_target_to_frame)
            xyz = xyz_target_to_frame[0:3]
            
        except:
            xyz = [0, 0, 0]
        return xyz


    def xyz_to_pixel(self, xyz):
        """
        Given a 3D point in your “frame” coords, return the [u,v] pixel in the unrotated image.
        """
        try:
            # 1) Build transform from frame → camera (same as before)
            xyzabc = np.hstack((xyz, [0.0, 0.0, 0.0]))
            T_tgt_to_frame = np.array(dorna_pose.xyzabc_to_T(xyzabc))
            frame_mat    = np.linalg.inv(self.frame_mat_inv)
            T_tgt_to_cam = frame_mat @ T_tgt_to_frame
            xyz_cam      = dorna_pose.T_to_xyz(T_tgt_to_cam)[:3]

            # 2) Project with your pure function
            u, v = self.camera.pixel(xyz_cam, self.camera_data["depth_int"])

            # 3) Compensate for any image rotation
            h, w, _ = self.camera_data["depth_img"].shape
            if   self.rot ==  90: pxl = [h - 1 - int(round(v)), int(round(u))]
            elif self.rot == 180: pxl = [w - 1 - int(round(u)), h - 1 - int(round(v))]
            elif self.rot == 270: pxl = [int(round(v)), w - 1 - int(round(u))]
            else:                 pxl = [int(round(u)),     int(round(v))]
        except:
            pxl = [0, 0]
        return pxl


    def xyz(self, pxl):
        return self.pixel_to_xyz(pxl)


    def pixel(self, xyz):
        return self.xyz_to_pixel(xyz)


    def init_retval(self):
        return {"all":[], "valid":[], "camera_data": None, "frame_mat_inv": None}


    def run(self, data=None, **kwargs):
        # return
        self.retval = self.init_retval()
        retval = []
        try:
            # assign the new value
            for key, value in kwargs.items():
                # Check if the attribute already exists in the class
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # update camera_data
            camera_data = self.get_camera_data(data)
            _img = camera_data[self.feed].copy()

            # ori
            if self.rot != 0:
                _img = rotate_and_flip(_img, rotate=self.rot)
            
            # frame
            self.frame_mat_inv = np.linalg.inv(dorna_pose.xyzabc_to_T(np.array(self.frame)))
            if self.robot is not None and camera_data["joint"] is not None:
                joint = camera_data["joint"][0:6]
                if "type" in self.camera_mount and "T" in self.camera_mount and self.camera_mount["type"].startswith("dorna_ta_j4"):
                    T_camholder_to_base = self.robot.kinematic.Ti_r_world(i=5, joint=joint[0:6])
                    T_cam_to_camholder = np.matrix(self.camera_mount["T"])
                    T_cam_to_base = np.matmul(T_camholder_to_base, T_cam_to_camholder)
                    self.frame_mat_inv = np.matmul(self.frame_mat_inv, T_cam_to_base)

            # intensity
            img_adjust = intensity(_img.copy(), **self.intensity)

            # color
            img_adjust = color_mask(img_adjust, **self.color)
            height, width = img_adjust.shape[0:2]

            # roi
            _roi = ROI(img_adjust.copy(), **self.roi)
            img_roi = _roi.img

            # thr
            self.img_thr = np.zeros(img_roi.shape[:2], dtype=np.uint8)
            if "cmd" in self.detection:
                # detection
                if self.detection["cmd"] == "elp":
                    # [pxl, corners, (pxl, (major_axis, minor_axis), rot),...]
                    result = ellipse(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": self.detection["cmd"], "conf": 1, "center": _roi.pxl_to_orig(r[0]), 
                    "corners": [_roi.pxl_to_orig(x) for x in r[1]]} for r in result]

                if self.detection["cmd"] in ["poly", "cnt"]:
                    # thr
                    self.img_thr = binary_thr(img_roi, **self.detection)

                    # find contour: [[pxl, corners, cnt], ...]
                    result = contour(self.img_thr, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": self.detection["cmd"], "conf": 1, "center": _roi.pxl_to_orig(r[0]), 
                    "corners": [_roi.pxl_to_orig(x) for x in r[1]]} for r in result]

                elif self.detection["cmd"] == "barcode":
                    # [pxl, corners, (pxl, (major_axis, minor_axis), rot),...]
                    result = barcode(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[1], "format": r[0], "conf": 1, "center": _roi.pxl_to_orig(r[3]), 
                    "corners": [_roi.pxl_to_orig(x) for x in r[2]]} for r in result]

                elif self.detection["cmd"] == "blob":
                    # The schema exposes `threshold` and `area` as range
                    # pairs (lo, hi) so the GUI can't put min > max. Split
                    # them back into the {min,max}Threshold / {min,max}Area
                    # kwargs that find.blob() (and OpenCV) actually expect.
                    blob_args = dict(self.detection)
                    thr_pair = blob_args.pop("threshold", None)
                    if isinstance(thr_pair, (list, tuple)) and len(thr_pair) == 2:
                        lo, hi = thr_pair
                        # OpenCV's SimpleBlobDetector defaults minRepeatability=2
                        # — it needs at least 2 distinct threshold steps to
                        # validate a blob. If the user collapses the range to
                        # a single value, push max up by 1 so we still produce
                        # results (and avoid the noisy "incompatible" warning).
                        if hi <= lo:
                            hi = lo + 1
                        blob_args["minThreshold"], blob_args["maxThreshold"] = lo, hi
                    area_pair = blob_args.pop("area", None)
                    if isinstance(area_pair, (list, tuple)) and len(area_pair) == 2:
                        blob_args["minArea"], blob_args["maxArea"] = area_pair
                    # [pxl, corners, (pxl, (major_axis, minor_axis), rot),...]
                    result = blob(img_roi, **blob_args)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": "blob", "conf": 1, "center": _roi.pxl_to_orig(r["center"]),
                    "corners": [_roi.pxl_to_orig(x) for x in r["corners"]]} for r in result]

                elif self.detection["cmd"] == "mser":
                    # Schema exposes `area` as a (lo, hi) range pair so the
                    # GUI can't put min > max — split it back into the
                    # min_area / max_area kwargs MSER expects.
                    mser_args = dict(self.detection)
                    area_pair = mser_args.pop("area", None)
                    if isinstance(area_pair, (list, tuple)) and len(area_pair) == 2:
                        mser_args["min_area"], mser_args["max_area"] = area_pair
                    result = mser(img_roi, **mser_args)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": "mser", "conf": 1, "center": _roi.pxl_to_orig(r["center"]),
                    "corners": [_roi.pxl_to_orig(x) for x in r["corners"]]} for r in result]

                elif self.detection["cmd"] == "elp_fit":
                    # [pxl, corners, (pxl, (major_axis, minor_axis), rot),...]
                    result = elp_fit(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": "elp_fit", "conf": 1, "center": _roi.pxl_to_orig(r["center"]),
                    "corners": [_roi.pxl_to_orig(x) for x in r["corners"]]} for r in result]


                elif self.detection["cmd"] == "aruco" and camera_data["K"] is not None:
                    try:
                        result = aruco(img_roi, camera_data["K"], camera_data["D"], **self.detection)
                        # build _retval  (FIX: stop indexing [0] on rvec/tvec/id)
                        _retval = [{
                            "timestamp": camera_data["timestamp"],
                            "cls": str(int(np.asarray(r[2][0]).item())),                              # CHANGED
                            "conf": 1,
                            "center": _roi.pxl_to_orig(r[0]),
                            "corners": [_roi.pxl_to_orig(x) for x in r[1]],
                            "rvec": dorna_pose.rvec_to_abc(r[2][2].flatten().tolist()),  # CHANGED
                            "tvec": r[2][3].flatten().tolist(),                              # CHANGED
                        } for r in result]
                        
                        # draw
                        draw_aruco(img_adjust, np.array([r[2][0] for r in result]),
                                    np.array([[r["corners"] for r in _retval]], dtype=np.float32), 
                                    [r[2][2] for r in result], [r[2][3] for r in result], 
                                    camera_data["K"], camera_data["D"],
                                    self.detection["marker_length"])
                       
                        # add to retval
                        retval = _retval

                        # --- frame application (unchanged logic, but now r["tvec"]/r["rvec"] are clean 3-lists) ---
                        for r in retval:
                            T_target_to_cam = np.array(dorna_pose.xyzabc_to_T(np.array(r["tvec"] + r["rvec"])))
                            T_target_to_frame = self.frame_mat_inv @ T_target_to_cam
                            xyzabc = dorna_pose.T_to_xyzabc(T_target_to_frame)
                            r["tvec"], r["rvec"] = xyzabc[:3], xyzabc[3:]

                    except Exception as ex:
                        print("Exception: ", ex)
                        pass
                elif self.detection["cmd"] == "charuco" and camera_data["K"] is not None:
                    try:
                        # [[pxl, corners, (id, rvec, tvec)], ...]
                        result = charuco(img_roi, camera_data["K"], camera_data["D"], **self.detection)
                        if result:
                            retval =[{"timestamp": camera_data["timestamp"], "cls": "charuco", "conf": 1, "center": [0, 0],
                            "corners": [0, 0], "rvec": dorna_pose.rvec_to_abc(result[0].flatten().tolist()), "tvec": result[1].flatten().tolist(), "err": result[4]}]

                            # draw
                            if self.display and "label" in self.display and self.display["label"] >= 0:
                                draw_charuco(img_adjust, result[2], result[3], self.detection["sqr_x"]*self.detection["sqr_length"], camera_data["K"], camera_data["D"], result[0], result[1])
                            # xyz_target_2_cam
                            for r in retval:
                                T_target_to_cam = dorna_pose.xyzabc_to_T(np.array(r["tvec"]+ r["rvec"]))

                                # apply frame
                                T_target_to_frame = np.matmul(self.frame_mat_inv, T_target_to_cam)
                                xyzabc_target_to_frame = dorna_pose.T_to_xyzabc(T_target_to_frame)
                                r["tvec"] = xyzabc_target_to_frame[0:3]
                                r["rvec"] = xyzabc_target_to_frame[3:6]
                    except Exception as ex:
                        pass
                elif self.detection["cmd"] == "ocr":
                    # Lazy-init: OCR ships its own bundled models, so it
                    # doesn't need to be wired at __init__ time. We only
                    # pay the OpenVINO compile cost on first OCR run.
                    if not hasattr(self, "ocr") or self.ocr is None:
                        self.init_ocr()
                    result = self.ocr.ocr(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[1][0], "conf": r[1][1], "center": _roi.pxl_to_orig([(sum([p[0] for p in r[0]])/len(r[0])), sum([p[1] for p in r[0]])/len(r[0])]), 
                    "corners": [_roi.pxl_to_orig(sublist) for sublist in r[0]]} for r in result]
                elif self.detection["cmd"] == "od":
                    result = self.od(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r.cls, "conf": r.prob, "center": _roi.pxl_to_orig([r.rect.x+r.rect.w/2, r.rect.y+r.rect.h/2]),
                    "corners": [_roi.pxl_to_orig(pxl) for pxl in [[r.rect.x, r.rect.y], [r.rect.x+r.rect.w, r.rect.y], [r.rect.x+r.rect.w, r.rect.y+r.rect.h], [r.rect.x, r.rect.y+r.rect.h]]]} for r in result]
                elif self.detection["cmd"] == "rod":
                    # Rotated detection: corners come from the ROD class as
                    # the four (already rotated) rectangle vertices, and the
                    # center is `rect.cx, rect.cy` rather than a midpoint of
                    # an axis-aligned box. The corners themselves encode
                    # orientation — no separate `angle` field needed.
                    result = self.rod(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r.cls, "conf": r.prob,
                    "center": _roi.pxl_to_orig([r.rect.cx, r.rect.cy]),
                    "corners": [_roi.pxl_to_orig(pxl) for pxl in r.corners]} for r in result]
                elif self.detection["cmd"] == "anom":
                    # Anomaly: one verdict (pass/fail) per image. Same
                    # ROI-wide rectangle treatment as cls, with 25-px
                    # inset so the label drawn above the top edge stays
                    # inside the frame. No localization / hot-spots —
                    # anomaly is a one-class problem; per-pixel info is
                    # diagnostic and intentionally not surfaced here.
                    roi_height, roi_width = img_roi.shape[0:2]
                    result = self.anom(img_roi, **self.detection)
                    pad = 25
                    x0 = min(pad, max(0, roi_width  - 1))
                    y0 = min(pad, max(0, roi_height - 1))
                    x1 = max(0,  roi_width  - 1 - pad)
                    y1 = max(0,  roi_height - 1 - pad)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": result.cls, "conf": float(result.score),
                    "center": [int(width/2), int(height/2)],
                    "corners": [_roi.pxl_to_orig([x0, y0]),
                                _roi.pxl_to_orig([x1, y0]),
                                _roi.pxl_to_orig([x1, y1]),
                                _roi.pxl_to_orig([x0, y1])]}]
                elif self.detection["cmd"] == "cls":
                    roi_height, roi_width = img_roi.shape[0:2]

                    # Inset the per-image classification rectangle so the
                    # label text drawn above its top edge stays inside the
                    # frame (cv.getTextSize for our default font is ~14px,
                    # plus a 5px gap → 25 covers it with breathing room).
                    pad = 25
                    x0 = min(pad, max(0, roi_width  - 1))
                    y0 = min(pad, max(0, roi_height - 1))
                    x1 = max(0,  roi_width  - 1 - pad)
                    y1 = max(0,  roi_height - 1 - pad)

                    result = self.cls(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[0], "conf": r[1], "center": [int(width/2), int(height/2)],
                    "corners": [_roi.pxl_to_orig([x0, y0]),
                                _roi.pxl_to_orig([x1, y0]),
                                _roi.pxl_to_orig([x1, y1]),
                                _roi.pxl_to_orig([x0, y1])]} for r in result]
                elif self.detection["cmd"] == "kp":
                    # New KP flow: ONE keypoint set per image (no OD
                    # first-stage iteration). The new KP class runs
                    # top-down on the supplied bbox; passing bbox=None
                    # uses the whole img_roi, with `padding` controlling
                    # the affine crop margin (defaults to 1.25× = 25%
                    # scale-up over the bbox, matching training).
                    kp_prm = {}
                    if "conf" in self.detection:
                        kp_prm["conf"] = self.detection["conf"]
                    if "padding" in self.detection:
                        kp_prm["padding"] = self.detection["padding"]

                    # bbox is exposed in the API as 4 corner points
                    # (same shape as ROI), so it's pickable with the
                    # polygon UI. Convert to (x, y, w, h) axis-aligned
                    # rect and remap from original-image coords to
                    # img_roi coords before handing to KP.
                    bbox_xywh = None
                    bbox_corners = self.detection.get("bbox")
                    if isinstance(bbox_corners, (list, tuple)) and len(bbox_corners) >= 3:
                        roi_pts = [_roi.pxl_orig_to_roi(pxl) for pxl in bbox_corners]
                        xs = [float(p[0]) for p in roi_pts]
                        ys = [float(p[1]) for p in roi_pts]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        bbox_xywh = (x_min, y_min, x_max - x_min, y_max - y_min)

                    kp_out = self.kp(img_roi, bbox=bbox_xywh, **kp_prm)

                    # Flatten: one result entry per keypoint, matching
                    # the shape of every other detection method (od,
                    # rod, blob, cnt, ...). Each entry's `cls` is the
                    # keypoint's name; `corners` is a small axis-aligned
                    # box around its center so the standard draw / bb
                    # filter / sort code keeps working uniformly. `xyz`
                    # gets attached later by the depth-mapping loop.
                    KP_PAD = 5  # px on each side of the center
                    retval = []
                    for kp in kp_out.keypoints:
                        cx, cy = _roi.pxl_to_orig([kp.x, kp.y])
                        retval.append({
                            "timestamp": camera_data["timestamp"],
                            "cls":     kp.name,
                            "conf":    kp.conf,
                            "center":  [cx, cy],
                            "corners": [
                                [cx - KP_PAD, cy - KP_PAD],
                                [cx + KP_PAD, cy - KP_PAD],
                                [cx + KP_PAD, cy + KP_PAD],
                                [cx - KP_PAD, cy + KP_PAD],
                            ],
                        })

            # add id
            for i in range(len(retval)):
                retval[i]["id"] = i

            # sort
            if "cmd" in self.sort:
                if self.sort["cmd"] == "shuffle":
                    retval = Sort().shuffle(retval)
                elif self.sort["cmd"] == "conf":
                    retval = Sort().conf(retval, ascending=self.sort["ascending"])
                elif self.sort["cmd"] == "pxl":
                    retval = Sort().pixel(retval, pxl=self.sort["pxl"], ascending=self.sort["ascending"])
                elif self.sort["cmd"] == "area":
                    retval = Sort().area(retval, ascending=self.sort["ascending"])

            # ej
            if "ej" in self.camera_mount:
                for r in retval:
                    r["ej"] = list(self.camera_mount["ej"])

            # all
            self.retval["all"] = list(retval)
            
            # valid
            if self.sort["max_det"] <= 0:
                self.sort["max_det"] = len(retval)

            for r in retval:
                # max det
                if len(self.retval["valid"]) >= self.sort["max_det"]:
                    break

                # valid area, aspect ratio
                if "bb" in self.limit:
                    if not Valid().bb(r["corners"], **self.limit["bb"]):
                        continue

                # valid center
                if "center" in self.limit:
                    if not Valid().center(r["center"], **self.limit["center"]):
                        continue

                # draw bb
                if "cmd" in self.detection and self.detection["cmd"] not in ["aruco", "charuco"] and "label" in self.display and self.display["label"]>=0:
                    color_label = self._color_for(r.get("cls"))
                    draw_corners(img_adjust, r["cls"], r["conf"], r["corners"], color=color_label, label=self.display["label"])

                # assign xyz if the data comes from camera
                if camera_data["depth_frame"] is not None and camera_data["K"] is not None:
                    r["xyz"] = self.xyz(r["center"])

                # xyz limit
                if "xyz" in self.limit:
                    if "xyz" not in r or not Valid().xyz(r["xyz"], **self.limit["xyz"]):
                        continue

                # 6D pose (plane / kp / PNP) and rvec/tvec limits used to
                # run here as a stage inside Detection.run(). They moved
                # to opt-in helper methods on this class:
                #   self.pose_plane(detection, plane)
                #   self.pose_kp(kp_list, kp_geometry)
                #   self.filter_rvec(results, **limits)
                #   self.filter_tvec(results, **limits)
                # Call them on the result list after run(). ArUco /
                # ChArUco still emit rvec/tvec from their cmd branches
                # because pose is intrinsic to those detectors.

                # append
                self.retval["valid"].append(dict(r))

            
            # save image
            if "save_img" in self.display and self.display["save_img"]:
                if isinstance(self.display["save_img"], str):
                    # save image path name
                    save_img_path = self.display["save_img"]
                else:
                    # make directory if not exists
                    os.makedirs("output", exist_ok=True)
                    save_img_path = "output/"+str(int(camera_data["timestamp"]))+".jpg"
                # Create a thread to perform the file write operation
                thread = threading.Thread(target=cv.imwrite, args=(save_img_path, img_adjust))
                thread.start()
                self.thread_list.append(thread)

            # save image
            if "save_img_roi" in self.display and self.display["save_img_roi"]:
                if isinstance(self.display["save_img_roi"], str):
                    # save image path name
                    save_img_path = self.display["save_img_roi"]
                else:
                    # make directory if not exists
                    os.makedirs("output", exist_ok=True)
                    save_img_path = "output/roi_"+str(int(camera_data["timestamp"]))+".jpg"
                # Create a thread to perform the file write operation
                thread = threading.Thread(target=cv.imwrite, args=(save_img_path, img_roi))
                thread.start()
                self.thread_list.append(thread)


            # img
            self.img = img_adjust

            # retval — build a single dict and assign atomically so nothing
            # can observe `self.retval["camera_data"]` as None mid-update.
            cd_out = dict(camera_data) if isinstance(camera_data, dict) else {}
            cd_out["img"] = img_adjust.copy()
            cd_out["img_roi"] = img_roi.copy()
            self.retval["camera_data"] = cd_out
            self.retval["frame_mat_inv"] = self.frame_mat_inv.copy()
        except Exception as ex:
            import traceback
            traceback.print_exc()

        return list(self.retval["valid"])


    # ── Draw-time color resolution ──────────────────────────────────
    # Color is a presentation concern (annotated overlay), not result
    # data — so it lives here, not in the result dicts. Looks up the
    # color the active ML model assigned to this class name; falls
    # back to green for classical detectors (no class palette) or
    # when the model wasn't loaded.
    def _color_for(self, cls_name):
        cmd   = (self.detection or {}).get("cmd")
        model = getattr(self, cmd, None) if cmd in ("od", "rod", "cls", "kp", "anom") else None
        if model is None or cls_name is None:
            return (0, 255, 0)
        # KP keeps a flat list of keypoint names + per-index palette.
        if cmd == "kp":
            try:
                idx = list(model.keypoint_names).index(cls_name)
                return model._kp_color(idx)
            except (ValueError, AttributeError):
                return (0, 255, 0)
        # OD / ROD / CLS / ANOM share the (cls list, colors map) shape.
        try:
            idx = list(model.cls).index(cls_name)
            return _color_for_class(idx, model.cls, model.colors)
        except (ValueError, AttributeError):
            return (0, 255, 0)


    def grasp(self, target_id, target_rvec, gripper_opening,
              finger_wdith, finger_location,
              mask_type="bb", prune_factor=2,
              num_steps=360, search_angle=(0, 360)):
        from dorna_vision.grasp import collision_free_rvec
        return collision_free_rvec(
            target_id=target_id,
            target_rvec=target_rvec,
            gripper_opening=gripper_opening,
            finger_wdith=finger_wdith,
            finger_location=finger_location,
            detection_obj=self,
            mask_type=mask_type,
            prune_factor=prune_factor,
            num_steps=num_steps,
            search_angle=search_angle,
        )


    # ── 6D pose helpers ─────────────────────────────────────────────
    # Replace the old self.pose["cmd"] stage that ran inside Detection.run().
    # The user calls these AFTER detection on a result they care about,
    # which (a) keeps detection results free of conditionally-attached
    # rvec/tvec fields, (b) lets the user pick which method to apply
    # without baking it into the detection config, and (c) mirrors how
    # `grasp()` already works as an opt-in post-processing step.
    #
    # Both helpers read camera state from self.camera_data and the user
    # frame from self.frame_mat_inv — both are populated by the most
    # recent `run()`. Calling a pose helper before any run() yields no
    # camera context and the underlying pose function will raise.
    #
    # ArUco / ChArUco detections still carry rvec/tvec directly in their
    # result dicts (pose is intrinsic to those detectors); these helpers
    # are for the methods that aren't.

    def pose_plane(self, detection, plane, **kwargs):
        """
        Fit a plane to the depth points inside the detection's bounding
        box. Requires a depth frame from the most recent run().

        plane: list of at least 3 [x, y, z] points defining the plane
               geometry in the user's frame.

        Returns (rvec, tvec) in the user frame, or None if the fit
        failed (e.g. degenerate depth, fewer than 3 plane points).
        """
        cd = self.camera_data
        if (cd is None or cd.get("depth_frame") is None
                or not isinstance(plane, (list, tuple)) or len(plane) < 3):
            return None
        result = Plane().pose(
            detection["corners"], plane, self.camera,
            cd["depth_frame"], cd["depth_int"], self.frame_mat_inv,
            **kwargs,
        )
        if not result:
            return None
        return result[0], result[1]


    def pose_kp(self, kp_list, kp_geometry, **kwargs):
        """
        Compute 6D pose from a keypoint list. Picks the underlying
        method by geometry length:
          2 points  → Pose_two_point (depth-anchored 2-point pose)
          ≥4 points → PNP (perspective-n-point with RANSAC)
          (3 pts)   → not supported, returns None.

        kp_list:     the flat list returned by `detection.run()` when
                     cmd=="kp" — each entry is one keypoint with
                     {cls, conf, center, corners, xyz?}. Just pass the
                     `results` list directly.
        kp_geometry: dict mapping keypoint name → [x, y, z] in the
                     object's reference frame. Names not in kp_list
                     are ignored; extra kp_list entries with names
                     not in kp_geometry are also ignored.

        Returns (rvec, tvec) in the user frame, or None if pose failed.
        """
        cd = self.camera_data
        if cd is None or cd.get("K") is None or not kp_list or not kp_geometry:
            return None
        n = len(kp_geometry)
        if n == 2:
            result = Pose_two_point().pose(
                kp_list=kp_list, kp_geometry=kp_geometry,
                camera_matrix=cd["K"], dist_coeffs=cd["D"],
                frame_mat_inv=self.frame_mat_inv, **kwargs,
            )
        elif n >= 4:
            result = PNP().pose(
                kp_list=kp_list, kp_geometry=kp_geometry,
                camera_matrix=cd["K"], dist_coeffs=cd["D"],
                frame_mat_inv=self.frame_mat_inv, **kwargs,
            )
        else:
            return None
        if not result:
            return None
        return result[0], result[1]


    # ── Pose-based filters ──────────────────────────────────────────
    # Returns a NEW filtered list — never mutates the input. Entries
    # without rvec/tvec are dropped (they obviously can't pass an
    # rvec/tvec constraint). Pair these with the pose helpers above:
    # compute pose first, attach to results, then filter.

    def filter_rvec(self, detections, **limits):
        """
        Keep detections whose rvec satisfies the angle constraints.
        See Valid().rvec for accepted kwargs (rvec_base, x_angle,
        y_angle, z_angle, inv).
        """
        return [
            d for d in detections
            if "rvec" in d and Valid().rvec(d["rvec"], **limits)
        ]


    def filter_tvec(self, detections, **limits):
        """
        Keep detections whose tvec lies inside the x/y/z range. See
        Valid().tvec for accepted kwargs (x, y, z, inv).
        """
        return [
            d for d in detections
            if "tvec" in d and Valid().tvec(d["tvec"], **limits)
        ]


    def close(self):
        # save image
        for thread in self.thread_list:
            thread.join()

        # object detection
        if hasattr(self, "od"):
            self.od.__del__()

        # ocr
        if hasattr(self, "ocr"):
            self.ocr.__del__()