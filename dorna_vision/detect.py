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
import matplotlib.pyplot as plt
from IPython.display import clear_output

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
                     #"rvec":{"rvec_base":[], "x_angle":[], "y_angle":[], "z_angle":[], "inv":0}, 
                     #"tvec":{"x": [], "y": [], "z": [], "inv":0}
                     },
            pose = {}, # {"cmd":"kp", "kp":{"plate": {"o":[0, 0, 0], ...}}}
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
        self.pose = pose
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


    def init_ocr(self):
        self.ocr = OCR()


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
                    # [pxl, corners, (pxl, (major_axis, minor_axis), rot),...]
                    result = blob(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": "blob", "conf": 1, "center": _roi.pxl_to_orig(r["center"]), 
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
                    result = self.ocr.ocr(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[1][0], "conf": r[1][1], "center": _roi.pxl_to_orig([(sum([p[0] for p in r[0]])/len(r[0])), sum([p[1] for p in r[0]])/len(r[0])]), 
                    "corners": [_roi.pxl_to_orig(sublist) for sublist in r[0]]} for r in result]
                elif self.detection["cmd"] == "od":
                    result = self.od(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r.cls, "conf": r.prob, "center": _roi.pxl_to_orig([r.rect.x+r.rect.w/2, r.rect.y+r.rect.h/2]), 
                    "corners": [_roi.pxl_to_orig(pxl) for pxl in [[r.rect.x, r.rect.y], [r.rect.x+r.rect.w, r.rect.y], [r.rect.x+r.rect.w, r.rect.y+r.rect.h], [r.rect.x, r.rect.y+r.rect.h]]],"color": r.color} for r in result]
                elif self.detection["cmd"] == "cls":
                    roi_height, roi_width = img_roi.shape[0:2]

                    result = self.cls(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[0], "conf": r[1], "center": [int(width/2), int(height/2)], 
                    "corners": [_roi.pxl_to_orig([min(5,roi_width-1), min(5,roi_height-1)]), 
                                _roi.pxl_to_orig([max(0,roi_width-6), min(5,roi_height-1)]), 
                                _roi.pxl_to_orig([max(0,roi_width-6), max(0,roi_height-6)]), 
                                _roi.pxl_to_orig([min(5,roi_width-1), max(0,roi_height-6)])],"color": r[2]} for r in result]
                elif self.detection["cmd"] == "kp":
                    # kp parameters
                    kp_prm={}
                    if "conf" in self.detection:
                        kp_prm["conf"] = self.detection["conf"]
                    if "cls" in self.detection and  isinstance(self.detection["cls"], dict):
                        kp_prm["cls"] = list(self.detection["cls"].keys())


                    result = self.kp.od(img_roi, **kp_prm)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r.cls, "conf": r.prob, "center": _roi.pxl_to_orig([r.rect.x+r.rect.w/2, r.rect.y+r.rect.h/2]), 
                    "corners": [_roi.pxl_to_orig(pxl) for pxl in [[r.rect.x, r.rect.y], [r.rect.x+r.rect.w, r.rect.y], [r.rect.x+r.rect.w, r.rect.y+r.rect.h], [r.rect.x, r.rect.y+r.rect.h]]],
                    "color": r.color} for r in result]

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
                
                # init
                pose_result = []
                kp_pxl = []

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
                    color_label = (0,255,0)
                    if "color" in r:
                        color_label = r["color"]
                    draw_corners(img_adjust, r["cls"], r["conf"], r["corners"], color=color_label, label=self.display["label"])

                # find keypoints
                if "cmd" in self.detection and self.detection["cmd"] == "kp":
                    # init
                    r["kp"] = []

                    # kp parameters
                    kp_prm={}
                    if "conf" in self.detection:
                        kp_prm["conf"] = self.detection["conf"]
                    if r["cls"] in self.detection["cls"]:
                        kp_prm["cls"] = self.detection["cls"][r["cls"]]
                    
                    # run
                    kps = self.kp.kp(img_roi, label=r["cls"], bb=[_roi.pxl_orig_to_roi(pxl) for pxl in r["corners"]], **kp_prm)
                    for kp in kps:
                        r["kp"].append({
                            "cls": kp.cls,
                            "conf": kp.prob,
                            "center": _roi.pxl_to_orig(kp.center)
                        })
                    
                    kp_pxl = [[k["cls"], k["center"]] for k in r["kp"]]

                # draw kp
                if kp_pxl and "label" in self.display and self.display["label"]>=0:
                    # draw template
                    for pxl in kp_pxl:
                        draw_point(img_adjust, pxl[1], pxl[0], display_label=self.display["label"])


                # assign xyz and filter if the data comes from camera
                if camera_data["depth_frame"] is not None and camera_data["K"] is not None:
                    # assign xyz
                    r["xyz"] = self.xyz(r["center"])

                    # assign xyz for kps
                    if "kp" in r:
                        for i in range(len(r["kp"])):
                            r["kp"][i]["xyz"] = self.xyz(r["kp"][i]["center"])

                # xyz limit
                if "xyz" in self.limit:
                    if "xyz" not in r or not Valid().xyz(r["xyz"], **self.limit["xyz"]):
                        continue

                # plane: rvec, tvec
                if "cmd" in self.detection and self.detection["cmd"] not in ["aruco", "charuco"] and "cmd" in self.pose and self.pose["cmd"] == "plane" and "plane" in self.pose and len(self.pose["plane"]) > 2 and camera_data["depth_frame"] is not None:
                    pose_result = Plane().pose(r["corners"], self.pose["plane"], self.camera, camera_data["depth_frame"], camera_data["depth_int"], self.frame_mat_inv)
                    if not pose_result:
                        continue
                    r["rvec"] = pose_result[0]
                    r["tvec"] = pose_result[1]
                    kp_pxl = [[i, pose_result[4][i]] for i in range(len(pose_result[4]))]
                
                # kp pose: rvec, tvec
                if "cmd" in self.detection and self.detection["cmd"] == "kp" and "cmd" in self.pose and self.pose["cmd"] == "kp" and "kp" in self.pose and r["cls"] in self.pose["kp"] and camera_data["depth_frame"] is not None:
                    pose_kp_prm = {}
                    if "thr" in self.pose:
                        pose_kp_prm["thr"] = self.pose["thr"]
                    
                    # number of keypoints
                    if len(self.pose["kp"][r["cls"]]) == 2:
                        pose_result = Pose_two_point().pose(kp_list=r["kp"], kp_geometry=self.pose["kp"][r["cls"]], camera_matrix=camera_data["K"], dist_coeffs=camera_data["D"], frame_mat_inv=self.frame_mat_inv, **pose_kp_prm)
                    if len(self.pose["kp"][r["cls"]]) >= 4:
                        pose_result = PNP().pose(kp_list=r["kp"], kp_geometry=self.pose["kp"][r["cls"]], camera_matrix=camera_data["K"], dist_coeffs=camera_data["D"], frame_mat_inv=self.frame_mat_inv, **pose_kp_prm)

                    if not pose_result:
                        continue
                    r["rvec"] = pose_result[0]
                    r["tvec"] = pose_result[1]

                # assign pose meta
                if pose_result:
                    try:
                        pose_center = frame_center_pixel(
                                                        rvec=pose_result[2], 
                                                        tvec=pose_result[3], 
                                                        camera_matrix=camera_data["K"], 
                                                        dist_coeffs=camera_data["D"])
                        pose_xyz = self.xyz(pose_center)
                        r["pose"] = {"center": pose_center, "xyz":pose_xyz}
                    except:
                        pass
                    
                # draw rvec
                if pose_result and "label" in self.display and self.display["label"]>=0:
                    # draw template
                    draw_3d_axis(img_adjust, rvec=pose_result[2], tvec=pose_result[3], camera_matrix=camera_data["K"], dist_coeffs=camera_data["D"])
                    
                # rvec valid
                if "rvec" in self.limit:
                    if "rvec" not in r or not Valid().rvec(r["rvec"], **self.limit["rvec"]):
                        continue

                # tvec valid
                if "tvec" in self.limit:
                    if "tvec" not in r or not Valid().tvec(r["tvec"], **self.limit["tvec"]):
                        continue

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

            # retval
            self.retval["camera_data"] = camera_data
            self.retval["camera_data"]["img"] = img_adjust.copy()
            self.retval["camera_data"]["img_roi"] = img_roi.copy()
            self.retval["frame_mat_inv"] = self.frame_mat_inv.copy()
        except Exception as ex:
            print("Exception: ", ex)    
        
        return list(self.retval["valid"])

    def draw(self,
            title="",
            axis=False,
            poly=[],
            poly_color=(0, 255, 0),
            poly_thickness=2,
            poly_fill=False,
            clear_jupyter_plot=False):

        try:
            img = self.retval["camera_data"]["img"].copy()
        except:
            return

        if poly:
            for p in poly:
                if len(p) == 0:
                    continue
                pts = np.array(p, dtype=np.int32)
                if poly_fill:
                    cv2.fillPoly(img, [pts], poly_color)
                else:
                    cv2.polylines(img, [pts], isClosed=True, color=poly_color, thickness=poly_thickness)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if clear_jupyter_plot and in_jupyter():
            clear_output(wait=True)

        plt.figure(figsize=(8, 5))
        plt.imshow(img_rgb)
        if not axis:
            plt.axis('off')
        if title:
            plt.title(title)
        plt.show()

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