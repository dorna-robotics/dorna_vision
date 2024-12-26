from dorna_vision.visual import *
from dorna_vision.find import *
from dorna_vision.ai import *
from dorna_vision.util import *
from dorna_vision.draw import *
from dorna_vision.pose import *

import time
import cv2 as cv
import os
import threading
import numpy as np
import random

from dorna2 import Kinematic

class Detection(object):
    """docstring for Detect"""
    def __init__(self,
            camera=None,
            robot=None,
            camera_mount="dorna_ta_j4_1",
            frame=[0, 0, 0, 0, 0, 0], 
            feed="color_img", 
            intensity={"a":1.0, "b":0},
            color={"low_hsv":[0, 0, 0], "high_hsv":[255, 255, 255], "inv":0},
            roi={"corners": [], "inv": 0, "crop": 0},
            detection={"cmd":None},
            limit = {"area":[], "aspect_ratio":[], "xyz":[], "inv":0},
            plane = [],
            output={"shuffle": 1, "max_det":1, "save_img":0, "save_img_roi":0},
            **kwargs
        ):
        super(Detection, self).__init__()
        
        self.camera = camera
        self.robot = robot
        self.camera_mount_label = camera_mount
        self.camera_mount = self.set_camera_mount(camera_mount)
        self.frame = frame
        self.feed = feed
        self.intensity = intensity
        self.color = color
        self.roi = roi
        self.detection = detection
        self.limit = limit
        self.plane = plane
        self.output = output
        self.kwargs = kwargs

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
        elif "cmd" in self.detection and self.detection["cmd"] == "ocr":
            self.init_ocr()


    def set_camera_mount(self, camera_mount):
        retval = {"ej": [0, 0, 0, 0, 0, 0, 0, 0]}
        if self.robot is None:
            pass
        elif type(camera_mount) == dict and "type" in camera_mount and "T" in camera_mount and "ej" in camera_mount:
            retval = {
                "type": camera_mount["type"],
                "T": self.robot.kinematic.xyzabc_to_mat(np.array(camera_mount["T"])),
                "ej": camera_mount["ej"]
            }
        elif type(camera_mount) == str:
            for i in range(len(self.robot.config["camera_mount"])):
                if self.robot.config["camera_mount"][i]["type"] == camera_mount:
                    tmp =dict(self.robot.config["camera_mount"][i])
                    retval = {
                        "type": tmp["type"],
                        "T": self.robot.kinematic.xyzabc_to_mat(np.array(tmp["T"])),
                        "ej": tmp["ej"]
                    }
                    break

        return retval

            


    def init_cls(self, path):
        self.cls = CLS(path)


    def init_od(self, path):
        self.od = OD(path)
    

    def init_ocr(self):
        self.ocr = OCR()


    def get_camera_data(self, data=None):
        if type(data) == str: # read from file
            data = cv.imread(data)
            self.camera_data = {key:None for key in ["depth_frame", "ir_frame", "color_frame", "depth_img", "ir_img", "color_img", "depth_int", "frames", "joint"]}
            self.camera_data[self.feed] = data
            self.camera_data["timestamp"] = time.time()
        
        elif type(data) == dict: # keep the current
            self.camera_data = data
        
        else: # update
            joint = None
            depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, timestamp = self.camera.get_all()
            try:
                joint = self.robot.get_all_joint()
                if "ej" in self.camera_mount:
                    for i in range(min(len(self.camera_mount["ej"]), len(joint))):
                        joint[i] += self.camera_mount["ej"][i]

            except:
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
            }
        return self.camera_data


    def pixel_to_xyz(self, pxl):
        try:
            # xyz_target_2_cam
            xyz_target_to_cam = self.camera.xyz(pxl, self.camera_data["depth_frame"], self.camera_data["depth_int"])[0].tolist()
            T_target_to_cam = self.kinematic.xyzabc_to_mat(np.concatenate((np.array(xyz_target_to_cam), np.array([0, 0, 0]))))
            
            # apply frame
            T_target_to_frame = np.matmul(self.frame_mat_inv, T_target_to_cam)
            xyz_target_to_frame = self.kinematic.mat_to_xyzabc(T_target_to_frame).tolist()
            xyz = xyz_target_to_frame[0:3]
            
        except:
            xyz = [0, 0, 0]
        return xyz


    def run(self, data=None, **kwargs):
        # return
        retval = []
        try:
            # assign the new value
            for key, value in kwargs.items():
                # Check if the attribute already exists in the class
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # update camera_data
            camera_data = self.get_camera_data(data)
            _img = camera_data[self.feed]
            # frame
            self.frame_mat_inv = np.linalg.inv(self.kinematic.xyzabc_to_mat(np.array(self.frame)))
            if self.robot is not None:
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
                    "corners": [_roi.pxl_to_orig(x) for x in r[1]], "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]

                if self.detection["cmd"] in ["poly", "cnt"]:
                    # thr
                    self.img_thr = binary_thr(img_roi, **self.detection)

                    # find contour: [[pxl, corners, cnt], ...]
                    result = contour(self.img_thr, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": self.detection["cmd"], "conf": 1, "center": _roi.pxl_to_orig(r[0]), 
                    "corners": [_roi.pxl_to_orig(x) for x in r[1]], "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]

                elif self.detection["cmd"] == "aruco" and camera_data["depth_int"] is not None:
                    try:
                        # [[pxl, corners, (id, rvec, tvec)], ...]
                        result = aruco(img_roi, self.camera.camera_matrix(camera_data["depth_int"]), self.camera.dist_coeffs(camera_data["depth_int"]), **self.detection)
                        _retval = [{"timestamp": camera_data["timestamp"], "cls": str(r[2][0][0]), "conf": 1, "center": _roi.pxl_to_orig(r[0]),
                        "corners": [_roi.pxl_to_orig(x) for x in r[1]], "xyz": [0, 0, 0], "rvec": [x*180/np.pi for x in r[2][2][0].tolist()], "tvec": r[2][3][0].tolist()} for r in result]
                        draw_aruco(img_adjust, np.array([r[2][0] for r in result]), np.array([[r["corners"] for r in _retval]], dtype=np.float32), [r[2][2] for r in result], [r[2][3] for r in result], self.camera.camera_matrix(camera_data["depth_int"]), self.camera.dist_coeffs(camera_data["depth_int"]))
                        retval = _retval
                        # rvec tvec
                        # xyz_target_2_cam
                        for r in retval:
                            T_target_to_cam = self.kinematic.xyzabc_to_mat(np.array(r["tvec"]+ r["rvec"]))

                            # apply frame
                            T_target_to_frame = np.matmul(self.frame_mat_inv, T_target_to_cam)
                            xyzabc_target_to_frame = self.kinematic.mat_to_xyzabc(T_target_to_frame).tolist()
                            r["tvec"] = xyzabc_target_to_frame[0:3]
                            r["rvec"] = xyzabc_target_to_frame[3:6]

                    except Exception as ex:
                        pass
                elif self.detection["cmd"] == "ocr":
                    result = self.ocr.ocr(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[1][0], "conf": r[1][1], "center": _roi.pxl_to_orig([(sum([p[0] for p in r[0]])/len(r[0])), sum([p[1] for p in r[0]])/len(r[0])]), 
                    "corners": [_roi.pxl_to_orig(sublist) for sublist in r[0]], "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]
                elif self.detection["cmd"] == "od":
                    result = self.od(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r.cls, "conf": r.prob, "center": _roi.pxl_to_orig([r.rect.x+r.rect.w/2, r.rect.y+r.rect.h/2]), 
                    "corners": [_roi.pxl_to_orig(pxl) for pxl in [[r.rect.x, r.rect.y], [r.rect.x+r.rect.w, r.rect.y], [r.rect.x+r.rect.w, r.rect.y+r.rect.h], [r.rect.x, r.rect.y+r.rect.h]]],
                    "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]
                elif self.detection["cmd"] == "cls":
                    result = self.cls(img_roi, **self.detection)
                    retval = [{"timestamp": camera_data["timestamp"], "cls": r[0], "conf": r[1], "center": [int(width/2), int(height/2)], 
                    "corners": [[min(20,width-1), min(20,height-1)], [max(0,width-21), min(20,height-1)], [max(0,width-21), max(0,height-21)], [min(20,width-1), max(0,height-21)]],
                    "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]


            # limit
            # area
            if "area" in self.limit and len(self.limit["area"])== 2:
                for r in retval[:]:
                    corners = np.array(r["corners"])
                    # Get x and y coordinates
                    x = corners[:, 0]
                    y = corners[:, 1]                
                    # Calculate area using the Shoelace formula
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    if area < min(self.limit["area"]) or area > max(self.limit["area"]):
                        retval.remove(r)

            # ratio
            if "aspect_ratio" in self.limit and len(self.limit["aspect_ratio"])== 2:
                for r in retval[:]:
                    corners = np.array(r["corners"])
                    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
                    aspect_ratio = np.min(sides) / np.max(sides)
                    if aspect_ratio < min(self.limit["aspect_ratio"]) or aspect_ratio > max(self.limit["aspect_ratio"]):
                        retval.remove(r)

            # xyz only if the data comes from camera
            if all([camera_data["depth_frame"] is not None, camera_data["depth_int"] is not None]):
                for r in retval:
                    r["xyz"] = self.pixel_to_xyz(r["center"])

            # xyz limit
            if "xyz" in self.limit and len(self.limit["xyz"])== 3 and len(self.limit["xyz"][0]) == 2 and len(self.limit["xyz"][1]) == 2 and len(self.limit["xyz"][2]) == 2:
                for r in retval[:]:
                    if all([self.limit["xyz"][i][0] <= r["xyz"][i] <= self.limit["xyz"][i][1] for i in range(3)]):
                        if self.limit["inv"]:
                            retval.remove(r)
                    else:
                        if not self.limit["inv"]:    
                            retval.remove(r)

            # plane
            if "cmd" in self.detection and self.detection["cmd"] in ["elp", "poly", "cnt", "ocr", "od"] and len(self.plane) > 2 and camera_data["depth_frame"] is not None:
                # tmp pixels from poi
                tmp_pxls = np.array(self.plane)
                
                for r in retval:
                    # Compute the rotated rectangle from the points
                    center, dim, rot = cv.minAreaRect(np.array(r["corners"], dtype=np.float32))

                    # pose from tmp
                    valid, center_3d, X, Y, Z, pxl_map = pose_3_point(camera_data["depth_frame"], camera_data["depth_int"], tmp_pxls, center, dim, rot, self.camera)

                    if valid: # add pose                    
                        tvec_target_to_cam = center_3d.tolist()
                        rodrigues, _= cv.Rodrigues(np.matrix([[X[0], Y[0], Z[0]],
                                                [X[1], Y[1], Z[1]],
                                                [X[2], Y[2], Z[2]]])) 
                        
                        rvec_target_to_cam = [rodrigues[i, 0]*180/np.pi for i in range(3)]

                        # xyz_target_2_cam
                        T_target_to_cam = self.kinematic.xyzabc_to_mat(np.array(tvec_target_to_cam+ rvec_target_to_cam))

                        # apply frame
                        T_target_to_frame = np.matmul(self.frame_mat_inv, T_target_to_cam)
                        xyzabc_target_to_frame = self.kinematic.mat_to_xyzabc(T_target_to_frame).tolist()
                        r["tvec"] = xyzabc_target_to_frame[0:3]
                        r["rvec"] = xyzabc_target_to_frame[3:6]

                        # draw template
                        for pxl in pxl_map:
                            draw_point(img_adjust, pxl)

                        # draw axes
                        draw_3d_axis(img_adjust, center_3d, X, Y, Z, self.camera.camera_matrix(camera_data["depth_int"]), self.camera.dist_coeffs(camera_data["depth_int"]))                

            # draw corners
            if "cmd" in self.detection and self.detection["cmd"] in ["elp", "poly", "cnt", "ocr", "od", "cls"]: # corners and axes
                for r in retval:
                    draw_corners(img_adjust, r["cls"], r["conf"], r["corners"])
            
            # shuffle
            if self.output["shuffle"]:
                random.shuffle(retval)
            
            # max_det
            if self.output["max_det"] > 0:
                retval = retval[0:self.output["max_det"]]

            # ej
            if "ej" in self.camera_mount:
                for r in retval:
                    r["ej"] = list(self.camera_mount["ej"])

            # save image
            if self.output["save_img"]:
                # make directory if not exists
                os.makedirs("output", exist_ok=True)
                # Create a thread to perform the file write operation
                thread = threading.Thread(target=cv.imwrite, args=("output/"+str(int(camera_data["timestamp"]))+".jpg", img_adjust))
                thread.start()
                self.thread_list.append(thread)

            # save image
            if self.output["save_img_roi"]:
                # make directory if not exists
                os.makedirs("output", exist_ok=True)
                # Create a thread to perform the file write operation
                thread = threading.Thread(target=cv.imwrite, args=("output/roi_"+str(int(camera_data["timestamp"]))+".jpg", img_roi))
                thread.start()
                self.thread_list.append(thread)


            # img
            self.img = img_adjust
        except Exception as ex:
            print(ex)    
        # return
        return retval


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