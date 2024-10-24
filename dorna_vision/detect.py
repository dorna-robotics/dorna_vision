import json
import cv2 as cv
from dorna_vision.visual import *
from dorna_vision.find import *
from dorna_vision.ai import *
from dorna_vision.util import *
from dorna_vision.draw import *
from dorna_vision.visual import *
from dorna_vision.pose import *
import threading
import numpy as np
import random
from dorna2 import Kinematic

class Detection(object):
    """docstring for Detect"""
    def __init__(self,
            camera=None,
            robot=None,
            frame=[0, 0, 0, 0, 0, 0], 
            kind="color_img", 
            intensity={"a":1.0, "b":0},
            color={"low_hsv":[0, 0, 0], "high_hsv":[255, 255, 255], "inv":0},
            roi={"corners": [], "inv": 0, "crop": 0},
            detection={"cmd":"poly", },
            limit = {"area":[], "ratio":[], "xyz":[], "inv":0},
            plane = [],
            output={"shuffle": 1, "max_det":1, "save_path": None},
            **kwargs
        ):
        super(Detection, self).__init__()
        
        self.camera = camera
        self.robot = robot
        self.frame = frame
        self.kind = kind
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

        # img
        self.img = np.zeros((10, 10))

        # kinematic
        self.kinematic = Kinematic()

        # object detection
        if self.detection["cmd"] == "od":
            self.od = OD(self.detection["path"])
        elif self.detection["cmd"] == "ocr":
            self.ocr = OCR()

    def get_camera_data(self):
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, timestamp = self.camera.get_all()
        self.camera_data = {
            "depth_frame": depth_frame,
            "ir_frame": ir_frame,
            "color_frame": color_frame,
            "depth_img": depth_img,
            "ir_img": ir_img,
            "color_img": color_img,
            "depth_int": depth_int,
            "frames": frames,
            "timestamp": timestamp
        }
        return self.camera_data


    def img(self):
        return self.img.copy()


    def run(self, img=None):               
        # return
        retval = []

        # frame
        frame_mat_inv = np.linalg.inv(self.robot.kinematic.xyzabc_to_mat(np.array(self.frame)))
        if self.robot:
            jonit = self.robot.get_all_joint()[0:6]
            T_camholder_to_base = self.robot.kinematic.Ti_r_world(i=5, joint=jonit[0:6])
            T_cam_to_camholder = np.matrix(self.robot.config["T_camera_j4"])
            T_cam_to_base = np.matmul(T_camholder_to_base, T_cam_to_camholder)
            frame_mat_inv = np.matmul(frame_mat_inv, T_cam_to_base)

        # img
        if img:
            _img = cv.imread(img)
        else:
            camera_data = self.get_camera_data()
            _img = camera_data[self.kind]

        # intensity
        img_adjust = intensity(_img.copy(), **self.intensity)

        # color
        img_adjust = color_mask(img_adjust, **self.color)
        
        # roi
        _roi = ROI(img_adjust.copy(), **self.roi)
        img_roi = _roi.img

        # thr
        self.img_thr = np.zeros(img_roi.shape[:2], dtype=np.uint8)

        # detection
        if self.detection["cmd"] == "ellipse":
            # [pxl, corners, (pxl, (major_axis, minor_axis), rot),...]
            result = ellipse(img_roi, **self.detection)
            retval = [{"timestamp": camera_data["timestamp"], "cls": 0, "conf": 1, "center": _roi.pxl_to_orig(r[0]), 
            "corners": [_roi.pxl_to_orig(x) for x in r[1]], "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]

        if self.detection["cmd"] in ["poly", "contour"]:
            # thr
            self.img_thr = binary_thr(img_roi, **self.detection)

            # find contour: [[pxl, corners, cnt], ...]
            result = contour(self.img_thr, **self.detection):
            retval = [{"timestamp": camera_data["timestamp"], "cls": 0, "conf": 1, "center": _roi.pxl_to_orig(r[0]), 
            "corners": [_roi.pxl_to_orig(x) for x in r[1]], "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]

        elif self.detection["cmd"] == "aruco":
            # [[pxl, corners, (id, rvec, tvec)], ...]
            result = aruco(img_roi, **self.detection)
            retval = [{"timestamp": camera_data["timestamp"], "cls": r[2][0], "conf": 1, "center": _roi.pxl_to_orig(r[0]),
            "corners": [_roi.pxl_to_orig(x) for x in r[1]], "xyz": [0, 0, 0], "rvec": [x*180/np.pi for x in r[2][1]], "tvec": r[2][2].tolist()} for r in result]

        elif self.detection["cmd"] == "ocr":
            result = self.ocr(img_roi, **self.detection)
            retval = [{"timestamp": camera_data["timestamp"], "cls": r[1][0], "conf": r[1][1], "center": _roi.pxl_to_orig([sum([p[0] for p in r[0]])/len(r[0]), sum([p[1] for p in r[0]])/len(r[0])]), 
            "corners": [_roi.pxl_to_orig(sublist) for sublist in r[0]], "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result[0]]

        elif self.detection["cmd"] == "od":
            result = self.od(img_roi, **self.detection)
            retval = [{"timestamp": camera_data["timestamp"], "cls": r.cls, "conf": r.conf, "center": _roi.pxl_to_orig([r.x+r.w/2, r.y+r.h/2]), 
            "corners": [_roi.pxl_to_orig(pxl) for pxl in [[r.x, r.y], [r.x+r.w, r.y], [r.x+r.w, r.y+r.h], [r.x, r.y+r.h]]],
            "xyz": [0, 0, 0], "rvec": [0, 0, 0], "tvec": [0, 0, 0]} for r in result]

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
        if "ratio" in self.limit and len(self.limit["ratio"])== 2:
            for r in retval[:]:
                corners = np.array(r["corners"])
                sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
                ratio = np.min(sides) / np.max(sides)
                if ratio < min(self.limit["ratio"]) or ratio > max(self.limit["ratio"]):
                    retval.remove(r)

        # xyz only if the data comes from camera
        if not img:
            for r in retval:
                # xyz_target_2_cam
                xyz_target_to_cam, _ = self.camera.xyz(r["center"], camera_data["depth_frame"], camera_data["depth_int"])[0].tolist()
                T_target_to_cam = self.kinematic.xyzabc_to_mat(np.concatenate((np.array(xyz_target_to_cam), np.array([0, 0, 0]))))
                
                # apply frame
                T_target_to_frame = np.matmul(frame_mat_inv, T_target_to_cam)
                xyz_target_to_frame = self.kinematic.mat_to_xyzabc(T_target_to_frame).tolist()
                r["xyz"] = xyz_target_to_frame[0:3]
                
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
        if self.detection["cmd"] in ["ellipse", "poly", "contour", "ocr", "od"] and len(self.plane) > 2 and camera_data["depth_frame"] is not None:
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
                    T_target_to_cam = self.kinematic.xyzabc_to_mat(np.array([tvec_target_to_cam, rvec_target_to_cam]))
                    
                    # apply frame
                    T_target_to_frame = np.matmul(frame_mat_inv, T_target_to_cam)
                    xyzabc_target_to_frame = self.kinematic.mat_to_xyzabc(T_target_to_frame).tolist()
                    r["tvec"] = xyzabc_target_to_frame[0:3]
                    r["rvec"] = xyzabc_target_to_frame[3:6]

                    # draw template
                    for pxl in pxl_map:
                        draw_point(img_adjust, pxl)

                    # draw axes
                    draw_3d_axis(img_adjust, center_3d, X, Y, Z, self.camera.camera_matrix(camera_data["depth_int"]), self.camera.dist_coeffs(camera_data["depth_int"]))                

        # draw corners
        if self.detection["cmd"] in ["ellipse", "poly", "contour", "ocr", "od"]: # corners and axes
            for r in retval:
                draw_corners(img_adjust, r["cls"], r["corners"])
        
        # shuffle
        if self.output["shuffle"]:
            random.shuffle(retval)
        
        # max_det
        if self.output["max_det"] > 0:
            retval = retval[0:self.output["max_det"]]

        # save
        if self.output["save_path"]:
            # Create a thread to perform the file write operation
            thread = threading.Thread(target=cv.imwrite, args=(self.output["save_path"], img_adjust))
            thread.start()

            self.thread_list.append(thread)

        # img
        self.img = img_adjust

        # return
        return retval


    """
    timestamp
    cls
    conf
    corners
    xyz
    rvec
    tvec
    """
    def _pattern(self, camera, camera_data, **kwargs):
        # init retval

        retval = []
        """bgr_img"""
        bgr_img = camera_data["color_img"]

        """adjust and out img"""
        adjust_img = bgr_img.copy()
        thr_img = bgr_img.copy()

        # intensity
        if kwargs["intensity_enb"]:
            adjust_img = intensity(adjust_img, kwargs["intensity_alpha"], kwargs["intensity_beta"])
                
        # color mask
        if kwargs["color_enb"]:
            adjust_img = color_mask(adjust_img, (kwargs["color_h"][0], kwargs["color_s"][0], kwargs["color_v"][0]), (kwargs["color_h"][1], kwargs["color_s"][1], kwargs["color_v"][1]), kwargs["color_inv"])

        # roi
        mask_img = adjust_img.copy()
        if kwargs["roi_enb"]:
            mask_img = roi_mask(mask_img, kwargs["roi_value"], kwargs["roi_inv"])

        """methods"""
        # method
        if kwargs["method_value"] == 0: # ellipse
            # edge drawing
            elps = edge_drawing(mask_img, min_path_length=kwargs["m_elp_min_path_length"], min_line_length=kwargs["m_elp_min_line_length"], nfa_validation=kwargs["m_elp_nfa_validation"], sigma=kwargs["m_elp_sigma"], gradient_threshold_value=kwargs["m_elp_gradient_threshold_value"], pf_mode=kwargs["m_elp_pf_mode"], axes=kwargs["m_elp_axes"], ratio=kwargs["m_elp_ratio"])
            
            # draw elps
            draw_ellipse(adjust_img, elps, axis=False)
            
            # corners
            corners = [get_obb_corners(elp[0], [2*elp[1][0], 2*elp[1][1]], elp[2]) for elp in elps]

            # return
            retval = [
                {"timestamp": camera_data["timestamp"],
                "cls": 0,
                "conf":1,
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                } for elp in elps
            ]
            for ret in retval:
                #??? center is not given
                draw_obb(adjust_img, ret["id"], ret["center"], ret["corners"])

        elif kwargs["method_value"] == 2: # polygon
            # thr
            thr_img = binary_thr(mask_img, kwargs["m_poly_type"], kwargs["m_poly_inv"], kwargs["m_poly_blur"], kwargs["m_poly_thr"], kwargs["m_poly_mean_sub"])

            # find contour
            draws = find_cnt(thr_img, kwargs["m_poly_area"], kwargs["m_poly_perimeter"], kwargs["m_poly_value"])

            # draw contours
            draw_cnt(adjust_img, draws, axis=False)

            # corners
            corners = [get_obb_corners(draw[0], draw[1], draw[2]) for draw in draws]

            # return
            retval = [
                {"timestamp": camera_data["timestamp"],
                "cls":0,
                "conf":1,
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                } for draw in draws
            ]
            for ret in retval:
                # ??? center is not valid
                draw_obb(adjust_img, ret["id"], ret["center"], ret["corners"])
        
        elif kwargs["method_value"] == 3: # contour
            # thr
            thr_img = binary_thr(mask_img, kwargs["m_cnt_type"], kwargs["m_cnt_inv"], kwargs["m_cnt_blur"], kwargs["m_cnt_thr"], kwargs["m_cnt_mean_sub"])

            # find contour
            draws = find_cnt(thr_img, kwargs["m_cnt_area"], kwargs["m_cnt_perimeter"])

            # draw contours
            out_img = draw_cnt(adjust_img, draws, axis=False)

            # corners
            corners = [get_obb_corners(draw[0], draw[1], draw[2]) for draw in draws]

            # return
            retval = [
                {"timestamp": camera_data["timestamp"],
                "cls": 0,
                "conf":1,
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                } for draw in draws
            ]
            for ret in retval:
                # ??? adjust center
                draw_obb(adjust_img, ret["id"], ret["center"], ret["corners"])
        
        elif kwargs["method_value"] == 4: # aruco #??? echck this the id does not match
            retval = []

            # pose: [id, corner, rvec, tvec]
            aruco_data = find_aruco(mask_img, camera.camera_matrix(camera_data["depth_int"]), camera.dist_coeffs(camera_data["depth_int"]), kwargs["m_aruco_dictionary"], kwargs["m_aruco_marker_length"], kwargs["m_aruco_refine"], kwargs["m_aruco_subpix"])

            for i in range(len(aruco_data)):
                _id = int(aruco_data[i][0][0])
                _timestamp = camera_data["timestamp"]

                # corner and center
                _corners = aruco_data[i][1].reshape((4,2))
                _center = [int(sum([c[0] for c in _corners])/4), int(sum([c[1] for c in _corners])/4)]
                _corners = [[int(c[0]), int(c[1])] for c in _corners]
                
                # rvec, tvec
                _rvec = aruco_data[i][2][0].tolist()
                _rvec = [r*180/np.pi for r in _rvec]
                _tvec = aruco_data[i][3][0].tolist()
                
                retval.append({
                    "id":i,
                    "center": _center,
                    "corners": _corners,
                    "xyz": [0, 0, 0],
                    "rvec": _rvec,
                    "tvec": _tvec
                    })

            # draw
            draw_aruco(adjust_img, aruco_data, camera.camera_matrix(camera_data["depth_int"]), camera.dist_coeffs(camera_data["depth_int"]))

        if kwargs["run"] == "ocr": # ocr
            retval = self.ocr.ocr(mask_img, conf=kwargs["conf"])
        if kwargs["run"] == "od": # object detection
            # run detection
            objects = self.od(mask_img, **kwargs)
            
            # format the result
            retval = [
                        {"timestamp": camera_data["timestamp"],
                        "cls": obj.cls,
                        "label": self.od["classes"][obj.cls],
                        "conf": obj.conf,
                        "corners": [[obj.rect.x, obj.rect.y], [obj.rect.x+obj.rect.w, obj.rect.y], [obj.rect.x+obj.rect.w, obj.rect.y+obj.rect.h], [obj.rect.x, obj.rect.y+obj.rect.h]],
                        "xyz": [0, 0, 0],
                        "rvec": [0, 0, 0],
                        "tvec": [0, 0, 0],
                        }
                    for obj in objects
            ]
            
            # edge drawing
            results = edge_drawing(mask_img, min_path_length=kwargs["m_elp_min_path_length"], min_line_length=kwargs["m_elp_min_line_length"], nfa_validation=kwargs["m_elp_nfa_validation"], sigma=kwargs["m_elp_sigma"], gradient_threshold_value=kwargs["m_elp_gradient_threshold_value"], pf_mode=kwargs["m_elp_pf_mode"], axes=kwargs["m_elp_axes"], ratio=kwargs["m_elp_ratio"])
            
            # draw elps
            draw_ellipse(adjust_img, elps, axis=False)
            
            # corners
            corners = [get_obb_corners(elp[0], [2*elp[1][0], 2*elp[1][1]], elp[2]) for elp in elps]

            # return
            retval = [
                {"timestamp": camera_data["timestamp"],
                "cls": 0,
                "conf":1,
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                } for elp in elps
            ]
            for ret in retval:
                #??? center is not given
                draw_obb(adjust_img, ret["id"], ret["center"], ret["corners"])

        # xyz:
        if camera_data["depth_frame"] is not None:
            for i in range(len(retval)):
                # xyz
                retval[i]["xyz"] = camera.xyz(retval[i]["center"], camera_data["depth_frame"], camera_data["depth_int"])[0].tolist()

        # pose
        if kwargs["method_value"] in [0, 1, 2, 3]:
            # tmp pixels from poi
            tmp_pxls = np.array(kwargs["poi_value"])
            
            if len(tmp_pxls) > 2 and camera_data["depth_frame"] is not None:
                for i in range(len(retval)):

                    # axis
                    center = retval[i]["center"]
                    dim = retval[i]["obb"]["wh"] # (w, h)
                    rot = retval[i]["obb"]["rot"]
                    del retval[i]["obb"]
                    
                    # pose from tmp
                    valid, center_3d, X, Y, Z, pxl_map = pose_3_point(camera_data["depth_frame"], camera_data["depth_int"], tmp_pxls, center, dim, rot, camera)

                    if valid: # add pose
                        draw_3d_axis(adjust_img, center_3d, X, Y, Z, camera.camera_matrix(camera_data["depth_int"]), camera.dist_coeffs(camera_data["depth_int"]))
                        
                        retval[i]["tvec"] = center_3d.tolist()
                        rodrigues, _= cv.Rodrigues(np.matrix([[X[0], Y[0], Z[0]],
                                                [X[1], Y[1], Z[1]],
                                                [X[2], Y[2], Z[2]]])) 
                        retval[i]["rvec"] = [rodrigues[i, 0]*180/np.pi for i in range(3)]
                    

                    # draw template
                    for pxl in pxl_map:
                        draw_point(adjust_img, pxl)

        return camera_data["timestamp"], retval, adjust_img, thr_img, bgr_img
