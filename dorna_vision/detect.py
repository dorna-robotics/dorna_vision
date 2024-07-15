import json
import cv2 as cv
from dorna_vision.util import *
from dorna_vision.draw import *
import os
import time
import threading
import numpy as np

class Detect(object):
    """docstring for Detect"""
    def __init__(self, camera):
        super(Detect, self).__init__()
        # add the module path

        # Construct the path to the JSON file
        self.default_path = os.path.join(os.path.dirname(__file__))

        # init camera
        self.camera = camera

        # pattern config
        self.config = {"poi_value": [], "color_enb": 0, "color_h": [60, 120], "color_s": [85, 170], "color_v": [85, 170], "color_inv": 0, "roi_enb": 0, "roi_value": [[353.84, 171.66], [324.84, 524.06], [828.9, 564.2], [811.06, 180.58]], "roi_inv": 0, "intensity_enb": 0, "intensity_alpha": 2.0, "intensity_beta": 50, "method_value": 0, "m_elp_pf_mode": 0, "m_elp_nfa_validation": 1, "m_elp_min_path_length": 50, "m_elp_min_line_length": 10, "m_elp_sigma": 1, "m_elp_gradient_threshold_value": 20, "m_elp_axes": [20, 100], "m_elp_ratio": [0.0, 1.0], "m_circle_inv": 1, "m_circle_type": 0, "m_circle_thr": 127, "m_circle_blur": 3, "m_circle_mean_sub": 0, "m_circle_radius": [1, 30], "m_poly_inv": 1, "m_poly_type": 0, "m_poly_thr": 127, "m_poly_blur": 3, "m_poly_mean_sub": 0, "m_poly_value": 3, "m_poly_area": [100, 100000], "m_poly_perimeter": [10, 100000], "m_cnt_inv": 1, "m_cnt_type": 0, "m_cnt_thr": 127, "m_cnt_blur": 3, "m_cnt_mean_sub": 0, "m_cnt_area": [100, 100000], "m_cnt_perimeter": [10, 100000], "m_aruco_dictionary": "DICT_6X6_250", "m_aruco_marker_length": 10, "m_aruco_refine": "CORNER_REFINE_NONE", "m_aruco_subpix": 0}

        # thread list
        self.thread_list = []

        self.adjust_img = np.zeros((10, 10))


    # terminate the detect
    def close(self):
        while self.thread_list:
            # wait for the thread to end
            self.thread_list[0].join()

            # pop it 
            self.thread_list.pop(0) 

    def update_camera_data(self):
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
        return self.adjust_img


    def pattern(self, config=None, save_path=None):
        if type(config) == dict:
            pass
        
        elif type(config) == str: 
            # Load JSON data from the file
            with open(config, 'r') as json_file:
                config = json.load(json_file)
        
        # update config
        self.config = config
        kwargs = dict(self.config)
        
        # camera data
        camera_data = self.update_camera_data()

        # run pattern detection
        timestamp, retval, adjust_img, thr_img, bgr_img = self._pattern(self.camera, camera_data, **kwargs)
        self.adjust_img = adjust_img

        # save the plots
        if save_path:
            # Create a thread to perform the file write operation
            thread = threading.Thread(target=cv.imwrite, args=(save_path, adjust_img))
            thread.start()

            self.thread_list.append(thread)

        return retval


    """
    obb: pxl, (w,h), rot
    pose: valid, rvec, tvec
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
                {"id": i,
                "timestamp": camera_data["timestamp"],
                "center": elp[0],
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                "obb": {"wh": [2*elp[1][0], 2*elp[1][1]], "rot": elp[2]},
                } for elp, i in zip(elps, range(len(elps)))
            ]
            for ret in retval:
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
                {"id": i,
                "timestamp": camera_data["timestamp"],
                "center": draw[0],
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                "obb": {"wh": draw[1], "rot": draw[2]}
                } for draw, i in zip(draws, range(len(draws)))
            ]
            for ret in retval:
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
                {"id":i,
                "center": draw[0],
                "corners": corners[i],
                "xyz": [0, 0, 0],
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 0],
                "obb": {"wh": draw[1], "rot": draw[2]}
                } for draw, i in zip(draws, range(len(draws)))
            ]
            for ret in retval:
                draw_obb(adjust_img, ret["id"], ret["center"], ret["corners"])
        
        elif kwargs["method_value"] == 4: # aruco
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


def main_pattern():
    from camera import Camera

    camera = Camera()
    camera.connect()

    d = Detect(camera)

    for i in range(10):
        retval = d.pattern()
    camera.close()
    d.close()

if __name__ == '__main__':
    main_pattern()