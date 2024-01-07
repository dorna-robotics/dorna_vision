import json
import cv2 as cv
from util import *
from draw import *
import os
import time
import threading


# fix ast
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
        self.config_pattern = None

        # thread list
        self.thread_list = []

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

    def pattern(self, config_path=None):
        # config file
        if config_path:
            # check given path and if not exists 
            if not os.path.isfile(config_path):
                config_path = os.path.join(self.default_path, "config", config_path)

            # Load JSON data from the file
            with open(config_path, 'r') as json_file:
                self.config_pattern = json.load(json_file)
        # execute detection
        kwargs = dict(self.config_pattern)
        
        # camera data
        self.update_camera_data()

        # run pattern detection
        timestamp, retval, adjust_img, thr_img = self._pattern(self.camera, self.camera_data, **kwargs)

        print(retval)
        # save the plots
        # path
        save_path = os.path.join(self.default_path, "tmp", "img.jpg")
        # Add the text to the image
        cv.putText(adjust_img, "timestamp: "+str(round(timestamp,3)), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # Create a thread to perform the file write operation
        #thread = threading.Thread(target=cv.imwrite, args=(save_path, cv.cvtColor(adjust_img, cv.COLOR_BGR2RGB)))
        thread = threading.Thread(target=cv.imwrite, args=(save_path, adjust_img))
        thread.start()

        self.thread_list.append(thread)

        return retval


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
            
            # return
            retval = [
                {"id": i,
                "rect_center_pxl": [elp[0][0], elp[0][1]],
                "rect_wh": [2*elp[1][0], 2*elp[1][1]],
                "rect_rot": elp[2],
                "center_mass_pxl": [elp[0][0], elp[0][1]],
                } for elp, i in zip(elps, range(len(elps)))
            ]
                                   
        elif kwargs["method_value"] == 2: # polgon
            # thr
            thr_img = binary_thr(mask_img, kwargs["m_poly_type"], kwargs["m_poly_inv"], kwargs["m_poly_blur"], kwargs["m_poly_thr"], kwargs["m_poly_mean_sub"])

            # find contour
            draws = find_cnt(thr_img, kwargs["m_poly_area"], kwargs["m_poly_perimeter"], kwargs["m_poly_value"])

            # draw contours
            draw_cnt(adjust_img, draws, axis=False)

            # return
            retval = [
                {"id": i,
                "rect_center_pxl": [draw[0][0], draw[0][1]],
                 "rect_wh": draw[1],
                 "rect_rot": draw[2],
                 "center_mass_pxl": [draw[3][0], draw[3][1]],
                } for draw, i in zip(draws, range(len(draws)))
            ]

        elif kwargs["method_value"] == 3:
            # thr
            thr_img = binary_thr(mask_img, kwargs["m_cnt_type"], kwargs["m_cnt_inv"], kwargs["m_cnt_blur"], kwargs["m_cnt_thr"], kwargs["m_cnt_mean_sub"])

            # find contour
            draws = find_cnt(thr_img, kwargs["m_cnt_area"], kwargs["m_cnt_perimeter"])

            # draw contours
            out_img = draw_cnt(adjust_img, draws, axis=False)

            # return
            retval = [
                {"id":i,
                "rect_center_pxl": [draw[0][0], draw[0][1]],
                 "rect_wh": draw[1],
                 "rect_rot": draw[2],
                 "center_mass_pxl": [draw[3][0], draw[3][1]],
                } for draw, i in zip(draws, range(len(draws)))
            ]

        elif kwargs["method_value"] == 4: # aruco
            # pose: [id, corner, rvec, tvec]
            aruco_data = find_aruco(adjust_img, camera.camera_matrix(camera_data["depth_int"]), camera.dist_coeffs(camera_data["depth_int"]), kwargs["m_aruco_dictionary"], kwargs["m_aruco_marker_length"], kwargs["m_aruco_refine"], kwargs["m_aruco_subpix"])
            
            # draw
            draw_aruco(adjust_img, aruco_data, camera.camera_matrix(camera_data["depth_int"]), camera.dist_coeffs(camera_data["depth_int"]))

            # return
            retval = [
                {"id":int(draw[0]),
                "rect_center_pxl": [int((draw[1][0][0][1]+draw[1][0][2][1])/2), int((draw[1][0][0][0]+draw[1][0][2][0])/2)],
                "rect_wh": [0, 0],
                "rect_rot": 0,
                "center_mass_pxl": [int((draw[1][0][0][1]+draw[1][0][2][1])/2), int((draw[1][0][0][0]+draw[1][0][2][0])/2)],
                "pose_tvec": draw[3][0].tolist(),
                "pose_rvec": draw[2][0].tolist(),
                } for draw in aruco_data
            ]

        #timestamp
        for ret in retval:
            ret["timestamp"] = camera_data["timestamp"]
        
        # pose
        if kwargs["method_value"] in [0, 1, 2, 3]:
            # tmp pixels from poi
            tmp_pxls = np.array(kwargs["poi_value"])
            
            if len(tmp_pxls) > 2 and camera_data["depth_frame"] is not None:
                for i in range(len(retval)):

                    # axis
                    center = retval[i]["rect_center_pxl"]
                    dim = retval[i]["rect_wh"] # (w, h)
                    rot = retval[i]["rect_rot"]

                    # xyz
                    retval[i]["rect_center_xyz"] = camera.xyz(retval[i]["rect_center_pxl"], camera_data["depth_frame"], camera_data["depth_int"])[0].tolist()
                    retval[i]["center_mass_xyz"] = camera.xyz(retval[i]["center_mass_pxl"], camera_data["depth_frame"], camera_data["depth_int"])[0].tolist()

                    
                    # pose from tmp
                    valid, center_3d, X, Y, Z, pxl_map = pose_3_point(camera_data["depth_frame"], camera_data["depth_int"], tmp_pxls, center, dim, rot, camera)
                    retval[i]["pose"] = valid

                    if valid: # add pose
                        draw_3d_axis(adjust_img, center_3d, X, Y, Z, camera.camera_matrix(camera_data["depth_int"]), camera.dist_coeffs(camera_data["depth_int"]))
                        
                        retval[i]["pose_tvec"] = center_3d.tolist()
                        rodrigues, _= cv.Rodrigues(np.matrix([[X[0], Y[0], Z[0]],
                                                [X[1], Y[1], Z[1]],
                                                [X[2], Y[2], Z[2]]])) 
                        retval[i]["pose_rvec"] = [rodrigues[0, 0], rodrigues[1,0], rodrigues[2,0]]
                    
                    else:
                        retval[i]["pose_tvec"] = [0, 0, 0]
                        retval[i]["pose_rvec"] = [0, 0, 0]

                    # draw template do
                    for pxl in pxl_map:
                        draw_point(adjust_img, pxl)

        return camera_data["timestamp"], retval, adjust_img, thr_img


def main_pattern():
    from camera import Camera

    camera = Camera()
    camera.connect()

    d = Detect(camera)

    d.pattern("test.json")
    for i in range(10):
        d.pattern()
    camera.close()
    d.close()

if __name__ == '__main__':
    main_pattern()