import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ipywidgets import interactive, widgets
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Ellipse
import json
import textwrap
import ast
import pickle as pkl
import itertools
import random
from datetime import datetime

from camera import Camera
from dorna2 import Dorna

from dorna_vision.detect import *
from dorna_vision.draw import *
from dorna_vision.util import *
from dorna_vision.calibration import *


class default_widget(object):
    """docstring for ClassName"""
    def __init__(self,):
        super(default_widget, self).__init__()
        continuous_update = False
        style={'description_width': '150px'}
        style_2={'description_width': '0px'}
        widgets.IntText(value=7, description='Any:', disabled=False)
        self.widget_helper = {
            "xyz_label": widgets.HTML(value="Convert pixel coordinates to its 3D spatial values based on the predefined reference frame.", layout={'width': '99%'}, style=style),
            "xyz_pxl": widgets.Text(value='', placeholder='[width, height]', description='Pixel coordinates (pxl)', disabled=False, style=style),
            "xyz_xyz": widgets.Text(value='', placeholder='', description='Result (mm)', disabled=True, style=style),
            "xyz_convert": widgets.Button( description='Convert', disabled=False, button_style="", tooltip='Convert'),

            "pixel_label": widgets.HTML(value="Convert 3D spatial coordinates to their corresponding pixel positions based on the predefined reference frame.", layout={'width': '99%'}, style=style),
            "pixel_xyz": widgets.Text(value='', placeholder='[x, y, z]', description='xyz (mm)', disabled=False, style=style),
            "pixel_pxl": widgets.Text(value='', placeholder='', description='Result (pxl)', disabled=True, style=style),
            "pixel_convert": widgets.Button( description='Convert', disabled=False, button_style="", tooltip='Convert'),

            "clb_label": widgets.HTML(value='<p>Calibrate the robot using this section. Print the Aruco marker from the following link: <a href="https://github.com/dorna-robotics/dorna_vision/blob/main/dorna_vision/asset/4x4_1000-0.pdf" target="_blank">Aruco Marker</a>. Ensure that the marker is printed at its original size, with a precise dimension of 20x20mm.</p>', layout={'width': '99%'}, style=style),
            "clb_data_label": widgets.HTML(value="Collected data (size: 0)", disabled=True, style=style),
            "clb_data": widgets.Textarea(value='', placeholder='', disabled=True, rows=5, layout={'width': '99%'}, style=style),
            "clb_result_label": widgets.HTML(value="Calibration result", layout={'width': '99%'}, style=style),
            "clb_result": widgets.Textarea(value='', placeholder='', disabled=True, rows=2, layout={'width': '99%'}, style=style),
            "clb_calibrate_b": widgets.Button( description='Calibrate', disabled=False, button_style="success", tooltip='Calibrate', style=style),
            "clb_clear_b": widgets.Button( description='Clear List', disabled=False, button_style="", tooltip='Clear List',style=style),
            "clb_robot_b": widgets.Button( description='Motor ON/OFF', disabled=False, button_style="warning", tooltip='Motor ON/OFF', style=style),
            "clb_capture_b": widgets.Button( description='Capture One', disabled=False, button_style="", tooltip='Capture One', style=style),
            "clb_capture_m_b": widgets.Button( description='Capture Multiple', disabled=False, button_style="", tooltip='Capture Multiple', style=style),
            "clb_apply_b": widgets.Button( description='Apply the Result', disabled=False, button_style="", tooltip='Apply the Result', style=style),
            "clb_thr" : widgets.FloatSlider(value=1, min=0.01, max=4, step=0.01, description='Threshold value', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "clb_aruco_marker_length": widgets.FloatSlider(value=20, min=1, max=100, step=0.1, description='Marker length (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "clb_aruco_enb": widgets.Checkbox(value=True, description='Use Aruco marker data', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

        }

        self.widget_init = {
            "camera_setup_label": widgets.HTML(value="<ul><li><strong>Eye-in-hand</strong>: When the camera is mounted on the robot, input the robot's IP address to synchronize the 3D data with the robot.</li><li><strong>Eye-to-hand</strong>: If the camera is stationary, leave the robot's IP address field blank.</li></ul>", layout={'width': '99%'}, style=style),
            "camera_setup_type": widgets.Dropdown(value=1, options=[('Eye-in-hand', 0), ('Eye-to-hand', 1),], description='Mounting setup', continuous_update=continuous_update, style=style),
            "camera_setup_robot_ip": widgets.Text(value='localhost', placeholder='localhost', description='Robot IP address', disabled=False, style=style),
            "camera_clb_apply": widgets.Checkbox(value=False, description='Use custom calibration data', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "camera_clb_T": widgets.Text(value='', placeholder='Camera and robot vector', description='T', disabled=False, style=style, layout={'width': '50%'}),
            "camera_clb_ej": widgets.Text(value='', placeholder='Joint error', description='ej', disabled=False, style=style, layout={'width': '50%'}),

            "frame_label": widgets.HTML(value="Specify the reference frame based on the camera's setup:    <ul> <li>If the camera is attached to the robot (eye-in-hand), set it relative to the robot's base frame.</li> <li>If the camera is fixed (eye-to-hand), set it relative to the camera frame.</li> </ul> All measurements will be reported concerning this selected frame.", layout={'width': '99%'}, style=style),
            "frame_value": widgets.Text(value='[0, 0, 0, 0, 0, 0]', placeholder='[0, 0, 0, 0, 0, 0]', description='Frame', disabled=False, style=style),
            "ml_label": widgets.HTML(value="Specify the path to the model, for any ML-based detection method.", layout={'width': '99%'}, style=style),
            "ml_detection_type": widgets.Dropdown(value=0, options=[('Object detection', 0), ('Image classification', 1), ('Keypoint detection', 2)], description='Detection method', continuous_update=continuous_update, style=style),
            "ml_detection_path": widgets.Text(value='', placeholder='Path to the detection model file (*.pkl), e.g., ai_models/test.pkl.', description='Model path', disabled=False, layout={'width': '99%'}, style=style),
            "ml_kp_geometry": widgets.Textarea(description='Keypoint geometry (mm)', value='', 
            placeholder='# provide the geometry of each keypoint in the object frame \n' \
            '{"object1":{"kp1":[0, 0, 0], "kp2":[0, 0, 0], "kp3":[0, 0, 0]}, "object2":{"kp1":[0, 0, 0], "kp2":[0, 0, 0], "kp3":[0, 0, 0]}}', 
            disabled=False, rows=5, layout={'width': '99%'}, style=style),

            "init": widgets.Button( description='Initialize Parameters', disabled=False, tooltip='Initialize Parameters', button_style="success"), 
        }
        self.widget_input = {
            "sort_label": widgets.Label(value="Arrange the detected objects by confidence score, pixel distance, bounding-box area, and other criteria.", layout={'width': '99%'}, style=style),
            "sort_value": widgets.Dropdown(value=0, options=[('No sorting', 0), ('Random shuffle', 1), ('Confiddence score', 2), ('Pixel distance', 3), ('Bounding box area', 4)], description='Sorting', continuous_update=continuous_update, style=style),
            "sort_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),

            "sort_no_max_det" : widgets.IntSlider(value=100, min=1, max=200, step=1, description='Max detections per run', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 

            "sort_shuffle_max_det" : widgets.IntSlider(value=100, min=1, max=200, step=1, description='Max detections per run', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 

            "sort_conf_max_det" : widgets.IntSlider(value=100, min=1, max=200, step=1, description='Max detections per run', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 
            "sort_conf_ascending": widgets.Checkbox(value=False, description='Ascending order', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "sort_pixel_max_det" : widgets.IntSlider(value=100, min=1, max=200, step=1, description='Max detections per run', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 
            "sort_pixel_ascending": widgets.Checkbox(value=True, description='Ascending order', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "sort_pixel_value" : widgets.Text(value="", placeholder='[width, height]', description='Pixel (pxl)', disabled=False, layout={'width': '99%'}, style=style), 

            "sort_area_max_det" : widgets.IntSlider(value=100, min=1, max=200, step=1, description='Max detections per run', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 
            "sort_area_ascending": widgets.Checkbox(value=False, description='Ascending order', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "sort_xyz_max_det" : widgets.IntSlider(value=100, min=1, max=200, step=1, description='Max detections per run', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 
            "sort_xyz_ascending": widgets.Checkbox(value=True, description='Ascending order', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "sort_xyz_value" : widgets.Text(value="", placeholder='[x, y, z]', description='xyz (mm)', disabled=False, layout={'width': '99%'}, style=style), 

            "pose_label": widgets.Label(value="Estimate each object’s 6D pose (rotation rvec & translation tvec).", layout={'width': '99%'}, style=style),
            "pose_value": widgets.Dropdown(value=0, options=[('No pose', 0), ('Fit plane', 1), ('Keypoint', 2)], description='6D-pose method', continuous_update=continuous_update, style=style),
            "pose_kp_thr" : widgets.IntSlider(value=5, min=0, max=10, step=1, description='Error threshold (pxl)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "pose_plane_value": widgets.Text(value="[]", placeholder='[]', description='Plane', disabled=True, layout={'width': '99%'}, style=style),

            "color_label": widgets.Label(value="Apply a color mask to filter specific colors by adjusting hue, saturation, and value. Fine-tune these settings to isolate the desired color range for better detection results.", layout={'width': '99%'}, style=style),
            "color_enb": widgets.Checkbox(value=False, description='Apply color mask', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "color_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "color_h": widgets.IntRangeSlider(value=[60, 120], min=0, max=179, step=1, description='Hue', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "color_s": widgets.IntRangeSlider(value=[85, 170], min=0, max=255, step=1, description='Saturation', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "color_v": widgets.IntRangeSlider(value=[85, 170], min=0, max=255, step=1, description='Vue', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "color_inv": widgets.Checkbox(value=False, description='Invert color mask', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "ori_label": widgets.Label(value="Rotate the source image clockwise as needed.", layout={'width': '99%'}, style=style),
            "ori_rotate": widgets.Dropdown(value=0, options=[('No rotation', 0), ('90 degrees', 90), ('180 degrees', 180), ('270 degrees', 270)], description='Clockwise rotation', continuous_update=continuous_update, style=style),

            "roi_label": widgets.Label(value="Select the region of interest where the detection method is applied. Use the blue polygon selector on the output image to define this area.", layout={'width': '99%'}, style=style),
            "roi_enb": widgets.Checkbox(value=False, description='Apply ROI', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "roi_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "roi_value": widgets.Text(value='[]', placeholder='[]', description='ROI', disabled=True, layout={'width': '99%'}, style=style),
            "roi_offset" : widgets.IntSlider(value=0, min=-200, max=200, step=1, description='Offset', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "roi_inv": widgets.Checkbox(value=False, description='Invert region', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "roi_crop": widgets.Checkbox(value=False, description='Crop region', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "intensity_label": widgets.Label(value="Adjust brightness and contrast if necessary to enhance image details. Use the sliders for optimal visibility and improved detection results.", layout={'width': '99%'}, style=style),
            "intensity_enb": widgets.Checkbox(value=False, description='Apply intensity', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "intensity_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "intensity_a" : widgets.FloatSlider(value=1, min=0, max=4, step=0.01, description='Contrast', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "intensity_b" : widgets.IntSlider(value=0, min=-255, max=255, step=1, description='Brightness', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "bb_range_label": widgets.Label(value="Filter detected objects by their bounding box shape and size to improve detection results.", layout={'width': '99%'}, style=style),
            "bb_range_enb": widgets.Checkbox(value=False, description='Apply limit', continuous_update=continuous_update,layout={'width': '99%'}, style=style),
            "bb_range_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "bb_range_aspect_ratio": widgets.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.01, description='Aspect ratio', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "bb_range_area": widgets.IntRangeSlider(value=[0, 100000], min=0, max=100000, step=100, description='Area (pxl X pxl)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "2d_range_label": widgets.Label(value="Filter detected objects by limiting their bounding box center to a specific area in the image.", layout={'width': '99%'}, style=style),
            "2d_range_enb": widgets.Checkbox(value=False, description='Apply limit', continuous_update=continuous_update,layout={'width': '99%'}, style=style),
            "2d_range_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "2d_range_w": widgets.IntRangeSlider(value=[100, 200], min=0, max=2000, step=1, description='Width (pxl)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "2d_range_h": widgets.IntRangeSlider(value=[100, 200], min=0, max=2000, step=1, description='Height (pxl)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "2d_range_inv": widgets.Checkbox(value=False, description='Invert the range', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "3d_range_label": widgets.Label(value="Apply 3D constraints to remove detected objects outside the specified x, y, z range relative to the frame.", layout={'width': '99%'}, style=style),
            "3d_range_enb": widgets.Checkbox(value=False, description='Apply limit', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "3d_range_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "3d_range_x": widgets.IntRangeSlider(value=[250, 350], min=-1000, max=1000, step=1, description='x (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "3d_range_y": widgets.IntRangeSlider(value=[0, 50], min=-1000, max=1000, step=1, description='y (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "3d_range_z": widgets.IntRangeSlider(value=[0, 50], min=-1000, max=1000, step=1, description='z (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "3d_range_inv": widgets.Checkbox(value=False, description='Invert the range', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "rvec_range_label": widgets.Label(value="Apply orientation constraints relative to the given base rotation vector: remove detections whose rotation vector deviates outside the specified x, y, z angle ranges.", layout={'width': '99%'}, style=style),
            "rvec_range_enb": widgets.Checkbox(value=False, description='Apply limit', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "rvec_range_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "rvec_range_rvec_base" : widgets.Text(value="[0, 0, 0]", placeholder='[0, 0, 0]', description='Base rvec', disabled=False, layout={'width': '99%'}, style=style), 
            "rvec_range_x_angle": widgets.IntRangeSlider(value=[0, 180], min=0, max=180, step=1, description='x angle (deg)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "rvec_range_y_angle": widgets.IntRangeSlider(value=[0, 180], min=0, max=180, step=1, description='y angle (deg)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "rvec_range_z_angle": widgets.IntRangeSlider(value=[0, 180], min=0, max=180, step=1, description='z angle (deg)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "rvec_range_inv": widgets.Checkbox(value=False, description='Invert the range', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "tvec_range_label": widgets.Label(value="Remove detections whose translation vector (tvec) lies outside the specified x, y, z distance ranges.", layout={'width': '99%'}, style=style),
            "tvec_range_enb": widgets.Checkbox(value=False, description='Apply limit', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "tvec_range_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "tvec_range_x": widgets.IntRangeSlider(value=[250, 350], min=-1000, max=1000, step=1, description='x (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "tvec_range_y": widgets.IntRangeSlider(value=[0, 50], min=-1000, max=1000, step=1, description='y (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "tvec_range_z": widgets.IntRangeSlider(value=[0, 50], min=-1000, max=1000, step=1, description='z (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "tvec_range_inv": widgets.Checkbox(value=False, description='Invert the range', continuous_update=continuous_update, layout={'width': '99%'}, style=style),



            "method_value": widgets.Dropdown(value=0, options=[('No detection', 0), ('Ellipse detection', 2), ('Polygon detection', 3), ('Contour detection', 4), ('Aruco detection', 5), ('Barcode reader', 6), ('OCR', 7)], description='Detection method', continuous_update=continuous_update, style=style),

            "m_line_min_line_length": widgets.IntSlider(value=100, min=1, max=1000, step=1, description='Min line length', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "m_elp_pf_mode": widgets.Checkbox(value=False, description='Auto detection', continuous_update=continuous_update,layout={'width': '99%'}, style=style),
            "m_elp_nfa_validation": widgets.Checkbox(value=True, description='False alarm validation', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_elp_min_path_length": widgets.IntSlider(value=50, min=1, max=1000, step=1, description='Min path length', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_elp_min_line_length": widgets.IntSlider(value=10, min=1, max=1000, step=1, description='Min line length', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_elp_sigma": widgets.IntSlider(value=1, min=0, max=20, step=0.1, description='Blur', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_elp_gradient_threshold_value": widgets.IntSlider(value=20, min=1, max=100, step=1, description='Gradient', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
    

            "m_poly_inv": widgets.Checkbox(value=True, description='Inverse', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_poly_type": widgets.Dropdown(value=0, options=[('0: Otsu (auto)', 0), ('1: Binary', 1), ('2: Gaussian', 2)], description='Type', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_poly_thr" : widgets.IntSlider(value=127, min=0, max=255, step=1, description='Threshold value', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_poly_blur": widgets.IntSlider(value=3, min=1, max=20, step=1, description='Smoothing blur', continuous_update=continuous_update, layout={'width': '99%'}, style=style),                    
            "m_poly_mean_sub": widgets.IntSlider(value=0, min=-200, max=200, step=1, description='Mean subtract', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_poly_side" : widgets.IntSlider(value=3, min=3, max=20, step=1, description='Sides', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            
            "m_cnt_inv": widgets.Checkbox(value=True, description='Inverse', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_cnt_type": widgets.Dropdown(value=0, options=[('0: Otsu (auto)', 0), ('1: Binary', 1), ('2: Gaussian', 2)], description='Type', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_cnt_thr" : widgets.IntSlider(value=127, min=0, max=255, step=1, description='Threshold value', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_cnt_blur": widgets.IntSlider(value=3, min=1, max=20, step=1, description='Smoothing blur', continuous_update=continuous_update, layout={'width': '99%'}, style=style),                    
            "m_cnt_mean_sub": widgets.IntSlider(value=0, min=-200, max=200, step=1, description='Mean subtract', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "m_aruco_dictionary":widgets.Dropdown(value="DICT_4X4_100", options= ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000", "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000", "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000", "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"], description='Dictionary', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_aruco_marker_length": widgets.FloatSlider(value=20, min=1, max=100, step=0.1, description='Marker length (mm)', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_aruco_refine":widgets.Dropdown(value="CORNER_REFINE_APRILTAG", options=["CORNER_REFINE_NONE", "CORNER_REFINE_SUBPIX", "CORNER_REFINE_CONTOUR", "CORNER_REFINE_APRILTAG"], description='Refinement', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "m_aruco_subpix": widgets.Checkbox(value=False, description='Sub pixel', continuous_update=continuous_update, layout={'width': '99%'}, style=style),            

            "m_ocr_conf" : widgets.FloatSlider(value=0.1, min=0.01, max=1, step=0.01, description='Confidence', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 


            "m_barcode_format": widgets.Dropdown(value="Any", options=["Any", "Aztec", "Codabar", "Code39", "Code93", "Code128","DataMatrix",
                                "DataBar", "DataBarExpanded", "DataBarLimited", "DXFilmEdge", "EAN8", "EAN13", "ITF", "PDF417", "QRCode", "MicroQRCode",
                                "RMQRCode", "UPCA", "UPCE", "LinearCodes", "MaxiCode", "MatrixCodes"],
                                description='Format', continuous_update=continuous_update, layout={'width': '99%'}, style=style),

            "m_od_conf" : widgets.FloatSlider(value=0.5, min=0.01, max=1, step=0.01, description='Confidence', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 
            "m_od_cls" : widgets.Text(value="", placeholder='', description='Detection classes', disabled=False, layout={'width': '99%'}, style=style), 

            "m_cls_conf" : widgets.FloatSlider(value=0.5, min=0.01, max=1, step=0.01, description='Confidence', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 

            "m_kp_conf" : widgets.FloatSlider(value=0.5, min=0.01, max=1, step=0.01, description='Confidence', continuous_update=continuous_update, layout={'width': '99%'}, style=style), 
            "m_kp_cls" : widgets.Text(value="", placeholder='', description='Detection classes', disabled=False, layout={'width': '99%'}, style=style), 

            "display_text": widgets.Label(value="Configure how inference results are presented and saved", layout={'width': '99%'}, style=style),
            "display_enb": widgets.Checkbox(value=False, description='Apply formatting', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "display_line": widgets.HTML(value="<hr>", description=' ', layout={'width': '99%'}, style=style_2),
            "display_label": widgets.Dropdown(value=0, description='Display', options=[('None', -1), ('Bounding box only', 0), ('Bounding box and labels', 1)], continuous_update=continuous_update, style=style),
            "display_save": widgets.Checkbox(value=False, description='Save the annotated image in the "output/*.jpg"', continuous_update=continuous_update, layout={'width': '99%'}, style=style),
            "display_save_roi": widgets.Checkbox(value=False, description='Save the unannotated ROI image in the "output/*.jpg"', continuous_update=continuous_update, layout={'width': '99%'}, style=style),


        }
        
        self.widget_trigger = {
            "color_picker": widgets.ColorPicker(concise=False, description='Color picker', value='blue', disabled=False, style={'text_width': '0'}),
            "color_hsv": widgets.Text(value='Hue = 119, Saturation = 255, Value = 255', placeholder='', description='', disabled=True,),            

            "source_value": widgets.Dropdown(value=0, options=[('Stereo camera', 0), ('File', 1)], description='Image source', continuous_update=continuous_update, style=style),
            "source_feed": widgets.Dropdown(value="color_img", options=[('Color image', "color_img")], description='Feed', continuous_update=continuous_update, style=style, layout={'visible': 'none'}),

            "s_file_value": widgets.Text(value='', placeholder='Path to the file (*.jpg, *.jpeg, *.png, *.tiff, ...).Ex: img/test.jpg', description='File path', disabled=False, layout={'width': '99%'}, style=style),            
            "s_apply": widgets.Button( description='Capture Image', disabled=True, button_style="success", tooltip='Capture Image', style=style),
            #"s_update": widgets.Button( description='', disabled=False, button_style="", tooltip='Update source list', icon='refresh', layout={'width': '50px'}),
            "s_save_path": widgets.Text(value='', placeholder='*.jpg', description='Save image as', disabled=False, layout={'width': '99%'}),            
            "s_save": widgets.Button( description='Save', disabled=False, button_style="", tooltip='Save as'),

            #"model_path": widgets.Text(value='', placeholder='/full_path/to_the/object_detection_model.pkl', description='Object Detection Model', disabled=False, layout={'width': '99%'}, style=style),            
            #"model_save": widgets.Button( description='Set', disabled=False, button_style="", tooltip='Set'),

            #"robot_ip": widgets.Text(value='', placeholder='192.168.254.10', description='Robot IP Address', disabled=False, layout={'width': '99%'}, style=style),            
            #"robot_connect": widgets.Button( description='Connect', disabled=False, button_style="", tooltip='Connect'),

            #"camera_robot_calibration": widgets.Textarea(value='[[0.00525873615, -0.999894519, 0.0134620306, 46.5174596], [0.999959617, 0.00535678348, -0.00735796480, 32.0776662], [0.00728773209, -0.0135001806, 0.999882310, -4.24772615], [0.0, 0.0, 0.0, 1.0]]', placeholder='[[0.00525873615, -0.999894519, 0.0134620306, 46.5174596], [0.999959617, 0.00535678348, -0.00735796480, 32.0776662], [0.00728773209, -0.0135001806, 0.999882310, -4.24772615], [0.0, 0.0, 0.0, 1.0]]', description='Camera & Robot Calibration Matrix', disabled=False, layout={'width': '99%'}, style=style),            
            "out_prm_value_label": widgets.HTML(value="Preset", layout={'width': '99%'}, style=style),
            "out_prm_value": widgets.Textarea(value='', placeholder='',disabled=True,  rows=3, layout={'width': '99%'}),

            "out_prm_label": widgets.HTML(value="Sample API call", layout={'width': '99%'}, style=style),
            "out_prm": widgets.Textarea(value='', placeholder='',disabled=True,  rows=15, layout={'width': '99%'}),
            
            "out_return_label": widgets.HTML(value="Return value", layout={'width': '99%'}, style=style),
            "out_return": widgets.Textarea(value='', placeholder='', disabled=True, rows=5, layout={'width': '99%'}),

            "close": widgets.Button( description='Exit App', disabled=False, button_style="danger", tooltip='Exit App', layout={'justify-content': 'flex-end'}),
        }

class Detection_app(object):
    """docstring for App"""
    def __init__(self, **kwargs):
        super(Detection_app, self).__init__()
        self.retval =[]
        self.config = {}

        """widgets"""
        # widget
        self.widget_init = default_widget().widget_init
        self.widget_in = default_widget().widget_input
        self.widget_tr = default_widget().widget_trigger
        self.widget_helper = default_widget().widget_helper

        # create plots
        plt.close('all')
        self.plt = {"out":{"fig":None, "ax":None, "img":None}, "method":{"fig":None, "ax":None, "img":None}, "plane":{"fig":None, "ax":None, "img":None}, "clb":{"fig":None, "ax":None, "img":None}}

        """plots"""
        # out
        self.plt_out = widgets.Output()
        with self.plt_out:
            self.plt["out"]["fig"], self.plt["out"]["ax"] = plt.subplots(frameon=False)
            self.plt["out"]["img"] = self.plt["out"]["ax"].imshow(cv.cvtColor(np.zeros((5, 9), dtype=np.uint8), cv.COLOR_BGR2RGB))
            self.plt["out"]["fig"].canvas.header_visible = False
            self.plt["out"]["fig"].tight_layout()
            self.plt["out"]["ax"].axis('off')
            plt.show()
        self.plt_out.layout.display = "none" 

        # method
        self.plt_method = widgets.Output()
        with self.plt_method:
            self.plt["method"]["fig"], self.plt["method"]["ax"] = plt.subplots(frameon=False)
            self.plt["method"]["img"] = self.plt["method"]["ax"].imshow(np.zeros((5, 9), dtype=np.uint8))
            self.plt["method"]["fig"].canvas.header_visible = False
            self.plt["method"]["fig"].tight_layout()
            self.plt["method"]["fig"].set_size_inches((4.5, 2.5), forward=True)
            self.plt["method"]["ax"].axis('off')
            plt.show()
        self.plt_method.layout.visibility = "hidden"
        
        # plane
        self.plt_plane = widgets.Output()
        with self.plt_plane:
            # init fig and ax
            self.plane_plt_maker()
            plt.show()  
            # init plane
            self.plane_value = poly_select(self.widget_in["pose_plane_value"])
            # Initialize plane selector
            self.plane_selector = PolygonSelector(self.plt["plane"]["ax"], onselect=self.plane_value.onselect, useblit=True, props=dict(color='orange', linestyle='--'))
        self.plt_plane.layout.visibility = "hidden"

        # calibration plot
        self.plt_clb = widgets.Output(layout=widgets.Layout(flex="1 1 auto"))
        with self.plt_clb:
            self.plt["clb"]["fig"], self.plt["clb"]["ax"] = plt.subplots(frameon=False)
            self.plt["clb"]["img"] = self.plt["clb"]["ax"].imshow(np.zeros((5, 9), dtype=np.uint8))
            self.plt["clb"]["fig"].canvas.header_visible = False
            self.plt["clb"]["fig"].tight_layout()
            self.plt["clb"]["fig"].set_size_inches((4.5, 2.5), forward=True)
            self.plt["clb"]["ax"].axis('off')
            plt.show()


        """accordion for adjust the image"""
        # adjust_image
        color_picker_box = widgets.HBox([self.widget_tr[k] for k in [key for key in self.widget_tr.keys() if key.startswith('color_')]])
        acc_adjust_img = widgets.Accordion()
        acc_adjust_img.children = [
            widgets.VBox([self.widget_tr["source_value"], self.widget_tr["s_file_value"]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('ori_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('roi_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('intensity_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('color_')]]+[widgets.HTML("<hr>")]+[color_picker_box]),
        ]
        for i, title in enumerate(['Source', 'Orientation', 'Region of Interest', 'Intensity', 'Color Mask']):
            acc_adjust_img.set_title(i, title)    

        """init vbox"""
        acc_init_vbox = widgets.Accordion()
        acc_init_vbox.children = [
            widgets.VBox([self.widget_init[k] for k in [key for key in self.widget_init.keys() if key.startswith('camera_')]]),
            widgets.VBox([self.widget_init[k] for k in [key for key in self.widget_init.keys() if key.startswith('frame_')]]),
            widgets.VBox([self.widget_init[k] for k in [key for key in self.widget_init.keys() if key.startswith('ml_')]]),
        ] 
        for i, title in enumerate(['1. Camera Mounting', '2. Frame', '3. AI Models']):
            acc_init_vbox.set_title(i, title)
        
        init_vbox = widgets.VBox([
            acc_init_vbox,
            widgets.HBox([self.widget_init[k] for k in [key for key in ["init"]]]),
        ])  


        """method vbox"""
        method_vbox = widgets.VBox([
            self.widget_in["method_value"],
            widgets.VBox([widgets.HTML("<hr>")]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_line_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_elp_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_poly_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_cnt_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_aruco_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_barcode_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_ocr_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_od_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_cls_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('m_kp_')]]),
        ], layout={'width': '100%'})        

        """accordion for settings"""
        # acc setting
        acc_helper = widgets.Accordion()

        acc_helper.children = [
            widgets.VBox([self.widget_helper[k] for k in [key for key in self.widget_helper.keys() if key.startswith('xyz_')]]),
            widgets.VBox([self.widget_helper[k] for k in [key for key in self.widget_helper.keys() if key.startswith('pixel_')]]),
            widgets.VBox([
                self.widget_helper["clb_label"],
                widgets.HBox([
                    widgets.VBox([self.widget_helper["clb_aruco_enb"],
                                self.widget_helper["clb_aruco_marker_length"],
                                self.widget_helper["clb_thr"],
                                widgets.VBox([widgets.HTML("<hr>")]),
                                widgets.HBox([self.widget_helper["clb_capture_b"], self.widget_helper["clb_capture_m_b"], widgets.Label(layout=widgets.Layout(flex="1")), self.widget_helper["clb_robot_b" ], self.widget_helper["clb_clear_b" ]]),
                                self.widget_helper["clb_data"],
                                widgets.HBox([self.widget_helper["clb_calibrate_b"], widgets.Label(layout=widgets.Layout(flex="1")), self.widget_helper["clb_data_label"]]),
                                widgets.VBox([widgets.HTML("<hr>")]),
                                self.widget_helper["clb_result"],
                                self.widget_helper["clb_apply_b"],], layout=widgets.Layout(flex="4 1 auto")),
                    
                    self.plt_clb,
                ])

            ]),  
      ]

        for i, title in enumerate(["Pixel to XYZ", "XYZ to Pixel", "Eye-in-Hand Calibration"]):
            acc_helper.set_title(i, title)  

        """6d-pose vbox"""
        pose_vbox = widgets.VBox([
            self.widget_in["pose_value"],
            widgets.VBox([widgets.HTML("<hr>")]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('pose_kp_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('pose_plane_')]]),
        ], layout={'width': '100%'})        


        """accordion for settings"""
        # acc setting
        acc_setting = widgets.Accordion()
        acc_setting.children = [
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('sort_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('bb_range_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('2d_range_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('3d_range_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('rvec_range_')]]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('tvec_range_')]]),
            widgets.HBox([pose_vbox, self.plt_plane]),
            widgets.VBox([self.widget_in[k] for k in [key for key in self.widget_in.keys() if key.startswith('display_')]]),
        ]

        for i, title in enumerate(["Sorting Result", "Bounding Box Limits", "Center Pixel Limits", "XYZ Limits", "Rotation Vector Limits", "Translation Vector Limits", "6D Pose", "Display"]):
            acc_setting.set_title(i, title)    


        """result"""
        result_vbox = widgets.VBox([self.widget_tr[k] for k in [key for key in self.widget_tr.keys() if key.startswith('out_')]])

        """tab"""
        tabs = [
            init_vbox,
            acc_adjust_img,
            widgets.HBox([method_vbox, self.plt_method]),
            acc_setting,
            result_vbox,
            acc_helper,
        ]
        self.tab = widgets.Tab()
        self.tab.children = tabs

        # hide
        for i in range(1, len(self.tab.children)):
            self.tab.children[i].layout.display = 'none'
        
        self.tab.set_title(0, 'Initialization')
        self.tab.set_title(1, 'Image')
        self.tab.set_title(2, 'Detection')
        self.tab.set_title(3, 'Setting')
        self.tab.set_title(4, 'Result')
        self.tab.set_title(5, 'Helper Functions')
        
        # header
        header = widgets.HBox([
            self.widget_tr["s_apply"],
            self.widget_tr["close"],
        ])

        # display
        display(widgets.VBox([header, self.tab, self.plt_out]))
        
        # init parameters
        self.widget_init["init"].on_click(self.init_parameter)


        """roi"""
        # init roi
        self.roi_value = poly_select(self.widget_in["roi_value"])

        # Initialize PolygonSelector
        self.roi_selector = PolygonSelector(self.plt["out"]["ax"], onselect=self.roi_value.onselect, useblit=True, props=dict(color='blue', linestyle='--'))

        # interactive for source
        interactive(self.hide_show_source, source_value=self.widget_tr["source_value"])

        # interactive for ip
        interactive(self.hide_show_ip, source_value=self.widget_init["camera_setup_type"])

        # interactive for ml model
        interactive(self.hide_show_model, ml_detection_type=self.widget_init["ml_detection_type"])

        # interactive color_picker
        self.widget_tr["color_picker"].observe(self.hex_to_hsv, names='value')
        
        # capture
        self.widget_tr["s_apply"].on_click(self.capture_camera_data)

        # capture
        #self.widget_tr["s_update"].on_click(self.update_source_list)

        # save
        self.widget_tr["s_save"].on_click(self.save_as_source)

        # close
        self.widget_tr["close"].on_click(self.__del__)

        # pixel to xyz
        self.widget_helper["xyz_convert"].on_click(self.pixel_to_xyz)

        # xyz to pixel
        self.widget_helper["pixel_convert"].on_click(self.xyz_to_pixel)


        # calibrate
        self.widget_helper["clb_capture_b"].on_click(self.clb_capture_image)
        self.widget_helper["clb_capture_m_b"].on_click(self.clb_capture_multiple_image)
        self.widget_helper["clb_robot_b"].on_click(self.clb_robot)
        self.widget_helper["clb_clear_b"].on_click(self.clb_clear_data)
        self.widget_helper["clb_calibrate_b"].on_click(self.clb_calibrate)
        self.widget_helper["clb_apply_b"].on_click(self.clb_apply)


    def __del__(self, b):
        # buttons
        self.widget_tr["close"].layout.display = "none"
        self.widget_tr["s_apply"].layout.display = "none"

        try:
            self.d.camera.close()
        except Exception as ex:
            pass

        try:
            self.d.close()
        except Exception as ex:
            pass
        
        # plot
        plt.close('all')

        # tabs
        self.tab.close()


    def init_parameter(self, b):
        # disable elements in init tab
        for k in self.widget_init.keys():
            self.widget_init[k].disabled = True
        self.widget_init["init"].layout.display = 'none'

        # data
        self.data = None
        
        # robot and camera mount
        camera_mount = None
        robot = None
        if self.widget_init["camera_setup_type"].value == 0:
            try:
                robot_tmp = Dorna()
                if robot_tmp.connect(self.widget_init["camera_setup_robot_ip"].value):
                    robot = robot_tmp
                    camera_mount = "dorna_ta_j4_1"
                    if self.widget_init["camera_clb_apply"].value:
                        camera_mount = {
                            "type": "dorna_ta_j4_1",
                            "T": ast.literal_eval(self.widget_init["camera_clb_T"].value),
                            "ej": ast.literal_eval(self.widget_init["camera_clb_ej"].value),
                        }
        
            except Exception as ex:
                pass
        
        # frame
        try:
            frame = ast.literal_eval(self.widget_init["frame_value"].value)
            if len(frame) != 6:
                frame = [0, 0, 0, 0, 0, 0]
        except Exception as ex:
            frame = [0, 0, 0, 0, 0, 0]

        # camera
        camera_connected = False
        camera = Camera()
        try:
            if camera.connect():
                camera_connected = True
        except Exception as ex:
            pass
        
        # calobrate
        self.clb = None
        if robot and camera_connected:
            clb_prm = {'detection': {'cmd': 'aruco', 'dictionary': "DICT_4X4_100", 'marker_length': 20, 'refine': "CORNER_REFINE_APRILTAG", 'subpix': False}}
            clb_detection = Detection(camera=camera, robot=None, **clb_prm)
            self.clb = Calibration(robot, clb_detection)

        # detect
        self.d = Detection(camera=camera, robot=robot, camera_mount=camera_mount, frame=frame)
        
        # ocr
        self.d.init_ocr()
        
        # object_detection
        ml_detection_path = self.widget_init["ml_detection_path"].value
        if ml_detection_path:
            if self.widget_init["ml_detection_type"].value == 0:
                self.d.init_od(ml_detection_path) 
                self.widget_in["method_value"].options = list(self.widget_in["method_value"].options) + [("Object detection", 8)]
            elif self.widget_init["ml_detection_type"].value == 1:
                self.d.init_cls(ml_detection_path) 
                self.widget_in["method_value"].options = list(self.widget_in["method_value"].options) + [("Image classification", 9)]
            elif self.widget_init["ml_detection_type"].value == 2:
                self.d.init_kp(ml_detection_path) 
                self.widget_in["method_value"].options = list(self.widget_in["method_value"].options) + [("Keypoint detection", 10)]


        # hide element in method
        for k in [key for key in self.widget_in.keys() if key.startswith('m_')]:
            self.widget_in[k].layout.display = 'none'

        # hide element in pose
        for k in [key for key in self.widget_in.keys() if key.startswith('pose_plane') or key.startswith('pose_kp')]:
            self.widget_in[k].layout.display = 'none'

        # enable capture image
        self.widget_tr["s_apply"].disabled = False

        # show tabs
        for i in range(1, len(self.tab.children)):
            self.tab.children[i].layout.display = "flex"

        # interactive run with no changing data
        interactive(self._detect_pattern, **self.widget_in)

        # display plot
        #self.plt["out"]["img"].set_visible(True)
        self.plt_out.layout.display = "flex"


    def clb_capture_image(self, b):
        marker_length=self.widget_helper["clb_aruco_marker_length"].value
        use_aruco=self.widget_helper["clb_aruco_enb"].value
        thr=self.widget_helper["clb_thr"].value
        # capture image
        self.clb.capture_image(marker_length=marker_length, use_aruco=use_aruco, thr=thr)
        self.clb.add_data()

        # update plot
        self.plt["clb"]["img"].set_data(cv.cvtColor(self.clb.img, cv.COLOR_BGR2RGB))
        self.plt["clb"]["fig"].canvas.draw()
        self.plt["clb"]["fig"].canvas.flush_events()
        # let it refresh
        
        # update lsit
        self.widget_helper["clb_data"].value = "".join(f"{json.dumps(d)}\n" for d in [x["cmd"] for data in self.clb.collected_data for x in data])
        self.widget_helper["clb_data_label"].value = f"Collected data (size: {sum([len(x) for x in self.clb.collected_data])})"


    def clb_capture_multiple_image(self, b):
        joint_variance = [[-2+random.uniform(-0.5, 0.5), 2+random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
                        [0], 
                        [0],
                        [0],
                        [-2+random.uniform(-0.5, 0.5), 2+random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
                        [0]]
        joint_deviation = list(itertools.product(*joint_variance))
        current_joint = np.array(self.clb.robot.get_all_joint()[0:6])
        target_joint_list = [current_joint+joint for joint in joint_deviation]
        for j in target_joint_list:
            self.clb.robot.jmove(rel=0, j0=j[0], j1=j[1], j2=j[2], j3=j[3], j4=j[4], j5=j[5], vel=50, accel=800, jerk=1000, cont=0)
            self.clb.robot.sleep(0.2)
            self.clb_capture_image(None)

    def clb_robot(self, b):
        self.clb.robot.set_motor(int(self.clb.robot.get_motor())^1)


    def clb_clear_data(self, b):
        # clear data
        self.clb.clear_data()

        # update list
        self.widget_helper["clb_data"].value = ""
        self.widget_helper["clb_data_label"].value = "Collected data (size: 0)"


    def clb_calibrate(self, b):        
        # disable everything
        self.widget_helper["clb_calibrate_b"].disabled = True

        # calibrating
        self.widget_helper["clb_result"].value = "Calibrating..."

        
        # calibrate
        result, error = self.clb.calibrate()
        
        # update result
        self.widget_helper["clb_result"].value = "camera_mount = "+json.dumps(result)+"\n error = "+str(error)
        
        # disable everything
        self.widget_helper["clb_calibrate_b"].disabled = False

    

    def clb_apply(self, b):
        # apply
        self.d.set_camera_mount = self.d.set_camera_mount(self.clb.result)

        # update init
        self.widget_init["camera_clb_T"].value = json.dumps(self.clb.result["T"])
        self.widget_init["camera_clb_ej"].value = json.dumps(self.clb.result["ej"])
        self.widget_init["camera_clb_apply"].value = True


    def hex_to_hsv(self, change):
        # Remove '#' if present
        hex_color = change['new'].lstrip('#')

        # Convert hex to RGB
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Normalize RGB values to the range [0, 1]
        normalized_rgb = tuple(value / 255.0 for value in rgb_color)

        # Convert RGB to HSV
        hsv_color = colorsys.rgb_to_hsv(*normalized_rgb)

        # Adjust HSV values to the common OpenCV conventions
        h = int(hsv_color[0] * 179)
        s = int(hsv_color[1] * 255)
        v = int(hsv_color[2] * 255)
        self.widget_tr["color_hsv"].value = f"Hue = {h}, Saturation = {s}, Value = {v}"

        # color
        self.widget_in["color_h"].value = [max(0, h-20), min(179, h+20)]
        self.widget_in["color_s"].value = [max(0, s-20), min(255, s+20)]
        self.widget_in["color_v"].value = [max(0, v-20), min(255, v+20)]

    
    def save_as_source(self, b):
        file_path = self.widget_tr["s_save_path"].value
        
        # opencv
        cv.imwrite(file_path, self.d.camera_data["color_img"])

        
    def open_pkl(self, file_path):
        with open(file_path, 'rb') as file:
            loaded_data = pkl.load(file)
        return loaded_data


    def pixel_to_xyz(self, b):
        pxl = ast.literal_eval(self.widget_helper["xyz_pxl"].value)
        xyz = self.d.pixel_to_xyz(pxl)
        self.widget_helper["xyz_xyz"].value = f"[{', '.join(f'{value:.1f}' for value in xyz)}]"

    def xyz_to_pixel(self, b):
        xyz = ast.literal_eval(self.widget_helper["pixel_xyz"].value)
        pxl = self.d.xyz_to_pixel(xyz)
        self.widget_helper["pixel_pxl"].value = f"[{', '.join(f'{value:.1f}' for value in pxl)}]"


    def capture_camera_data(self, b):
        self.data = None
        if self.widget_tr["source_value"].value == 1: # read from file
            # file_path
            self.data = self.widget_tr["s_file_value"].value

        # call detect pattern
        kwargs = {k:self.widget_in[k].value for k in self.widget_in.keys()}
        self._detect_pattern(**kwargs)


    def update_source_list(self, b):
        all_device = self.d.camera.all_device()
        
        i = 0
        options = []
        for device in all_device:
            options.append((device["name"] +" (S/N: "+device["serial_number"], ")", i))
            i += 1
        options.append(('Image file', i))
        self.widget_tr["source_value"].options = options


    def hide_show_source(self, **kwargs):
        if kwargs["source_value"] == 1:
            self.widget_tr["s_file_value"].layout.display = "flex"
            #self.widget_tr["source_feed"].layout.display = "none"
        elif kwargs["source_value"] == 0:
            self.widget_tr["s_file_value"].layout.display = "none"
            #self.widget_tr["source_feed"].layout.display = "flex"


    def hide_show_ip(self, **kwargs):
        if kwargs["source_value"] == 0:
            self.widget_init["camera_setup_robot_ip"].layout.display = "flex"
            self.widget_init["camera_clb_apply"].layout.display = "flex"
            self.widget_init["camera_clb_T"].layout.display = "flex"
            self.widget_init["camera_clb_ej"].layout.display = "flex"
        elif kwargs["source_value"] == 1:
            self.widget_init["camera_setup_robot_ip"].layout.display = "none"
            self.widget_init["camera_clb_apply"].layout.display = "none"
            self.widget_init["camera_clb_T"].layout.display = "none"
            self.widget_init["camera_clb_ej"].layout.display = "none"

    def hide_show_model(self, **kwargs):
        if kwargs["ml_detection_type"] == 2:
            self.widget_init["ml_kp_geometry"].layout.display = "flex"
        else:
            self.widget_init["ml_kp_geometry"].layout.display = "none"


    def plane_plt_maker(self):
        # create
        self.plt["plane"]["fig"], self.plt["plane"]["ax"] = plt.subplots(frameon=False)
        #fig.suptitle("Select 3 points of interest")
        self.plt["plane"]["fig"].canvas.header_visible = False
        self.plt["plane"]["fig"].tight_layout()
        self.plt["plane"]["fig"].set_size_inches((4.5, 2.5), forward=True)
        self.plt["plane"]["ax"].axis('off')
        
        # Set the height and calculate the width based on the golden ratio
        height = 1.0
        width = 1.0

        # Draw the ellipse in magenta
        ellipse = Ellipse((0, 0), 2 * width, 2 * height, linewidth=1, edgecolor='#FF00FF', facecolor='none')
        self.plt["plane"]["ax"].add_patch(ellipse)

        # Draw the minimum bounding box around the ellipse in magenta
        min_bounding_box = plt.Rectangle((-width, -height), 2 * width, 2 * height, linewidth=1, edgecolor='#FF00FF', facecolor='none', label='Oriented Bounding Box')
        self.plt["plane"]["ax"].add_patch(min_bounding_box)

        # Draw major and minor axes
        major_axis = plt.Line2D([0, width], [0, 0], color='red', linestyle='dashed', linewidth=1)
        major_axis.set_label('Major Axis')
        minor_axis = plt.Line2D([0, 0], [0, height], color='green', linestyle='dashed', linewidth=1)
        minor_axis.set_label('Minor Axis')
        self.plt["plane"]["ax"].add_line(major_axis)
        self.plt["plane"]["ax"].add_line(minor_axis)

        # Plot the center of the rectangle in blue
        self.plt["plane"]["ax"].plot(0, 0, marker='o', markersize=6, color='blue', label='Center')

        # Set axis limits with x and y axes twice as large
        self.plt["plane"]["ax"].set_xlim(-2 * width, 2 * width)
        self.plt["plane"]["ax"].set_ylim(-2 * height, 2 * height)

        # Display the legend
        self.plt["plane"]["ax"].legend()

        # Set aspect ratio
        self.plt["plane"]["ax"].set_aspect(1/1.68)

        # invert y
        self.plt["plane"]["ax"].invert_yaxis()


    def api_call(self, prm):
        code = textwrap.dedent(
f"""# imports
from dorna2 import Dorna
from camera import Camera
from dorna_vision import Detection

# robot
robot = Dorna()
robot.connect(_robot_ip_address_)

# camera
camera = Camera()
camera.connect()

# detection 
preset = {prm}
detection = Detection(camera=camera, robot=robot, **preset)

# call the detection
retval = detection.run()

# sample motion
if len(retval) > 0:
    robot.go(retval[0]["xyz"], ej=retval[0]["ej"], tcp=tcp)

# close
robot.close()
camera.close()
detection.close()
""")
        return code
    

    def _detect_pattern(self, **kwargs):
        try:
            # adjust kwargs
            prm = {}
            _prm ={}

            # camera_mount
            if self.widget_init["camera_clb_apply"].value:
                _c_mount = {"type": "dorna_ta_j4_1", "T": json.loads(self.widget_init["camera_clb_T"].value), "ej": json.loads(self.widget_init["camera_clb_ej"].value)}
                _prm["camera_mount"] = _c_mount

            # feed
            prm["feed"] = self.widget_tr["source_feed"].value
            
            # ori
            prm["rot"] = kwargs["ori_rotate"]
            if kwargs["ori_rotate"]:
                _prm["rot"] = prm["rot"]

            # intensity
            prm["intensity"] = {"a": 1.0, "b": 0}
            if kwargs["intensity_enb"]:
                prm["intensity"] = {"a": kwargs["intensity_a"], "b": kwargs["intensity_b"]}
                _prm["intensity"] = prm["intensity"]
            
            # color
            prm["color"] = {"low_hsv": [0, 0, 0], "high_hsv": [255, 255, 255], "inv": 0}
            if kwargs["color_enb"]:
                prm["color"] = {"low_hsv": [kwargs[k][0] for k in ["color_h", "color_s", "color_v"]], "high_hsv": [kwargs[k][1] for k in ["color_h", "color_s", "color_v"]], "inv": kwargs["color_inv"]}
                _prm["color"] = prm["color"]

            # roi
            prm["roi"] = {"corners": [], "inv": 0, "crop": 0, "offset": 0}
            if kwargs["roi_enb"]:
                prm["roi"] = {"corners": ast.literal_eval(kwargs["roi_value"]), "inv": kwargs["roi_inv"], "crop": kwargs["roi_crop"], "offset": kwargs["roi_offset"]}
                _prm["roi"] = prm["roi"]
            
            # detection
            prm["detection"] = {}
            if kwargs["method_value"] != 0:
                if kwargs["method_value"] == 1:
                    prm["detection"] = {"cmd":"line", "min_line_length": kwargs["m_line_min_line_length"]}
                if kwargs["method_value"] == 2:
                    prm["detection"] = {"cmd":"elp", "min_path_length": kwargs["m_elp_min_path_length"], "min_line_length": kwargs["m_elp_min_line_length"], "nfa_validation": kwargs["m_elp_nfa_validation"], "sigma": kwargs["m_elp_sigma"], "gradient_threshold_value": kwargs["m_elp_gradient_threshold_value"], "pf_mode": kwargs["m_elp_pf_mode"]}
                elif kwargs["method_value"] == 3:
                    prm["detection"] = {"cmd":"poly", "type": kwargs["m_poly_type"], "inv": kwargs["m_poly_inv"], "blur": kwargs["m_poly_blur"], "thr": kwargs["m_poly_thr"], "mean_sub": kwargs["m_poly_mean_sub"], "side": kwargs["m_poly_side"]}
                elif kwargs["method_value"] == 4:
                    prm["detection"] = {"cmd":"cnt", "type": kwargs["m_cnt_type"], "inv": kwargs["m_cnt_inv"], "blur": kwargs["m_cnt_blur"], "thr": kwargs["m_cnt_thr"], "mean_sub": kwargs["m_cnt_mean_sub"]}
                elif kwargs["method_value"] == 5:
                    prm["detection"] = {"cmd":"aruco", "dictionary": kwargs["m_aruco_dictionary"], "marker_length": kwargs["m_aruco_marker_length"], "refine": kwargs["m_aruco_refine"] , "subpix": kwargs["m_aruco_subpix"]}
                elif kwargs["method_value"] == 6:
                    prm["detection"] = {"cmd":"barcode", "format": kwargs["m_barcode_format"]}
                elif kwargs["method_value"] == 7:
                    prm["detection"] = {"cmd":"ocr", "conf": kwargs["m_ocr_conf"]}
                elif kwargs["method_value"] == 8:
                    try:
                        cls_name =  ast.literal_eval(kwargs["m_od_cls"])
                    except:
                        cls_name = []
                    prm["detection"] = {"cmd":"od", "path": self.widget_init["ml_detection_path"].value, "conf": kwargs["m_od_conf"], "cls": cls_name}   
                elif kwargs["method_value"] == 9:
                    prm["detection"] = {"cmd":"cls", "path": self.widget_init["ml_detection_path"].value, "conf": kwargs["m_cls_conf"]}   
                elif kwargs["method_value"] == 10:
                    try:
                        cls_name =  ast.literal_eval(kwargs["m_kp_cls"])
                    except:
                        cls_name = {}
                    prm["detection"] = {"cmd":"kp", "path": self.widget_init["ml_detection_path"].value, "conf": kwargs["m_kp_conf"], "cls": cls_name}   
                _prm["detection"] = prm["detection"]
            
            #sort
            prm["sort"] = {"cmd": None, "max_det": kwargs["sort_no_max_det"]}
            if kwargs["sort_value"] != 0:
                if kwargs["sort_value"] == 1: # shuffle
                    prm["sort"] = {"cmd":"shuffle", "max_det": kwargs["sort_shuffle_max_det"]}
                elif kwargs["sort_value"] == 2: # conf
                    prm["sort"] = {"cmd":"conf", "ascending": kwargs["sort_conf_ascending"], "max_det": kwargs["sort_conf_max_det"]}
                elif kwargs["sort_value"] == 3: # pixel
                    try:
                        sort_pixel_value = ast.literal_eval(kwargs["sort_pixel_value"])
                    except:
                        sort_pixel_value = [0 ,0]
                    prm["sort"] = {"cmd":"pxl", "pxl": sort_pixel_value, "ascending": kwargs["sort_pixel_ascending"], "max_det": kwargs["sort_pixel_max_det"]}
                elif kwargs["sort_value"] == 4: # area
                    prm["sort"] = {"cmd":"area", "ascending": kwargs["sort_area_ascending"], "max_det": kwargs["sort_area_max_det"]}
                _prm["sort"] = prm["sort"]

            #limit
            prm["limit"] = {}
            if kwargs["bb_range_enb"]: # bb
                prm["limit"]["bb"] = {}
                prm["limit"]["bb"]["aspect_ratio"] = list(kwargs["bb_range_aspect_ratio"])
                prm["limit"]["bb"] ["area"] = list(kwargs["bb_range_area"])
            if kwargs["2d_range_enb"]: # center
                prm["limit"]["center"] = {}
                prm["limit"]["center"]["width"] = list(kwargs["2d_range_w"])
                prm["limit"]["center"]["height"] = list(kwargs["2d_range_h"])
                prm["limit"]["center"]["inv"] = int(kwargs["2d_range_inv"])
            if kwargs["3d_range_enb"]: # xyz
                prm["limit"]["xyz"] = {}
                prm["limit"]["xyz"]["x"] = list(kwargs["3d_range_x"])
                prm["limit"]["xyz"]["y"] = list(kwargs["3d_range_y"])
                prm["limit"]["xyz"]["z"] = list(kwargs["3d_range_z"])
                prm["limit"]["xyz"]["inv"] = int(kwargs["3d_range_inv"])
            if kwargs["tvec_range_enb"]: # tvec
                prm["limit"]["tvec"] = {}
                prm["limit"]["tvec"]["x"] = list(kwargs["tvec_range_x"])
                prm["limit"]["tvec"]["y"] = list(kwargs["tvec_range_y"])
                prm["limit"]["tvec"]["z"] = list(kwargs["tvec_range_z"])
                prm["limit"]["tvec"]["inv"] = int(kwargs["tvec_range_inv"])
            if kwargs["rvec_range_enb"]: # rvec
                prm["limit"]["rvec"] = {}
                prm["limit"]["rvec"]["rvec_base"] = ast.literal_eval(kwargs["rvec_range_rvec_base"])
                prm["limit"]["rvec"]["x_angle"] = list(kwargs["rvec_range_x_angle"])
                prm["limit"]["rvec"]["y_angle"] = list(kwargs["rvec_range_y_angle"])
                prm["limit"]["rvec"]["z_angle"] = list(kwargs["rvec_range_z_angle"])
                prm["limit"]["rvec"]["inv"] = int(kwargs["rvec_range_inv"])
            if prm["limit"]:
                _prm["limit"] = prm["limit"]

            # pose
            if kwargs["pose_value"]:
                prm["pose"] = {}
                if kwargs["pose_value"] == 1: # plane
                    try:
                        plane_value = ast.literal_eval(kwargs["pose_plane_value"])
                    except:
                        plane_value = []
                    prm["pose"] = {"cmd":"plane", "plane": plane_value}
                elif kwargs["pose_value"] == 2: #keypoint
                    try:
                        kp_value = ast.literal_eval(self.widget_init["ml_kp_geometry"].value)
                    except:
                        kp_value = {}
                    prm["pose"] = {"cmd":"kp", "kp": kp_value, "thr": kwargs["pose_kp_thr"]}
                _prm["pose"] = prm["pose"]
            
            # display
            if kwargs["display_enb"]:
                prm["display"] = {"label": kwargs["display_label"], "save_img": kwargs["display_save"], "save_img_roi": kwargs["display_save_roi"]}
                _prm["display"] = prm["display"]


            """hide and show inputs"""
            show_key = [[key for key in self.widget_in.keys() if key.startswith(term)] for term in ["m_nothing", "m_line", "m_elp", "m_poly", "m_cnt", "m_aruco", "m_barcode", "m_ocr", "m_od", "m_cls", "m_kp"]][kwargs["method_value"]]
            hide_key = [key for key in self.widget_in.keys() if key.startswith('m_') and key not in show_key] 
            for k in show_key:
                if self.widget_in[k].layout.display != "flex":
                    self.widget_in[k].layout.display = "flex"
            for k in hide_key:
                if self.widget_in[k].layout.display != "none":
                    self.widget_in[k].layout.display = "none"

            """hide and show pose"""
            show_key = [[key for key in self.widget_in.keys() if key.startswith(term)] for term in ["pose_nothing", "pose_plane", "pose_kp"]][kwargs["pose_value"]]
            hide_key = [key for key in self.widget_in.keys() if (key.startswith('pose_plane') or key.startswith('pose_kp')) and key not in show_key] 
            for k in show_key:
                if self.widget_in[k].layout.display != "flex":
                    self.widget_in[k].layout.display = "flex"
            for k in hide_key:
                if self.widget_in[k].layout.display != "none":
                    self.widget_in[k].layout.display = "none"

            """hide and show sort"""
            show_key = [[key for key in self.widget_in.keys() if key.startswith(term)] for term in ["sort_no", "sort_shuffle", "sort_conf", "sort_pixel", "sort_area", "sort_xyz"]][kwargs["sort_value"]]
            hide_key = [key for key in self.widget_in.keys() if (key.startswith('sort_no') or key.startswith('sort_shuffle') or key.startswith('sort_conf') or key.startswith('sort_pixel') or key.startswith('sort_area') or key.startswith('sort_xyz')) and key not in show_key] 
            for k in show_key:
                if self.widget_in[k].layout.display != "flex":
                    self.widget_in[k].layout.display = "flex"
            for k in hide_key:
                if self.widget_in[k].layout.display != "none":
                    self.widget_in[k].layout.display = "none"

            # display thr
            if kwargs["pose_value"] in [1]: # polygon and contour
                self.plt_plane.layout.visibility = "visible"

            else:      
                self.plt_plane.layout.visibility = "hidden"

            # run pattern detection
            self.prm = prm
            retval = self.d.run(data=self.data, **prm)
            self.data = dict(self.d.camera_data)
            
            # adjust the frame size
            self.plt["out"]["img"].set_extent([0, self.d.camera_data[prm["feed"]].shape[1], self.d.camera_data[prm["feed"]].shape[0], 0])
            self.plt["method"]["img"].set_extent([0, self.d.camera_data[prm["feed"]].shape[1], self.d.camera_data[prm["feed"]].shape[0], 0])
            
            # display thr
            if kwargs["method_value"] in [3, 4]: # polygon and contour
                self.plt["method"]["img"].set_data(cv.cvtColor(self.d.img_thr, cv.COLOR_GRAY2RGB))
                self.plt["method"]["fig"].canvas.draw_idle()
                self.plt_method.layout.visibility = "visible"

            else:      
                self.plt_method.layout.visibility = "hidden"

            # remove old text when updating
            for txt in self.plt["out"]["ax"].texts:
                txt.remove()

            # Update the existing plot
            self.plt["out"]["img"].set_data(cv.cvtColor(self.d.img, cv.COLOR_BGR2RGB))
            
            # Update the existing plot with the timestamped image
            ts = self.d.retval["camera_data"]["timestamp"]
            dt = datetime.fromtimestamp(ts)
            caption = f'Img captured at: {dt.strftime("%Y-%m-%d %H:%M:%S")}'
            
            #draw the caption on top of the image
            self.plt["out"]["ax"].text(
                0.5, 1.05, caption,
                transform=self.plt["out"]["ax"].transAxes,
                ha="center", va="bottom",
                clip_on=False, 
                color="black",
                fontsize=9,
            )
            self.plt["out"]["fig"].canvas.draw_idle() 

            # type retval
            self.retval = retval
            json_str = json.dumps(retval)
            converted_retval = json.loads(json_str, parse_int=lambda x: int(x), parse_float=lambda x: float(x), parse_constant=lambda x: x, object_hook=lambda d: {k: 1 if v is True else 0 if v is False else v for k, v in d.items()}) 
            self.widget_tr["out_return"].value = json.dumps(converted_retval)

            # api call
            self.widget_tr["out_prm"].value = self.api_call(_prm)
            self.config = kwargs
            
            # prm
            self.widget_tr["out_prm_value"].value = textwrap.dedent(f"""{_prm}""")


        except Exception as ex:
            pass