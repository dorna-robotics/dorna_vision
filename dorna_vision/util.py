import cv2 as cv
import numpy as np
import colorsys
from dorna_vision.board import Aruco

# rgb_img -> binary
def binary_thr(bgr_img, type=0, inv=True, blur=3, thr_val=127, mean_sub=2):    
    # gray image
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_RGB2GRAY)    

    # Apply GaussianBlur to further reduce noise
    #blur_img = cv.bilateralFilter(gray_img,90,75,75)
    #blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    blur_img = cv.blur(gray_img,(blur,blur))
    
    # set the mode
    mode = 2*type + inv
    
    if mode < 1:
        _, thr_img = cv.threshold(blur_img, thr_val, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    elif mode == 1:
        _, thr_img = cv.threshold(blur_img, thr_val, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    elif mode == 2:
        _, thr_img = cv.threshold(blur_img, thr_val, 255, cv.THRESH_BINARY)
    elif mode == 3:
        _, thr_img = cv.threshold(blur_img, thr_val, 255, cv.THRESH_BINARY_INV)
    elif mode == 4:
        thr_img = cv.adaptiveThreshold(blur_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,2*thr_val+3,mean_sub) 
    else:
        thr_img = cv.adaptiveThreshold(blur_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,2*thr_val+3,mean_sub)

    return thr_img


# [[center, (w,h), rot, center_m, cnt]]
def find_cnt(thr_img, area=(1, None), perimeter=(1, None), poly=None):
    retval = []
    
    # find contours 
    cnts, _ = cv.findContours(thr_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # list for storing names of shapes 
    for cnt in cnts:
        
        # perimeter filter
        p = cv.arcLength(cnt,True)
        if any([perimeter[1] and p > perimeter[1], p < perimeter[0]]):
            continue
        
        # area filter
        a = cv.contourArea(cnt)
        if any([area[1] and a > area[1], a < area[0]]):
            continue
        
        # check for poly
        if poly:
            approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
            if len(approx) != poly:
                continue
        
        # min rect
        rect = cv.minAreaRect(cnt)
        
        # Calculate the center coordinates and rot angle
        center = (int(rect[0][0]), int(rect[0][1]))
        
        # rotation
        rot = rect[2]
        
        # center_m
        M = cv.moments(cnt)
        center_m = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        if poly:
            cnt = approx
        
        retval.append([center, rect[1], rot, center_m, cnt])
        
    return retval


def find_circle(thr_img, radius=(10,30), dp=1.5, prm=(300,0.8)):
    retval = []    
    
    rows = thr_img.shape[0]
    circles = cv.HoughCircles(thr_img, cv.HOUGH_GRADIENT_ALT, 1, minDist=rows / 50,
                               param1=prm[0], param2=prm[1],
                               minRadius=radius[0], maxRadius=radius[1])
    
    if circles is not None:
        retval = np.uint16(np.around(circles))[0, :]
    
    return retval

# find aruco and its pose
"""
pre define dictionary 
    DICT_4X4_50,
    DICT_4X4_100,
    DICT_4X4_250,
    DICT_4X4_1000,
    DICT_5X5_50,
    DICT_5X5_100,
    DICT_5X5_250,
    DICT_5X5_1000,
    DICT_6X6_50,
    DICT_6X6_100,
    DICT_6X6_250,
    DICT_6X6_1000,
    DICT_7X7_50,
    DICT_7X7_100,
    DICT_7X7_250,
    DICT_7X7_1000,
    DICT_ARUCO_ORIGINAL,
    DICT_APRILTAG_16h5,
    DICT_APRILTAG_25h9,
    DICT_APRILTAG_36h10,
    DICT_APRILTAG_36h11
    [("DICT_4X4_50", 0), ("DICT_4X4_100", 1), ("DICT_4X4_250", 2), ("DICT_4X4_1000", 3), ("DICT_5X5_50", 4), ("DICT_5X5_100", 5), ("DICT_5X5_250", 6), ("DICT_5X5_1000", 7), ("DICT_6X6_50", 8), ("DICT_6X6_100", 9), ("DICT_6X6_250", 10), ("DICT_6X6_1000", 11), ("DICT_7X7_50", 12), ("DICT_7X7_100", 13), ("DICT_7X7_250", 14), ("DICT_7X7_1000", 15), ("DICT_ARUCO_ORIGINAL", 16), ("DICT_APRILTAG_16h5", 17), ("DICT_APRILTAG_25h9", 18), ("DICT_APRILTAG_36h10", 19), ("DICT_APRILTAG_36h11", 20)]
refine method
    CORNER_REFINE_NONE,
    CORNER_REFINE_SUBPIX,
    CORNER_REFINE_CONTOUR,
    CORNER_REFINE_APRILTAG
    [("CORNER_REFINE_NONE", 0), ("CORNER_REFINE_SUBPIX", 1), ("CORNER_REFINE_CONTOUR", 2), ("CORNER_REFINE_APRILTAG", 3)]

"""
def find_aruco(img, camera_matrix, dist_coeffs, dictionary="DICT_6X6_250", marker_length=10, refine="CORNER_REFINE_APRILTAG", subpix=False, coordinate="cw"):
    retval = []

    # pose
    board = Aruco(dictionary=dictionary, refine=refine, subpix=subpix, marker_length=marker_length)
    rvecs, tvecs, aruco_corner, aruco_id, img_gray = board.pose(img, camera_matrix, dist_coeffs, coordinate)

    # empty
    if aruco_id is None:
        return retval

    return [[id, corner, rvec, tvec] for id, corner, rvec, tvec in zip(aruco_id, aruco_corner, rvecs, tvecs)]


#[[center(x,y), (major_axis, minor_axis), rot], ...]    
def edge_drawing(rgb_img, min_path_length=50, min_line_length = 10, nfa_validation = True, sigma=1, gradient_threshold_value=20, pf_mode = False, axes=(), ratio=(0,1)):
        retval = [] 

        # init
        e_d = cv.ximgproc.createEdgeDrawing()

        EDParams = cv.ximgproc_EdgeDrawing_Params()
        EDParams.MinPathLength = min_path_length     # try changing this value between 5 to 1000
        EDParams.PFmode = pf_mode         # defaut value try to swich it to True
        EDParams.MinLineLength = min_line_length     # try changing this value between 5 to 100
        EDParams.NFAValidation = nfa_validation   # defaut value try to swich it to False
        EDParams.Sigma = sigma # gaussian blur
        EDParams.GradientThresholdValue = gradient_threshold_value # gradient
        
        # set parameters
        e_d.setParams(EDParams)

        # gray image
        gray_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)
        
        # detect edges
        edges = e_d.detectEdges(gray_img)
        #segments =  e_d.getSegments()
        #lines = e_d.detectLines()
        elps = e_d.detectEllipses()
    
        if elps is None:
            return retval

        # format
        #elps = np.uint16(np.around(elps))
        for elp in elps:
            # axes
            major_axis = int(elp[0][2]+elp[0][3])
            minor_axis = int(elp[0][2]+elp[0][4])

            # center 
            center = (int(elp[0][0]), int(elp[0][1]))

            # angle
            angle = round(elp[0][5]%360, 2)

            # filter axes
            if axes and len(axes) == 2:
                if any([major_axis > max(axes), minor_axis < min(axes)]):
                    continue

            # filter ratio
            if ratio and len(ratio) == 2:
                r = minor_axis / major_axis
                if any([r > max(ratio), r < min(ratio)]):
                    continue
         
            # add to return
            retval.append([center, (major_axis, minor_axis), angle])
            
        return retval


def roi_mask(img, roi=[], inv=False):
    if len(roi) < 3:
        masked_img = img.copy()
    else:
        # Change data type
        roi = np.array(roi, dtype=np.int32)

        # Create a binary mask with the same shape as the image
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)

        # Create a polygon and fill it with white (1)
        cv.fillPoly(mask, [roi], 1)

        if inv:
            # Invert the mask
            mask = 1 - mask

        # Apply the mask to the image
        masked_img = cv.bitwise_and(img, img, mask=mask)
    
    return masked_img


def color_mask(rgb_img, low_hsv, high_hsv, inv=False):
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv_img, low_hsv, high_hsv)
    
    if inv:
        mask = cv.bitwise_not(mask)
    
    # Apply the mask to the image
    masked_img = cv.bitwise_and(hsv_img, hsv_img, mask=mask)
    
    # Convert the final masked image back to RGB
    rgb_img = cv.cvtColor(masked_img, cv.COLOR_HSV2RGB)
    
    return rgb_img


def intensity(img, alpha, beta):
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)


"""
pose
"""
def pose_3_point(depth_frame, depth_int, tmp_pxls, center, dim, rot, camera):
    valid = 0
    # rotation matrix
    _cos = np.cos(np.radians(rot))
    _sin = np.sin(np.radians(rot))
    screen_x = np.array([_cos , _sin])
    screen_y = np.array([-_sin , _cos])

    # tmp to pxl: pick only the first 3
    pxls = np.array([center + 0.5 * dim[0] * tmp[0] * screen_x + 0.5 * dim[1] * tmp[1] * screen_y for tmp in tmp_pxls[0:3]])

    #
    tmp_pxl_21 = tmp_pxls[1] - tmp_pxls[0]
    tmp_pxl_31 = tmp_pxls[2] - tmp_pxls[0]

    # xyz
    xyzs = [camera.xyz(pxl, depth_frame, depth_int)[0] for pxl in pxls]
    try:
        #project center and center + dx and center +dy vectors onto 3d space and onto the plane
        g1 = (tmp_pxl_21[1] *tmp_pxl_31[0] - tmp_pxl_21[0]* tmp_pxl_31[1])
        g2 = -(tmp_pxls[0][1]* tmp_pxl_31[0]- tmp_pxls[0][0]* tmp_pxl_31[1])
        g3 = -(-tmp_pxls[0][1]* tmp_pxl_21[0] + tmp_pxls[0][0]* tmp_pxl_21[1])
        center_3d = xyzs[0]  + g2*(xyzs[1]-xyzs[0])/ g1  + g3*(xyzs[2]-xyzs[0]) /g1

        # X
        X = (xyzs[1]-xyzs[0])*(-tmp_pxl_31[1])/g1 + (xyzs[2]-xyzs[0])*(tmp_pxl_21[1])/g1
        X = X/np.linalg.norm(X)
        
        # Z
        #Z = np.cross(xyzs[1] - xyzs[0], xyzs[2] - xyzs[0])
        Z = np.cross(xyzs[2] - xyzs[0], xyzs[1] - xyzs[0])
        Z = Z / np.linalg.norm(Z)

        if Z[2] <= 0: #  z is always positive
            Z = -Z
    
        # Y
        Y = np.cross(Z,X)

        valid = 1
    except Exception as e:
        X = np.zeros(3)
        Y = np.zeros(3)
        Z = np.zeros(3)
        center_3d = np.zeros(3)

    return valid, center_3d, X, Y , Z, pxls


def hex_to_hsv(hex_color):
    # Remove '#' from the hex color string
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Normalize RGB values to the range [0, 1]
    normalized_rgb = tuple(value / 255.0 for value in rgb_color)
    
    # Convert RGB to HSV
    hsv_color = colorsys.rgb_to_hsv(*normalized_rgb)
    
    # adjust
    ratio = [179, 255, 255]
    hsv_adj = np.array([int(hsv_color[i] * ratio[i]) for i in range(3)])

    return hsv_adj


class poly_select(object):
    """docstring for poly_select"""
    def __init__(self, widget):
        super(poly_select, self).__init__()
        self.widget = widget
        self.vert = []

    def onselect(self, vert):
        self.vert = [[round(v[0],2), round(v[1],2)]for v in vert]
        self.widget.value = str(self.vert)
