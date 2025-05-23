import cv2 as cv
import numpy as np
from dorna_vision.board import Aruco
import zxingcpp



def barcode(img, **kwargs):
    # format, text, corners, center
    retval = []
    try:
        if "format" in kwargs and kwargs["format"] != "Any":
            barcodes = zxingcpp.read_barcodes(img, getattr(zxingcpp.BarcodeFormat, kwargs["format"]))
        else:
            barcodes = zxingcpp.read_barcodes(img)
        
        for barcode in barcodes:
            try:
                corners = [[int(barcode.position.top_left.x), int(barcode.position.top_left.y)],
                        [int(barcode.position.top_right.x), int(barcode.position.top_right.y)],
                        [int(barcode.position.bottom_right.x), int(barcode.position.bottom_right.y)],
                        [int(barcode.position.bottom_left.x), int(barcode.position.bottom_left.y)]]

                x_coords = [pt[0] for pt in corners]
                y_coords = [pt[1] for pt in corners]
                center = [int(sum(x_coords) / len(corners)), int(sum(y_coords) / len(corners))]
                # format, text, corners, center    
                retval.append([barcode.format.name,
                                barcode.text,
                                corners,
                                center])
            except:
                pass
    except:
        pass

    return retval

# [[pxl, corners, cnt], ...]
def contour(thr_img, **kwargs):
    retval = []
    
    # find contours 
    cnts, _ = cv.findContours(thr_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # list for storing names of shapes 
    for cnt in cnts:
        # check for poly
        if "side" in kwargs and kwargs["side"] is not None:
            approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
            if len(approx) != kwargs["side"]:
                continue

        # moments
        M = cv.moments(cnt)

        # Avoid division by zero (when area is zero)
        if M["m00"] == 0:
            continue
        
        # Calculate the center coordinates and rot angle
        pxl = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

        # min rect
        rect = cv.minAreaRect(cnt)
    
        # Get the 4 corners of the rectangle
        box = cv.boxPoints(rect).astype(int)

        # Convert to list of tuples
        corners = [[point[0], point[1]] for point in box]
                
        retval.append([pxl, corners, cnt])
        
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
# [[pxl, corners, (id, rvec, tvec)], ...]
def aruco(img, camera_matrix, dist_coeffs, dictionary="DICT_6X6_250", marker_length=10, refine="CORNER_REFINE_APRILTAG", subpix=False, coordinate="ccw", **kwargs):

    retval = []

    # pose
    board = Aruco(dictionary=dictionary, refine=refine, subpix=subpix, marker_length=marker_length)
    rvecs, tvecs, aruco_corner, aruco_id, _ = board.pose(img, camera_matrix, dist_coeffs, coordinate)

    # empty
    if aruco_id is None:
        return retval

    for i in range(len(aruco_id)):
        if len(aruco_corner[i][0]) == 4:
            corners = aruco_corner[i][0].reshape((4,2))
            
            # pxl
            A1 = corners[2][1] - corners[0][1]
            B1 = corners[0][0] - corners[2][0]
            C1 = A1 * corners[0][0] + B1 * corners[0][1]
            
            # Coefficients for the second line
            A2 = corners[3][1] - corners[1][1]
            B2 = corners[1][0] - corners[3][0]
            C2 = A2 * corners[1][0] + B2 *corners[1][1]
            
            # Calculate the determinant
            det = A1 * B2 - A2 * B1
            
            if det == 0:
                continue
            
            # Calculate the intersection point
            pxl = [int((C1 * B2 - C2 * B1) / det), int((A1 * C2 - A2 * C1) / det)]

            # make corners int
            corners = [[int(c[0]), int(c[1])] for c in corners]

            retval.append([pxl, corners, [aruco_id[i], aruco_corner[i], rvecs[i], tvecs[i]]])
    return retval


#[[pxl, corners, (major_axis, minor_axis, rot)], ...]    
def ellipse(bgr_img, min_path_length=50, min_line_length = 10, nfa_validation = True, sigma=1, gradient_threshold_value=20, pf_mode = False, **kwargs):
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
        gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
        
        # detect edges
        edges = e_d.detectEdges(gray_img)
        #segments =  e_d.getSegments()
        #lines = e_d.detectLines()
        elps = e_d.detectEllipses()
    
        if elps is None:
            return retval

        # format
        for elp in elps:
            # axes
            major_axis = int(2*(elp[0][2]+elp[0][3]))
            minor_axis = int(2*(elp[0][2]+elp[0][4]))

            # center 
            center = (int(elp[0][0]), int(elp[0][1]))

            # rotation
            rot = round(elp[0][5]%360, 2)

            # rect
            rect = (center, (major_axis, minor_axis), rot)

            # Get the 4 corners of the bounding box using boxPoints
            box = cv.boxPoints(rect).astype(int)

            # Convert to list of tuples
            corners = [[point[0], point[1]] for point in box]

            # add to return
            retval.append([center, corners, rect])
            
        return retval

