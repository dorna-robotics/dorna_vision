import cv2 as cv
import numpy as np
from matplotlib.patches import Ellipse


def draw_aruco(img, aruco_data, camera_matrix, dist_coeffs, length=20, thickness=1):
    # unpack
    ids = np.array([a[0] for a in aruco_data])
    corners = [a[1] for a in aruco_data]
    rvecs = [a[2] for a in aruco_data]
    tvecs = [a[3] for a in aruco_data]
    
    # draw
    if len(ids):
        cv.aruco.drawDetectedMarkers(img, corners, ids)

        # draw
        for i in range(len(ids)):
            # Draw axes
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], length, thickness)


def draw_2d_axis(img, center, angle, label=False, length=20, thickness=1):
        
        if label:
            # Draw the center coordinates on the image
            cv.putText(img, f'c: ({center[0]}, {center[1]})'+ ", rot: {:.1f}".format(angle), center,
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness, cv.LINE_AA)

        # Calculate the rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)

        # Calculate the axis endpoints
        x_axis_end = (int(center[0] + length * np.cos(np.radians(angle))),
                      int(center[1] + length * np.sin(np.radians(angle))))

        y_axis_end = (int(center[0] - length * np.sin(np.radians(angle))),
                      int(center[1] + length * np.cos(np.radians(angle))))

        # Draw the rotated axes
        cv.line(img, center, x_axis_end, (0, 0, 255), thickness)  # Red X-axis
        cv.line(img, center, y_axis_end, (0, 255, 0), thickness)  # Green Y-axis
   
    
def draw_cnt(cnt_img, draws, axis=True, label=False, length=20, thickness=1):
    
    # [[center, (w,h), rot, center_m, cnt]]
    for draw in draws:
        # Draw the cnt
        cv.drawContours(cnt_img, [draw[4]], 0, (255, 0, 255), thickness)
        
        if axis:
            # center axes        
            draw_2d_axis(cnt_img, draw[0], draw[2], label, length)
        
        # draw center mass
        #cv.circle(cnt_img, draw[3], 1, (255, 255, 0), thickness)


def draw_poly(img, vertices, color=(255, 0, 0), thickness=1):
    cv.polylines(poly_img, [np.array(vertices, np.int32)], isClosed=True, color=color, thickness=thickness)

    
def draw_ellipse(img, elps, axis=True, label=False, length=20, thickness=1):
    
    for elp in elps:
        if axis:
            # ellipse center and axes
            draw_2d_axis(img, elp[0], elp[2], label, length)
        
        # ellipse
        cv.ellipse(img, elp[0], elp[1], elp[2] ,0, 360, (255, 0, 255), thickness, cv.LINE_AA)


def draw_circle(img, circles, color=(255, 0, 255), thickness=1):    
    for circle in circles:
        # circle center
        cv.circle(img, (circle[0], circle[1]), 1, color, thickness)
        # circle outline
        cv.circle(img, (circle[0], circle[1]), circle[2], color, thickness)


"""
draw 3d_axis
"""
def draw_3d_axis(img, center, X, Y, Z, camera_matrix, dist_coeffs, length=10, thick=1):
    #Perform projection
    p_list = np.array([center, center+ X*length, center+Y*length,center+Z*length])

    res_list, _ =  cv.projectPoints(p_list, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)

    try:    
        pc = (int(res_list[0][0][0]), int(res_list[0][0][1]))
        px = (int(res_list[1][0][0]), int(res_list[1][0][1]))
        py = (int(res_list[2][0][0]), int(res_list[2][0][1]))
        pz = (int(res_list[3][0][0]), int(res_list[3][0][1]))
        # draw
        cv.line(img, pz, pc, (255, 0, 0), thickness=thick)
        cv.line(img, px, pc, (0, 0, 255), thickness=thick)
        cv.line(img, py, pc, (0, 255, 0), thickness=thick)
    except:
        pass

def draw_point(img, pxl, radius=1, color=(153,255,255), thickness=2):
    cv.circle(img, (int(pxl[0]), int(pxl[1])), radius, color, thickness)