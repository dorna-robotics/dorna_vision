import cv2 as cv
import numpy as np

def draw_aruco(img, ids, corners, rvecs, tvecs, camera_matrix, dist_coeffs, length=20, thickness=2):
    # draw
    if len(ids):
        cv.aruco.drawDetectedMarkers(img, corners, ids)

        # draw
        for i in range(len(ids)):
            # Draw axes
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], length, thickness)


def draw_2d_axis(img, center, angle, label=False, length=20, thickness=2):
        
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
   
    
def draw_cnt(cnt_img, draws, axis=True, label=False, length=20, color=(0, 255, 0), thickness=2):
    
    # [[center, (w,h), rot, center_m, cnt]]
    for draw in draws:
        # Draw the cnt
        cv.drawContours(cnt_img, [draw[4]], 0, color, thickness)
        
        if axis:
            # center axes        
            draw_2d_axis(cnt_img, draw[0], draw[2], label, length)
        


def draw_poly(img, vertices, color=(0, 255, 0), thickness=2):
    cv.polylines(img, [np.array(vertices, np.int32)], isClosed=True, color=color, thickness=thickness)

    
def draw_ellipse(img, elps, axis=True, label=False, color=(0, 255, 0), length=20, thickness=2):
    
    for elp in elps:
        if axis:
            # ellipse center and axes
            draw_2d_axis(img, elp[0], elp[2], label, length)
        
        # ellipse
        cv.ellipse(img, elp[0], elp[1], elp[2] ,0, 360, color, thickness, cv.LINE_AA)


def draw_circle(img, circles, color=(255, 0, 255), thickness=2):    
    for circle in circles:
        # circle center
        cv.circle(img, (circle[0], circle[1]), 1, color, thickness)
        # circle outline
        cv.circle(img, (circle[0], circle[1]), circle[2], color, thickness)


def draw_point(img, pxl, radius=1, color=(153,255,255), thickness=3):
    cv.circle(img, (int(pxl[0]), int(pxl[1])), radius, color, thickness)


# draw oriented bounding box
def _draw_obb(img, id, center, wh, rot, xyz, color= (255,0,255), thickness=2):
    # Create the rotated rectangle
    rect = (center, wh, rot)

    # Get the 4 points of the rotated rectangle
    box = cv.boxPoints(rect)
    box = np.int0(box)

    # Draw the rotated rectangle
    cv.polylines(img, [box], isClosed=True, color=color, thickness=thickness)

    # put label
    # Draw the center coordinates on the image
    cv.putText(img, f"id={id}", center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness, cv.LINE_AA)

# draw oriented bounding box
def draw_obb(img, id, center, corners, color= (255,0,255), thickness=2):
    top_left, top_right, bottom_right, bottom_left = corners

    cv.line(img, top_left, top_right, color, thickness)
    cv.line(img, top_right, bottom_right, color, thickness)
    cv.line(img, bottom_right, bottom_left, color, thickness)
    cv.line(img, bottom_left, top_left, color, thickness)

    # Draw the center coordinates on the image
    cv.putText(img, f"id={id}", center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness, cv.LINE_AA)


def draw_corners(img, cls, conf, corners, color= (0,255,0)):
    # Draw the rotated rectangle
    converted_corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(img, [converted_corners], isClosed=True, color=color, thickness=2)

    # Draw the center coordinates on the image
    #cv.putText(img, "cls:"+cls+", conf:"+str(f"{conf:.2g}"), corners[0], cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)
    # Draw a green rectangle as the background
    x, y = corners[0]  # Top-left corner of the text
    text = "cls:" + cls + ", conf:" + str(f"{conf:.2g}")
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Get the size of the text to calculate the background
    (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)

    # Define rectangle position and size (background)
    background_rect = (x, y - text_height - 5, text_width + 10, text_height + 5)

    # Draw the green background rectangle
    cv.rectangle(img, (background_rect[0], background_rect[1]), 
                (background_rect[0] + background_rect[2], background_rect[1] + background_rect[3]), 
                color, -1)  # Green color, filled rectangle

    # Draw the text in white on top of the background
    cv.putText(img, text, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)  # black text


"""
draw 3d_axis
"""
def draw_3d_axis(img, center, X, Y, Z, camera_matrix, dist_coeffs, length=10, thickness=2, draw=True):
    #Perform projection
    p_list = np.array([center, center+ X*length, center+Y*length,center+Z*length])

    res_list, _ =  cv.projectPoints(p_list, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)

    try:    
        pc = (int(res_list[0][0][0]), int(res_list[0][0][1]))
        px = (int(res_list[1][0][0]), int(res_list[1][0][1]))
        py = (int(res_list[2][0][0]), int(res_list[2][0][1]))
        pz = (int(res_list[3][0][0]), int(res_list[3][0][1]))
        if draw:
            # draw
            cv.line(img, pz, pc, (255, 0, 0), thickness=thickness)
            cv.line(img, px, pc, (0, 0, 255), thickness=thickness)
            cv.line(img, py, pc, (0, 255, 0), thickness=thickness)

        return [pc, px, py, pz]
    except:
        pass

    return None