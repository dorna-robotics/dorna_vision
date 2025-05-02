import cv2 as cv
import numpy as np

def draw_aruco(img, ids, corners, rvecs, tvecs, camera_matrix, dist_coeffs, length=20, thickness=1):
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
   
    
def draw_cnt(cnt_img, draws, axis=True, label=False, length=20, color=(0, 255, 0), thickness=1):
    
    # [[center, (w,h), rot, center_m, cnt]]
    for draw in draws:
        # Draw the cnt
        cv.drawContours(cnt_img, [draw[4]], 0, color, thickness)
        
        if axis:
            # center axes        
            draw_2d_axis(cnt_img, draw[0], draw[2], label, length)
        


def draw_poly(img, vertices, color=(0, 255, 0), thickness=1):
    cv.polylines(img, [np.array(vertices, np.int32)], isClosed=True, color=color, thickness=thickness)

    
def draw_ellipse(img, elps, axis=True, label=False, color=(0, 255, 0), length=20, thickness=2):
    
    for elp in elps:
        if axis:
            # ellipse center and axes
            draw_2d_axis(img, elp[0], elp[2], label, length)
        
        # ellipse
        cv.ellipse(img, elp[0], elp[1], elp[2] ,0, 360, color, thickness, cv.LINE_AA)


def draw_circle(img, circles, color=(255, 0, 255), thickness=1):    
    for circle in circles:
        # circle center
        cv.circle(img, (circle[0], circle[1]), 1, color, thickness)
        # circle outline
        cv.circle(img, (circle[0], circle[1]), circle[2], color, thickness)


def draw_point(img, pxl, label, radius=3, color=(153, 255, 255), thickness=3, font_scale=0.5, text_offset=(5, -5)):
    # Draw circle
    center = (int(pxl[0]), int(pxl[1]))
    cv.circle(img, center, radius, color, thickness)

    # Draw label text near the point
    text_pos = (center[0] + text_offset[0], center[1] + text_offset[1])
    cv.putText(img, str(label), text_pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv.LINE_AA)


# draw oriented bounding box
def _draw_obb(img, id, center, wh, rot, xyz, color= (255,0,255), thickness=1):
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
def draw_obb(img, id, center, corners, color= (255,0,255), thickness=1):
    top_left, top_right, bottom_right, bottom_left = corners

    cv.line(img, top_left, top_right, color, thickness)
    cv.line(img, top_right, bottom_right, color, thickness)
    cv.line(img, bottom_right, bottom_left, color, thickness)
    cv.line(img, bottom_left, top_left, color, thickness)

    # Draw the center coordinates on the image
    cv.putText(img, f"id={id}", center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness, cv.LINE_AA)


def draw_corners(img, cls, conf, corners, color= (0,255,0), thickness=1, label=True):
    # Draw the rotated rectangle
    converted_corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(img, [converted_corners], isClosed=True, color=color, thickness=thickness)

    if label:
        # Draw a green rectangle as the background
        x, y = corners[0]  # Top-left corner of the text
        text = "cls:" + cls + ", conf:" + str(f"{conf:.2g}")
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6

        # Get the size of the text to calculate the background
        (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness=thickness)

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



def draw_3d_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length=10, thickness=2, draw=True):
    img = cv.drawFrameAxes(img, camera_matrix,
        dist_coeffs, np.radians(rvec).astype(np.float32).reshape(3, 1),
        np.array(tvec, dtype=np.float32).reshape(3, 1), 
        length)