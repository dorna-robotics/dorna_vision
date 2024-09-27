import cv2 as cv
import numpy as np
import colorsys
from dorna_vision.board import Aruco

# bgr_img -> binary
def binary_thr(bgr_img, type=0, inv=True, blur=3, thr=127, mean_sub=2, **kwargs):    
    # gray image
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    # Apply GaussianBlur to further reduce noise
    #blur_img = cv.bilateralFilter(gray_img,90,75,75)
    #blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    blur_img = cv.blur(gray_img,(blur,blur))
    
    # set the mode
    mode = 2*type + inv
    
    if mode < 1:
        _, thr_img = cv.threshold(blur_img, thr, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    elif mode == 1:
        _, thr_img = cv.threshold(blur_img, thr, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    elif mode == 2:
        _, thr_img = cv.threshold(blur_img, thr, 255, cv.THRESH_BINARY)
    elif mode == 3:
        _, thr_img = cv.threshold(blur_img, thr, 255, cv.THRESH_BINARY_INV)
    elif mode == 4:
        thr_img = cv.adaptiveThreshold(blur_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,2*thr+3,mean_sub) 
    else:
        thr_img = cv.adaptiveThreshold(blur_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,2*thr+3,mean_sub)

    return thr_img

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


def get_obb_corners(center, wh, rot):
    # Center of the ellipse
    cx, cy = center

    # Half-width and half-height
    a, b = wh[0] / 2, wh[1] / 2

    # Rotation angle in radians
    theta = np.radians(rot)

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Initial corners relative to the center
    corners = np.array([
        [a, b],
        [a, -b],
        [-a, -b],
        [-a, b]
    ])

    # Rotate and translate corners
    rotated_corners = np.dot(corners, R.T) + np.array([cx, cy])
    return [[int(corner[0]), int(corner[1])] for corner in rotated_corners]





