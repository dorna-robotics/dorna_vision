import cv2 as cv
import numpy as np
import colorsys

def color_mask(bgr_img, low_hsv=[0, 0, 0], high_hsv=[255, 255, 255], inv=False):
    if low_hsv in [[0, 0, 0], []] and high_hsv in [[255, 255, 255], []] and not inv:
        return bgr_img
    
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_img, np.array(low_hsv), np.array(high_hsv))
    
    if inv:
        mask = cv.bitwise_not(mask)
    
    # Apply the mask to the image
    masked_img = cv.bitwise_and(hsv_img, hsv_img, mask=mask)
    
    # Convert the final masked image back to BGR
    bgr_img = cv.cvtColor(masked_img, cv.COLOR_HSV2BGR)
    
    return bgr_img


def intensity(img, a=1.0, b=0):
    if a == 1.0 and b == 0:
        return img
    return np.clip(a * img + b, 0, 255).astype(np.uint8)


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

class ROI(object):

    """docstring for Crop"""
    def __init__(self, img, corners=[], inv=False, crop=False):
        """
        Parameters
        ----------
        img : numpy array
            Original image
        corners : list of points
            Region of interest
        inv : bool
            Invert the mask
        crop : bool
            Crop the image to the region of interest

        Attributes
        ----------
        img : numpy array
            Cropped image
        corners : list of points
            Region of interest
        masked_img : numpy array
            Masked image
        x : int
            x-coordinate of the top-left corner of the bounding box
        y : int
            y-coordinate of the top-left corner of the bounding box
        w : int
            Width of the bounding box
        h : int
            Height of the bounding box
        """
        
        super(ROI, self).__init__()        
        
        # initialize
        self.roi = np.array(corners, dtype=np.int32)
        self.x = 0
        self.y = 0
        self.h, self.w = img.shape[0:2]
        
        if len(self.roi) < 3:
            self.img = img 
        else:           
            # Create a binary mask with the same shape as the image
            mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)

            # Create a polygon and fill it with white (1)
            cv.fillPoly(mask, [self.roi], 1)

            if inv:
                # Invert the mask
                mask = 1 - mask

            # Apply the mask to the image
            masked_img = cv.bitwise_and(img, img, mask=mask) 
            
            if crop:
                # contour
                self.cnt = np.array(self.roi, dtype=np.int32).reshape((-1, 1, 2))

                # Find the bounding box
                self.x, self.y, self.w, self.h = cv.boundingRect(self.cnt)
            

            # cropped image
            self.img = masked_img[self.y:self.y+self.h, self.x:self.x+self.w].copy()

    
    def pxl_to_orig(self, pxl):
        return [int(pxl[0] + self.x), int(pxl[1] + self.y)]
