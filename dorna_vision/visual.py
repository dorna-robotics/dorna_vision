import cv2 as cv
import numpy as np
import colorsys


def rotate_and_flip(img, rotate=0, flip_h=False, flip_v=False):    
    # Step 1: Rotation
    if rotate not in [0, 90, 180, 270]:
        return img
    
    if rotate == 90:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        img = cv.rotate(img, cv.ROTATE_180)
    elif rotate == 270:
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)  # Equivalent to 270° clockwise
    
    # Step 2: Horizontal flip (mirror over y-axis)
    if flip_h:
        img = cv.flip(img, 1)
    
    # Step 3: Vertical flip (mirror over x-axis)
    if flip_v:
        img = cv.flip(img, 0)
    
    return img

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
    def __init__(self, img, corners=[], inv=False, crop=False, offset=0, color=(128, 128, 128)):
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
        margin : int
            positive, goes around. negative, goes inside 

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
        #ratio = self.w / self.h


        # adjust roi
        if len(self.roi) == 0:
            self.img = img.copy()
        elif len(self.roi) == 1:
            x, y = self.roi[0]
            self.roi = np.array([[x,y], [x+1, y], [x+1, y+1], [x, y+1]], dtype=np.int32)
        elif len(self.roi) == 2:
            x1, y1 = self.roi[0]
            x2, y2 = self.roi[1]
            self.roi = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)

        if len(self.roi) > 0:         
            # Create a binary mask with the same shape as the image
            mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)

            # adjust the polygon based on the offset
            if offset:
                self.roi = self.offset_polygon(list(self.roi), offset)
            
            # bound the roi
            self.roi = np.array([[min(self.w-1, max(0, v[0])), min(self.h-1, max(0, v[1]))] for v in self.roi], dtype=np.int32)

            # Create a polygon and fill it with white (1)
            cv.fillPoly(mask, [self.roi], 1)

            if inv:
                # Invert the mask
                mask = 1 - mask

            # Apply the mask to the image
            #masked_img = cv.bitwise_and(img, img, mask=mask) 
            # Create a gray background image with the same shape and type as the original image.
            gray_background = np.full(img.shape, color, dtype=img.dtype)

            # Convert your single channel mask to a boolean mask.
            mask_bool = mask.astype(bool)

            # For a multi-channel image, expand the mask's dimensions so it works across all channels.
            # Use np.where to select pixels: from img where mask is True, and from gray_background otherwise.
            masked_img = np.where(mask_bool[:, :, None], img, gray_background)

            if crop:
                # contour
                self.cnt = np.array(self.roi, dtype=np.int32).reshape((-1, 1, 2))

                # Find the bounding box
                self.x, self.y, self.w, self.h = cv.boundingRect(self.cnt)

            # cropped image
            self.img = masked_img[self.y:self.y+self.h, self.x:self.x+self.w].copy()

            """
            if crop:
                self.img = self.adjust_aspect_ratio(self.img, ratio)
            """

    def offset_polygon(self, polygon, offset_px):
        """
        Offset a polygon by a fixed pixel distance around its centroid.
        :param polygon: List of [x, y] points.
        :param offset_px: Positive to expand, negative to shrink.
        :return: Offset polygon as list of points.
        """
        polygon_np = np.array(polygon, dtype=np.float32)
        
        # Compute centroid
        moments = cv.moments(polygon_np.astype(np.int32))
        if moments["m00"] == 0:
            return polygon  # Avoid division by zero
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centroid = np.array([cx, cy], dtype=np.float32)

        # Compute vectors from centroid to polygon points
        vectors = polygon_np - centroid
        
        # Compute distances and directions
        distances = np.linalg.norm(vectors, axis=1, keepdims=True)
        directions = vectors / np.where(distances > 0, distances, 1e-8)  # Avoid division by zero
        
        # Apply offset (expand/shrink along radial direction)
        new_vectors = vectors + directions * offset_px
        new_polygon = centroid + new_vectors
        
        return new_polygon.astype(np.int32)

    def adjust_aspect_ratio(self, image, target_ratio):
        """
        Adjust image to a target aspect ratio by adding black margins.
        
        Args:
            image: Input image (numpy array).
            target_ratio: Target width/height ratio (e.g., 16/9 for 16:9).
            
        Returns:
            Padded image with exact target aspect ratio.
        """
        h, w = image.shape[:2]
        current_ratio = w / h

        if current_ratio < target_ratio:
            # Image is taller: add right margin
            new_width = int(round(h * target_ratio))
            new_height = h
            #pad_x = new_width - w
            #pad_y = 0
        else:
            # Image is wider: add bottom margin
            new_height = int(round(w / target_ratio))
            new_width = w
            #pad_x = 0
            #pad_y = new_height - h

        # Create black canvas
        if len(image.shape) == 3:
            canvas = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((new_height, new_width), dtype=image.dtype)
            
        # Place original image in top-left corner
        canvas[:h, :w] = image        
        return canvas


    def pxl_to_orig(self, pxl):
        return [int(pxl[0] + self.x), int(pxl[1] + self.y)]


    def pxl_orig_to_roi(self, pxl):
        return [int(pxl[0] - self.x), int(pxl[1] - self.y)]
