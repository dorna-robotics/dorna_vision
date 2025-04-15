from dorna_vision.util import *


class Limit:
    def __init__(self, **kwargs):
        pass


    def bb(self, detections, area_range, aspect_ratio_range, inv=0, **kwargs):
        area_min = min(area_range)
        area_max = max(area_range)
        aspect_ratio_min = min(aspect_ratio_range)
        aspect_ratio_max = max(aspect_ratio_range)
        
        for d in detections:
            # area
            corners = np.array(d["corners"])
            x = corners[:, 0]
            y = corners[:, 1]                
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            # ratio
            sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            aspect_ratio = np.min(sides) / np.max(sides)

            # inside
            if not inside((area, aspect_ratio), [(area_min, area_max), (aspect_ratio_min, aspect_ratio_max)], inv=inv):
                detections.remove(d)
        return detections


    def xyz(self, detections, x_range, y_range, z_range, inv=0, **kwargs):
        x_min = min(x_range)
        x_max = max(x_range)
        y_min = min(y_range)
        y_max = max(y_range)
        z_min = min(z_range)
        z_max = max(z_range)
        
        for d in detections:
            if not inside(d["xyz"], [(x_min, x_max), (y_min, y_max), (z_min, z_max)], inv=inv):
                detections.remove(d)
        return detections


    def pixel(self, detections, width_range, height_range, inv=0, **kwargs):
        width_min = min(width_range)
        width_max = max(width_range)
        height_min = min(height_range)
        height_max = max(height_range)
        
        for d in detections:
            if not inside(d["center"], [(width_min, width_max), (height_min, height_max)], inv=inv):
                detections.remove(d)
        return detections