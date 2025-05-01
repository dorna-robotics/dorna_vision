import numpy as np
import cv2

def inside(p, limit, inv=0, **kwargs):
    retval = True
    
    # list format
    if isinstance(limit, list):
        for i in range(min(len(p), len(limit))):
            if isinstance(limit[i], list) and len(limit[i]) == 2:
                if not inv and not(min(limit[i]) <= p[i] <= max(limit[i])):
                    retval = False
                    break
                elif inv and (min(limit[i]) <= p[i] <= max(limit[i])):
                    retval = False
                    break

    return retval


def get_nested(d, keys, default=None):
    """
    Recursively get value from nested dictionary.

    Args:
        d (dict): The dictionary to search.
        keys (list or tuple): Sequence of keys (e.g., ["hasan", "hossein"]).
        default: Value to return if path doesn't exist.

    Returns:
        The found value or default.
    """
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


class Valid(object):
    def bb(self, bb, area=[], aspect_ratio=[], inv=0, **kwargs):
        retval = True
        try:
            # area and aspect ratio
            corners = np.array(bb)
            x, y = corners[:, 0], corners[:, 1]
            area_bb = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            aspect_ratio_bb = np.min(sides) / np.max(sides)

            retval = inside([area_bb, aspect_ratio_bb], [area, aspect_ratio], inv=inv)
        except:
            pass
        return retval

    def xyz(self, xyz, x=None, y=None, z=None, inv=0, **kwargs):
        return inside(xyz, [x, y, z], inv=inv)
    

    def center(self, pxl, width=None, height=None, inv=0, **kwargs):
        return inside(pxl, [width, height], inv=inv)
    

    def rvec(self, rvec, rvec_base, x_angle=None, y_angle=None, z_angle=None, inv=0, **kwargs):
        keep = True
        try:
            limits = {"x": x_angle, "y": y_angle, "z": z_angle}

            # Convert base rvec to rotation matrix
            R_base, _ = cv2.Rodrigues(np.radians(np.array(rvec_base).reshape(3, 1)))
            x_base, y_base, z_base = R_base[:, 0], R_base[:, 1], R_base[:, 2]

            axis_map_base = {"x": x_base, "y": y_base, "z": z_base}

            rvec_test = np.radians(np.array(rvec).reshape(3, 1))
            R_test, _ = cv2.Rodrigues(rvec_test)
            x_test, y_test, z_test = R_test[:, 0], R_test[:, 1], R_test[:, 2]
            axis_map_test = {"x": x_test, "y": y_test, "z": z_test}

            
            for axis in ["x", "y", "z"]:
                lim = limits.get(axis)
                if lim is None:
                    continue

                min_deg, max_deg = lim
                dot = np.clip(np.dot(axis_map_base[axis], axis_map_test[axis]), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))

                if not inv:
                    if not (min_deg <= angle <= max_deg):
                        keep = False
                        break
                else:
                    if (min_deg <= angle <= max_deg):
                        keep = False
                        break
        except:
            pass
        return keep

    def tvec(self, tvec, x=None, y=None, z=None, inv=0, **kwargs):
        return self.xyz(tvec, x, y, z, inv)


class Limit(object):
    def bb(self, detections, area=None, aspect_ratio=None, inv=0, **kwargs):
        output = []
        for d in detections:
            if Valid().bb(d["corners"], area=area, aspect_ratio=aspect_ratio, inv=inv):
                output.append(d)
        return output


    def xyz(self, detections, key=["xyz"], x=None, y=None, z=None, inv=0, **kwargs):
        output = []
        for d in detections:
            if Valid().xyz(get_nested(d, key), x, y, z, inv=inv):
                output.append(d)
        return output


    def center(self, detections, key=["center"], width=None, height=None, inv=0, **kwargs):
        output = []
        for d in detections:
            if Valid().pixel(get_nested(d, key), width, height, inv=inv):
                output.append(d)
        return output


    def rvec(self, detections, rvec_base, x_angle=None, y_angle=None, z_angle=None, inv=0):
        result = []
        for d in detections:
            if Valid().rvec(d["rvec"], rvec_base=rvec_base, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle, inv=inv):
                result.append(d)
        return result


    def tvec(self, detections, x=None, y=None, z=None, inv=0, **kwargs):
        return self.xyz(detections, x=x, y=y, z=z, inv=inv)
