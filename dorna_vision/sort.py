import numpy as np
import random

class Sort:
    @staticmethod
    def _get(d, key, default=0):
        """Pull key from dict or attribute from object."""
        if isinstance(d, dict):
            return d.get(key, default)
        return getattr(d, key, default)

    def shuffle(self, detections, max_det=-1, **kwargs):
        if max_det <= 0:
            max_det = len(detections)
        random.shuffle(detections)
        return detections[0:max_det]

    def conf(self, detections, ascending=False, max_det=-1, key="conf",  **kwargs):
        """
        Sort by confidence score.
          ascending=False → highest confidences first (default)
          ascending=True  → lowest first
        """
        if max_det <= 0:
            max_det = len(detections)

        # build a NumPy array of confidences
        confs = np.array([self._get(d, key, 0) for d in detections])
        # argsort ascending or descending
        order = np.argsort(confs) if ascending else np.argsort(-confs)
        idx = order[:max_det]
        return [detections[i] for i in idx]

    def pixel(self, detections, pxl, ascending=True, max_det=-1, key="center", **kwargs):
        """
        Sort by squared-pixel distance to (x0,y0).
          ascending=True → closest first (default)
        """
        if max_det <= 0:
            max_det = len(detections)

        x0, y0 = pxl
        # collect centers
        centers = np.array([self._get(d, key, (0, 0)) for d in detections])
        # compute squared distances
        deltas = centers - np.array([x0, y0])
        dists = np.einsum('ij,ij->i', deltas, deltas)
        order = np.argsort(dists) if ascending else np.argsort(-dists)
        idx = order[:max_det]
        return [detections[i] for i in idx]


    def area(self, detections, ascending=False, max_det=-1, key="corners", **kwargs):
        """
        Sort by polygon area (shoelace) of the list of corner pixels in `key`.
        `key` should point to an iterable of (x,y) pairs.
        """
        if max_det <= 0:
            max_det = len(detections)

        def _compute_area(corners):
            # corners: list of [x,y] or array Nx2
            pts = np.asarray(corners)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                return 0.0
            x, y = pts[:,0], pts[:,1]
            # shoelace
            return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

        areas = np.array([_compute_area(self._get(d, key, [])) for d in detections])
        order = np.argsort(areas) if ascending else np.argsort(-areas)
        return [detections[i] for i in order[:max_det]]


    def xyz(self, detections, xyz, ascending=True, max_det=-1, key="xyz", **kwargs):
        """
        Sort by squared-3D distance to (x0,y0,z0).
          ascending=True → nearest first
        """
        if max_det <= 0:
            max_det = len(detections)

        x0, y0, z0 = xyz
        pts = np.array([self._get(d, key, (0,0,0)) for d in detections])
        deltas = pts - np.array([x0, y0, z0])
        dists = np.einsum('ij,ij->i', deltas, deltas)
        order = np.argsort(dists) if ascending else np.argsort(-dists)
        idx = order[:max_det]
        return [detections[i] for i in idx]

    def tvec(self, detections, tvec, ascending=True, max_det=-1, key="tvec", **kwargs):
        return self.xyz(detections, xyz=tvec, ascending=ascending, max_det=max_det, key=key, **kwargs)

