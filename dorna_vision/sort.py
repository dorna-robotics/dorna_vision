import heapq


class Sort:
    def conf(self, detections, increasing=False, max_det=1000, **kwargs):
        # Sort by .conf and slice the top N
        return sorted(
            detections,
            key=lambda d: d.conf,
            reverse=not increasing
        )[:max_det]


    def pixel(self, detections, pixel, increasing=True, max_det=1000, **kwargs):
        x0, y0 = pixel
        dist_fn = lambda d: (d.center[0] - x0)**2 + (d.center[1] - y0)**2

        if increasing:
            return heapq.nsmallest(max_det, detections, key=dist_fn)
        else:
            return heapq.nlargest(max_det, detections, key=dist_fn)


    def xyz(self, detections, xyz, increasing=True, max_det=1000, **kwargs):
        x0, y0, z0 = xyz
        dist_fn = lambda d: (d.xyz[0] - x0)**2 + (d.xyz[1] - y0)**2 + (d.xyz[2] - z0)**2

        if increasing:
            return heapq.nsmallest(max_det, detections, key=dist_fn)
        else:
            return heapq.nlargest(max_det, detections, key=dist_fn)


    def tvec(self, detections, tvec, increasing=True, max_det=1000, **kwargs):
        x0, y0, z0 = tvec
        dist_fn = lambda d: (d.tvec[0] - x0)**2 + (d.tvec[1] - y0)**2 + (d.tvec[2] - z0)**2

        if increasing:
            return heapq.nsmallest(max_det, detections, key=dist_fn)
        else:
            return heapq.nlargest(max_det, detections, key=dist_fn)


    def robot_distance(self, detections, xyz, increasing=True, max_det=1000, **kwargs):
        pass

