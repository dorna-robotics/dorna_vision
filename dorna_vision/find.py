import math
import cv2 as cv
import numpy as np
from dorna_vision.board import Aruco, Charuco
import zxingcpp


def _ransac_fit_ellipse(contour, iters=40, inlier_tol=2.0, sample=8, rng=None):
    """
    RANSAC-wrapped ellipse fit. cv.fitEllipse uses least-squares over
    every contour point, so a single noisy/spurious vertex (specular
    glints, sub-pixel artifacts, bordering objects) skews the fit. This
    samples small subsets, fits each, and scores by how many *contour*
    points lie within `inlier_tol` pixels of the resulting ellipse —
    then refits on the inlier set. The fit returned is biased toward
    the dominant elliptic structure rather than the noise tail.

    Returns ((cx, cy), (MA, ma), angle) like cv.fitEllipse, or None if
    no fit was found.
    """
    n = len(contour)
    if n < 5:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)
    if n < sample:
        # Too few points to subsample meaningfully; just use direct fit.
        try:
            return cv.fitEllipse(contour)
        except cv.error:
            return None

    rng = rng if rng is not None else np.random.default_rng(0)
    best_inliers = None
    best_count = 0
    best_fit = None

    for _ in range(int(iters)):
        idx = rng.choice(n, size=sample, replace=False)
        sub = pts[idx].reshape(-1, 1, 2)
        try:
            fit = cv.fitEllipse(sub)
        except cv.error:
            continue
        (cx, cy), (MA, ma), angle = fit
        if MA <= 0 or ma <= 0:
            continue
        a = MA / 2.0
        b = ma / 2.0
        # Distance proxy: |x'^2/a^2 + y'^2/b^2 - 1| in the ellipse's
        # local frame. Scaled by min(a,b) so the tolerance stays in
        # pixels regardless of the ellipse's size.
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        # cv.fitEllipse uses the angle convention where the major axis
        # points along (sin(angle), -cos(angle)), so the rotation that
        # brings the ellipse axis-aligned is the inverse of that.
        xp =  s * dx - c * dy
        yp =  c * dx + s * dy
        residual = np.abs((xp * xp) / (a * a) + (yp * yp) / (b * b) - 1.0)
        scaled = residual * min(a, b)
        inliers = scaled < float(inlier_tol)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_fit = fit

    if best_fit is None:
        return None
    # Refit on the inlier set for sub-pixel polish.
    if best_inliers is not None and best_count >= 5:
        inlier_pts = pts[best_inliers].reshape(-1, 1, 2)
        try:
            return cv.fitEllipse(inlier_pts)
        except cv.error:
            return best_fit
    return best_fit


def elp_fit(
    img,
    *,
    use_otsu=True,
    area_range=(300, 500000),
    aspect_ratio_tol=0.1,
    circularity_min=0.7,
    convexity_min=0.9,
    use_ransac=True,
    ransac_iters=40,
    ransac_tol=2.0,
    visualize=False,
    **kwargs
):
    vis = img.copy() if visualize else None

    # --- Convert to grayscale and denoise ---
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0.6)

    # --- Equalize histogram for better contrast ---
    gray_eq = cv.equalizeHist(gray)

    # --- Threshold (Otsu or Adaptive) ---
    if use_otsu:
        _, bw = cv.threshold(gray_eq, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        bw = cv.adaptiveThreshold(
            gray_eq, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 5
        )

    # --- Automatically invert if needed (dark object on light background) ---
    if np.mean(gray_eq[bw == 255]) > np.mean(gray_eq[bw == 0]):
        bw = cv.bitwise_not(bw)

    # --- Morphological cleanup ---
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    # --- Find contours ---
    cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    results = []

    rng = np.random.default_rng(0)
    for c in cnts:
        area = cv.contourArea(c)
        if not (area_range[0] < area < area_range[1]):
            continue
        if len(c) < 5:
            continue

        # RANSAC fit is more robust to noisy contour points (e.g. specular
        # highlights, jagged edges from imperfect thresholding) than the
        # plain least-squares fit. Falls back to direct fit on tiny
        # contours where subsampling is meaningless.
        if use_ransac:
            fit = _ransac_fit_ellipse(
                c, iters=int(ransac_iters), inlier_tol=float(ransac_tol), rng=rng,
            )
        else:
            try:
                fit = cv.fitEllipse(c)
            except cv.error:
                fit = None
        if fit is None:
            continue
        (x, y), (MA, ma), angle = fit
        if ma == 0 or MA == 0:
            continue

        aspect_ratio = min(MA, ma) / max(MA, ma)
        if aspect_ratio < (1 - aspect_ratio_tol):
            continue

        perim = cv.arcLength(c, True)
        circularity = 4 * np.pi * area / (perim * perim + 1e-6)
        if circularity < circularity_min:
            continue

        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if convexity < convexity_min:
            continue

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        bbox_corners = [tuple(map(float, pt)) for pt in box]

        results.append({
            "center": (float(x), float(y)),
            "corners": bbox_corners,
            "ellipse": ((float(x), float(y)), (float(MA), float(ma)), float(angle)),
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "convexity": convexity,
        })

        if visualize:
            cv.drawContours(vis, [c], -1, (0, 255, 255), 1)
            for i in range(4):
                p1 = tuple(map(int, box[i]))
                p2 = tuple(map(int, box[(i + 1) % 4]))
                cv.line(vis, p1, p2, (255, 255, 0), 1)
            cv.ellipse(vis, (int(x), int(y)),
                       (int(MA / 2), int(ma / 2)),
                       angle, 0, 360, (255, 0, 0), 1)
            cv.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)

    if visualize:
        cv.imshow("Detected Ellipses", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return results


def blob(
    img,
    *,
    roi=None,
    visualize=False,

    # --- Blob detector ---
    minThreshold=10,
    maxThreshold=400,
    minArea=100,
    maxArea=5000,
    minCircularity=0.30,

    # --- Ellipse refinement ---
    pad=10,
    morph_open=1,
    use_subpix=True,
    win=(3,3),

    # --- Preprocessing ---
    use_gaussian=True,
    use_clahe=False,
    clahe_clip=2.0,
    clahe_grid=(8,8),

    # --- Blob polarity ---
    blob_is_dark=True,
    **kwargs,
):
    """
    Returns list of dicts:
      {'center': (cx, cy),
       'corners': [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]}
    """

    # --- grayscale + vis ---
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        vis  = img.copy()
    else:
        gray = img.copy()
        vis  = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    H, W = gray.shape[:2]

    if use_gaussian:
        gray = cv.GaussianBlur(gray, (3,3), 0.6)
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)

    # --- ROI ---
    x_min, y_min, x_max, y_max = 0, 0, W, H
    if roi is not None:
        (x1, y1), (x2, y2) = roi
        x_min, x_max = sorted([int(round(x1)), int(round(x2))])
        y_min, y_max = sorted([int(round(y1)), int(round(y2))])
        x_min = max(0, min(W-1, x_min)); x_max = max(1, min(W, x_max))
        y_min = max(0, min(H-1, y_min)); y_max = max(1, min(H, y_max))
    gray_roi = gray[y_min:y_max, x_min:x_max]

    # --- blob detector ---
    p = cv.SimpleBlobDetector_Params()
    p.minThreshold = float(minThreshold)
    p.maxThreshold = float(maxThreshold)
    p.filterByArea = True
    p.minArea = float(minArea)
    p.maxArea = float(maxArea)
    p.filterByCircularity = True
    p.minCircularity = float(minCircularity)
    p.filterByColor = True
    p.blobColor = 0 if blob_is_dark else 255

    detector = cv.SimpleBlobDetector_create(p)
    kps = detector.detect(gray_roi)

    results = []
    if not kps:
        return results

    # --- refinement per blob ---
    for i, kp in enumerate(kps):
        bx, by = kp.pt
        bx += x_min; by += y_min
        # SimpleBlobDetector reports kp.size as the equivalent-circle
        # diameter, which underestimates the major axis of elongated
        # ellipses (e.g. a 200×80 blob → kp.size ≈ 127). Using
        # kp.size/2 + pad as the search radius would clip the contour
        # at the crop edges and the bounding rect would come out
        # smaller than the actual blob. Start with a generous radius
        # (kp.size = 2× the original) and re-expand once if the
        # threshold result still touches the ROI border.
        base_r = max(2, int(round(kp.size)) + int(pad))

        def _crop(r):
            xa = max(0, int(round(bx - r)))
            xb = min(W, int(round(bx + r)))
            ya = max(0, int(round(by - r)))
            yb = min(H, int(round(by + r)))
            return xa, xb, ya, yb

        br = base_r
        x0, x1, y0, y1 = _crop(br)

        roi_local = gray[y0:y1, x0:x1]
        _, bw = cv.threshold(roi_local, 0, 255,
                             cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if morph_open > 0:
            bw = cv.morphologyEx(bw, cv.MORPH_OPEN,
                                 np.ones((3,3), np.uint8),
                                 iterations=morph_open)

        # If the foreground touches any border of the local crop, the
        # blob extends outside our search window — re-crop with double
        # the radius and re-threshold so boundingRect captures the
        # full silhouette.
        h_l, w_l = bw.shape[:2]
        if w_l > 1 and h_l > 1:
            touches_border = (
                bw[0, :].any() or bw[-1, :].any() or
                bw[:, 0].any() or bw[:, -1].any()
            )
            if touches_border:
                br = base_r * 2
                x0, x1, y0, y1 = _crop(br)
                roi_local = gray[y0:y1, x0:x1]
                _, bw = cv.threshold(roi_local, 0, 255,
                                     cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
                if morph_open > 0:
                    bw = cv.morphologyEx(bw, cv.MORPH_OPEN,
                                         np.ones((3,3), np.uint8),
                                         iterations=morph_open)

        cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts and len(max(cnts, key=cv.contourArea)) >= 5:
            largest = max(cnts, key=cv.contourArea)
            (ex, ey), (MA, ma), angle = cv.fitEllipse(largest)
            cx, cy = ex + x0, ey + y0
            # Tight axis-aligned box around the actual blob contour, in
            # image coords. The search crop above used a `pad`-inflated
            # ROI to give the threshold/morph step room to recover the
            # full silhouette — but that's a search box, not a result
            # box, so don't return it as `corners`.
            bx_l, by_l, bw_l, bh_l = cv.boundingRect(largest)
            tx0, ty0 = bx_l + x0, by_l + y0
            tx1, ty1 = tx0 + bw_l, ty0 + bh_l
        else:
            cx, cy = bx, by
            MA = ma = 2*br; angle = 0.0
            # Fallback: contour fit failed, so the best estimate of blob
            # extent is the detector's keypoint size (diameter).
            r = max(1, int(round(kp.size/2.0)))
            tx0 = max(0, int(round(bx - r)))
            tx1 = min(W, int(round(bx + r)))
            ty0 = max(0, int(round(by - r)))
            ty1 = min(H, int(round(by + r)))

        if use_subpix:
            pt = np.array([[[cx, cy]]], np.float32)
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            sub = cv.cornerSubPix(gray, pt, win, (-1,-1), term)
            cx, cy = sub[0,0,0], sub[0,0,1]

        corners = [(tx0,ty0),(tx1,ty0),(tx1,ty1),(tx0,ty1)]
        results.append({"center": (float(cx), float(cy)), "corners": corners})

        if visualize:
            # ROI box
            cv.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 1)
            # blob circle (red)
            cv.circle(vis, (int(round(bx)), int(round(by))),
                      int(round(kp.size/2.0)), (0,0,255), 1)
            cv.circle(vis, (int(round(bx)), int(round(by))), 2, (0,0,255), -1)
            # ellipse (blue)
            cv.ellipse(vis, (int(round(cx)), int(round(cy))),
                       (max(1,int(round(MA/2.0))), max(1,int(round(ma/2.0)))),
                       angle, 0, 360, (255,0,0), 1)
            cv.circle(vis, (int(round(cx)), int(round(cy))), 2, (255,0,0), -1)
            # shift vector (orange)
            cv.line(vis, (int(round(bx)), int(round(by))),
                    (int(round(cx)), int(round(cy))), (0,200,255), 1)
            # index
            cv.putText(vis, f"{i}", (int(round(cx))+6, int(round(cy))-6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)

    if visualize:
        cv.imshow("blobs/ellipse", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return results

def mser(
    img,
    *,
    delta=5,
    min_area=60,
    max_area=14400,
    max_variation=0.25,
    min_diversity=0.2,
    blob_is_dark=True,
    nms_iou=0.5,
    **kwargs,
):
    """
    MSER (Maximally Stable Extremal Regions): finds blob-like regions that
    stay coherent across a sweep of binarization thresholds. The big win
    over `blob`/`cnt` is that there's no threshold value to tune — the
    algorithm itself is threshold-invariant — so the same parameters
    survive lighting changes that would break Otsu/SimpleBlobDetector.

    Polarity is filtered post-hoc: MSER intrinsically detects both bright
    and dark extremal regions, but `blob_is_dark` (matching find.blob's
    knob) keeps only the side the user asked for, by comparing each
    region's mean intensity to the image median.

    Returns a list shaped like find.blob():
      [{'center': (cx, cy),
        'corners': [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]}, ...]
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    detector = cv.MSER_create(
        delta=int(delta),
        min_area=int(min_area),
        max_area=int(max_area),
        max_variation=float(max_variation),
        min_diversity=float(min_diversity),
    )
    regions, bboxes = detector.detectRegions(gray)

    if len(regions) == 0:
        return []

    # MSER returns nested regions (same blob detected at multiple scales),
    # so filter both by polarity and overlap before returning.
    median_intensity = float(np.median(gray))
    raw = []
    for region, bbox in zip(regions, bboxes):
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or len(region) == 0:
            continue
        region_mean = float(gray[region[:, 1], region[:, 0]].mean())
        is_dark = region_mean < median_intensity
        if blob_is_dark and not is_dark:
            continue
        if (not blob_is_dark) and is_dark:
            continue
        cx = float(region[:, 0].mean())
        cy = float(region[:, 1].mean())
        raw.append({
            "center": (cx, cy),
            "bbox":   (int(x), int(y), int(w), int(h)),
            # Larger regions are usually preferred when nested duplicates
            # collide in NMS — area becomes the score.
            "score":  float(w * h),
        })

    if not raw:
        return []

    boxes  = [r["bbox"]  for r in raw]
    scores = [r["score"] for r in raw]
    keep   = cv.dnn.NMSBoxes(boxes, scores, 0.0, float(nms_iou))
    if len(keep) == 0:
        return []
    keep_idx = keep.flatten() if hasattr(keep, "flatten") else list(keep)

    results = []
    for i in keep_idx:
        x, y, w, h = raw[i]["bbox"]
        x0, y0 = x, y
        x1, y1 = x + w, y + h
        results.append({
            "center": raw[i]["center"],
            "corners": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        })
    return results


def barcode(img, **kwargs):
    # format, text, corners, center
    retval = []
    try:
        if "format" in kwargs and kwargs["format"] != "Any":
            barcodes = zxingcpp.read_barcodes(img, getattr(zxingcpp.BarcodeFormat, kwargs["format"]))
        else:
            barcodes = zxingcpp.read_barcodes(img)
        
        for barcode in barcodes:
            try:
                corners = [[int(barcode.position.top_left.x), int(barcode.position.top_left.y)],
                        [int(barcode.position.top_right.x), int(barcode.position.top_right.y)],
                        [int(barcode.position.bottom_right.x), int(barcode.position.bottom_right.y)],
                        [int(barcode.position.bottom_left.x), int(barcode.position.bottom_left.y)]]

                x_coords = [pt[0] for pt in corners]
                y_coords = [pt[1] for pt in corners]
                center = [int(sum(x_coords) / len(corners)), int(sum(y_coords) / len(corners))]
                # format, text, corners, center    
                retval.append([barcode.format.name,
                                barcode.text,
                                corners,
                                center])
            except:
                pass
    except:
        pass

    return retval

# [[pxl, corners, cnt], ...]
def contour_orig(thr_img, **kwargs):
    retval = []
    
    # find contours 
    cnts, _ = cv.findContours(thr_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # list for storing names of shapes 
    for cnt in cnts:
        # check for poly
        if "side" in kwargs and kwargs["side"] is not None:
            approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
            if len(approx) != kwargs["side"]:
                continue

        # moments
        M = cv.moments(cnt)

        # Avoid division by zero (when area is zero)
        if M["m00"] == 0:
            continue
        
        # Calculate the center coordinates and rot angle
        pxl = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

        # min rect
        rect = cv.minAreaRect(cnt)
    
        # Get the 4 corners of the rectangle
        box = cv.boxPoints(rect).astype(int)

        # Convert to list of tuples
        corners = [[point[0], point[1]] for point in box]
                
        retval.append([pxl, corners, cnt])
        
    return retval



def contour(
    thr_img,
    *,
    elp=False,
    subpix=False,
    visualize=False,
    circularity_min=0,
    convexity_min=0,
    **kwargs
):
    retval = []

    # find contours 
    cnts, _ = cv.findContours(thr_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        # optional polygon side filter
        if "side" in kwargs and kwargs["side"] is not None:
            approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
            if len(approx) != kwargs["side"]:
                continue

        # skip very small contours
        if len(cnt) < 5:
            continue

        # circularity filter
        area = cv.contourArea(cnt)
        perim = cv.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < circularity_min:
            continue

        # convexity filter
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if convexity < convexity_min:
            continue

        # optional sub-pixel refinement
        if subpix:
            cnt_f = cnt.astype(np.float32)
            gray = thr_img if len(thr_img.shape) == 2 else cv.cvtColor(thr_img, cv.COLOR_BGR2GRAY)
            term_crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv.cornerSubPix(gray, cnt_f, (3, 3), (-1, -1), term_crit)
            cnt = cnt_f.astype(np.int32)

        if elp:
            # fit ellipse (for elp center)
            (x, y), (MA, ma), angle = cv.fitEllipse(cnt)
            pxl = [float(x), float(y)]  # use ellipse center
        else:
            # pixel center via moments
            M = cv.moments(cnt)
            if M["m00"] == 0:
                continue
            pxl = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]  # use moment center

        # bounding rectangle corners
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect).astype(int)
        corners = [[p[0], p[1]] for p in box]

        retval.append([pxl, corners, cnt])

        # visualization
        if visualize:
            vis = cv.cvtColor(thr_img, cv.COLOR_GRAY2BGR)
            cv.drawContours(vis, [cnt], -1, (0, 255, 255), 1)
            cv.circle(vis, (int(round(pxl[0])), int(round(pxl[1]))), 3, (0, 255, 0), -1)
            if elp:
                cv.ellipse(vis, (int(x), int(y)),
                           (int(MA / 2), int(ma / 2)),
                           angle, 0, 360, (255, 0, 0), 1)
                cv.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

    if visualize:
        cv.imshow("Contours", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return retval



# find aruco and its pose
"""
pre define dictionary 
    DICT_4X4_50,
    DICT_4X4_100,
    DICT_4X4_250,
    DICT_4X4_1000,
    DICT_5X5_50,
    DICT_5X5_100,
    DICT_5X5_250,
    DICT_5X5_1000,
    DICT_6X6_50,
    DICT_6X6_100,
    DICT_6X6_250,
    DICT_6X6_1000,
    DICT_7X7_50,
    DICT_7X7_100,
    DICT_7X7_250,
    DICT_7X7_1000,
    DICT_ARUCO_ORIGINAL,
    DICT_APRILTAG_16h5,
    DICT_APRILTAG_25h9,
    DICT_APRILTAG_36h10,
    DICT_APRILTAG_36h11
    [("DICT_4X4_50", 0), ("DICT_4X4_100", 1), ("DICT_4X4_250", 2), ("DICT_4X4_1000", 3), ("DICT_5X5_50", 4), ("DICT_5X5_100", 5), ("DICT_5X5_250", 6), ("DICT_5X5_1000", 7), ("DICT_6X6_50", 8), ("DICT_6X6_100", 9), ("DICT_6X6_250", 10), ("DICT_6X6_1000", 11), ("DICT_7X7_50", 12), ("DICT_7X7_100", 13), ("DICT_7X7_250", 14), ("DICT_7X7_1000", 15), ("DICT_ARUCO_ORIGINAL", 16), ("DICT_APRILTAG_16h5", 17), ("DICT_APRILTAG_25h9", 18), ("DICT_APRILTAG_36h10", 19), ("DICT_APRILTAG_36h11", 20)]
refine method
    CORNER_REFINE_NONE,
    CORNER_REFINE_SUBPIX,
    CORNER_REFINE_CONTOUR,
    CORNER_REFINE_APRILTAG
    [("CORNER_REFINE_NONE", 0), ("CORNER_REFINE_SUBPIX", 1), ("CORNER_REFINE_CONTOUR", 2), ("CORNER_REFINE_APRILTAG", 3)]

"""
# [[pxl, corners, (id, rvec, tvec)], ...]
def aruco(img, camera_matrix, dist_coeffs, **kwargs):

    retval = []

    # pose
    board = Aruco(**kwargs)
    rvecs, tvecs, aruco_corner, aruco_id, _ = board.pose(img, camera_matrix, dist_coeffs)

    # empty
    if aruco_id is None:
        return retval

    for i in range(len(aruco_id)):
        c = aruco_corner[i][0]      # shape (4, 2), OpenCV order: TL, TR, BR, BL
        if len(c) != 4:
            continue
        """
        # Diagonals: p0–p2 and p1–p3
        x0,y0 = c[0]; x2,y2 = c[2]
        x1,y1 = c[1]; x3,y3 = c[3]

        # Line through (x0,y0)-(x2,y2): A1 x + B1 y = C1
        A1 = y2 - y0
        B1 = x0 - x2
        C1 = A1 * x0 + B1 * y0

        # Line through (x1,y1)-(x3,y3): A2 x + B2 y = C2
        A2 = y3 - y1
        B2 = x1 - x3
        C2 = A2 * x1 + B2 * y1

        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-6:         # more robust than det == 0
            continue

        # Intersection (center pixel)
        cx = (C1 * B2 - C2 * B1) / det
        cy = (A1 * C2 - A2 * C1) / det
        pxl = [int(round(cx)), int(round(cy))]
        """
        pxl = [int(round(c[0][0])), int(round(c[0][1]))]  # TL, for example

        corners = [[int(round(px)), int(round(py))] for (px, py) in c]

        retval.append([pxl, corners, [aruco_id[i], aruco_corner[i], rvecs[i], tvecs[i]]])

    return retval



def charuco(img, camera_matrix, dist_coeffs, **kwargs):

    retval = []

    # pose
    board = Charuco(**kwargs)
    rvec, tvec, charuco_corners, charuco_ids, _, mean_err = board.pose(img, camera_matrix, dist_coeffs, disp=False)

    # empty
    if rvec is not None:
        retval = [rvec, tvec, charuco_corners, charuco_ids, mean_err]

    return retval

#[[pxl, corners, (major_axis, minor_axis, rot)], ...]    
# How far an image edge may be from an ellipse's outline and still count
# for it (_sector_support): a real edge wanders 2–3 px from a perfect
# circle — at 2 a good disc's inner circle scored 0.64, at 3 it scores 1.0.
SECTOR_TOL = 3


def _near_edges(edge_img, tol=SECTOR_TOL):
    """The edge image grown by ``tol`` px — every pixel within ``tol`` of an
    edge. Made ONCE per image and shared by every ellipse's _sector_support:
    a busy frame finds dozens of ellipses (86 on a machined holder), and
    growing it per ellipse cost ~1.5 ms each."""
    return cv.dilate(edge_img, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8))


def _sector_support(near, rect, sector_deg):
    """How much of an ellipse's outline the image shows, sector by sector:
    the outline is sampled about every pixel, a sample counts when it falls
    on ``near`` (_near_edges: an edge within SECTOR_TOL px), and each ``sector_deg`` sector
    (angle around the centre, from +x towards +y) scores its counted share,
    0..1. Returns the list, sector 0 first — its minimum is the ellipse's
    conf: a chip, a break or a dent in one place pulls it down even when the
    rest of the outline is perfect."""
    (cx, cy), (major, minor), rot = rect
    a, b = major / 2.0, minor / 2.0
    n = max(360, int(np.pi * (a + b)))            # about one sample per pixel of outline
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.deg2rad(rot)
    xs = cx + a * np.cos(t) * np.cos(r) - b * np.sin(t) * np.sin(r)
    ys = cy + a * np.cos(t) * np.sin(r) + b * np.sin(t) * np.cos(r)
    h, w = near.shape[:2]
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    seen = np.zeros(n, bool)
    seen[inside] = near[yi[inside], xi[inside]] > 0      # off the image: not seen
    sector = ((np.degrees(np.arctan2(ys - cy, xs - cx)) % 360) // sector_deg).astype(int)
    k = int(math.ceil(360 / sector_deg))
    total = np.bincount(sector, minlength=k)
    hit = np.bincount(sector, weights=seen, minlength=k)
    return [round(float(h / t), 3) for h, t in zip(hit, total) if t]


# The edge's DENTS (_dent): every DENT_STEP° a ray crosses the outline,
# sampling the image DENT_WIN px each side of it; the edge on that ray is
# where the profile crosses half-way between its level inside and outside
# (the outer DENT_EDGE px of each end) — so the edge follows the image's
# own contrast, whatever the background. A dent is the edge's distance from
# its running median over ±DENT_SMOOTH° (the outline's slow wander — a
# tilt, a lens — is not a dent), held over DENT_HOLD° (one noisy ray is
# not). A ray that shows no edge near the outline — its two levels closer
# than DENT_CONTRAST, or no crossing — has its edge at least DENT_WIN px
# away: a dent that deep (a bite that took the band). Rays off the image
# are skipped.
DENT_STEP = 0.5
DENT_WIN = 15
DENT_EDGE = 4
DENT_SMOOTH = 15
DENT_HOLD = 1.5
DENT_CONTRAST = 25


def _channel(bgr_img, name):
    """One channel of the image by name: "gray", "h" / "s" / "v" (HSV), or
    "l" / "a" / "b" (Lab — b is yellow against blue)."""
    if name == "gray":
        return cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    if name in ("h", "s", "v"):
        return cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)[..., "hsv".index(name)]
    if name in ("l", "a", "b"):
        return cv.cvtColor(bgr_img, cv.COLOR_BGR2LAB)[..., "lab".index(name)]
    raise ValueError(f"channel {name!r}: gray, h, s, v, l, a or b")


def _half_cross(prof, lvl_in, lvl_out):
    """Where each row of ``prof`` (samples along a ray, its middle on the
    outline) crosses half-way from ``lvl_in`` to ``lvl_out`` — the crossing
    nearest the middle, to a fraction of a px: ``(edge, crossed)``, edge in
    px from the middle (+ outward), crossed False on a row that never does."""
    n = len(prof)
    above = (prof - ((lvl_in + lvl_out) / 2)[:, None]) * np.sign(lvl_in - lvl_out)[:, None]
    flip = (above[:, :-1] >= 0) & (above[:, 1:] < 0)
    mid = (prof.shape[1] - 1) / 2
    idx = np.arange(prof.shape[1] - 1)
    j = np.where(flip, np.abs(idx + 0.5 - mid), np.inf).argmin(1)
    y0, y1 = above[np.arange(n), j], above[np.arange(n), j + 1]
    return j + y0 / np.where(y0 - y1 == 0, 1, y0 - y1) - mid, flip[np.arange(n), j]


def _dent(chan, rect):
    """The deepest dent in an ellipse's edge, on one image channel (see
    DENT_*): ``(depth, at)`` — depth in px (at most DENT_WIN), at its angle
    around the centre (from +x towards +y, degrees, as _sector_support's
    sectors). (0.0, None) when no ray shows an edge."""
    (cx, cy), (major, minor), rot = rect
    a, b = major / 2.0, minor / 2.0
    n = int(round(360 / DENT_STEP))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.deg2rad(rot)
    px = cx + a * np.cos(t) * np.cos(r) - b * np.sin(t) * np.sin(r)
    py = cy + a * np.cos(t) * np.sin(r) + b * np.sin(t) * np.cos(r)
    dx, dy = px - cx, py - cy
    rad = np.hypot(dx, dy)
    ux, uy = dx / rad, dy / rad
    s = np.arange(-DENT_WIN, DENT_WIN + 1, dtype=np.float32)
    mx = (px[:, None] + ux[:, None] * s).astype(np.float32)
    my = (py[:, None] + uy[:, None] * s).astype(np.float32)
    h, w = chan.shape[:2]
    inside = (mx.min(1) >= 0) & (mx.max(1) <= w - 1) & (my.min(1) >= 0) & (my.max(1) <= h - 1)
    prof = cv.remap(chan.astype(np.float32), mx, my, cv.INTER_LINEAR)
    prof = (prof[:, :-2] + prof[:, 1:-1] + prof[:, 2:]) / 3            # 3 px along the ray
    # outside: the far end's level; inside: the inside half's value farthest
    # from it — a band (a ridge, not a step) is judged by its own peak, not by
    # whatever lies past it
    lvl_out = np.median(prof[:, -DENT_EDGE:], 1)
    half = prof[:, :prof.shape[1] // 2]
    lvl_in = half[np.arange(len(half)), np.abs(half - lvl_out[:, None]).argmax(1)]
    edged = np.abs(lvl_in - lvl_out) >= DENT_CONTRAST
    # the half-way crossing nearest the outline, px from it, + outward
    edge, crossed = _half_cross(prof, lvl_in, lvl_out)
    edged &= crossed
    if not (inside & edged).any():
        return 0.0, None
    edge[~(inside & edged)] = np.median(edge[inside & edged])
    k = int(round(DENT_SMOOTH / DENT_STEP))
    ref = np.median(np.lib.stride_tricks.sliding_window_view(np.concatenate([edge[-k:], edge, edge[:k]]), 2 * k + 1), 1)
    dev = np.minimum(np.abs(edge - ref), DENT_WIN)
    dev[inside & ~edged] = DENT_WIN                                    # no edge near the outline
    dev[~inside] = 0
    hold = max(1, int(round(DENT_HOLD / DENT_STEP)))
    held = np.lib.stride_tricks.sliding_window_view(np.concatenate([dev, dev[:hold - 1]]), hold).min(1)
    i = int(held.argmax())
    at = float(np.degrees(np.arctan2(dy[i], dx[i])) % 360)
    return round(float(held[i]), 2), round(at, 1)


def ellipse(bgr_img, min_path_length=50, min_line_length = 10, nfa_validation = True, sigma=1, gradient_threshold_value=20, pf_mode = False,
            sector_deg=None, dent=None, dent_img=None, **kwargs):
        """Edge-based ellipses. ``sector_deg`` None (default): every
        ellipse's conf is 1. Set (e.g. 10): conf is how much of its outline
        the image shows in its worst ``sector_deg`` sector — 1.0 a whole
        outline, lower where a sector is missing its edge (_sector_support,
        an edge counting within SECTOR_TOL px of the outline). ``dent``
        None (default): off. Set to a channel ("gray", "h", "s", "v"): each
        ellipse also gets its edge's deepest dent on that channel (_dent),
        read on ``dent_img`` — the same view before any colour mask (a
        mask's cut is a hard edge of its own: the shadow beside the apc
        band sits at the cut and speckles into dents) — or on ``bgr_img``
        when None."""
        retval = []

        # init
        e_d = cv.ximgproc.createEdgeDrawing()

        EDParams = cv.ximgproc_EdgeDrawing_Params()
        EDParams.MinPathLength = min_path_length     # try changing this value between 5 to 1000
        EDParams.PFmode = pf_mode         # defaut value try to swich it to True
        EDParams.MinLineLength = min_line_length     # try changing this value between 5 to 100
        EDParams.NFAValidation = nfa_validation   # defaut value try to swich it to False
        EDParams.Sigma = sigma # gaussian blur
        EDParams.GradientThresholdValue = gradient_threshold_value # gradient
        
        # set parameters
        e_d.setParams(EDParams)

        # gray image
        gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
        
        # detect edges
        edges = e_d.detectEdges(gray_img)
        #segments =  e_d.getSegments()
        #lines = e_d.detectLines()
        elps = e_d.detectEllipses()
    
        if elps is None:
            return retval

        near = _near_edges(e_d.getEdgeImage()) if sector_deg is not None else None
        chan = _channel(bgr_img if dent_img is None else dent_img, dent) if dent is not None else None

        # format
        for elp in elps:
            # axes
            major_axis = int(2*(elp[0][2]+elp[0][3]))
            minor_axis = int(2*(elp[0][2]+elp[0][4]))

            # center 
            center = (int(elp[0][0]), int(elp[0][1]))

            # rotation
            rot = round(elp[0][5]%360, 2)

            # rect
            rect = (center, (major_axis, minor_axis), rot)

            # Get the 4 corners of the bounding box using boxPoints
            box = cv.boxPoints(rect).astype(int)

            # Convert to list of tuples
            corners = [[point[0], point[1]] for point in box]

            # conf: 1, or the worst sector's share of the outline the image
            # shows — scored on the UNROUNDED ellipse: the edge is one pixel
            # wide, and the integer centre above alone moved a whole disc's
            # score from 0.83 to 0.59. ``sectors``: every sector's share,
            # for a view that draws them (None when sector_deg is unset)
            cx, cy, r0, a0, b0, th = (float(v) for v in elp[0])
            exact = ((cx, cy), (2 * (r0 + a0), 2 * (r0 + b0)), th)
            conf, sectors = 1, None
            if sector_deg is not None:
                sectors = _sector_support(near, exact, float(sector_deg))
                conf = min(sectors)
            # the edge's deepest dent (None when ``dent`` is unset)
            dent_r = _dent(chan, exact) if chan is not None else None

            # add to return
            retval.append([center, corners, rect, conf, sectors, dent_r])
            
        return retval


# ── ring: a band's two edges, measured along rays ─────────────────────
# For a band too thin or too faint for elp to find its edges (the apc
# disc's top: a ~9 px olive band between a bright face and a machined
# holder). The ROI is the part, roughly centred. (1) Every RING_STEP° a ray
# from the ROI's centre, RING_FROM..RING_TO of its half-size, finds the
# inner channel's steepest step — the face's edge, the strongest edge
# there — and a circle is fit to those. (2) From that circle, the rays are
# averaged in RING_BIN° bins; on each, the INNER edge is where the inner
# channel crosses half-way between its levels RING_WIN px each side (the
# median of the RING_END px at each end), and the OUTER edge where the
# outer channel, past the band's extreme, first crosses half-way to the
# outside's level (the median of RING_LVL px past ``width``). The band is
# sought from RING_END px past that bin's inner edge (the face is behind
# it) to ``width`` px past it, on the side of the outside the band lies on
# in most bins (a bluish machining streak beside a yellowish band is the
# other side, never the band); a bin whose band stands out less than
# RING_WEAK of the bins' median is skipped. A
# circle is fit to each (_fit_circle); one whose points sit further than
# RING_MAX_RMS px from it on average is no circle the image shows (a
# shadow across the part) and is left out. Tuned on the apc disc's top
# (2026-10-23, 89 images): fits 0.2–2.1 px on 87, 7.5 / 11.3 px on the two
# with a shadow or no band; split-half centres agree to 0.05 px (inner)
# and 0.2 px (outer).
RING_STEP = 0.5
RING_FROM, RING_TO = 0.5, 0.98
RING_BIN = 2
RING_WIN = 15
RING_END = 4
RING_LVL = 8
RING_WEAK = 0.5
RING_MIN_SHARE = 1 / 3          # a circle needs this share of its bins
RING_MAX_RMS = 4


def _rays(chan, cx, cy, r0, r1, n):
    """``chan`` sampled along n rays from (cx, cy), radius r0..r1 in 1 px
    steps, 3 px smoothed along each: ``(prof, angles)``, prof (n, samples)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = np.arange(r0, r1 + 1, dtype=np.float32)
    mx = (cx + np.cos(t)[:, None] * s).astype(np.float32)
    my = (cy + np.sin(t)[:, None] * s).astype(np.float32)
    prof = cv.remap(chan.astype(np.float32), mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return cv.blur(prof, (3, 1)), t


def _fit_circle(xs, ys):
    """The least-squares circle through points, refit three times without
    the points over 3 MAD (at least 1 px) off it: ``(cx, cy, r, rms,
    kept)`` — rms of the kept points' distance from it, px."""
    keep = np.ones(len(xs), bool)
    for _ in range(3):
        A = np.c_[2 * xs[keep], 2 * ys[keep], np.ones(keep.sum())]
        cx, cy, c = np.linalg.lstsq(A, xs[keep] ** 2 + ys[keep] ** 2, rcond=None)[0]
        r = math.sqrt(c + cx * cx + cy * cy)
        res = np.hypot(xs - cx, ys - cy) - r
        keep = np.abs(res) <= max(3 * 1.4826 * np.median(np.abs(res[keep])), 1.0)
    return cx, cy, r, float(np.sqrt(np.mean(res[keep] ** 2))), keep


def ring(bgr_img, inner="l", outer="b", width=40, **kwargs):
    """A band's two edges as two circles (see RING_*): ``inner`` / ``outer``
    the channel each edge is read on (_channel), ``width`` how far past the
    inner edge the outer one is looked for, px. Returns a list of up to two
    {"edge": "inner" | "outer", "center", "corners", "r", "rms", "conf"} —
    conf the share of the bins its fit kept; a circle with under
    RING_MIN_SHARE of them, or a fit over RING_MAX_RMS px, is left out."""
    h, w = bgr_img.shape[:2]
    half = min(w, h) / 2
    ci, co = _channel(bgr_img, inner), _channel(bgr_img, outer)
    n = int(round(360 / RING_STEP))

    # (1) the inner edge, roughly: each ray's steepest step
    r0 = int(RING_FROM * half)
    prof, t = _rays(ci, w / 2, h / 2, r0, int(RING_TO * half), n)
    j = np.abs(np.diff(prof, axis=1)).argmax(1) + r0 + 0.5
    cx, cy, r, _, _ = _fit_circle(w / 2 + j * np.cos(t), h / 2 + j * np.sin(t))

    # (2) both edges from that circle, in bins
    k = int(round(RING_BIN / RING_STEP))
    bins = lambda p: p.reshape(n // k, k, -1).mean(1)
    a = (np.arange(n // k) + 0.5) * RING_BIN * np.pi / 180
    pin, _ = _rays(ci, cx, cy, int(r) - RING_WIN, int(r) + RING_WIN, n)
    pin = bins(pin)
    e_in, ok_in = _half_cross(pin, np.median(pin[:, :RING_END], 1), np.median(pin[:, -RING_END:], 1))
    r_in = int(r) + e_in
    base = int(r) - RING_WIN                                            # the outer profile's first px
    pout, _ = _rays(co, cx, cy, base, int(r) + width + RING_END, n)
    pout = bins(pout)
    lvl = np.median(pout[:, -RING_LVL:], 1)
    pos = np.arange(pout.shape[1])[None, :]
    start = np.round(r_in - base).astype(int)[:, None] + RING_END       # past each bin's inner edge
    inband = (pos >= start) & (pos < start - RING_END + width)
    dev = pout - lvl[:, None]
    side = 1.0 if np.median([d[b].mean() for d, b in zip(dev, inband) if b.any()]) >= 0 else -1.0
    ext = np.where(inband, dev * side, -np.inf).argmax(1)
    peak = pout[np.arange(len(pout)), ext]
    rise = (peak - lvl) * side                                         # how far the band stands out
    above = (pout - ((peak + lvl) / 2)[:, None]) * side
    flip = (pos[:, :-1] >= ext[:, None]) & (above[:, :-1] >= 0) & (above[:, 1:] < 0)
    ok_out = flip.any(1) & ok_in & (rise >= RING_WEAK * np.median(rise[ok_in]))
    m = flip.argmax(1)
    y0, y1 = above[np.arange(len(m)), m], above[np.arange(len(m)), m + 1]
    r_out = base + m + y0 / np.where(y0 - y1 == 0, 1, y0 - y1)

    out = []
    for edge, rr, ok in (("inner", r_in, ok_in), ("outer", r_out, ok_out)):
        if ok.mean() < RING_MIN_SHARE:
            continue
        fx, fy, fr, rms, keep = _fit_circle((cx + rr * np.cos(a))[ok], (cy + rr * np.sin(a))[ok])
        if keep.sum() < RING_MIN_SHARE * len(a) or rms > RING_MAX_RMS:
            continue
        out.append({"edge": edge, "center": (fx, fy), "r": fr, "rms": round(rms, 2),
                    "conf": round(float(keep.sum() / len(a)), 3),
                    "corners": [[fx - fr, fy - fr], [fx + fr, fy - fr], [fx + fr, fy + fr], [fx - fr, fy + fr]]})
    return out
