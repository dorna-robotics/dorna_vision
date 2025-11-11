import cv2 as cv
import numpy as np
from dorna_vision.board import Aruco, Charuco
import zxingcpp

def elp_fit(
    img,
    *,
    use_otsu=True,
    area_range=(300, 500000),
    aspect_ratio_tol=0.1,
    circularity_min=0.7,
    convexity_min=0.9,
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

    for c in cnts:
        area = cv.contourArea(c)
        if not (area_range[0] < area < area_range[1]):
            continue
        if len(c) < 5:
            continue

        (x, y), (MA, ma), angle = cv.fitEllipse(c)
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
        br = max(2, int(round(kp.size/2.0)) + int(pad))

        x0 = max(0, int(round(bx - br)))
        x1 = min(W, int(round(bx + br)))
        y0 = max(0, int(round(by - br)))
        y1 = min(H, int(round(by + br)))

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
        else:
            cx, cy = bx, by
            MA = ma = 2*br; angle = 0.0

        if use_subpix:
            pt = np.array([[[cx, cy]]], np.float32)
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            sub = cv.cornerSubPix(gray, pt, win, (-1,-1), term)
            cx, cy = sub[0,0,0], sub[0,0,1]

        corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
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

    vis = cv.cvtColor(thr_img, cv.COLOR_GRAY2BGR) if visualize else None

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

        # fit ellipse (for elp center)
        (x, y), (MA, ma), angle = cv.fitEllipse(cnt)

        # pixel center via moments
        M = cv.moments(cnt)
        if M["m00"] == 0:
            continue

        if elp:
            pxl = [float(x), float(y)]  # use ellipse center
        else:
            pxl = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]  # use moment center

        # bounding rectangle corners
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect).astype(int)
        corners = [[p[0], p[1]] for p in box]

        retval.append([pxl, corners, cnt])

        # visualization
        if visualize:
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
def ellipse(bgr_img, min_path_length=50, min_line_length = 10, nfa_validation = True, sigma=1, gradient_threshold_value=20, pf_mode = False, **kwargs):
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

            # add to return
            retval.append([center, corners, rect])
            
        return retval

