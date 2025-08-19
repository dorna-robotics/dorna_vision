import cv2 as cv
import numpy as np

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
refine method
    CORNER_REFINE_NONE,
    CORNER_REFINE_SUBPIX,
    CORNER_REFINE_CONTOUR,
    CORNER_REFINE_APRILTAG
"""

class Aruco(object):
    """docstring for aruco"""
    def __init__(self, dictionary="DICT_4X4_100", refine="CORNER_REFINE_APRILTAG",
                 subpix=False, marker_length=20, marker_size=100,
                 win_size=(11,11), scale=1):
        super(Aruco, self).__init__()
        self.dictionary   = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary))
        self.refine       = refine
        self.subpix       = subpix
        self.marker_length= marker_length     # units of your choice; tvec will match this
        self.marker_size  = marker_size
        self.win_size     = win_size
        self.scale        = scale

        self.prms = cv.aruco.DetectorParameters()
        # Adaptive threshold
        self.prms.adaptiveThreshWinSizeMin    = 3
        self.prms.adaptiveThreshWinSizeMax    = 23
        self.prms.adaptiveThreshWinSizeStep   = 10
        self.prms.adaptiveThreshConstant      = 7
        # Contour filtering
        self.prms.minMarkerPerimeterRate      = 0.04
        self.prms.maxMarkerPerimeterRate      = 4.0
        self.prms.polygonalApproxAccuracyRate = 0.03
        # Corner refinement
        self.prms.cornerRefinementMethod      = getattr(cv.aruco, self.refine)
        self.prms.cornerRefinementWinSize     = 21
        self.prms.cornerRefinementMaxIterations = 100
        self.prms.cornerRefinementMinAccuracy   = 1e-6

    def create(self, board_path="board.png", marker_id=0):
        board = cv.aruco.generateImageMarker(self.dictionary, marker_id, self.marker_size)
        cv.imwrite(board_path, board)

    def corner(self, img):
        """
        Detect ArUco corners with upsampling + CLAHE (+ optional subpix),
        then scale corners back to the original resolution.
        Returns: corners(list of (1,4,2) float32), ids, rejected, gray_orig
        """
        gray_orig = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        gray = cv.resize(gray_orig, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_CUBIC)
        gray = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        gray = cv.GaussianBlur(gray, (5,5), 0)

        corners_up, ids, rejected_up = cv.aruco.detectMarkers(gray, self.dictionary, parameters=self.prms)

        if self.subpix and corners_up:
            for c in corners_up:
                cv.cornerSubPix(
                    gray, c, winSize=self.win_size, zeroZone=(-1,-1),
                    criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
                )

        corners  = [c.astype(np.float32) / self.scale for c in corners_up]
        rejected = [r.astype(np.float32) / self.scale for r in rejected_up]

        return corners, ids, rejected, gray_orig

    def pose(self, img, camera_matrix, dist_coeffs):
        """
        Pose via OpenCV's ArUco estimator (no manual solvePnP).
        Returns rvecs, tvecs with shape (N,3), corners, ids, gray.
        """
        corners, ids, rejected, gray = self.corner(img)

        if ids is None or len(corners) == 0:
            rvecs = np.empty((0,3), np.float32)
            tvecs = np.empty((0,3), np.float32)

            return rvecs, tvecs, corners, ids, gray

        # 1) Pose from OpenCV estimator
        # corners must be a list of (1,4,2) float32 arrays in original pixel scale
        prm = cv.aruco.EstimateParameters()
        prm.pattern = cv.aruco.ARUCO_CW_TOP_LEFT_CORNER
        rvecs_e, tvecs_e, _ = cv.aruco.estimatePoseSingleMarkers(
            corners, float(self.marker_length), camera_matrix, dist_coeffs,
            estimateParameters=prm
        )  # rvecs_e, tvecs_e: (N,1,3)

        rvecs = rvecs_e.reshape(-1,3).astype(np.float32)
        tvecs = tvecs_e.reshape(-1,3).astype(np.float32)

        # 2) Cheirality enforcement per marker. If any Z<=0, retry with reversed winding and keep better/front-facing.
        # Build object points from marker_length (centered square at Z=0).
        m = float(self.marker_length)
        obj_cw  = np.array([[-m/2,  m/2, 0.0],
                            [ m/2,  m/2, 0.0],
                            [ m/2, -m/2, 0.0],
                            [-m/2, -m/2, 0.0]], dtype=np.float32)
        obj_ccw = obj_cw[[0,3,2,1], :]

        for i in range(len(corners)):
            if tvecs[i,2] > 0:
                continue  # good

            # Try reversed corner order for this marker
            img_pts = corners[i].reshape(4,2).astype(np.float32)

            # Estimate again with reversed winding by reordering the image corners to match CCW model
            # (equivalently reorder obj points; here we reorder image to keep API simple)
            img_rev = img_pts[[0,3,2,1], :][None, ...].astype(np.float32)  # shape (1,4,2)
            r_alt, t_alt, _ = cv.aruco.estimatePoseSingleMarkers(
                [img_rev.astype(np.float32)],  # list of (1,4,2)
                float(self.marker_length), camera_matrix, dist_coeffs
            )
            r_alt = r_alt.reshape(3).astype(np.float32)
            t_alt = t_alt.reshape(3).astype(np.float32)

            # Choose front-facing with lower reprojection error
            def reproj_err(rvec, tvec, obj):
                proj, _ = cv.projectPoints(obj, rvec.reshape(3,1), tvec.reshape(3,1), camera_matrix, dist_coeffs)
                return float(np.mean(np.linalg.norm(proj.reshape(-1,2) - img_pts, axis=1)))

            err_cur = reproj_err(rvecs[i], tvecs[i], obj_cw)
            err_alt = reproj_err(r_alt,     t_alt,     obj_ccw)

            if (t_alt[2] > 0) and (err_alt <= err_cur):
                rvecs[i] = r_alt
                tvecs[i] = t_alt
            elif tvecs[i,2] <= 0 and t_alt[2] > 0:
                rvecs[i] = r_alt
                tvecs[i] = t_alt
            # else keep current (even if negative) only if both are negative—should be rare

        return rvecs, tvecs, corners, ids, gray


import cv2 as cv
import numpy as np

class Charuco(object):
    def __init__(
        self,
        sqr_x=7, sqr_y=7,
        sqr_length=30, marker_length=24,
        dictionary="DICT_5X5_1000",
        refine="CORNER_REFINE_SUBPIX",
        subpix=False,
        win_size=(11, 11),
        scale=3,                # upsample factor for sub‑pixel
        # --- new knobs ---
        pnp="SQPNP",           # "SQPNP" | "ITERATIVE" | "AP3P"
        use_ransac=True,
        ransac_reproj=2.0,     # px
        ransac_iters=200,
        ransac_conf=0.999,
        refine_lm=True,
        min_corners=4,
        use_norm_points=True    # undistort to normalized coords before PnP
    ):
        # subpix flag for chess intersections
        self.subpix = subpix
        self.scale  = scale
        self.win_size = win_size

        # PnP options
        self.pnp = pnp
        self.use_ransac = use_ransac
        self.ransac_reproj = float(ransac_reproj)
        self.ransac_iters  = int(ransac_iters)
        self.ransac_conf   = float(ransac_conf)
        self.refine_lm = refine_lm
        self.min_corners = int(min_corners)
        self.use_norm_points = bool(use_norm_points)

        # build a matching ArUco detector (with sub‑pixel or not)
        self.aruco = Aruco(
            dictionary=dictionary,
            refine=refine,
            subpix=False,     # we do sub‑pix ourselves on Charuco corners
            marker_length=marker_length,
            win_size=self.win_size,
            scale=self.scale
        )
        # create a Charuco board
        self.board = cv.aruco.CharucoBoard(
            (sqr_x, sqr_y),
            sqr_length,
            marker_length,
            self.aruco.dictionary
        )
        # use the contrib factory for the best defaults
        self.prms = cv.aruco.DetectorParameters()
        self.prms.adaptiveThreshConstant = 7

    # --- helpers (added) ---
    def _pnp_flag(self):
        name = (self.pnp or "").upper()
        if name == "SQPNP":
            return cv.SOLVEPNP_SQPNP
        if name == "AP3P":
            return cv.SOLVEPNP_AP3P
        # default
        return cv.SOLVEPNP_ITERATIVE

    def _solve_with_pipeline(self, objPts_all, imgPts_all, K, D, rvec0, tvec0):
        """
        Runs: (optional) RANSAC -> solvePnP (chosen flag) -> LM refine.
        Returns rvec, tvec, inlier_indices (or None).
        """
        flag = self._pnp_flag()
        rvec, tvec = rvec0, tvec0
        inliers = None

        # RANSAC (on all points)
        if self.use_ransac and len(imgPts_all) >= self.min_corners:
            ok, rvec_r, tvec_r, inliers = cv.solvePnPRansac(
                objectPoints=objPts_all,
                imagePoints=imgPts_all,
                cameraMatrix=K,
                distCoeffs=D,
                useExtrinsicGuess=True,
                iterationsCount=self.ransac_iters,
                reprojectionError=self.ransac_reproj,
                confidence=self.ransac_conf,
                flags=flag
            )
            if ok and inliers is not None and len(inliers) >= self.min_corners:
                rvec, tvec = rvec_r, tvec_r
                idx = inliers.reshape(-1)
                objPts = objPts_all[idx]
                imgPts = imgPts_all[idx]
            else:
                objPts, imgPts = objPts_all, imgPts_all
                inliers = None
        else:
            objPts, imgPts = objPts_all, imgPts_all

        # main solve with chosen flag
        ok2, rvec_s, tvec_s = cv.solvePnP(
            objPts, imgPts, K, D,
            rvec, tvec, useExtrinsicGuess=True, flags=flag
        )
        if ok2:
            rvec, tvec = rvec_s, tvec_s

        # optional LM refine
        if self.refine_lm:
            try:
                cv.solvePnPRefineLM(objPts, imgPts, K, D, rvec, tvec)
            except Exception:
                pass

        return rvec, tvec, (inliers.reshape(-1) if inliers is not None else None)

    """
    save charuco board, and save the result in a file
    """
    def create(self, board_path="board.png", width=1000, height=1000, margin=0):
        cv.imwrite(board_path, self.board.generateImage((width,height), margin, margin))

    """
    detect charuco chess board corners
    """
    def corner(self, img, camera_matrix, dist_coeffs):
        """
        1) Call your robust ArUco.corner() to get:
            - marker corners already refined at upsampled scale
            - the ORIGINAL gray (gray0)
        2) Reconstruct the UPSAMPLED gray (exactly as Aruco.corner does)
        3) Prune & refine those same marker corners (but scaled up) on that gray
        4) Interpolate Charuco intersections on that gray
        5) SubPix-refine the intersections (still on upsampled gray)
        6) Down-scale the Charuco intersections and return them
        """
        # 1) get marker corners + gray0 from your Aruco pipeline
        marker_corners, marker_ids, marker_rejected, gray0 = self.aruco.corner(img)
        if not marker_corners or marker_ids is None:
            return 0, [], [], gray0

        # 2) rebuild the exact upsampled gray
        gray_up = cv.resize(
            gray0, None,
            fx=self.scale, fy=self.scale,
            interpolation=cv.INTER_CUBIC
        )
        gray_up = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_up)
        gray_up = cv.GaussianBlur(gray_up, (5,5), 0)

        # 3) scale your marker corners *back up* to match gray_up coords
        mk_up = [ (c * self.scale).astype(np.float32) for c in marker_corners ]

        #    prune + snap to board
        cv.aruco.refineDetectedMarkers(
            gray_up,
            self.board,
            mk_up,
            marker_ids,
            marker_rejected,
            camera_matrix,
            dist_coeffs
        )

        # 4) interpolate chessboard intersections
        resp, charuco_up, charuco_ids = cv.aruco.interpolateCornersCharuco(
            markerCorners=mk_up,
            markerIds=marker_ids,
            image=gray_up,
            board=self.board
        )

        # 5) optional sub-pixel refine on the UPSAMPLED intersections
        if resp > 0 and self.subpix:
            cv.cornerSubPix(
                gray_up,
                charuco_up,
                winSize=self.win_size,
                zeroZone=(-1,-1),
                criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
            )

        # 6) scale the Charuco intersections back to original res
        if resp > 0:
            charuco_up /= self.scale

        # return: count, corners @ original resolution, ids, original gray
        return resp, charuco_up, charuco_ids, gray0

    def pose(self, img, camera_matrix, dist_coeffs, disp=True):
        # 1) Detect Charuco corners
        resp, charuco_corners, charuco_ids, gray_orig = self.corner(
            img, camera_matrix, dist_coeffs
        )
        if resp is None or resp <= 0 or charuco_ids is None or len(charuco_ids) < self.min_corners:
            return None, None, charuco_corners, charuco_ids, gray_orig, None

        # 2) Initial estimate (seed)
        rvec0 = np.zeros((3,1), dtype=np.float64)
        tvec0 = np.zeros((3,1), dtype=np.float64)
        retval, rvec_seed, tvec_seed = cv.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.board,
            camera_matrix,
            dist_coeffs,
            rvec0,
            tvec0
        )
        if not retval:
            return None, None, charuco_corners, charuco_ids, gray_orig, None
        rvec, tvec = rvec_seed, tvec_seed

        # 3) Build the matching 3D points for each detected corner
        size_x, size_y = self.board.getChessboardSize()   # e.g. (7,7) squares
        sq = self.board.getSquareLength()
        obj_all = np.array(
            [[i*sq, j*sq, 0]
             for j in range(1, size_y)    # j=0 top row
             for i in range(1, size_x)],  # i=0 left column
            dtype=np.float32
        )
        ids = charuco_ids.flatten()
        objPts_all = obj_all[ids].astype(np.float32)                # (N,3)

        # 4) Prepare 2D points; optionally undistort to normalized coords
        #    Keep a copy of original pixels for final error computation.
        imgPts_px = charuco_corners.reshape(-1,2).astype(np.float32)

        if self.use_norm_points:
            # normalized coordinates (no distortion)
            undist = cv.undistortPoints(
                charuco_corners.astype(np.float32), camera_matrix, dist_coeffs
            )  # (N,1,2)
            imgPts_all = undist.reshape(-1,2).astype(np.float32)
            Kn = np.eye(3, dtype=np.float32)
            Dn = None
        else:
            imgPts_all = imgPts_px
            Kn, Dn = camera_matrix, dist_coeffs

        # 5) Robust PnP pipeline on chosen coordinate space
        rvec, tvec, inliers = self._solve_with_pipeline(
            objPts_all, imgPts_all, Kn, Dn, rvec, tvec
        )

        # 6) Mean reprojection error (pixels in the ORIGINAL image space)
        proj, _ = cv.projectPoints(objPts_all, rvec, tvec, camera_matrix, dist_coeffs)
        proj_pts2d = proj.reshape(-1,2)
        errs = np.linalg.norm(imgPts_px - proj_pts2d, axis=1)
        mean_err = float(errs.mean()) if len(errs) else None

        # 7) Draw if requested
        if disp:
            cv.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
            axis_len = sq * (size_x)
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len)

        # 8) Return everything plus error (same signature/order as yours)
        return rvec, tvec, charuco_corners, charuco_ids, gray_orig, mean_err


def main_charuco_corenr():
    test = Charuco(8, 8, 12, 8)

    img = cv.imread("board.png")
    response, charuco_corner, charuco_id, _ = test.corner(img)
    print(response, "\n", charuco_corner,"\n", charuco_id )
    cv.imshow("charuco", img)
    cv.waitKey(0)


def main_chess_corenr():
    test = Charuco(8, 8, 12, 8)

    img = cv.imread("board.png")
    response, corners = test.chess_corner(img)
    cv.imshow("charuco", img)
    cv.waitKey(0)


def main_charuco_create():
    test = Charuco(8, 8, 12, 8)
    test.create("board.png")
    

def main_aruco_create():
    arc = Aruco()
    arc.create()


if __name__ == '__main__':
    main_charuco_create()
    #main_aruco_create()
    #main_chess_corenr()
    main_charuco_corenr()