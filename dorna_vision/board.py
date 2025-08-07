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
    def __init__(self, dictionary = "DICT_4X4_100", refine="CORNER_REFINE_APRILTAG", subpix=False, marker_length=20, marker_size=100, win_size=(11,11), scale=4):
        super(Aruco, self).__init__()
        self.dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary))
        self.refine = refine
        self.subpix = subpix
        self.marker_length = marker_length
        self.marker_size = marker_size
        self.win_size = win_size
        self.scale = scale
        """"
        # prms and refine
        prms =  cv.aruco.DetectorParameters()
        prms.cornerRefinementMethod = getattr(cv.aruco, self.refine)
        """
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
        self.prms.cornerRefinementWinSize     = 21   # bigger window
        self.prms.cornerRefinementMaxIterations = 100
        self.prms.cornerRefinementMinAccuracy   = 1e-6

    def create(self, board_path="board.png", marker_id=0):
        # Generate the marker image
        board = cv.aruco.generateImageMarker(self.dictionary, marker_id, self.marker_size)

        # write
        cv.imwrite(board_path, board)


    def corner(self, img):
        """
        Detect ArUco corners with upsampling + CLAHE + optional blur,
        then (if self.subpix) run cv.cornerSubPix on the upsampled image
        before scaling back to the original resolution.
        Returns: corners, ids, rejected, gray_orig
        """
        # 1) convert to gray and keep the original for return
        gray_orig = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 2) upscale so sub‑pixel has more pixels to work with
        gray = cv.resize(
            gray_orig,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv.INTER_CUBIC
        )

        # 3) apply CLAHE (Contrast Limited AHE) for more uniform grads
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # 4) optional light blur to reduce sensor noise
        gray = cv.GaussianBlur(gray, (5,5), 0)

        # 5) detect markers on the upsampled, equalized image
        corners_up, ids, rejected_up = cv.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.prms
        )

        # 6) if subpix refinement is on, refine each upsampled corner
        if self.subpix and corners_up:
            for c in corners_up:
                cv.cornerSubPix(
                    gray,
                    c,
                    winSize=self.win_size,
                    zeroZone=(-1,-1),
                    criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
                )

        # 7) scale corners & rejected candidates back down to original size
        corners = [c.astype(np.float32) / self.scale for c in corners_up]
        rejected = [r.astype(np.float32) / self.scale for r in rejected_up]

        return corners, ids, rejected, gray_orig


    def pose(self, img, camera_matrix, dist_coeffs, coordinate="ccw"):
        """
        Same as before, but uses the new, robust corner() filter.
        """
        corners, ids, rejected, gray = self.corner(img)
        rvecs, tvecs = [], []
        m = float(self.marker_length)

        # choose object‑points based on coordinate flag
        if coordinate.lower() == "cw":
            obj_pts = np.array([
                [0,   0,   0],
                [m,   0,   0],
                [m,   m,   0],
                [0,   m,   0]
            ], dtype=np.float32)
        else:
            obj_pts = np.array([
                [-m/2,  m/2, 0],
                [ m/2,  m/2, 0],
                [ m/2, -m/2, 0],
                [-m/2, -m/2, 0]
            ], dtype=np.float32)

        # solvePnP per remaining marker
        for c in corners:
            img_pts = c.reshape(4,2).astype(np.float32)
            ok, rvec, tvec = cv.solvePnP(
                obj_pts, img_pts,
                camera_matrix, dist_coeffs,
                flags=cv.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                rvecs.append(rvec)
                tvecs.append(tvec)

        # format output arrays
        if rvecs:
            rvecs = np.array(rvecs, dtype=np.float32)
            tvecs = np.array(tvecs, dtype=np.float32)
        else:
            rvecs = np.empty((0,1,3), np.float32)
            tvecs = np.empty((0,1,3), np.float32)

        return rvecs, tvecs, corners, ids, gray


class Charuco(object):
    def __init__(
        self,
        sqr_x=7, sqr_y=7,
        sqr_length=30, marker_length=24,
        dictionary="DICT_5X5_1000",
        refine="CORNER_REFINE_APRILTAG",
        subpix=False,
        win_size=(11, 11),
        scale=3               # upsample factor for sub‑pixel
    ):
        # subpix flag for chess intersections
        self.subpix = subpix
        self.scale  = scale
        self.win_size = win_size

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

        # tighten thresholding if needed
        self.prms.adaptiveThreshConstant = 7


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
        #    (gray0 is the original-resolution gray)
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
        if resp <= 0:
            return None, None, charuco_corners, charuco_ids, gray_orig, None

        # 2) Initial estimate
        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        retval, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.board,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec
        )
        if not retval:
            return None, None, charuco_corners, charuco_ids, gray_orig, None

        # 3) Build the matching 3D points for each detected corner
        size_x, size_y = self.board.getChessboardSize()   # e.g. (7,7) squares
        sq = self.board.getSquareLength()
        # intersections = (size_x-1)*(size_y-1), left→right, top→bottom
        obj_all = np.array(
            [[i*sq, j*sq, 0] 
            for j in range(1, size_y)    # j=0 top row
            for i in range(1, size_x)],   # i=0 left column
            dtype=np.float32
        )
        objPts = obj_all[ charuco_ids.flatten() ]  # shape = (N,3)

        # 4) Iterative LM refine on the full set
        ok, rvec, tvec = cv.solvePnP(
            objPts,
            charuco_corners.reshape(-1,2),
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE
        )

        # 5) Compute mean reprojection error
        proj, _    = cv.projectPoints(objPts, rvec, tvec, camera_matrix, dist_coeffs)
        proj_pts2d = proj.reshape(-1,2)
        detected2d = charuco_corners.reshape(-1,2)
        errs       = np.linalg.norm(detected2d - proj_pts2d, axis=1)
        mean_err   = float(errs.mean())

        # 6) Draw if requested
        if disp:
            cv.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
            axis_len = sq * (size_x)
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len)

        # 7) Return everything plus error
        return rvec, tvec, charuco_corners, charuco_ids, gray_orig, mean_err
    

    def chess_corner(self, img):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1, 0.001)

        # bgr to gray
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray_img, (self.sqr_y-1, self.sqr_x-1), None)
        color_img = cv.drawChessboardCorners(img, (self.sqr_y-1, self.sqr_x-1), corners, ret)

        # If found, add object points, image points (after refining them)
        corners2 = []
        if ret == True:
            corners2 = cv.cornerSubPix(gray_img, corners,(11,11),(-1,-1),criteria)

        return ret , corners2


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