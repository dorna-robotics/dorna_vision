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
    def __init__(self, dictionary = "DICT_6X6_250", refine="CORNER_REFINE_APRILTAG", subpix=False, marker_length=10, marker_size=100):
        super(Aruco, self).__init__()
        self.dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary))
        self.refine = refine
        self.subpix = subpix
        self.marker_length = marker_length
        self.marker_size = marker_size

    def create(self, board_path="board.png", marker_id=0):
        # Generate the marker image
        board = cv.aruco.generateImageMarker(self.dictionary, marker_id, self.marker_size)

        # write
        cv.imwrite(board_path, board)

    def corner(self, img):
        # bgr to gray
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # prms and refine
        prms =  cv.aruco.DetectorParameters()
        prms.cornerRefinementMethod = getattr(cv.aruco, self.refine)

        # Detect ArUco markers in the image
        aruco_corner, aruco_id, aruco_reject = cv.aruco.detectMarkers(img_gray, self.dictionary, parameters=prms)

        # corner subpix
        if self.subpix:
            [cv.cornerSubPix(img_gray, corner, winSize=(11, 11), zeroZone=(-1, -1), criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1, 0.001)) for corner in aruco_corner]

        return aruco_corner, aruco_id, aruco_reject, img_gray

    def pose(self, img, camera_matrix, dist_coeffs):
        # corner detection
        aruco_corner, aruco_id, aruco_reject, img_gray = self.corner(img)
        
        # Estimate pose
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(aruco_corner, markerLength=self.marker_length, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        return rvecs, tvecs, aruco_corner, aruco_id, img_gray


class Charuco(object):
    """docstring for charuco"""
    def __init__(self, sqr_x=8, sqr_y=8, sqr_length=12, marker_length=8, dictionary="DICT_6X6_250", refine="CORNER_REFINE_APRILTAG", subpix=False):
        super(Charuco, self).__init__()
        self.aruco = Aruco(dictionary=dictionary, refine=refine, subpix=subpix, marker_length=marker_length)
        self.subpix = subpix
        self.board = cv.aruco.CharucoBoard((sqr_x, sqr_y), sqr_length, marker_length, self.aruco.dictionary)
        self.sqr_x = sqr_x
        self.sqr_y = sqr_y


    """
    save charuco board, and save the result in a file
    """
    def create(self, board_path="board.png", width=1000, height=1000, margin=0):
        cv.imwrite(board_path, self.board.generateImage((width,height), margin, margin))


    """
    detect charuco chess board corners
    """
    def corner(self, img):
        # init
        response = 0
        charuco_corner = []
        charuco_id = []

        # aruco markers
        aruco_corner, aruco_id, aruco_reject, img_gray = self.aruco.corner(img)

        # Get charuco corners and ids from detected aruco markers
        if aruco_corner:
            response, charuco_corner, charuco_id = cv.aruco.interpolateCornersCharuco(
                markerCorners=aruco_corner,
                markerIds=aruco_id,
                image=img_gray,
                board=self.board)

            if response:
                # refine
                if self.subpix:
                    charuco_corner = cv.cornerSubPix(img_gray,
                        charuco_corner,
                        (11,11),
                        (-1,-1),
                        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1, 0.001))

                cv.aruco.drawDetectedCornersCharuco(img, charuco_corner, charuco_id, cornerColor=(0, 255, 0))

        return response, charuco_corner, charuco_id, img_gray


    def pose(self, img, camera_matrix, dist_coeffs):
        # init
        rvec = np.zeros((3, 3))
        tvec = np.zeros((3, 1))
        
        # find corners
        response, charuco_corner, charuco_id, img_gray = self.corner(img)

        if response:

            # estimate pose
            retval, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(charuco_corner, charuco_id, self.board, camera_matrix, dist_coeffs, rvec, tvec)    

            # draw axis
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, self.board.getChessboardSize()[0]*self.board.getSquareLength(), 1)

        return rvec, tvec, charuco_id, img_gray 
        

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


def main_chess_corenr():
    test = Charuco(8, 8, 12, 8)

    img = cv.imread("board.png")
    response, corners = test.chess_corner(img)
    print(corners)
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