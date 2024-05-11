from camera import Camera
from board import Charuco
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import itertools
"""
T_cam_2_j4 = np.matrix([[-4.93641500e-01, -8.67863349e-01,  5.59578205e-02,  6.44697848e+01],
                        [ 8.66291643e-01, -4.85045214e-01,  1.19456808e-01, -1.33825844e+02],
                        [-7.65301123e-02,  1.07444630e-01,  9.91261213e-01,  6.06569858e+01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
"""
def eye_in_hand_dorna_ta_embeded_camera(robot, kinematic, camera, charuco_board):
    data = {
        
    }
    # init window
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)

    # initialization
    R_target_2_cam_list = []
    t_target_2_cam_list = []
    R_j4_2_base_list = []
    t_j4_2_base_list = []

    # Generate all possible lists of size 5
    initial_joint = np.array([0, 45, -120, 0, -15, 0])
    deviations = np.array(list(itertools.product([-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5],[0])))
    all_joints = [(initial_joint +  joint).tolist() for joint in deviations]

    for joint in all_joints:
        # move to the joint
        robot.jmove(rel=0, vel=50, accel=800, jerk=1000, j0=joint[0], j1=joint[1], j2=joint[2], j3=joint[3], j4=joint[4], j5=joint[5])
        robot.sleep(0.5)

        # capture image
        _, _, _, _, _, color_img, depth_int, _, _= camera.get_all()

        # current joint and pose
        joint = robot.get_all_joint()
        T_j4_2_base = kinematic.Ti_r_world(i=5, joint=joint[0:6])
        R_j4_2_base_list.append(T_j4_2_base[:3, :3])
        t_j4_2_base_list.append(T_j4_2_base[:3, 3])


        # target_pose
        R_target_2_cam, t_target_2_cam, charuco_id, img_gray = charuco_board.pose(color_img, camera.camera_matrix(depth_int), camera.dist_coeffs(depth_int))   
        R_target_2_cam_list.append(R_target_2_cam)
        t_target_2_cam_list.append(t_target_2_cam)

        # show axis
        cv2.imshow("color",color_img)
        
        # wait 1ms
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'): # press q to exit
            break

    # run calibration    
    R_cam_2_j4, t_cam_2_j4 = cv2.calibrateHandEye(R_j4_2_base_list, t_j4_2_base_list, R_target_2_cam_list, t_target_2_cam_list)
    
    # transformation matrix format
    T_cam_2_j4 = np.eye(4)
    T_cam_2_j4[:3, :3] = R_cam_2_j4
    T_cam_2_j4[:3, 3] = np.ravel(t_cam_2_j4)


    return T_cam_2_j4


def main_eye_in_hand_dorna_ta_embeded_camera():
    # parameters
    robot_ip = "192.168.254.54" 
    model = "dorna_ta"
    sqr_x=8
    sqr_y=8
    sqr_length=12
    marker_length=8
    dictionary="DICT_6X6_250"
    refine="CORNER_REFINE_APRILTAG"
    subpix=False

    # camera
    camera = Camera()
    camera.connect()

    # board
    charuco_board = Charuco(sqr_x, sqr_y, sqr_length, marker_length, dictionary, refine, subpix)
    
    # Robot
    robot = Dorna()
    robot.connect(robot_ip)

    # kinematics
    kinematic = Kinematic(model)

    # pose
    T_cam_2_j4 = eye_in_hand_dorna_ta_embeded_camera(robot, kinematic, camera, charuco_board)

    # close the connections
    camera.close()
    robot.close()

    print("### T_cam_2_j4 ###\n", T_cam_2_j4)


if __name__ == '__main__':
    main_eye_in_hand_dorna_ta_embeded_camera()