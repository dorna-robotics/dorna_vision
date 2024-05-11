import sys
sys.path.append('C:/Users/hossein/Desktop/github/dorna_vision/')
from camera import Camera
from dorna_vision.board import Charuco
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import time

"""
The camera locate the charuco center
Guide the robot to go above the target with a given rotation matrix
"""
def guide_robot_to_charuco(robot, kinematic, camera, charuco_board, T_cam_2_j4, distance):
    # init window
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)


    while True:
        # capture image
        _, _, _, _, _, color_img, depth_int, _, _= camera.get_all()

        # target_pose
        R_target_2_cam, t_target_2_cam, charuco_id, img_gray = charuco_board.pose(color_img, camera.camera_matrix(depth_int), camera.dist_coeffs(depth_int))   
        T_target_2_cam = np.eye(4)
        T_target_2_cam[:3, :3] = R_target_2_cam
        T_target_2_cam[:3, 3] = np.ravel(t_target_2_cam)
        
        # show axis
        cv2.imshow("color",color_img)
        
        # wait 1ms
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'): # press q to exit
            break
        elif key & 0xFF == ord('g'): # press g to exit
            # current joint and pose
            joint = robot.get_all_joint()
            T_j4_2_base = kinematic.Ti_r_world(i=5, joint=joint[0:6])
            
            # target
            T_target_2_base = np.matmul(T_j4_2_base, np.matmul(T_cam_2_j4, T_target_2_cam) )
            xyz_target_2_base =T_target_2_base[:3, 3].flatten().tolist()[0]
            print(T_target_2_cam)
            print(T_target_2_base)
            print("###")
            
            """
            # robot
            robot.set_motor(1)
            robot.sleep(1)
            robot.lmove(rel=0, vel=50, accel=800, jerk=1000, x=xyz_target_2_base[0], y=xyz_target_2_base[1], z=xyz_target_2_base[2] + distance, a=0, b=0)
            """
        time.sleep(0.01)

    return True



def main_guide_robot_to_charuco():
    # parameters
    T_cam_2_j4 = np.matrix([[-4.93641500e-01, -8.67863349e-01,  5.59578205e-02,  6.44697848e+01],
                        [ 8.66291643e-01, -4.85045214e-01,  1.19456808e-01, -1.33825844e+02],
                        [-7.65301123e-02,  1.07444630e-01,  9.91261213e-01,  6.06569858e+01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    distance = 20
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

    # test
    guide_robot_to_charuco(robot, kinematic, camera, charuco_board, T_cam_2_j4, distance)
    
    # close the connections
    camera.close()
    robot.close()


if __name__ == '__main__':
    main_guide_robot_to_charuco()