import sys
sys.path.append('C:/Users/hossein/Desktop/github/dorna_vision/')
from camera import Camera
from dorna_vision.board import Charuco
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

"""
The camera locate the charuco center
Guide the robot to go above the target with a given rotation matrix
"""
def guide_robot_to_charuco(robot, kinematic, camera, T_cam_2_j4, distance):
    # init window
    #cv2.namedWindow('color', cv2.WINDOW_NORMAL)

    #create axes
    ax1 = plt.subplot(111)

    #create image plot
    im1 = ax1.imshow(np.zeros((100, 100, 3)))

    plt.ion()

    while True:
        # capture image
        depth_frame, _, _, _, _, color_img, depth_int, _, _= camera.get_all()
        
        # show axis
        #cv2.imshow("color",color_img)
        # Show color image
        im1.set_data(color_img)
        plt.pause(0.01)
        
        # wait 1ms
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'): # press q to exit
            break
        elif key & 0xFF == ord('g'): # press g to exit
            # Prompt the user to input x and y coordinates
            x = int(input("Enter the width coordinate: "))
            y = int(input("Enter the height coordinate: "))
            
            # T_target_2_cam
            xyz_target_2_cam, _ = camera.xyz((x, y), depth_frame, depth_int)
            T_target_2_cam = np.eye(4)
            T_target_2_cam[:3, 3] = np.ravel(xyz_target_2_cam)
            
            # current joint and pose
            joint = robot.get_all_joint()
            T_j4_2_base = kinematic.Ti_r_world(i=5, joint=joint[0:6])
            
            # target_2_base
            T_target_2_base = np.matmul(T_j4_2_base, np.matmul(T_cam_2_j4, T_target_2_cam) )
            xyz_target_2_base =T_target_2_base[:3, 3].flatten().tolist()[0]            
            
            # robot
            robot.set_motor(1)
            robot.sleep(1)
            robot.lmove(rel=0, vel=50, accel=800, jerk=1000, x=xyz_target_2_base[0], y=xyz_target_2_base[1], z=xyz_target_2_base[2] + distance, a=0, b=0)
            

    plt.ioff() # due to infinite loop, this gets never called.
    plt.show()
    return True



def main_guide_robot_to_charuco():
    # parameters
    T_cam_2_j4 = np.matrix([[-9.99920230e-01, -3.42457264e-03,  1.22432824e-02,  7.68865611e+00],
                            [ 3.26439435e-03, -9.99907934e-01,  1.30910399e-02,  4.83039759e+01],
                            [ 1.22015316e-02, -1.31299625e-02,  9.99839350e-01,  3.74904752e+01],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    distance = 50
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

    # Robot
    robot = Dorna()
    robot.connect(robot_ip)

    # kinematics
    kinematic = Kinematic(model)

    # test
    guide_robot_to_charuco(robot, kinematic, camera, T_cam_2_j4, distance)
    
    # close the connections
    camera.close()
    robot.close()


if __name__ == '__main__':
    main_guide_robot_to_charuco()