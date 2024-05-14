from camera import Camera
from board import Charuco
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import itertools
import pickle
import time
from scipy.optimize import minimize

"""
T_cam_2_j4 = np.matrix([[-4.93641500e-01, -8.67863349e-01,  5.59578205e-02,  6.44697848e+01],
                        [ 8.66291643e-01, -4.85045214e-01,  1.19456808e-01, -1.33825844e+02],
                        [-7.65301123e-02,  1.07444630e-01,  9.91261213e-01,  6.06569858e+01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
"""

def Euler_matrix(abg,v):
    cv0 = np.cos(abg[0])
    sv0 = np.sin(abg[0])
    cv1 = np.cos(abg[1])
    sv1 = np.sin(abg[1])
    cv2 = np.cos(abg[2])
    sv2 = np.sin(abg[2])
    return np.matrix([
        [cv1* cv0   , sv2*sv1*cv0 - cv2*sv0 , cv2*sv1*cv0 - sv2*sv0 , v[0]  ],
        [cv1 * sv0  , sv2*sv1*sv0 + cv2*cv0 , cv2*sv1*sv0 + sv2*cv0 , v[1]  ],
        [-sv1       , sv2*cv1               , cv2*cv1               , v[2]  ],
        [0,0,0,1]])


def minimizer(joints, R_target_2_cam_list, t_target_2_cam_list, kinematic, force_z = None, use_rotation = False):

    R_j4_2_base_list = []
    t_j4_2_base_list = []

    for j in joints:
        T_j4_2_base = kinematic.Ti_r_world(i=5, joint=j)
        R_j4_2_base_list.append(np.array(T_j4_2_base[:3, :3])) 
        t_j4_2_base_list.append(np.array(T_j4_2_base[:3, 3]) )

    data_t = [np.array(d) for d in t_target_2_cam_list]
    data_R = []

    for r in R_target_2_cam_list:
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(r, rotation_matrix)
        data_R.append(np.array(rotation_matrix))

    #R_cam_2_j4, t_cam_2_j4 = cv2.calibrateHandEye(R_j4_2_base_list, t_j4_2_base_list, data_R, data_t 
    #   , method =  cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)

    T_cam_2_j4 = np.eye(4)
    #T_cam_2_j4[:3, :3] = R_cam_2_j4
    #T_cam_2_j4[:3, 3] = np.ravel(t_cam_2_j4)


    def likelihood(p):
        T = Euler_matrix([p[3],p[4],p[5]],[p[0],p[1],p[2]])
        v =[]
        for test_index in range(len(joints)):
            R_test = np.eye(4)
            R_test[:3, :3] =  data_R[test_index]
            R_test[:3, 3] = np.ravel(t_target_2_cam_list[test_index])
            g = np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=joints[test_index]),np.matrix(T)), np.matrix(R_test))
            if not use_rotation:
                v.append([g[0,3],g[1,3],g[2,3]])
            else:
                g[0,3] = g[0,3]/100.0
                g[1,3] = g[1,3]/100.0
                g[2,3] = g[2,3]/100.0
                v.append(g)

        squared_distances = 0
        v = np.array(v)
        centroid = np.mean(v, axis=0)

        if not use_rotation:
            squared_distances = np.sum((v - centroid)**2, axis=1) 
            
            if force_z!= None:
                squared_distances += ((np.array([ (g[2] - force_z) for g in v]))**2 ) 

            squared_distances = np.mean(squared_distances)

        else:
            squared_distances = np.sum( np.square(np.ravel(v - centroid)))

        return np.sqrt(squared_distances)


    f = minimize(likelihood, [0,0,0,0,0,0])#np.transpose(t_cam_2_j4).tolist()[0])
    #T_cam_2_j4[:3, 3] = f.x
    T_cam_2_j4 = Euler_matrix([f.x[3],f.x[4],f.x[5]],[f.x[0],f.x[1],f.x[2]])

    return T_cam_2_j4


def eye_in_hand_dorna_ta_embeded_camera(robot, kinematic, camera, charuco_board):
    # init window
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)

    # initialization
    joints = []
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
        robot.sleep(1)

        # capture image
        _, _, _, _, _, color_img, depth_int, _, _= camera.get_all()

        # joint
        joint = robot.get_all_joint()
        joints.append(joint)
        
        # pose
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


    """
    # run calibration    
    R_cam_2_j4, t_cam_2_j4 = cv2.calibrateHandEye(R_j4_2_base_list, t_j4_2_base_list, R_target_2_cam_list, t_target_2_cam_list)
    
    # transformation matrix format
    T_cam_2_j4 = np.eye(4)
    T_cam_2_j4[:3, :3] = R_cam_2_j4
    T_cam_2_j4[:3, 3] = np.ravel(t_cam_2_j4)
    """
    T_cam_2_j4 = minimizer( joints = joints, 
                            R_target_2_cam_list = R_target_2_cam_list, 
                            t_target_2_cam_list = t_target_2_cam_list, 
                            kinematic = kinematic, 
                            force_z = None, 
                            use_rotation = False)



    # save data
    data = {
        "joints":joints,
        "R_target_2_cam_list": R_target_2_cam_list,
        "t_target_2_cam_list": t_target_2_cam_list,
        "R_j4_2_base_list":R_j4_2_base_list,
        "t_j4_2_base_list":t_j4_2_base_list,       
    }

    
    with open("data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    return True


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
    #filter={"decimate":2, "temporal":[0.1, 40]} # {"decimate":2, "spatial":[2, 0.5, 20], "temporal":[0.4, 20], "hole_filling":1}
    #filter={"decimate":2, "spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1}

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


if __name__ == '__main__':
    main_eye_in_hand_dorna_ta_embeded_camera()