from camera import Camera
from board import Charuco
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import itertools
from scipy.optimize import minimize
import pickle

def Euler_matrix(abg,xyz):
    cv0 = np.cos(abg[0])
    sv0 = np.sin(abg[0])
    cv1 = np.cos(abg[1])
    sv1 = np.sin(abg[1])
    cv2 = np.cos(abg[2])
    sv2 = np.sin(abg[2])
    return np.matrix([
        [cv1* cv0   , sv2*sv1*cv0 - cv2*sv0 , cv2*sv1*cv0 - sv2*sv0 , xyz[0]  ],
        [cv1 * sv0  , sv2*sv1*sv0 + cv2*cv0 , cv2*sv1*sv0 + sv2*cv0 , xyz[1]  ],
        [-sv1       , sv2*cv1               , cv2*cv1               , xyz[2]  ],
        [0,0,0,1]])


def minimizer(joints, R_target_2_cam_list, t_target_2_cam_list, kinematic, ground_truth, use_rotation = False, joint_calibration=False):

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
        #cv2.Rodrigues(r, rotation_matrix)
        data_R.append(np.array(rotation_matrix))

    #T_cam_2_j4 = np.eye(4)

    def likelihood(p):


        T_cam_2_j4 = Euler_matrix([p[3],p[4],p[5]],[p[0],p[1],p[2]]) #[1.57079632679, 0, 0] , [p[0],p[1],p[2]])#
        v =[]
        for test_index in range(len(joints)):
            g = np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=joints[test_index]),np.matrix(T_cam_2_j4)), np.vstack((t_target_2_cam_list[test_index], np.array([[1]]))))
            v.append([g[0,0],g[1,0],g[2,0]])


        centroid = np.mean(v, axis=0)
        #centroid2 = np.array([343.557786, 23.676558, 0.607504])



        a = np.array([np.linalg.norm(g - centroid) for g in v])
        l = np.mean(a)
        print("Centroid position: ",np.mean(v,axis=0))
        print("Mean error: ", l , " Max error: ", np.max(a))

        return l

    f = minimize(likelihood, [0,0,0,0,0,0])


    T_cam_2_j4 = Euler_matrix([f.x[3],f.x[4],f.x[5]],[f.x[0],f.x[1],f.x[2]])
    """
    T_cam_2_j4 = np.matrix([[-1,  0,  0,  f.x[0]],
                            [0, -1,  0,  f.x[1]],
                            [ 0, 0,  1,  f.x[2]],
                            [ 0,  0,  0,  1]])
    """
    return T_cam_2_j4




def dorna_ta_eye_in_hand_embeded_camera(robot, kinematic, camera, charuco_board, joint_list, ground_truth, joint_calibration, file_path):
    # search_id
    search_id = np.floor(((charuco_board.sqr_x-1)*(charuco_board.sqr_y-1)-1)/2)

    print("search_id: ", search_id) 
    # init window
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)

    # initialization
    joints = []
    R_target_2_cam_list = []
    t_target_2_cam_list = []
    R_j4_2_base_list = []
    t_j4_2_base_list = []

    #motor
    robot.set_motor(1)

    # Generate all possible lists of size 5    
    for joint in joint_list:
        robot.jmove(rel=0, vel=50, accel=800, jerk=1000, j0=joint[0], j1=joint[1], j2=joint[2], j3=joint[3], j4=joint[4], j5=joint[5])
        robot.sleep(1)

        # capture image
        depth_frame, _, _, _, _, color_img, depth_int, _, _= camera.get_all()

        # joint
        joint = robot.get_all_joint()
        # search_id pose
        try:
            response, charuco_corner, charuco_id, _ = charuco_board.corner(color_img)
            search_id_index,_ = np.where(charuco_id == search_id)
            search_id_index = search_id_index[0]
        except:
            print("no charuco detected")
            continue

        # append
        joints.append(joint)
        
        # pose
        T_j4_2_base = kinematic.Ti_r_world(i=5, joint=joint[0:6])
        R_j4_2_base_list.append(T_j4_2_base[:3, :3])
        t_j4_2_base_list.append(T_j4_2_base[:3, 3])

        # pixel
        px, py = charuco_corner[search_id_index][0]
        xyz_target_2_cam, _ = camera.xyz((px, py), depth_frame, depth_int)
        R_target_2_cam_list.append(np.eye(3))
        t_target_2_cam_list.append(xyz_target_2_cam.reshape(-1, 1))

        # show axis
        cv2.imshow("color",color_img)

        # wait 1ms
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'): # press q to exit
            break


    # cam to robot transformation matrix
    T_cam_2_j4 = minimizer( joints = joints, 
                            R_target_2_cam_list = R_target_2_cam_list, 
                            t_target_2_cam_list = t_target_2_cam_list, 
                            kinematic = kinematic, 
                            ground_truth = ground_truth, 
                            use_rotation = False,
                            joint_calibration=joint_calibration)

    # save data
    if file_path:
        with open(file_path, 'wb') as f:
            data = {
                "joints": joints,
                "R_target_2_cam_list": R_target_2_cam_list,
                "t_target_2_cam_list": t_target_2_cam_list,
                "R_j4_2_base_list": R_j4_2_base_list,
                "t_j4_2_base_list": t_j4_2_base_list,
                "T_cam_2_j4": T_cam_2_j4
            }
            pickle.dump(data, f)
    return T_cam_2_j4


def main_dorna_ta_eye_in_hand_embeded_camera():
    # parameters
    robot_ip = "192.168.254.30" 
    model = "dorna_ta"
    sqr_x=8
    sqr_y=8
    sqr_length=12
    marker_length=8
    dictionary="DICT_6X6_250"
    refine="CORNER_REFINE_APRILTAG"
    subpix=False
    ground_truth = (None, None, 0.25)
    joint_calibration = False
    file_path = "data.pkl"

    # joint_list
    #initial_joint = np.array([0, 25, -90, 0, -25, 0])
    initial_joint = np.array([0, 28, -100, 0, -17, 0])
    deviations = np.array(list(itertools.product([-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10],[0])))
    joint_list = [(initial_joint +  joint).tolist() for joint in deviations]


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
    #T_cam_2_j4 = dorna_ta_eye_in_hand_embeded_camera(robot, kinematic, camera, charuco_board, joint_list, ground_truth, joint_calibration, file_path)
    
    # cam to robot transformation matrix
    with open(file_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)
    T_cam_2_j4 = minimizer( joints = data["joints"], 
                            R_target_2_cam_list = data["R_target_2_cam_list"], 
                            t_target_2_cam_list = data["t_target_2_cam_list"], 
                            kinematic = kinematic, 
                            ground_truth = ground_truth, 
                            use_rotation = False,
                            joint_calibration=joint_calibration)

    # close the connections
    camera.close()
    robot.close()

    print(T_cam_2_j4)


if __name__ == '__main__':
    main_dorna_ta_eye_in_hand_embeded_camera()