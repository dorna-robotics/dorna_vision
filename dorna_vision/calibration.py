import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.optimize import minimize
from IPython.display import display, clear_output

"""
helper functions
"""
# euler to transformation matrix
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


def likelihood(p, kinematic, data, use_aruco, use_ground_truth):
    total_error = 0
    
    T_cam_2_j4 = Euler_matrix([p[3],p[4],p[5]],[p[0],p[1],p[2]])
    num_data = 0

    for point_set in data:
        v =[]
        for idx in range(len(point_set)):
            gt = []
            if use_aruco:
                gt = point_set[idx]["aruco_t_target_2_cam"]
            else:
                gt = point_set[idx]["t_target_2_cam"]

            g = np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=point_set[idx]["joint"]), np.matrix(T_cam_2_j4)), 
                np.vstack((np.reshape(gt, (3,1) ), np.array([[1]]))))
            v.append([g[0,0],g[1,0],g[2,0]])
            num_data = num_data + 1
        

        centroid = None

        if len(point_set[0]["t_target_2_base"])==0:
            centroid = np.mean(v, axis=0)
        else:
            if(use_ground_truth):
                centroid =  point_set[0]["t_target_2_base"]
            else:
                centroid = np.mean(v, axis=0)

        a = np.array([np.linalg.norm(np.array(g) - np.array(centroid)) for g in v])
        total_error += np.linalg.norm(a)**2
    print("centroid: ", centroid)
    return np.sqrt(total_error/num_data)


def validation(T_cam_2_j4, data, kinematic, gt):
    error_list = []
    for d in data:
        g = np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=d["joint"]), np.matrix(T_cam_2_j4)), 
            np.vstack((np.reshape(d["t_target_2_cam"], (3,1) ), np.array([[1]]))))
        detected = np.array([g[0,0],g[1,0],g[2,0]])
        #error_list.append(np.linalg.norm(np.array(detected) - np.array(gt)))
        error_list.append(np.array(detected) - np.array(gt))
    return error_list


def minimizer(data, kinematic, use_aruco=False, use_ground_truth=True):
    # minimize
    f = minimize(likelihood, x0=[0,0,0,0,0,0], args = (kinematic, data, use_aruco, use_ground_truth))
    print("avg error (mm): ",f.fun)
    
    # result
    T_cam_2_j4 = Euler_matrix([f.x[3],f.x[4],f.x[5]],[f.x[0],f.x[1],f.x[2]])
    
    return T_cam_2_j4


def dorna_ta_eye_in_hand_camera_kit(robot, detection, joint_deviation, aruco_id):
    # init
    retval = []
    
    # record t_target_2_base
    touching = input("The robot is touching the aruco marker (y or n):")
    touching = touching.lower()

    # touch
    if touching != "y":
        return retval
    
    print("claibration process start")

    # init centroid
    xyz_target_2_base = robot.get_all_pose()[0:3]
    
    #go up
    robot.set_motor(1)
    robot.sleep(1)
    robot.lmove(rel=1, z=140, vel=100, accel=1000, jerk=2000)
    robot.jmove(rel=0, j0=-2.9042938990811535, j1=34.475098, j2=-109.973145, j3=0, j4=-16.501465, j5=0, vel=50, accel=800, jerk=1000)
    
    robot.sleep(1)
    
    # initial joint
    initial_joint = np.array(robot.get_all_joint()[0:6])
    target_joint_list = [(initial_joint +  joint).tolist() for joint in joint_deviation]

    pxl_target_2_cam_list = []
    # Generate all possible lists of size 5    
    for i in range(len(target_joint_list)):
        # move and sleep
        robot.jmove(rel=0, j0=target_joint_list[i][0], j1=target_joint_list[i][1], j2=target_joint_list[i][2], j3=target_joint_list[i][3], j4=target_joint_list[i][4], j5=target_joint_list[i][5])
        robot.sleep(1)

        # current joint
        joint = np.array(robot.get_all_joint()[0:6]).tolist()
        
        # run detection
        results = detection.run()
        if results:
            for result in results:
                if result["cls"] == str(int(aruco_id)):
                    # append collect data
                    retval.append({"joint": joint, "t_target_2_cam": result["xyz"], "t_target_2_base": xyz_target_2_base, "aruco_t_target_2_cam": result["tvec"], "aruco_r_target_2_cam": result["rvec"]})
                    pxl_target_2_cam_list.append(result["center"])
                    break
            
        # draw centers
        for pxl in pxl_target_2_cam_list:
            cv2.circle(detection.img, (int(pxl[0]), int(pxl[1])), 5, (255,0,255), 2)

        # Clear the previous output
        clear_output(wait=True)
        plt.imshow(cv2.cvtColor(detection.img, cv2.COLOR_BGR2RGB)) # Display the image
        plt.axis('off')  # Turn off axis
        display(plt.gcf())  # Display the updated plot
        plt.close() # release the memory
        print("round: ", i+1, " / ", len(target_joint_list))

    return retval
