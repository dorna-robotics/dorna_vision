import cv2 as cv
import numpy as np
from scipy.optimize import differential_evolution


"""
helper functions
"""
def euler_matrix(abg,xyz):
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


# optimizing xyzabc
def rvec_calibration(p, data, xyz, ej=[0,0,0,0,0,0,0,0]): # no rotation only all the parameters for the euler matrix
    # init
    total_error = 0
    num_data = 0

    for set in data:
        v =[]
        for d in set:
            g = ((d["kin_approx"] @ euler_matrix(p, xyz)) @ np.matrix([[d["gt"][0]], [d["gt"][1]], [d["gt"][2]], [1]])).T

            v.append(g)
            num_data = num_data + 1

        # compute centroid
        centroid = np.mean(v, axis=0)
        
        # compute distance
        d = [np.linalg.norm(g - centroid) for g in v]
        total_error += np.mean(np.square(d))
    
    return total_error


# p: x, y, z, ej0, ej1, ej2, ej3, ej4, ej5
# T: [[0 ,-1, 0, x],[1, 0, 0, y],[0, 0, 1, z],[0,0,0,1]]
def total_error(p, data):
    # init
    error = 0
    for set in data:
        # v
        v =[((d["H"][0]+p[3]*d["H"][1]+p[4]*d["H"][2]+p[5]*d["H"][3]+p[6]*d["H"][4]+p[7]*d["H"][5]) @ np.matrix([[p[1]], [-p[0]], [p[2]], [1]])).T for d in set]
        
        # compute centroid
        centroid = np.mean(v, axis=0)
        
        # compute distance
        d = [np.linalg.norm(g - centroid) for g in v]
        error += np.mean(np.square(d))
    
    return error


class Calibration(object):
    def __init__(self, robot=None, detection=None):
        self.robot = robot
        self.detection = detection
        
        # init
        self.collected_data = [[]]
        self.data = {}
        self.img = np.zeros((5, 9), dtype=np.uint8)
        self.result = None
        self.error = None


    def capture_image(self, marker_length=20, use_aruco=1, thr=0.5):
        center = None
        # adjust marker length
        self.detection.detection["marker_length"] = marker_length

        # init
        self.data = {}
        text = "[No marker detected]"
        color = (0, 0, 255)

        # current joint
        joint = np.array(self.robot.get_all_joint()[0:6]).tolist()
        
        # run detection
        results = self.detection.run()
        if results and results[0]["cls"] == "0":
            result = results[0]
            center = result["center"]

            # distance
            error = np.linalg.norm(np.array(result["xyz"])-np.array(result["tvec"]))

            # threshold
            if error > thr:
                text = "[Bad] marker error: " + str(round(error,2)) + "mm"
            else:
                # target_to_camera
                if use_aruco:
                    gt = result["tvec"]
                else:
                    gt = result["xyz"]
                
                text = "[Good] marker error: " + str(round(error,2)) + "mm"
                color = (0, 255, 0)
                cmd = {
                    "cmd": "jmove", "rel": 0,
                    "j0": round(joint[0], 2), "j1": round(joint[1], 2), "j2": round(joint[2], 2), "j3": round(joint[3], 2), "j4": round(joint[4], 2), "j5": round(joint[5], 2)
                }
                self.data = dict({"joint": joint, "t_target_2_cam": result["xyz"], "aruco_t_target_2_cam": result["tvec"], "aruco_r_target_2_cam": result["rvec"], "cmd": cmd, "gt": gt})


        self.img = self.detection.img.copy()
        cv.putText(self.img, text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv.LINE_AA)
        if center is not None:
            cv.circle(self.img, (center[0], center[1]), 5, (255,0,255), 2)


    def add_data(self):
        if self.data:
            self.collected_data[0].append(self.data)


    def clear_data(self):
        self.collected_data = [[]]


    def prepare_data(self, data):
        retval =[]
        for batch in data:
            tmp = list()
            for result in batch:
                # target_to_camera
                gt = result["gt"]

                # target_2_cam
                target_2_cam = self.robot.kinematic.xyzabc_to_mat(np.array([gt[0], gt[1], gt[2], result["aruco_r_target_2_cam"][0], result["aruco_r_target_2_cam"][1], result["aruco_r_target_2_cam"][2]]))

                _kinematic = self.robot.kinematic.Ti_r_world(i=5, joint=result["joint"])

                # jacobian
                epsilon = 0.1
                joint_differential =  np.zeros((5, 6))

                for i in range(min(joint_differential.shape)):
                    joint_differential[i, i] = epsilon

                jac = [_kinematic] 

                for i in range(min(joint_differential.shape)):
                    diff_kin = (self.robot.kinematic.Ti_r_world(i=5, joint=(np.array(result["joint"])+joint_differential[i])) - jac[0])/epsilon
                    jac.append(diff_kin)

                # H matrix
                H = []
                gt_matrix = np.matrix([[0, -1, 0, -gt[1]], [1, 0, 0, gt[0]], [0, 0, 1, gt[2]], [0, 0, 0, 1]])
                for i in range(len(jac)):
                    H.append(np.matmul(jac[i], gt_matrix))
                
                # append
                tmp.append({"gt": gt, "kinematic": _kinematic, "joint": result["joint"], "target_2_cam": target_2_cam, "jac": jac, "H": H})
            
            if len(tmp) > 0:
                retval.append(tmp)
        
        return retval


    def calibrate(self, camera_mount_type="dorna_ta_j4_1", je_bound=0.5, tvec_bound=5, rvec_bound=0.01):
        # prepared data
        data = self.prepare_data(self.collected_data)
        
        if camera_mount_type == "dorna_ta_j4_1":
            for i in range(len(self.robot.config["camera_mount"])):
                if self.robot.config["camera_mount"][i]["type"] == camera_mount_type:
                    T = self.robot.config["camera_mount"][i]["T"]
                    ej = self.robot.config["camera_mount"][i]["ej"]
                    break
        
            bounds = [(T[0]-tvec_bound, T[0]+tvec_bound), 
                    (T[1]-tvec_bound, T[1]+tvec_bound), 
                    (T[2]-tvec_bound, T[2]+tvec_bound),
                    (ej[0]-je_bound, ej[0]+je_bound),
                    (ej[1]-je_bound, ej[1]+je_bound),
                    (ej[2]-je_bound, ej[2]+je_bound),
                    (ej[3]-je_bound, ej[3]+je_bound),
                    (ej[4]-je_bound, ej[4]+je_bound),
                    ]
            
            # additional prepare
            for set in data:
                for d in set:
                    # H matrix
                    H = []
                    gt_matrix = np.matrix([[0, -1, 0, -d["gt"][1]], [1, 0, 0, d["gt"][0]], [0, 0, 1, d["gt"][2]], [0, 0, 0, 1]])
                    for i in range(len(d["jac"])):
                        H.append(np.matmul(d["jac"][i], gt_matrix))
                    
                    d["H"] = H

            # return
            result = differential_evolution(total_error, bounds, args=(data, ), maxiter=1000, seed=42)

            # rvec calibration
            xyz = result.x[0:3].tolist()
            ej = result.x[3:].tolist()

            # prpare adta
            for set in data:
                for d in set:
                    d["kin_approx"] = d["jac"][0]
                    for i in range(len(ej)):
                        d["kin_approx"] += ej[i]*d["jac"][i+1]

            bound = [
                (np.pi/2-rvec_bound ,np.pi/2+rvec_bound),
                (-rvec_bound ,rvec_bound),
                (-rvec_bound ,rvec_bound),  
            ]

            result= differential_evolution(rvec_calibration, bound, args=(data, xyz, ej,), maxiter=1000, seed=42)
            
            # T matrix
            T = euler_matrix(result.x,xyz)
            xyzabc = self.robot.kinematic.mat_to_xyzabc(T)
            T = [round(x,3) for x in xyzabc.tolist()]

            self.result = {
                "type": camera_mount_type,
                "T": T,
                "ej": [round(x,3) for x in ej]+[0, 0, 0]
            }
            self.error = round(result.fun, 3)

            return dict(self.result), self.error