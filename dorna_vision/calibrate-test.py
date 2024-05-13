import pickle
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import random
from scipy.optimize import minimize


kinematic = Kinematic(model = "dorna_ta")

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

R_j4_2_base_list = []
t_j4_2_base_list = []

for j in data["joints"]:
	T_j4_2_base = kinematic.Ti_r_world(i=5, joint=j)
	R_j4_2_base_list.append(np.array(T_j4_2_base[:3, :3])) 
	t_j4_2_base_list.append(np.array(T_j4_2_base[:3, 3]) )



data_t = [np.array(d) for d in data["t_target_2_cam_list"] ]
data_R = []
for r in data["R_target_2_cam_list"]:
	rotation_matrix = np.zeros(shape=(3,3))
	cv2.Rodrigues(r, rotation_matrix)
	data_R.append(np.array(rotation_matrix))

R_cam_2_j4, t_cam_2_j4 = cv2.calibrateHandEye(R_j4_2_base_list, t_j4_2_base_list, data_R, data_t 
	, method =  cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)

T_cam_2_j4 = np.eye(4)
T_cam_2_j4[:3, :3] = R_cam_2_j4
T_cam_2_j4[:3, 3] = np.ravel(t_cam_2_j4)




#now test with scip

def likelihood(p):
	T = np.matrix([[-1,0,0,p[0]],
					[0,-1,0,p[1]],
					[0,0,1,p[2]],
					[0,0,0,1]])
	v =[]
	for test_index in range(len(data["joints"])):
		R_test = np.eye(4)
		R_test[:3, :3] =  data_R[test_index]
		R_test[:3, 3] = np.ravel(data["t_target_2_cam_list"][test_index])
		g = np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=data["joints"][test_index]),np.matrix(T)), np.matrix(R_test))
		v.append([g[0,3],g[1,3],g[2,3]])
	v = np.array(v)
	centroid = np.mean(v, axis=0)
	squared_distances = np.sum((v - centroid)**2, axis=1)
	return np.sqrt(np.mean(squared_distances))



f = minimize(likelihood, np.transpose(t_cam_2_j4).tolist()[0])

T_cam_2_j4[:3, 3] = f.x

for test_index in range(len(data["joints"])):
	R_test = np.eye(4)
	R_test[:3, :3] =  data_R[test_index]
	R_test[:3, 3] = np.ravel(data["t_target_2_cam_list"][test_index])

	
	g = (np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=data["joints"][test_index]),np.matrix(T_cam_2_j4)), np.matrix(R_test)) )

	print([g[0,3],g[1,3],g[2,3]])





"""


#test matrix
test_real_cam_to_j4 = np.matrix([[-2.82946320e-01, -9.56751237e-01,  6.75903175e-02,  4.67124569e+01],
 								[ 9.52429265e-01, -2.71950478e-01,  1.37555199e-01, -1.45016817e+02],
 								[-1.13224888e-01,  1.03295734e-01,  9.88185264e-01,  3.44112170e+01],
 								[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

test_real_board_world = np.matrix([	[1, 0, 0,  2.20631406e+02],
									[0, 1,0,  2.06204132e+02],
									[0, 0, 1, -1.08278187e+00],
									[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

joint_list = []
board_to_cam_list = []

random.seed(3)

for i in range(3):
	js = [random.uniform(-180,180) for _ in range(6)]
	T  = kinematic.Ti_r_world(i=5, joint=js)
	b_to_cam = np.matmul(np.linalg.inv(test_real_cam_to_j4), np.matmul(np.linalg.inv(T) , test_real_board_world))
	joint_list.append(js)
	board_to_cam_list.append(((b_to_cam)))
	#

#now do the test:
R_j4_2_base_list = []
t_j4_2_base_list = []
R_target_2_cam_list = []
t_target_2_cam_list = []
for i in range(len(joint_list)):
	T_j4_2_base = (kinematic.Ti_r_world(i=5, joint=joint_list[i]))
	R_j4_2_base_list.append(np.array(T_j4_2_base[:3, :3]))
	t_j4_2_base_list.append(np.array(((T_j4_2_base[:3, 3]))))

	R_target_2_cam_list.append(np.array(board_to_cam_list[i][:3, :3]))
	t_target_2_cam_list.append(np.array((board_to_cam_list[i][:3, 3])))

R_cam_2_j4, t_cam_2_j4 = cv2.calibrateHandEye(R_j4_2_base_list, t_j4_2_base_list,R_target_2_cam_list, t_target_2_cam_list,
	method =  cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)

T_cam_2_j4 = np.eye(4)
T_cam_2_j4[:3, :3] = R_cam_2_j4
T_cam_2_j4[:3, 3] = np.ravel(t_cam_2_j4)

print((T_cam_2_j4))

"""