import pickle
from dorna2 import Dorna, Kinematic
import cv2
import numpy as np
import random
from scipy.optimize import minimize



def Euler_matrix(abg,v):
	cv0 = np.cos(abg[0])
	sv0 = np.sin(abg[0])
	cv1 = np.cos(abg[1])
	sv1 = np.sin(abg[1])
	cv2 = np.cos(abg[2])
	sv2 = np.sin(abg[2])
	return np.matrix([
		[cv1* cv0 	, sv2*sv1*cv0 - cv2*sv0	, cv2*sv1*cv0 - sv2*sv0	, v[0]	],
		[cv1 * sv0	, sv2*sv1*sv0 + cv2*cv0	, cv2*sv1*sv0 + sv2*cv0	, v[1]	],
		[-sv1		, sv2*cv1 				, cv2*cv1				, v[2]	],
		[0,0,0,1]])


def calibrate_eye_in_hand(joints, R_target_2_cam_list, t_target_2_cam_list, kinematic, force_z_to_zero = 0, use_rotation = False):

	#set  force_z_to_zero = 0 if you don't want it to force z to be zero
	#set  force_z_to_zero = 1 if you want it to do this 

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
	#	, method =  cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)

	T_cam_2_j4 = np.eye(4)
	#T_cam_2_j4[:3, :3] = R_cam_2_j4
	#T_cam_2_j4[:3, 3] = np.ravel(t_cam_2_j4)


	def likelihood(p):
		T = Euler_matrix([p[3],p[4],p[5]],[p[0],p[1],p[2]])
		v =[]
		for test_index in range(len(data["joints"])):
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
			squared_distances = np.mean(np.sum((v - centroid)**2, axis=1) + ((np.array([g[2] for g in v]))**2 ) * force_z_to_zero)

		else:
			squared_distances = np.sum( np.square(np.ravel(v - centroid)))

		return np.sqrt(squared_distances)


	f = minimize(likelihood, [0,0,0,0,0,0])#np.transpose(t_cam_2_j4).tolist()[0])
	print(f.fun)
	#T_cam_2_j4[:3, 3] = f.x
	T_cam_2_j4 = Euler_matrix([f.x[3],f.x[4],f.x[5]],[f.x[0],f.x[1],f.x[2]])

	return T_cam_2_j4
	
	#test points in the resulted joints
	"""
	for test_index in range(len(joints)):
		R_test = np.eye(4)
		R_test[:3, :3] =  data_R[test_index]
		R_test[:3, 3] = np.ravel(data["t_target_2_cam_list"][test_index])
		g = (np.matmul(np.matmul(kinematic.Ti_r_world(i=5, joint=joints[test_index]),np.matrix(T_cam_2_j4)), np.matrix(R_test)) )
		
		print([g[0,3],g[1,3],g[2,3]])

	
"""
	


if __name__ == '__main__':

	with open('data_1.pkl', 'rb') as f:
		data = pickle.load(f)

	kinematic = Kinematic(model = "dorna_ta")

	T_cam_2_flange = calibrate_eye_in_hand(data["joints"], data["R_target_2_cam_list"], data["t_target_2_cam_list"], kinematic, 0, False)

	print(T_cam_2_flange)