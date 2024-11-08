import numpy as np

def pose_3_point(depth_frame, depth_int, tmp_pxls, center, dim, rot, camera):
    valid = 0
    # rotation matrix
    _cos = np.cos(np.radians(rot))
    _sin = np.sin(np.radians(rot))
    screen_x = np.array([_cos , _sin])
    screen_y = np.array([-_sin , _cos])

    # tmp to pxl: pick only the first 3
    pxls = np.array([center + 0.5 * dim[0] * tmp[0] * screen_x + 0.5 * dim[1] * tmp[1] * screen_y for tmp in tmp_pxls[0:3]])

    #
    tmp_pxl_21 = tmp_pxls[1] - tmp_pxls[0]
    tmp_pxl_31 = tmp_pxls[2] - tmp_pxls[0]

    # xyz
    xyzs = [camera.xyz(pxl, depth_frame, depth_int)[0] for pxl in pxls]
    try:
        #project center and center + dx and center +dy vectors onto 3d space and onto the plane
        g1 = (tmp_pxl_21[1] *tmp_pxl_31[0] - tmp_pxl_21[0]* tmp_pxl_31[1])
        g2 = -(tmp_pxls[0][1]* tmp_pxl_31[0]- tmp_pxls[0][0]* tmp_pxl_31[1])
        g3 = -(-tmp_pxls[0][1]* tmp_pxl_21[0] + tmp_pxls[0][0]* tmp_pxl_21[1])
        center_3d = xyzs[0]  + g2*(xyzs[1]-xyzs[0])/ g1  + g3*(xyzs[2]-xyzs[0]) /g1

        # X
        X = (xyzs[1]-xyzs[0])*(-tmp_pxl_31[1])/g1 + (xyzs[2]-xyzs[0])*(tmp_pxl_21[1])/g1
        X = X/np.linalg.norm(X)
        
        # Z
        #Z = np.cross(xyzs[1] - xyzs[0], xyzs[2] - xyzs[0])
        Z = np.cross(xyzs[2] - xyzs[0], xyzs[1] - xyzs[0])
        Z = Z / np.linalg.norm(Z)

        if Z[2] <= 0: #  z is always positive
            Z = -Z
    
        # Y
        Y = np.cross(Z,X)

        valid = 1
    except Exception as e:
        X = np.zeros(3)
        Y = np.zeros(3)
        Z = np.zeros(3)
        center_3d = np.zeros(3)

    return valid, center_3d, X, Y , Z, pxls


def camera_to_dorna_ta(T_target_2_cam, joint, kinematic, T_cam_2_j4, T_robot_2_frame=np.eye(4)):    
    # current joint and pose
    T_j4_2_base = kinematic.Ti_r_world(i=5, joint=joint[0:6])
    
    # target_2_base
    T_target_2_base = np.matmul(T_j4_2_base, np.matmul(T_cam_2_j4, T_target_2_cam) )
    
    # target_2_frame
    return np.matmul(T_robot_2_frame, T_target_2_base)