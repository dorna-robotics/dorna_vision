import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import cv2
from itertools import product


# return all the keypoint combinations
def keypoint_combinations(data, labels):
    # Group elements by class
    cls_map = {label: [] for label in labels}
    for item in data:
        cls = item["cls"]
        if cls in labels:
            cls_map[cls].append(item)

    # Check all labels have at least one element
    if any(len(cls_map[label]) == 0 for label in labels):
        return []

    # Generate all combinations (Cartesian product of class groups)
    return [list(combo) for combo in product(*(cls_map[label] for label in labels))]


def pose_pnp(detections, geometry, kinematic, camera_matrix, dist_coeffs, frame_mat_inv=np.eye(4), **kwargs):
    # retval
    retval = []

    # keypoint combinations
    keypoint_combinations = keypoint_combinations(detections, list(geometry.keys()))

    # loop over keypoint combinations
    for combination in keypoint_combinations:        
        # init success
        success = False
        
        # object points
        object_points = np.array([
            geometry[d["cls"]] for d in combination
        ], dtype=np.float32)
        
        # image points
        image_points = np.array([
            d["center"] for d in combination
        ], dtype=np.float32)

        if len(image_points) > 3:
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_P3P
            )

        elif len(image_points) == 3:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_P3P
            )


        if success:
            # Convert rvec (Rodrigues) to rotation matrix
            R_target_to_cam, _ = cv2.Rodrigues(rvec)  # 3x3

            # Create 4x4 transformation matrix
            T_target_to_cam = np.eye(4, dtype=np.float32)
            T_target_to_cam[:3, :3] = R_target_to_cam
            T_target_to_cam[:3, 3] = tvec.flatten()

            # to robot frame
            T_target_to_frame = np.matmul(frame_mat_inv, T_target_to_cam)
            xyzabc_target_to_frame = kinematic.mat_to_xyzabc(T_target_to_frame).tolist()
            _tvec = xyzabc_target_to_frame[0:3]
            _rvec = xyzabc_target_to_frame[3:6]

            retval.append({
                "rvec": _rvec,
                "tvec": _tvec
            })

    return retval


def project_to_plane(point, plane_coefficients):
    """
    Project a 3D point to a plane defined by coefficients.
    """
    x, y, z = point
    a, b, c, d = plane_coefficients
    z_plane = -(a * x + b * y + d) / c
    return np.array([x, y, z_plane])

def ransac_plane_fitting(xyzs, residual_threshold=5.0):
    """
    Fit a plane to 3D points using RANSAC while preserving inliers.
    
    Args:
        xyzs (list of tuples): 3D points (x, y, z).
        residual_threshold (float): The maximum distance for a point to be considered an inlier.

    Returns:
        tuple: (plane_coefficients, xyz_plane, inlier_mask)
            - plane_coefficients (tuple): Coefficients (a, b, c, d) of the plane equation.
            - xyz_plane (np.ndarray): Points projected onto the fitted plane.
            - inlier_mask (np.ndarray): Boolean mask of inliers.
    """
    xyzs = np.array(xyzs)  # Convert to a numpy array for easier slicing
    x, y = xyzs[:, :2], xyzs[:, 2]  # Use x, y as features and z as the target

    # Set up RANSAC with the specified residual threshold
    ransac = RANSACRegressor(residual_threshold=residual_threshold, random_state=42)
    poly = PolynomialFeatures(degree=1)  # For a linear plane: z = ax + by + c
    x_poly = poly.fit_transform(x)
    ransac.fit(x_poly, y)

    inlier_mask = ransac.inlier_mask_  # True for inliers, False for outliers

    # Extract plane coefficients
    coef = ransac.estimator_.coef_  # [intercept, a, b]
    intercept = ransac.estimator_.intercept_
    a, b = coef[1], coef[2]
    c, d = -1, intercept  # Assume z-coefficient as -1

    plane_coefficients = (a, b, c, d)

    return plane_coefficients, inlier_mask


def pose_3_point(depth_frame, depth_int, tmp_pxls, center, dim, rot, camera):
    # init
    valid = 0
    X = np.zeros(3)
    Y = np.zeros(3)
    Z = np.zeros(3)
    center_3d = np.zeros(3)

    # rotation matrix
    _cos = np.cos(np.radians(rot))
    _sin = np.sin(np.radians(rot))
    screen_x = np.array([_cos , _sin])
    screen_y = np.array([-_sin , _cos])

    # tmp to pxl: pick only the first 3
    pxls_list = np.array([center + 0.5 * dim[0] * tmp[0] * screen_x + 0.5 * dim[1] * tmp[1] * screen_y for tmp in tmp_pxls])

    # xyz
    xyzs = []
    pxls = []
    for pxl in pxls_list:
        tmp = camera.xyz(pxl, depth_frame, depth_int)[0]
        if tmp[2] > 0:
            pxls.append(pxl)
            xyzs.append(tmp)
    if len(xyzs) > 2:
        try:
            if len(xyzs) > 3:
                # ransac
                plane, inlier_mask = ransac_plane_fitting(xyzs, residual_threshold=0.5)

                # adjust
                A = []
                B = []
                for i in range(len(tmp_pxls)):
                    if len(A) > 2: 
                        break
                    if inlier_mask[i]:
                        A.append(tmp_pxls[i])
                        B.append(project_to_plane(xyzs[i], plane))
                tmp_pxls = A
                xyzs = B

            # tmp to pxl: pick only the first 3
            tmp_pxl_21 = tmp_pxls[1] - tmp_pxls[0]
            tmp_pxl_31 = tmp_pxls[2] - tmp_pxls[0]

            #project center and center + dx and center +dy vectors onto 3d space and onto the plane
            g1 = (tmp_pxl_21[1] *tmp_pxl_31[0] - tmp_pxl_21[0]* tmp_pxl_31[1])
            g2 = -(tmp_pxls[0][1]* tmp_pxl_31[0]- tmp_pxls[0][0]* tmp_pxl_31[1])
            g3 = -(-tmp_pxls[0][1]* tmp_pxl_21[0] + tmp_pxls[0][0]* tmp_pxl_21[1])
            center_3d = xyzs[0]  + g2*(xyzs[1]-xyzs[0])/ g1  + g3*(xyzs[2]-xyzs[0]) /g1

            # X
            X = (xyzs[1]-xyzs[0])*(-tmp_pxl_31[1])/g1 + (xyzs[2]-xyzs[0])*(tmp_pxl_21[1])/g1
            X = X/np.linalg.norm(X)
            
            # Z
            Z = np.cross(xyzs[2] - xyzs[0], xyzs[1] - xyzs[0])
            Z = Z / np.linalg.norm(Z)

            if Z[2] <= 0: #  z is always positive
                Z = -Z
        
            # Y
            Y = np.cross(Z,X)

            valid = 1
        except Exception as e:
            pass
    return valid, center_3d, X, Y , Z, pxls



def pose_3_point_backup(depth_frame, depth_int, tmp_pxls, center, dim, rot, camera):
    valid = 0
    X = np.zeros(3)
    Y = np.zeros(3)
    Z = np.zeros(3)
    center_3d = np.zeros(3)

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
    xyzs = []
    for pxl in pxls:
        tmp = camera.xyz(pxl, depth_frame, depth_int)[0]
        if tmp[2] > 0:
            xyzs.append(tmp)

    if len(xyzs) > 2:
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
            pass
    return valid, center_3d, X, Y , Z, pxls



def camera_to_dorna_ta(T_target_2_cam, joint, kinematic, T_cam_2_j4, T_robot_2_frame=np.eye(4)):    
    # current joint and pose
    T_j4_2_base = kinematic.Ti_r_world(i=5, joint=joint[0:6])
    
    # target_2_base
    T_target_2_base = np.matmul(T_j4_2_base, np.matmul(T_cam_2_j4, T_target_2_cam) )
    
    # target_2_frame
    return np.matmul(T_robot_2_frame, T_target_2_base)