import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import cv2
from collections import defaultdict
from itertools import product


class PNP(object):
    def keypoint_combinations(self, kp_list):
        # Group keypoints by 'cls'
        grouped = defaultdict(list)
        for kp in kp_list:
            cls = kp.get("cls")
            if cls:
                grouped[cls].append(kp)

        # Use only labels with at least one keypoint
        labels = [label for label in grouped if grouped[label]]

        if not labels:
            return []

        # Return all 1-per-label combinations
        return [
            dict(zip(labels, combo))
            for combo in product(*(grouped[label] for label in labels))
        ]


    def reprojection_error(self, obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs):
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        proj_pts = proj_pts.squeeze()
        img_pts = np.array(img_pts, dtype=np.float32)
        return np.mean(np.linalg.norm(proj_pts - img_pts, axis=1))


    """
    kp_list: [{"center": [x, y], "cls": "label1"}, {"center": [x, y], "cls": "label2"}, ...]
    kp: {"label1": [x1, y1, z1], "label2": [x2, y2, z2], ...}
    """
    def pose(self, kp_list, kp_geometry, kinematic, camera_matrix, dist_coeffs, frame_mat_inv=np.eye(4), thr=5.0, ransac_min_pts=4, **kwargs):

        best_error = float('inf')
        retval = []

        combos = self.keypoint_combinations(kp_list)

        for combo in combos:
            img_pts = []
            obj_pts = []

            for label, kp_data in combo.items():
                img_pts.append(kp_data["center"])
                if label not in kp_geometry:
                    raise KeyError(f"No 3D geometry defined for keypoint label '{label}'")
                obj_pts.append(kp_geometry[label])

            img_pts = np.array(img_pts, dtype=np.float32)
            obj_pts = np.array(obj_pts, dtype=np.float32)

            # Solve PnP
            if len(obj_pts) >= ransac_min_pts:
                # RANSAC+EPNP to get a robust initial pose
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, camera_matrix, dist_coeffs,
                    iterationsCount=2000,
                    reprojectionError=4.0,
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_EPNP
                )

                if not ok or inliers is None or len(inliers) < 4:
                    # fallback or bail out
                    continue

                # Stage 2: refine with all inliers using LM
                inl_obj = obj_pts[inliers[:,0]]
                inl_img = img_pts[inliers[:,0]]

                _, rvec_refined, tvec_refined = cv2.solvePnP(
                    inl_obj, inl_img, camera_matrix, dist_coeffs,
                    rvec, tvec,                   # use the RANSAC solution as a starting guess
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE  # Levenberg–Marquardt
                )
                rvec = rvec_refined
                tvec = tvec_refined
            else:
                continue

            # Compute reprojection error
            err = self.reprojection_error(obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs)
            # Save best result under threshold
            if err < min(thr, best_error):
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()

                T_rf = frame_mat_inv @ T
                xyzabc = kinematic.mat_to_xyzabc(T_rf)
                best_error = err

                retval = [
                    xyzabc[3:6].tolist(),  # rvec_target_to_robot
                    xyzabc[:3].tolist(),   # tvec_target_to_robot
                    [np.degrees(rvec[i, 0]) for i in range(3)],         # rvec_target_to_camera
                    tvec.flatten().tolist(),        # tvec_target_to_camera
                    img_pts                # 2D keypoints used
                ]

        return retval


class Plane(object):
    def project_to_plane(self, point, plane_coefficients):
        """
        Project a 3D point to a plane defined by coefficients.
        """
        x, y, z = point
        a, b, c, d = plane_coefficients
        z_plane = -(a * x + b * y + d) / c
        return np.array([x, y, z_plane])


    def ransac_plane_fitting(self, xyzs, residual_threshold=5.0):
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


    def pose_3_point(self, depth_frame, depth_int, tmp_pxls, center, dim, rot, camera):
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
                    plane, inlier_mask = self.ransac_plane_fitting(xyzs, residual_threshold=0.5)

                    # adjust
                    A = []
                    B = []
                    for i in range(len(tmp_pxls)):
                        if len(A) > 2: 
                            break
                        if inlier_mask[i]:
                            A.append(tmp_pxls[i])
                            B.append(self.project_to_plane(xyzs[i], plane))
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
        return center_3d, X, Y , Z, pxls, valid


    def pose(self, corners, plane, kinematic, camera, depth_frame, depth_int, frame_mat_inv=np.eye(4), **kwargs):
        # init
        retval = []

        # tmp pixels from poi
        tmp_pxls = np.array(plane, dtype=np.float32)
        
        # Compute the rotated rectangle from the points
        center, dim, rot = cv2.minAreaRect(np.array(corners, dtype=np.float32))

        # pose from tmp
        center_3d, X, Y, Z, pxl_map, valid = self.pose_3_point(depth_frame, depth_int, tmp_pxls, center, dim, rot, camera)
        if valid: # add pose                    
            tvec_target_to_cam = center_3d.tolist()
            rodrigues, _= cv2.Rodrigues(np.matrix([[X[0], Y[0], Z[0]],
                                    [X[1], Y[1], Z[1]],
                                    [X[2], Y[2], Z[2]]])) 
            
            rvec_target_to_cam = [np.degrees(rodrigues[i, 0]) for i in range(3)]

            # xyz_target_2_cam
            T_target_to_cam = kinematic.xyzabc_to_mat(np.array(tvec_target_to_cam+ rvec_target_to_cam))

            # apply frame
            T_target_to_frame = np.matmul(frame_mat_inv, T_target_to_cam)
            xyzabc_target_to_frame = kinematic.mat_to_xyzabc(T_target_to_frame).tolist()

            # rvec, tvec, rvec_target_to_cam, tvec_target_to_cam, pxl_map
            retval = [xyzabc_target_to_frame[3:6], xyzabc_target_to_frame[0:3], rvec_target_to_cam, tvec_target_to_cam, pxl_map]
                
        return retval


