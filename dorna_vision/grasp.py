import numpy as np
import cv2
from dorna2 import pose as dorna_pose

class BB:
    @staticmethod
    def elp(bbs, num_pts=100):
        angles = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
        ellipses = []
        for bb in bbs:
            arr = np.array(bb)
            if arr.ndim == 1 and arr.size == 4:
                x1, y1, x2, y2 = arr
            else:
                xs, ys = arr[:,0], arr[:,1]
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            rx, ry = abs(x2 - x1)/2, abs(y2 - y1)/2
            pts = np.vstack([
                cx + rx * np.cos(angles),
                cy + ry * np.sin(angles)
            ]).T.astype(np.int32)
            ellipses.append(pts)
        return ellipses

    @staticmethod
    def seg(bbs, img_bgr, iter_count=5, pad_frac=0.25, morph_k=5):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        all_polys = []
        for (x, y, w, h) in bbs:
            roi = img_rgb[y:y+h, x:x+w]
            mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
            mask[0,:] = mask[-1,:] = mask[:,0] = mask[:,-1] = cv2.GC_BGD
            p = int(min(w, h) * pad_frac)
            mask[p:h-p, p:w-p] = cv2.GC_FGD
            bgd, fgd = np.zeros((1,65)), np.zeros((1,65))
            cv2.grabCut(roi, mask, None, bgd, fgd, iter_count, mode=cv2.GC_INIT_WITH_MASK)
            m2 = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
            m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, kern)
            m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, kern)
            cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                all_polys.append(cnt.reshape(-1, 2) + np.array([x, y]))
        return all_polys


# -------------------------
# Vectorized point→segment distances
# -------------------------
def point_segments_distances(pts, v0, seg, seg_len2):
    w   = pts[:,None,:] - v0[None,:,:]                   # (N,S,2)
    dot = (w * seg[None,:,:]).sum(axis=2)                # (N,S)
    t   = np.clip(dot/seg_len2[None,:], 0.0, 1.0)         # (N,S)
    c   = v0[None,:,:] + t[:,:,None]*seg[None,:,:]       # (N,S,2)
    d2  = ((pts[:,None,:] - c)**2).sum(axis=2)           # (N,S)
    return np.sqrt(d2.min(axis=1))                       # (N,)


# -------------------------
def project_center_to_pixel(rvec, tvec, K, dist_coeffs=None):
    pts2d, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, dist_coeffs)
    u, v = pts2d.squeeze().tolist()
    return float(u), float(v)


class Gripper:
    def __init__(self, radius_mm, finger_width_mm, finger_angles_deg):
        self.radius_mm         = radius_mm
        self.finger_width_mm   = finger_width_mm
        self.finger_angles_deg = finger_angles_deg

# -------------------------
# Fast, vectorized grasp‐candidate search
# -------------------------
def find_grasp_candidates_3d(target_id, detections,
                             rvec, tvec, 
                             K, dist_coeffs,
                             gripper_cfg,
                             img_bgr, 
                             mask_type="bb", prune_factor=2.0,
                             num_steps=360, search_angle=(0, 360), thr_mm=None):

    # 1) image‐center + scale
    u0,v0 = project_center_to_pixel(rvec, tvec, K, dist_coeffs)
    px_per_mm  = K[0,0]/tvec[2]
    r_px   = gripper_cfg.radius_mm * px_per_mm
    w2_px  = (gripper_cfg.finger_width_mm * px_per_mm)/2
    thr_px = (thr_mm * px_per_mm) if thr_mm else w2_px

    # 2) collect & prune obstacles
    all_obs = [np.array(d["corners"], float)
               for d in detections if d["id"] != target_id]
    
    # if the list is empty, return the default rvec
    if len(all_obs) == 0:
        return [{
            "angle": 0.0,
            "score_mm": 9999.0,
            "rvec": rvec.ravel().tolist()
        }]
    thresh_px = prune_factor * r_px
    obs_pruned = [poly for poly in all_obs
        if np.any(np.hypot(poly[:,0]-u0, poly[:,1]-v0) <= thresh_px)]
    if not obs_pruned:
        obs_pruned = all_obs

    # 2a) mask type
    if mask_type == "elp":
        obs_pruned = BB.elp(obs_pruned, num_pts=100)
    elif mask_type == "seg":
        bb_list = []
        for poly in obs_pruned:
            x1 = int(poly[:,0].min())
            y1 = int(poly[:,1].min())
            w = int(poly[:,0].max() - poly[:,0].min())
            h = int(poly[:,1].max() - poly[:,1].min())
            bb_list.append((x1, y1, w, h))
        obs_pruned = BB.seg(bb_list, img_bgr)

    # 3) build global segment list
    v0_list, seg_list = [], []
    for poly in obs_pruned:
        verts = poly
        if not np.allclose(verts[0], verts[-1]):
            verts = np.vstack([verts, verts[0]])
        v0_list.append(verts[:-1])
        seg_list.append(verts[1:] - verts[:-1])
    v0_all       = np.vstack(v0_list)      # (S,2)
    seg_all      = np.vstack(seg_list)     # (S,2)
    seg_len2_all = (seg_all**2).sum(axis=1) # (S,)

    # --- ADD THIS: drop any zero‐length segments ---
    nonzero_mask = seg_len2_all > 0
    v0_all       = v0_all[nonzero_mask]
    seg_all      = seg_all[nonzero_mask]
    seg_len2_all = seg_len2_all[nonzero_mask]

    # 4) camera‐frame axes
    R_mat,_  = cv2.Rodrigues(rvec)
    axis_cam = R_mat[:,2]; axis_cam /= np.linalg.norm(axis_cam)
    tmp      = np.array([0,0,1.0]) if abs(axis_cam[2])<0.9 else np.array([0,1,0])
    e1       = np.cross(axis_cam, tmp); e1 /= np.linalg.norm(e1)
    e2       = np.cross(axis_cam, e1)

    # 5) generate all fingertip rays
    #thetas = np.linspace(0,2*np.pi,num_steps,endpoint=False)  # (T,)
    theta_min = np.deg2rad(search_angle[0])
    theta_max = np.deg2rad(search_angle[1])
    if theta_max < theta_min:
        theta_max += 2*np.pi  # wrap around

    thetas_full = np.linspace(0, 2*np.pi, num_steps, endpoint=False)
    thetas = thetas_full[(thetas_full >= theta_min) & (thetas_full <= theta_max)]

    offs   = np.deg2rad(gripper_cfg.finger_angles_deg)       # (F,)
    T, F   = len(thetas), len(offs)
    th_off = thetas[:,None] + offs[None,:]                    # (T,F)
    dir_cam = (np.cos(th_off)[...,None]*e1 +
               np.sin(th_off)[...,None]*e2)                  # (T,F,3)
    contacts = tvec.ravel()[None,None,:] + gripper_cfg.radius_mm*dir_cam  # (T,F,3)

    # 6) project all contacts
    X,Y,Z = contacts[...,0], contacts[...,1], contacts[...,2]
    us = (K[0,0]*X/Z) + K[0,2]
    vs = (K[1,1]*Y/Z) + K[1,2]
    pts = np.stack([us,vs],axis=2).reshape(-1,2)              # (T*F,2)

    # 7) compute distances in one shot
    dists = point_segments_distances(pts, v0_all, seg_all, seg_len2_all)
    dists = dists.reshape(T, F)                               # (T,F)
    scores= dists.min(axis=1)                                 # (T,)
    """
    # 8) reject any θ whose fingertip lands inside an obstacle
    inside = np.zeros(pts.shape[0], bool)
    for poly in all_obs:
        inside |= Path(poly).contains_points(pts)
    inside = inside.reshape(T, F)
    scores[inside.any(axis=1)] = -np.inf
    """
    # 8a) Reshape fingertips into (T,F,2)
    pts_tf = pts.reshape(T, F, 2).astype(np.float32)

    # 8b) Compute ray vectors from center
    center = np.array([u0, v0], dtype=np.float32)
    rays   = pts_tf - center[None, None, :]

    # 8c) Broadcast rays against all edges
    r    = rays[..., None, :]           # → (T, F, 1, 2)
    s    = seg_all[None, None, :, :]    # → (1, 1, S, 2)
    qmp  = v0_all[None, None, :, :] - center  # → (1, 1, S, 2)

    # 8d) 2D cross‑products
    cross_rs    = r[...,0]*s[...,1] - r[...,1]*s[...,0]   # (T,F,S)
    cross_qmp_s = qmp[...,0]*s[...,1] - qmp[...,1]*s[...,0]  # (T,F,S)
    cross_qmp_r = qmp[...,0]*r[...,1] - qmp[...,1]*r[...,0]  # (T,F,S)

    # 8e) Solve for ray (t) and edge (u) parameters
    mask_valid = cross_rs != 0
    t = np.where(mask_valid, cross_qmp_s / cross_rs, 0.0)
    u = np.where(mask_valid, cross_qmp_r / cross_rs, 0.0)

    # 8f) Intersection occurs when 0≤t≤1 and 0≤u≤1
    intersect = (
        mask_valid &
        (t >= 0) & (t <= 1) &
        (u >= 0) & (u <= 1)
    )  # shape (T, F, S)

    # 8g) Invalidate any angle if any finger intersects any edge
    hit_edge  = intersect.any(axis=2)  # (T, F)
    hit_angle = hit_edge.any(axis=1)   # (T,)
    scores[hit_angle] = -np.inf

    # 9) collect only scores ≥ threshold
    valid = np.nonzero(scores >= thr_px)[0]
    cands = []
    for i in valid:
        theta = thetas[i]
        x_cam = np.cos(theta)*e1 + np.sin(theta)*e2
        y_cam = np.cross(axis_cam, x_cam); y_cam /= np.linalg.norm(y_cam)
        R_cand = np.column_stack((x_cam,y_cam,axis_cam)).astype(np.float32)
        rvec_cand,_ = cv2.Rodrigues(R_cand)
        cands.append({
            "angle":    float(theta),
            "score_mm": float((scores[i] / px_per_mm).item()),
            "rvec":     rvec_cand.ravel().tolist()
        })
    cands.sort(key=lambda x: x["score_mm"], reverse=True)
    return cands


def collision_free_rvec(target_id, target_rvec, gripper_opening, finger_wdith, finger_location, detection_obj, mask_type="bb", prune_factor=2, num_steps=360, search_angle=(0, 360)):        
    best_rvec = None

    try:
        # detection
        target_tvec = [d["xyz"] for d in detection_obj.retval["valid"] if d["id"] == target_id][0]
    
        gripper = Gripper(radius_mm=gripper_opening/2, finger_width_mm=finger_wdith, finger_angles_deg=finger_location)
        
        detections = detection_obj.retval["valid"]
        frame_mat_inv = detection_obj.retval["frame_mat_inv"]
        camera_matrix = detection_obj.camera.camera_matrix(detection_obj.retval["camera_data"]["depth_int"])
        dist_coeffs = detection_obj.camera.dist_coeffs(detection_obj.retval["camera_data"]["depth_int"])
        depth_frme = detection_obj.retval["camera_data"]["depth_frame"]
        img_bgr = detection_obj.retval["camera_data"]["img"]

        
        # T traget to cam
        T_target_to_frame = np.array(dorna_pose.xyzabc_to_T(target_tvec+target_rvec))
        T_target_to_cam = np.linalg.inv(frame_mat_inv) @ T_target_to_frame

        rvec_to_cam, _ = cv2.Rodrigues(T_target_to_cam[:3, :3])
        tvec_to_cam = T_target_to_cam[:3, 3]

        # bring it to the right format
        rvec_to_cam = np.asarray(rvec_to_cam, dtype=np.float64).reshape((3, 1))
        tvec_to_cam = np.asarray(tvec_to_cam, dtype=np.float64).reshape((3, 1))
    
        # Find grasp candidates
        cands = find_grasp_candidates_3d(target_id, detections,
                            rvec_to_cam, tvec_to_cam, 
                            camera_matrix, dist_coeffs,
                            gripper, 
                            img_bgr,
                            mask_type=mask_type, prune_factor=prune_factor,
                            num_steps=num_steps, search_angle=search_angle, thr_mm=gripper.finger_width_mm/2)
        
        if cands:
            # best score
            best_rvec_target_to_cam = [x * 180 / np.pi for x in cands[0]["rvec"]]
            best_T_target_to_cam = dorna_pose.xyzabc_to_T([0, 0, 0]+best_rvec_target_to_cam)
            best_T_target_to_frame = frame_mat_inv @ best_T_target_to_cam
            best_xyzabc_target_to_frame = dorna_pose.T_to_xyzabc(best_T_target_to_frame)
            best_rvec = best_xyzabc_target_to_frame[3:]
            return best_rvec
    except Exception as e:
        pass

    return best_rvec    
    