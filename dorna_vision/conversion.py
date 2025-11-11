import numpy as np
import cv2


def raw_to_flat(img, rvec, K, dist=None, pts_raw=None):
    """
    Flatten a tilted raw image using rotation matrix, intrinsics, and (optional) distortion.
    Optionally maps pixel coordinates from raw → flat.

    Args:
        img      : np.ndarray (H, W, 3)
        rvec     : rotation vector in degrees (object in camera)
        K        : np.ndarray (3, 3) camera intrinsic matrix
        dist     : np.ndarray or None distortion coefficients [k1, k2, p1, p2, k3]
        pts_raw  : np.ndarray (N, 2) optional pixel coords in raw image

    Returns:
        img_flat : flattened (fronto-parallel) image (canvas auto-expanded)
        pts_flat : corresponding coords in flat image (if pts_raw provided)
        H_new    : 3×3 homography matrix (includes translation)
        offset   : (tx, ty) translation offset applied to keep canvas positive
    """
    # --- Step 1: rotation matrix ---
    R = cv2.Rodrigues(np.radians(rvec))[0]
    R_inv = R.T

    # --- Step 2: undistort image ---
    if dist is not None:
        img_undist = cv2.undistort(img, K, dist)
    else:
        img_undist = img.copy()

    # --- Step 3: compute homography (undo tilt) ---
    H = K @ R_inv @ np.linalg.inv(K)
    H /= H[2, 2]

    # --- Step 4: auto-expand canvas to fit warped image ---
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)

    xmin, ymin = np.floor(warped_corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(warped_corners.max(axis=0).ravel()).astype(int)
    tx, ty = -xmin, -ymin  # translation to keep image in positive coords

    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)

    H_new = T @ H
    new_w, new_h = xmax - xmin, ymax - ymin

    img_flat = cv2.warpPerspective(img_undist, H_new, (new_w, new_h), flags=cv2.INTER_LINEAR)

    # --- Step 5: map points if provided ---
    pts_flat = None
    if pts_raw is not None:
        pts_raw = np.array(pts_raw, dtype=np.float32).reshape(-1, 1, 2)

        # Undistort points if needed
        if dist is not None:
            pts_undist = cv2.undistortPoints(pts_raw, K, dist, P=K)
        else:
            pts_undist = pts_raw

        # Apply full homography (including translation)
        pts_undist_h = np.hstack([pts_undist.reshape(-1, 2), np.ones((len(pts_undist), 1))])
        pts_flat_h = (H_new @ pts_undist_h.T).T
        pts_flat = pts_flat_h[:, :2] / pts_flat_h[:, 2:3]

    # ✅ Return all useful outputs for reverse mapping
    return img_flat, pts_flat, H_new, (tx, ty)


def distort_points(pts, K, dist):
    """
    Apply lens distortion to undistorted 2D pixel coordinates.
    Used for re-applying distortion when mapping flat → raw.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalize points
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy

    k1, k2, p1, p2, k3 = dist[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Back to pixel coordinates
    x_pix = fx * x_d + cx
    y_pix = fy * y_d + cy
    return np.stack([x_pix, y_pix], axis=1)


def flat_to_raw(img_flat, rvec, K, dist=None, pts_flat=None, H_flat=None, offset=(0, 0), raw_shape=None):
    """
    Map a flattened (fronto-parallel) image back to the raw (distorted) camera image.
    Optionally maps pixel coordinates from flat → raw.

    Args:
        img_flat : np.ndarray (Hf, Wf, 3)
            Flattened (fronto-parallel) image (possibly larger than raw).
        rvec     : list or np.ndarray (3,)
            Rotation vector in degrees (object→camera).
        K        : np.ndarray (3, 3)
            Camera intrinsic matrix.
        dist     : np.ndarray or None
            Distortion coefficients [k1, k2, p1, p2, k3].
        pts_flat : np.ndarray (N, 2) or None
            Optional pixel coordinates in flat image.
        H_flat   : np.ndarray (3, 3) or None
            Homography returned from raw_to_flat (includes translation).
        offset   : (tx, ty)
            Translation offset applied in raw_to_flat (for safety).
        raw_shape: (H, W) or None
            Original raw image shape — if given, warps to match that canvas.

    Returns:
        img_raw  : np.ndarray
        pts_raw  : np.ndarray or None
    """

    # --- Step 1: if H_flat not given, rebuild it from rvec ---
    if H_flat is None:
        R = cv2.Rodrigues(np.radians(rvec))[0]
        R_inv = R.T
        H_raw = K @ R_inv @ np.linalg.inv(K)
        H_raw /= H_raw[2, 2]
        tx, ty = offset
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float32)
        H_flat = T @ H_raw

    # --- Step 2: invert the flat homography to go flat → raw (undistorted) ---
    H_inv = np.linalg.inv(H_flat)

    # --- Step 3: warp image back to raw (undistorted) coordinates ---
    if raw_shape is not None:
        raw_h, raw_w = raw_shape[:2]
    else:
        raw_h, raw_w = img_flat.shape[:2]

    img_undist = cv2.warpPerspective(img_flat, H_inv, (raw_w, raw_h), flags=cv2.INTER_LINEAR)

    # --- Step 4: reapply distortion if requested ---
    if dist is not None:
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (raw_w, raw_h), cv2.CV_32FC1)
        img_raw = cv2.remap(img_undist, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    else:
        img_raw = img_undist.copy()

    # --- Step 5: map points if provided ---
    pts_raw = None
    if pts_flat is not None:
        pts_flat = np.array(pts_flat, dtype=np.float32)
        pts_h = np.hstack([pts_flat, np.ones((len(pts_flat), 1))])
        pts_undist_h = (H_inv @ pts_h.T).T
        pts_undist = pts_undist_h[:, :2] / pts_undist_h[:, 2:3]
        if dist is not None:
            pts_raw = distort_points(pts_undist, K, dist)
        else:
            pts_raw = pts_undist

    return img_raw, pts_raw