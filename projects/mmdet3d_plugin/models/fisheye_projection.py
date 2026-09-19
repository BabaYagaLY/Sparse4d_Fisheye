import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.runner.base_module import Sequential


# ============================================================
# Fisheye projection: standard equidistant model
#   theta = atan2(r, z)  (supports z<0 for FOV > 180°)
#   theta_d = theta * (1 + k1*th^2 + k2*th^4 + k3*th^6 + k4*th^8)
#   u = fx * (theta_d / r) * x + cx
# ============================================================

def fisheye_project(
    points_3d, camera_extrinsic, K, D, image_wh=None,
    projection_mode="fisheye",
):
    """
    Project 3D LiDAR points to 2D fisheye image plane.

    Args:
        points_3d: (bs, num_anchor, num_pts, 3) in LiDAR coordinates
        camera_extrinsic: (bs, num_cams, 4, 4) LiDAR-to-camera transform
        K: (bs, num_cams, 3, 3) intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
        D: (bs, num_cams, 4) distortion coeffs [k1,k2,k3,k4]
        image_wh: (bs, num_cams, 2) [W,H] for normalization
        projection_mode: "fisheye" (equidistant + poly distortion)

    Returns:
        points_2d: (bs, num_cams, num_anchor, num_pts, 2) in pixel or normalized
    """
    return _fisheye_project_equidistant(points_3d, camera_extrinsic, K, D, image_wh)


def _transform_lidar_to_camera(points_3d, camera_extrinsic):
    """Transform LiDAR coords to each camera's coordinate system.

    camera_extrinsic is the 4x4 LiDAR-to-camera transform [R|t; 0|1],
    so the homogeneous matrix multiply already yields R*p + t.
    """
    pts_extend = torch.cat(
        [points_3d, torch.ones_like(points_3d[..., :1])], dim=-1
    )
    pts_cam = torch.matmul(
        camera_extrinsic[:, :, None, None, :3, :],
        pts_extend[:, None, ..., None],
    ).squeeze(-1)
    return pts_cam


# ============================================================
# Equidistant model (r = f * theta_d)
# ============================================================

def _fisheye_project_equidistant(points_3d, camera_extrinsic, K, D, image_wh=None):
    """
    Standard equidistant fisheye:
        theta = atan2(r, z)  — handles z<0 naturally (FOV > 180°)
        theta_d = theta * (1 + k1*th^2 + k2*th^4 + k3*th^6 + k4*th^8)
        u = fx * (theta_d / r) * x + cx

    Numerical safety:
      1. All intermediate compute done in FP32 to prevent fp16 underflow
      2. Clamp uv to [-2, 2] to prevent grid_sample overflow
      3. Points behind camera (z<=0) → uv=-1 (border, 0 gradient)
    """
    orig_dtype = points_3d.dtype

    # === 1. Coordinate transform in FP32 ===
    pts_cam = _transform_lidar_to_camera(
        points_3d.float(), camera_extrinsic.float()
    )
    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    z = pts_cam[..., 2]

    # === 2. Radial distance (FP32 safe: 1e-12 is representable) ===
    r = torch.sqrt(x * x + y * y).clamp(min=1e-12)

    # === 3. Fisheye angle ===
    theta = torch.atan2(r, z)
    # Allow fisheye FOV up to ~359 deg; keep a hard safety margin just below pi
    # to avoid the atan2 discontinuity/singularity.
    theta = theta.clamp(min=-3.13, max=3.13)

    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta6 * theta2

    k1 = D[..., 0].float()
    k2 = D[..., 1].float()
    k3 = D[..., 2].float()
    k4 = D[..., 3].float()

    theta_d = theta * (
        1.0
        + k1[:, :, None, None] * theta2
        + k2[:, :, None, None] * theta4
        + k3[:, :, None, None] * theta6
        + k4[:, :, None, None] * theta8
    )

    # === 4. Projection (FP32) ===
    scale = theta_d / r
    fx = K[:, :, None, None, 0, 0].float()
    fy = K[:, :, None, None, 1, 1].float()
    cx = K[:, :, None, None, 0, 2].float()
    cy = K[:, :, None, None, 1, 2].float()

    u = fx * scale * x + cx
    v = fy * scale * y + cy
    points_2d = torch.stack([u, v], dim=-1)

    if image_wh is not None:
        points_2d = points_2d / image_wh[:, :, None, None].float()

    # === 5. Mask points behind the camera (z <= 0) as off-screen.
    # This is per-camera: a point behind camera i should not contribute to
    # camera i's deformable attention, but may still be valid for camera j.
    in_front = (z > 0).unsqueeze(-1)
    points_2d = torch.where(
        in_front,
        points_2d,
        points_2d.new_tensor(-1.0),
    )

    # === 6. Safety: replace any NaN/Inf with off-screen marker (-1) ===
    points_2d = torch.nan_to_num(points_2d, nan=-1.0, posinf=-1.0, neginf=-1.0)
    points_2d = points_2d.clamp(-2.0, 2.0)

    # === 7. Cast back to original dtype (fp16/bf16/fp32) ===
    return points_2d.to(orig_dtype)


# ============================================================
# Camera encoder
# ============================================================

def build_fisheye_camera_encoder(embed_dims=256, in_channels=8):
    """MLP to encode fisheye camera parameters."""
    return Sequential(
        Linear(in_channels, embed_dims),
        nn.ReLU(inplace=True),
        Linear(embed_dims, embed_dims),
        nn.ReLU(inplace=True),
        nn.LayerNorm(embed_dims),
    )


# ============================================================
# Calibration: quaternion + translation → 4x4 extrinsic matrix
# ============================================================

def quaternion_to_rotation(qx, qy, qz, qw):
    """Quaternion [qx,qy,qz,qw] → 3x3 rotation matrix."""
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
    R[0, 1] = 2 * qx * qy - 2 * qz * qw
    R[0, 2] = 2 * qx * qz + 2 * qy * qw
    R[1, 0] = 2 * qx * qy + 2 * qz * qw
    R[1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
    R[1, 2] = 2 * qy * qz - 2 * qx * qw
    R[2, 0] = 2 * qx * qz - 2 * qy * qw
    R[2, 1] = 2 * qy * qz + 2 * qx * qw
    R[2, 2] = 1 - 2 * qx * qx - 2 * qy * qy
    return R


def build_extrinsic_from_cam2ego(extrinsic_list, camera_order=None):
    """
    Build LiDAR→Camera 4x4 extrinsics from camera-to-ego parameters.

    Args:
        extrinsic_list: list of [tx, ty, tz, qx, qy, qz, qw] per camera
        camera_order: list of camera names for ordering

    Returns:
        lidar2cam: (num_cams, 4, 4) LiDAR-to-camera transforms
    """
    if camera_order is None:
        camera_order = list(extrinsic_list.keys())

    lidar2cam_mats = []
    for cam_name in camera_order:
        ext = extrinsic_list[cam_name]  # [tx, ty, tz, qx, qy, qz, qw]
        tx, ty, tz = ext[0], ext[1], ext[2]
        qx, qy, qz, qw = ext[3], ext[4], ext[5], ext[6]

        # Camera-to-ego/LiDAR transform
        R_cam2ego = quaternion_to_rotation(qx, qy, qz, qw)
        t_cam2ego = np.array([tx, ty, tz])

        # Invert to get LiDAR-to-camera
        R_lidar2cam = R_cam2ego.T
        t_lidar2cam = -R_lidar2cam @ t_cam2ego

        lidar2cam = np.eye(4, dtype=np.float32)
        lidar2cam[:3, :3] = R_lidar2cam
        lidar2cam[:3, 3] = t_lidar2cam
        lidar2cam_mats.append(lidar2cam)

    return np.stack(lidar2cam_mats, axis=0)


def build_camera_params(camera_cfgs, camera_order=None):
    """
    Build K, D, extrinsics arrays from camera calibration dicts.

    Args:
        camera_cfgs: dict of camera_name → {focal, pp, inv_poly, extrinsic}
        camera_order: list of camera names

    Returns:
        K: (num_cams, 3, 3) intrinsics
        D: (num_cams, 4) inv_poly coefficients
        lidar2cam: (num_cams, 4, 4) LiDAR→Camera extrinsics
        image_wh: (num_cams, 2) [W, H]
    """
    if camera_order is None:
        camera_order = list(camera_cfgs.keys())

    num_cams = len(camera_order)
    K = np.zeros((num_cams, 3, 3), dtype=np.float32)
    D = np.zeros((num_cams, 4), dtype=np.float32)
    image_wh = np.zeros((num_cams, 2), dtype=np.float32)

    extrinsic_dict = {}
    for i, cam_name in enumerate(camera_order):
        cfg = camera_cfgs[cam_name]
        fx, fy = cfg["focal"]
        cx, cy = cfg["pp"]
        W, H = cfg["image_size"]

        K[i, 0, 0] = fx
        K[i, 1, 1] = fy
        K[i, 0, 2] = cx
        K[i, 1, 2] = cy
        K[i, 2, 2] = 1.0

        inv_poly = cfg["inv_poly"]
        for j in range(min(len(inv_poly), 4)):
            D[i, j] = inv_poly[j]

        image_wh[i, 0] = W
        image_wh[i, 1] = H

        extrinsic_dict[cam_name] = cfg["extrinsic"]

    lidar2cam = build_extrinsic_from_cam2ego(extrinsic_dict, camera_order)
    return K, D, lidar2cam, image_wh
