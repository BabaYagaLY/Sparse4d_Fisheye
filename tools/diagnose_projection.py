"""
Diagnose: compare the model's internal fisheye projection with the manually
projected points used in tools/visualize_dataset.py.

Usage:
    cd /root/ly/BEV/Sparse4D
    source /root/ly/BEV/mm_sparse4d/bin/activate
    export PYTHONPATH=/root/ly/BEV/Sparse4D:$PYTHONPATH
    python tools/diagnose_projection.py \
        work_dirs/fisheye_sparse4d/fisheye_v3_r50_4x_retrain.py \
        --frame-idx 0
"""
import argparse
import os

import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset


def fisheye_project_numpy(pts_cam, fx, fy, cx, cy, k):
    """Same as tools/visualize_dataset.py"""
    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    z = pts_cam[..., 2]
    r = np.sqrt(x * x + y * y) + 1e-12
    theta = np.arctan2(r, z)
    th2 = theta * theta
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4
    theta_d = theta * (1.0 + k[0] * th2 + k[1] * th4 + k[2] * th6 + k[3] * th8)
    scale = theta_d / r
    return np.stack([fx * scale * x + cx, fy * scale * y + cy], axis=-1)


def box3d_to_corners(box3d):
    """[x,y,z,w,l,h,yaw] -> (8,3)"""
    box3d = np.asarray(box3d)
    if box3d.ndim == 1:
        box3d = box3d[None]
    x, y, z = box3d[:, 0], box3d[:, 1], box3d[:, 2]
    w, l, h = box3d[:, 3], box3d[:, 4], box3d[:, 5]
    yaw = box3d[:, 6]

    corners_norm = np.stack(np.unravel_index(np.arange(8), [2, 2, 2]), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = np.stack([
        l[:, None] * corners_norm[:, 0],
        w[:, None] * corners_norm[:, 1],
        h[:, None] * corners_norm[:, 2],
    ], axis=-1)

    rot_cos = np.cos(yaw)
    rot_sin = np.sin(yaw)
    rot_mat = np.tile(np.eye(3)[None], (len(box3d), 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(-1)
    corners += box3d[:, None, :3]
    return corners[0] if len(box3d) == 1 else corners


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--frame-idx", type=int, default=0,
                        help="which training frame to test")
    args = parser.parse_args()
    return args


def to_numpy(x):
    """Handle DataContainer, Tensor, memoryview, ndarray."""
    if hasattr(x, "data"):
        x = x.data
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    elif isinstance(x, memoryview):
        x = np.asarray(x)
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = np.asarray(x)
    return x


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
        importlib.import_module(_module_path)

    ds_cfg = cfg.data.test.copy()
    ds_cfg["ann_file"] = cfg.data.train["ann_file"]
    dataset = build_dataset(ds_cfg)
    data = dataset[args.frame_idx]

    # Get GT boxes from raw info (test pipeline does not load annotations)
    info = dataset.data_infos[args.frame_idx]
    gt_bboxes = info["gt_boxes"][info["num_lidar_pts"] > 0]
    if len(gt_bboxes) == 0:
        print("No GT boxes in this frame, using a synthetic test point.")
        test_points = np.array([[5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    else:
        test_points = gt_bboxes[:5]

    # Get meta info from the processed sample
    camera_intrinsic = to_numpy(data["camera_intrinsic"])  # (4,3,3)
    camera_distortion = to_numpy(data["camera_distortion"])  # (4,4)
    camera_extrinsic = to_numpy(data["camera_extrinsic"])  # (4,4,4)
    image_wh = to_numpy(data["image_wh"])  # (4,2)

    print(f"Frame {args.frame_idx}: {len(test_points)} test boxes")
    print(f"Image WH per cam: {image_wh}")

    # Import model-side projection
    from projects.mmdet3d_plugin.models.fisheye_projection import fisheye_project

    max_err_all = 0.0
    for bi, box in enumerate(test_points):
        corners_ego = box3d_to_corners(box[:7])  # (8,3)
        for cam_idx in range(len(camera_intrinsic)):
            K = camera_intrinsic[cam_idx]
            D = camera_distortion[cam_idx]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            W, H = image_wh[cam_idx]
            R_l2c = camera_extrinsic[cam_idx][:3, :3]
            t_l2c = camera_extrinsic[cam_idx][:3, 3]

            # Manual projection (visualize_dataset.py style)
            corners_cam = (R_l2c @ corners_ego.T).T + t_l2c
            pts2d_manual = fisheye_project_numpy(corners_cam, fx, fy, cx, cy, D)

            # Model-side projection: feed all cameras at once (only once per box)
            pts_tensor = torch.from_numpy(corners_ego).float()[None, None]  # (1,1,8,3)
            K_tensor = torch.from_numpy(camera_intrinsic).float()[None]  # (1,4,3,3)
            D_tensor = torch.from_numpy(camera_distortion).float()[None]  # (1,4,4)
            E_tensor = torch.from_numpy(camera_extrinsic).float()[None]  # (1,4,4,4)
            wh_tensor = torch.from_numpy(image_wh).float()[None]  # (1,4,2)
            pts2d_model = fisheye_project(
                pts_tensor,
                E_tensor,
                K_tensor,
                D_tensor,
                wh_tensor,
            )[0, :, 0].cpu().numpy()  # (4,8,2), normalized by image_wh

            # Convert model output back to pixel coords for comparison
            pts2d_model_px = pts2d_model * wh_tensor.numpy()[0, :, None, :]  # (4,8,2)

        for cam_idx in range(len(camera_intrinsic)):
            K = camera_intrinsic[cam_idx]
            D = camera_distortion[cam_idx]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            W, H = image_wh[cam_idx]
            R_l2c = camera_extrinsic[cam_idx][:3, :3]
            t_l2c = camera_extrinsic[cam_idx][:3, 3]

            # Manual projection (visualize_dataset.py style)
            corners_cam = (R_l2c @ corners_ego.T).T + t_l2c
            pts2d_manual = fisheye_project_numpy(corners_cam, fx, fy, cx, cy, D)

            diff = np.abs(pts2d_manual - pts2d_model_px[cam_idx])
            valid = (
                (corners_cam[:, 2] > 0) &
                (pts2d_manual[:, 0] >= 0) & (pts2d_manual[:, 0] < W) &
                (pts2d_manual[:, 1] >= 0) & (pts2d_manual[:, 1] < H)
            )
            if valid.any():
                max_err = diff[valid].max()
                mean_err = diff[valid].mean()
            else:
                max_err = 0.0
                mean_err = 0.0
            max_err_all = max(max_err_all, max_err)

            in_front = (corners_cam[:, 2] > 0).sum()
            in_img = (
                (pts2d_manual[:, 0] >= 0) & (pts2d_manual[:, 0] < W) &
                (pts2d_manual[:, 1] >= 0) & (pts2d_manual[:, 1] < H)
            ).sum()

            # Debug: show per-corner error for in-front, in-image corners
            if in_front > 0 and in_img > 0:
                r = np.sqrt(corners_cam[:, 0]**2 + corners_cam[:, 1]**2) + 1e-12
                theta = np.arctan2(r, corners_cam[:, 2])
                for ci in range(8):
                    if corners_cam[ci, 2] > 0 and 0 <= pts2d_manual[ci, 0] < W and 0 <= pts2d_manual[ci, 1] < H:
                        if theta[ci] < 1.66:
                            print(
                                f"    corner {ci}: manual={list(pts2d_manual[ci])}, "
                                f"model={list(pts2d_model_px[cam_idx][ci])}, "
                                f"err={diff[ci].max():.4f}, theta={float(theta[ci]):.4f}"
                            )

            print(
                f"  box {bi} cam {cam_idx}: manual vs model max_err={max_err:.4f}px "
                f"mean_err={mean_err:.4f}px, corners_in_front={in_front}/8, "
                f"manual_in_image={in_img}/8"
            )

    print(f"\nOverall max projection error: {max_err_all:.4f} px")
    if max_err_all < 1.0:
        print("=> Model projection matches visualize_dataset.py well.")
    else:
        print("=> WARNING: projection mismatch detected!")


if __name__ == "__main__":
    main()
