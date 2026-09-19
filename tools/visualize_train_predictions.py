"""
Run inference on selected training frames using a checkpoint and visualize
predictions on fisheye images with per-view visibility filtering.

Projection logic strictly follows tools/visualize_dataset.py:
    corners_cam = R_l2c @ corners_ego + t_l2c
    corners_2d  = fisheye_project_pts(corners_cam, fx, fy, cx, cy, k)

A box is only drawn on a camera view if at least part of its projected
wireframe falls inside that view.

Usage:
    cd /root/ly/BEV/Sparse4D
    source /root/ly/BEV/mm_sparse4d/bin/activate
    export PYTHONPATH=/root/ly/BEV/Sparse4D:$PYTHONPATH
    python tools/visualize_train_predictions.py \
        work_dirs/fisheye_sparse4d_unfreeze_v2_120k_ffn_refine/fisheye_v3_r50_4x_unfreeze_v2.py \
        work_dirs/fisheye_sparse4d_unfreeze_v2_120k_ffn_refine/best_img_bbox_NuScenes/NDS_iter_107370.pth\
        --out-dir work_dirs/fisheye_sparse4d_unfreeze_v2_120k_ffn_refine/visual_train_pred \
        --num-frames 20 \
        --score-thr 0.05
"""
import argparse
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


RAW_CAM_KEYS = [
    "cam_hy_n5_avm_front",
    "cam_hy_n5_avm_back",
    "cam_hy_n5_avm_left",
    "cam_hy_n5_avm_right",
]
CAM_LABELS = ["FRONT", "BACK", "LEFT", "RIGHT"]

# BGR colors
TYPE_COLORS = [
    (59, 59, 238),    # 0 car
    (0, 255, 0),      # 1 truck
    (0, 0, 255),      # 2 construction_vehicle
    (255, 255, 0),    # 3 bus
    (0, 255, 255),    # 4 trailer
    (255, 0, 255),    # 5 barrier
    (255, 255, 255),  # 6 motorcycle
    (0, 127, 255),    # 7 bicycle
    (71, 130, 255),   # 8 pedestrian
    (127, 127, 0),    # 9 traffic_cone
]

class SubsetDataset:
    """Wrap a dataset so that single_gpu_test only sees selected indices."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def fisheye_project_pts(pts_cam, fx, fy, cx, cy, k):
    """Fisheye equidistant projection (same as tools/visualize_dataset.py)."""
    z = pts_cam[:, 2]
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
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
    """Convert [x,y,z,w,l,h,yaw,...] to 8 corners in ego/lidar frame."""
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
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
    return corners


def project_box_to_fisheye(corners_ego, R_l2c, t_l2c, fx, fy, cx, cy, k, W, H, n_pts_per_edge=30):
    """Project 3D box corners and edges to fisheye image with dense sampling.

    A box is only drawn on a camera if at least part of one edge is both
    *in front of the camera* (z > 0) and *inside the image*.  This avoids the
    behind-camera radial streak artifact common in fisheye visualization.

    Returns (edge_curves, corners_2d, visible_flag).
    """
    corners_cam = (R_l2c @ corners_ego.T).T + t_l2c
    corners_2d = fisheye_project_pts(corners_cam, fx, fy, cx, cy, k)

    edge_curves = []
    any_visible = False
    for i, j in EDGES:
        p0, p1 = corners_ego[i], corners_ego[j]
        t = np.linspace(0, 1, n_pts_per_edge)
        samples_ego = p0 + t[:, None] * (p1 - p0)
        samples_cam = (R_l2c @ samples_ego.T).T + t_l2c
        pts = fisheye_project_pts(samples_cam, fx, fy, cx, cy, k)
        visible = (
            (samples_cam[:, 2] > 0) &
            (pts[:, 0] >= 0) & (pts[:, 0] < W) &
            (pts[:, 1] >= 0) & (pts[:, 1] < H)
        )
        if visible.sum() >= 2:
            edge_curves.append(pts[visible].astype(np.int32))
            any_visible = True
        else:
            edge_curves.append(None)

    return edge_curves, corners_2d.astype(np.int32), any_visible


def draw_bev(bboxes_3d, bev_size=512, bev_range=80, colors=None):
    """Draw BEV view of 3D boxes."""
    bev = np.zeros([bev_size, bev_size, 3], dtype=np.uint8)
    res = bev_range / bev_size
    center = bev_size // 2

    for r in range(10, bev_range // 2 + 1, 10):
        cv2.circle(bev, (center, center), int(r / res), (80, 80, 80), 1)
    cv2.line(bev, (0, center), (bev_size, center), (80, 80, 80), 1)
    cv2.line(bev, (center, 0), (center, bev_size), (80, 80, 80), 1)

    if len(bboxes_3d) == 0:
        return bev

    corners = box3d_to_corners(bboxes_3d)
    bev_corners = corners[:, [0, 3, 4, 7]][..., [0, 1]]
    xs = bev_corners[..., 0] / res + center
    ys = -bev_corners[..., 1] / res + center

    for obj_idx, (x, y) in enumerate(zip(xs, ys)):
        color = colors[obj_idx] if colors is not None else (0, 255, 0)
        for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
            cv2.line(bev, (int(x[p1]), int(y[p1])), (int(x[p2]), int(y[p2])), color, 2)
    return bev


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on training fisheye images."
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out-dir", default="work_dirs/fisheye_sparse4d/visual_train_pred",
                        help="directory to save visualization")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="number of random frames to visualize")
    parser.add_argument("--frame-idx", type=int, nargs="+", default=None,
                        help="if provided, visualize these specific frame indices")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for sampling frames")
    parser.add_argument("--score-thr", type=float, default=0.05,
                        help="score threshold for visualization")
    parser.add_argument("--max-per-frame", type=int, default=50,
                        help="max boxes to draw per frame")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
        importlib.import_module(_module_path)

    # Build a test-mode dataset that reads train annotations but uses the
    # deterministic test pipeline/augmentation for visualization.
    ds_cfg = cfg.data.test.copy()
    ds_cfg["ann_file"] = cfg.data.train["ann_file"]
    dataset = build_dataset(ds_cfg)
    print(f"Train dataset size: {len(dataset)}")

    # Load fisheye calibration
    calib_path = os.path.join(
        os.path.dirname(cfg.data.train["ann_file"]), "fisheye_calib_4cam.npz"
    )
    if not os.path.exists(calib_path):
        calib_path = "fisheye_calib_4cam.npz"
    calib = np.load(calib_path, allow_pickle=True)
    calib_K = calib["K"].astype(np.float64)
    calib_D = calib["D"].astype(np.float64)
    calib_lidar2cam = calib["lidar2cam"].astype(np.float64)
    calib_order = list(calib["camera_order"])
    calib_image_wh = calib.get("image_wh", None)

    dataset_order = list(ds_cfg.get("camera_order", RAW_CAM_KEYS))
    calib_idx_map = [calib_order.index(c) for c in dataset_order]
    K = calib_K[calib_idx_map]
    D = calib_D[calib_idx_map]
    lidar2cam = calib_lidar2cam[calib_idx_map]
    image_wh = calib_image_wh[calib_idx_map] if calib_image_wh is not None else None

    # Build model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    # Choose frame indices
    if args.frame_idx is not None:
        indices = args.frame_idx
    else:
        np.random.seed(args.seed)
        indices = np.random.choice(len(dataset), args.num_frames, replace=False)
        indices = sorted(indices.tolist())

    print(f"Visualizing {len(indices)} frames: {indices[:10]}{'...' if len(indices) > 10 else ''}")

    # Use single_gpu_test on a subset for efficient inference
    subset = SubsetDataset(dataset, indices)
    data_loader = build_dataloader(
        subset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    print("Running inference on selected frames...")
    outputs = single_gpu_test(model, data_loader, False, None)
    print(f"Got {len(outputs)} predictions")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving visualization to {args.out_dir}")

    for fidx, idx in enumerate(indices):
        pred = outputs[fidx]["img_bbox"]
        scores = pred["scores_3d"].numpy()
        mask = scores >= args.score_thr
        bboxes = pred["boxes_3d"][mask].numpy()
        labels = pred["labels_3d"][mask].numpy()
        scores = scores[mask]

        # Keep top predictions if too many
        if len(bboxes) > args.max_per_frame:
            topk = np.argsort(scores)[::-1][:args.max_per_frame]
            bboxes = bboxes[topk]
            labels = labels[topk]
            scores = scores[topk]

        info = dataset.data_infos[idx]
        cam_names = list(info["cams"].keys())

        if len(bboxes) == 0:
            print(f"  frame {idx}: no predictions above threshold")
            continue

        # Normalize yaw
        bboxes[:, 6] = (
            bboxes[:, 6] - np.floor(bboxes[:, 6] / (2 * np.pi) + 0.5) * (2 * np.pi)
        )

        colors = [TYPE_COLORS[int(l) % len(TYPE_COLORS)] for l in labels]

        imgs = []
        for cam_idx, cam_name in enumerate(cam_names):
            img_path = info["cams"][cam_name]["data_path"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(ds_cfg["data_root"], img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: cannot read {img_path}")
                imgs.append(np.zeros((256, 704, 3), dtype=np.uint8))
                continue

            H, W = img.shape[:2]
            if image_wh is not None:
                W, H = int(image_wh[cam_idx][0]), int(image_wh[cam_idx][1])

            fx, fy = K[cam_idx][0, 0], K[cam_idx][1, 1]
            cx, cy = K[cam_idx][0, 2], K[cam_idx][1, 2]
            k = D[cam_idx]
            R_l2c = lidar2cam[cam_idx][:3, :3]
            t_l2c = lidar2cam[cam_idx][:3, 3]

            corners = box3d_to_corners(bboxes)
            for bi in range(len(bboxes)):
                edge_curves, corners_2d, visible = project_box_to_fisheye(
                    corners[bi], R_l2c, t_l2c, fx, fy, cx, cy, k, W, H
                )
                if not visible:
                    continue
                color = colors[bi]
                for curve in edge_curves:
                    if curve is not None:
                        cv2.polylines(img, [curve.reshape((-1, 1, 2))], False, color, 2)

            cv2.putText(
                img, CAM_LABELS[cam_idx], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            imgs.append(img)

        # Resize to same size and concatenate
        min_h = min(img.shape[0] for img in imgs)
        min_w = min(img.shape[1] for img in imgs)
        imgs = [cv2.resize(img, (min_w, min_h)) for img in imgs]

        top = np.concatenate(imgs[:2], axis=1)
        bot = np.concatenate(imgs[2:], axis=1)
        image = np.concatenate([top, bot], axis=0)

        bev = draw_bev(bboxes, bev_size=image.shape[0], bev_range=80, colors=colors)
        image = np.concatenate([image, bev], axis=1)

        out_path = os.path.join(args.out_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, image)
        print(f"  [{fidx + 1}/{len(indices)}] frame {idx}: {len(bboxes)} boxes saved {out_path}")

    print("Visualization done.")


if __name__ == "__main__":
    main()
