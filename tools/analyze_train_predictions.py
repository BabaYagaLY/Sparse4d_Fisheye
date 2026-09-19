"""
Analyze model predictions vs ground truth on training frames.

Usage:
    cd /root/ly/BEV/Sparse4D
    source /root/ly/BEV/mm_sparse4d/bin/activate
    export PYTHONPATH=/root/ly/BEV/Sparse4D:$PYTHONPATH
    python tools/analyze_train_predictions.py \
        work_dirs/fisheye_sparse4d/fisheye_v3_r50_4x.py \
        work_dirs/fisheye_sparse4d/iter_30000.pth \
        --num-frames 200 --score-thr 0.05
"""
import argparse
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from scipy.optimize import linear_sum_assignment


class SubsetDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def box3d_to_corners(box3d):
    """Convert [x,y,z,w,l,h,yaw,...] to 8 corners in ego frame."""
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


def fisheye_project_pts(pts_cam, fx, fy, cx, cy, k):
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


def compute_visible(corners_ego, R_l2c, t_l2c, fx, fy, cx, cy, k, W, H, n_pts_per_edge=10):
    """Check if any edge point of a 3D box is visible in this camera."""
    EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in EDGES:
        p0, p1 = corners_ego[i], corners_ego[j]
        t = np.linspace(0, 1, n_pts_per_edge)
        samples_ego = p0 + t[:, None] * (p1 - p0)
        samples_cam = (R_l2c @ samples_ego.T).T + t_l2c
        pts = fisheye_project_pts(samples_cam, fx, fy, cx, cy, k)
        visible = (
            (pts[:, 0] >= 0) & (pts[:, 0] < W) &
            (pts[:, 1] >= 0) & (pts[:, 1] < H) &
            (samples_cam[:, 2] > 0)
        )
        if visible.any():
            return True
    return False


def box_visible_in_any_camera(bbox, K, D, lidar2cam, image_wh):
    """Check if a 3D box is visible in at least one camera."""
    corners = box3d_to_corners(bbox[None, :7])[0]
    for cam_idx in range(len(K)):
        fx, fy = K[cam_idx][0, 0], K[cam_idx][1, 1]
        cx, cy = K[cam_idx][0, 2], K[cam_idx][1, 2]
        k = D[cam_idx]
        R_l2c = lidar2cam[cam_idx][:3, :3]
        t_l2c = lidar2cam[cam_idx][:3, 3]
        W, H = int(image_wh[cam_idx][0]), int(image_wh[cam_idx][1])
        if compute_visible(corners, R_l2c, t_l2c, fx, fy, cx, cy, k, W, H):
            return True
    return False


def match_predictions_to_gt(pred_boxes, gt_boxes, distance_threshold=2.0):
    """Bipartite matching based on 3D center distance."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    pred_centers = pred_boxes[:, :3]
    gt_centers = gt_boxes[:, :3]
    dists = np.linalg.norm(pred_centers[:, None] - gt_centers[None], axis=2)

    matches = linear_sum_assignment(dists)
    pred_indices, gt_indices = matches
    match_dists = dists[pred_indices, gt_indices]
    valid = match_dists < distance_threshold

    tp_pred = pred_indices[valid]
    tp_gt = gt_indices[valid]
    fp_mask = np.ones(len(pred_boxes), dtype=bool)
    fp_mask[tp_pred] = False
    fn_mask = np.ones(len(gt_boxes), dtype=bool)
    fn_mask[tp_gt] = False

    return tp_pred, tp_gt, np.where(fp_mask)[0], np.where(fn_mask)[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--distance-thr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
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

    ds_cfg = cfg.data.test.copy()
    ds_cfg["ann_file"] = cfg.data.train["ann_file"]
    dataset = build_dataset(ds_cfg)

    # Load calibration
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

    dataset_order = list(ds_cfg.get("camera_order", [
        "cam_hy_n5_avm_front", "cam_hy_n5_avm_back",
        "cam_hy_n5_avm_left", "cam_hy_n5_avm_right"
    ]))
    calib_idx_map = [calib_order.index(c) for c in dataset_order]
    K = calib_K[calib_idx_map]
    D = calib_D[calib_idx_map]
    lidar2cam = calib_lidar2cam[calib_idx_map]
    image_wh = calib_image_wh[calib_idx_map] if calib_image_wh is not None else None

    # Build model
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    np.random.seed(args.seed)
    indices = np.random.choice(len(dataset), min(args.num_frames, len(dataset)), replace=False)
    indices = sorted(indices.tolist())

    subset = SubsetDataset(dataset, indices)
    data_loader = build_dataloader(
        subset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
    )

    print(f"Running inference on {len(indices)} training frames...")
    outputs = single_gpu_test(model, data_loader, False, None)

    # Collect statistics
    all_scores = []
    tp_scores = []
    fp_scores = []
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_visible_gt = 0
    total_invisible_gt = 0
    per_class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0} for c in range(10)}

    for fidx, idx in enumerate(indices):
        pred = outputs[fidx]["img_bbox"]
        scores = pred["scores_3d"].numpy()
        mask = scores >= args.score_thr
        pred_boxes = pred["boxes_3d"][mask].numpy()
        pred_labels = pred["labels_3d"][mask].numpy()
        scores = scores[mask]

        info = dataset.data_infos[idx]
        gt_boxes = info["gt_boxes"]
        gt_names = info["gt_names"]
        num_pts = info["num_lidar_pts"]
        valid_mask = num_pts > 0
        gt_boxes = gt_boxes[valid_mask]
        gt_names = gt_names[valid_mask]

        # Map GT names to labels
        class_names = cfg.class_names
        gt_labels = np.array([class_names.index(n) if n in class_names else -1 for n in gt_names])
        keep = gt_labels >= 0
        gt_boxes = gt_boxes[keep]
        gt_labels = gt_labels[keep]

        # Normalize yaw
        if len(pred_boxes) > 0:
            pred_boxes[:, 6] = (
                pred_boxes[:, 6] - np.floor(pred_boxes[:, 6] / (2 * np.pi) + 0.5) * (2 * np.pi)
            )

        # Visibility analysis
        visible_mask = np.array([
            box_visible_in_any_camera(b, K, D, lidar2cam, image_wh)
            for b in gt_boxes
        ])
        total_visible_gt += visible_mask.sum()
        total_invisible_gt += (~visible_mask).sum()

        all_scores.extend(scores.tolist())
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        # Per-class GT counts
        for l in gt_labels:
            per_class_stats[int(l)]["gt"] += 1

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            total_fn += len(gt_boxes)
            for l in gt_labels:
                per_class_stats[int(l)]["fn"] += 1
            continue

        tp_pred, tp_gt, fp_idx, fn_idx = match_predictions_to_gt(
            pred_boxes, gt_boxes, args.distance_thr
        )

        total_tp += len(tp_pred)
        total_fp += len(fp_idx)
        total_fn += len(fn_idx)

        for p in tp_pred:
            tp_scores.append(float(scores[p]))
            l = int(pred_labels[p])
            per_class_stats[l]["tp"] += 1
        for p in fp_idx:
            fp_scores.append(float(scores[p]))
            l = int(pred_labels[p])
            per_class_stats[l]["fp"] += 1
        for g in fn_idx:
            l = int(gt_labels[g])
            per_class_stats[l]["fn"] += 1

    print("\n" + "="*60)
    print("Overall matching stats (center distance < %.2fm, score >= %.2f)" % (args.distance_thr, args.score_thr))
    print("="*60)
    print(f"Frames analyzed: {len(indices)}")
    print(f"Total predictions: {total_pred}")
    print(f"Total GT boxes: {total_gt}")
    print(f"  Visible in any camera: {total_visible_gt} ({100*total_visible_gt/total_gt:.1f}%)")
    print(f"  Not visible in any camera: {total_invisible_gt} ({100*total_invisible_gt/total_gt:.1f}%)")
    print(f"True positives: {total_tp}")
    print(f"False positives: {total_fp}")
    print(f"False negatives: {total_fn}")
    if total_tp + total_fp > 0:
        print(f"Precision: {100*total_tp/(total_tp+total_fp):.1f}%")
    if total_tp + total_fn > 0:
        print(f"Recall: {100*total_tp/(total_tp+total_fn):.1f}%")

    print("\nScore distribution:")
    if len(all_scores) > 0:
        print(f"  All pred scores: mean={np.mean(all_scores):.4f}, max={np.max(all_scores):.4f}, min={np.min(all_scores):.4f}, median={np.median(all_scores):.4f}")
    if len(tp_scores) > 0:
        print(f"  TP scores: mean={np.mean(tp_scores):.4f}, max={np.max(tp_scores):.4f}, median={np.median(tp_scores):.4f}")
    if len(fp_scores) > 0:
        print(f"  FP scores: mean={np.mean(fp_scores):.4f}, max={np.max(fp_scores):.4f}, median={np.median(fp_scores):.4f}")

    print("\nPer-class stats:")
    print(f"{'Class':<20} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Recall':>8} {'Precision':>10}")
    for c, name in enumerate(class_names):
        s = per_class_stats[c]
        recall = 100*s["tp"]/(s["tp"]+s["fn"]) if (s["tp"]+s["fn"]) > 0 else 0
        prec = 100*s["tp"]/(s["tp"]+s["fp"]) if (s["tp"]+s["fp"]) > 0 else 0
        print(f"{name:<20} {s['gt']:>6} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6} {recall:>7.1f}% {prec:>9.1f}%")


if __name__ == "__main__":
    main()
