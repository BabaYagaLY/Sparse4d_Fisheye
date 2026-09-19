"""
Compare model predictions with ground truth on selected training frames.
Draws GT boxes in green and predicted boxes in red on each fisheye view.
Also prints predicted and GT boxes numerically for diagnosis.

Usage:
    cd /root/ly/BEV/Sparse4D
    source /root/ly/BEV/mm_sparse4d/bin/activate
    export PYTHONPATH=/root/ly/BEV/Sparse4D:$PYTHONPATH
    python tools/compare_train_pred_gt.py \
        work_dirs/fisheye_sparse4d_retrain_longer/fisheye_v3_r50_4x_retrain_longer.py \
        work_dirs/fisheye_sparse4d_retrain_longer/best_img_bbox_NuScenes/NDS_iter_71580.pth \
        --out-dir work_dirs/fisheye_sparse4d_retrain_longer/compare_pred_gt \
        --frame-idx 3832 4534 4625 4416 2327 \
        --score-thr 0.05
"""
import argparse
import os

import cv2
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

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# BGR
PRED_COLOR = (0, 0, 255)  # red
GT_COLOR = (0, 255, 0)    # green

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


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


def box3d_to_corners(box3d):
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
    """Project a 3D box to a fisheye view with proper visibility filtering.

    A box is only drawn on a camera if at least part of one edge is both
    *in front of the camera* (z > 0) and *inside the image*.  This avoids the
    common fisheye visualization artifact where points behind the camera still
    project inside the image rectangle and create radial streaks.
    """
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
    return edge_curves, any_visible


def bev_iou_rotated(box_a, box_b):
    """BEV IoU for two 3D boxes [x,y,z,w,l,h,yaw]."""
    # Use cv2.rotatedRectangleIntersection on the BEV footprint.
    rect_a = (
        (float(box_a[0]), float(box_a[1])),
        (float(box_a[3]), float(box_a[4])),  # (w, l)
        float(np.degrees(box_a[6])),
    )
    rect_b = (
        (float(box_b[0]), float(box_b[1])),
        (float(box_b[3]), float(box_b[4])),
        float(np.degrees(box_b[6])),
    )
    inter_status, pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if inter_status == 0 or inter_status == 1 or pts is None:
        return 0.0
    area_inter = cv2.contourArea(pts)
    if area_inter <= 0:
        return 0.0
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    return float(area_inter / (area_a + area_b - area_inter + 1e-12))


def nms_3d_bev(boxes, scores, labels, iou_thr=0.5):
    """Class-aware BEV NMS.

    Args:
        boxes: (N, 7) array [x,y,z,w,l,h,yaw]
        scores: (N,) array
        labels: (N,) array
        iou_thr: BEV IoU threshold

    Returns:
        keep: list of indices to keep
    """
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j] or labels[j] != labels[i]:
                continue
            if bev_iou_rotated(boxes[i], boxes[j]) > iou_thr:
                suppressed[j] = True
    return keep


def draw_boxes_on_img(img, bboxes, R_l2c, t_l2c, fx, fy, cx, cy, k, W, H, color):
    """Draw a set of 3D boxes on an image. Returns number drawn."""
    if len(bboxes) == 0:
        return 0
    corners = box3d_to_corners(bboxes)
    drawn = 0
    for bi in range(len(bboxes)):
        edge_curves, visible = project_box_to_fisheye(
            corners[bi], R_l2c, t_l2c, fx, fy, cx, cy, k, W, H
        )
        if not visible:
            continue
        for curve in edge_curves:
            if curve is not None:
                cv2.polylines(img, [curve.reshape((-1, 1, 2))], False, color, 2)
        drawn += 1
    return drawn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out-dir", default="work_dirs/fisheye_sparse4d_retrain_longer/compare_pred_gt")
    parser.add_argument("--frame-idx", type=int, nargs="+", default=None)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="BEV IoU threshold for class-aware NMS; <=0 disables NMS")
    parser.add_argument("--max-per-frame", type=int, default=100)
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

    # Build test-mode dataset using train annotations
    ds_cfg = cfg.data.test.copy()
    ds_cfg["ann_file"] = cfg.data.train["ann_file"]
    dataset = build_dataset(ds_cfg)
    print(f"Dataset size: {len(dataset)}")

    # Load calibration
    calib_path = os.path.join(os.path.dirname(cfg.data.train["ann_file"]), "fisheye_calib_4cam.npz")
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

    if args.frame_idx is not None:
        indices = args.frame_idx
    else:
        np.random.seed(args.seed)
        indices = np.random.choice(len(dataset), args.num_frames, replace=False)
        indices = sorted(indices.tolist())
    print(f"Visualizing {len(indices)} frames: {indices}")

    subset = SubsetDataset(dataset, indices)
    data_loader = build_dataloader(
        subset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    print("Running inference...")
    outputs = single_gpu_test(model, data_loader, False, None)
    print(f"Got {len(outputs)} predictions")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving to {args.out_dir}")

    for fidx, idx in enumerate(indices):
        pred = outputs[fidx]["img_bbox"]
        scores = pred["scores_3d"].numpy()
        mask = scores >= args.score_thr
        pred_boxes = pred["boxes_3d"][mask].numpy()
        pred_labels = pred["labels_3d"][mask].numpy()
        scores = scores[mask]

        # Class-aware BEV NMS to remove duplicate predictions
        if args.nms_iou > 0 and len(pred_boxes) > 0:
            keep = nms_3d_bev(pred_boxes, scores, pred_labels, iou_thr=args.nms_iou)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            scores = scores[keep]

        if len(pred_boxes) > args.max_per_frame:
            topk = np.argsort(scores)[::-1][:args.max_per_frame]
            pred_boxes = pred_boxes[topk]
            pred_labels = pred_labels[topk]
            scores = scores[topk]

        # Normalize yaw
        pred_boxes[:, 6] = pred_boxes[:, 6] - np.floor(pred_boxes[:, 6] / (2 * np.pi) + 0.5) * (2 * np.pi)

        # GT boxes from raw info
        info = dataset.data_infos[idx]
        gt_boxes = info["gt_boxes"].copy()
        gt_names = info["gt_names"]
        # Filter valid GT (same as training uses: num_lidar_pts > 0)
        valid = info["num_lidar_pts"] > 0
        gt_boxes = gt_boxes[valid]
        gt_names = gt_names[valid]
        # Map names to class labels
        gt_labels = np.array([CLASS_NAMES.index(n) if n in CLASS_NAMES else -1 for n in gt_names])
        gt_boxes = gt_boxes[gt_labels >= 0]
        gt_labels = gt_labels[gt_labels >= 0]

        print(f"\n=== Frame {idx} ===")
        print(f"GT boxes: {len(gt_boxes)}")
        for bi, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            print(f"  GT {bi}: {CLASS_NAMES[label]} score=- box={box[:7].round(3)}")
        print(f"Pred boxes (thr={args.score_thr}): {len(pred_boxes)}")
        for bi, (box, label, s) in enumerate(zip(pred_boxes, pred_labels, scores)):
            print(f"  Pred {bi}: {CLASS_NAMES[label]} score={s:.3f} box={box[:7].round(3)}")

        cam_names = list(info["cams"].keys())
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

            # Draw GT first, then predictions on top
            draw_boxes_on_img(img, gt_boxes, R_l2c, t_l2c, fx, fy, cx, cy, k, W, H, GT_COLOR)
            draw_boxes_on_img(img, pred_boxes, R_l2c, t_l2c, fx, fy, cx, cy, k, W, H, PRED_COLOR)

            cv2.putText(img, CAM_LABELS[cam_idx], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            imgs.append(img)

        min_h = min(img.shape[0] for img in imgs)
        min_w = min(img.shape[1] for img in imgs)
        imgs = [cv2.resize(img, (min_w, min_h)) for img in imgs]
        top = np.concatenate(imgs[:2], axis=1)
        bot = np.concatenate(imgs[2:], axis=1)
        image = np.concatenate([top, bot], axis=0)

        out_path = os.path.join(args.out_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, image)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
