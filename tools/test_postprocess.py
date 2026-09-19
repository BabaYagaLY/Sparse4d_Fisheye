# Copyright (c) OpenMMLab. All rights reserved.
"""Run test/eval with optional 3D BEV NMS and score threshold post-processing.

This is a thin wrapper around tools/test.py that applies class-aware BEV NMS
and a configurable score threshold before saving/evaluating results.  It does
not change the model weights or architecture.
"""
import argparse
import mmcv
import os
import numpy as np
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmdet.apis import single_gpu_test, multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test


def bev_iou_rotated(box_a, box_b):
    """BEV IoU for two 3D boxes [x,y,z,w,l,h,yaw] using cv2 rotated rects."""
    # box order is [x, y, z, w, l, h, yaw]
    rect_a = (
        (float(box_a[0]), float(box_a[1])),
        (float(box_a[3]), float(box_a[4])),
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


def apply_postprocess(outputs, score_thr, nms_iou, max_per_frame=1000):
    """Apply score threshold and class-aware BEV NMS to a list of results."""
    import cv2  # noqa: F401  (bev_iou_rotated needs it)
    processed = []
    for result in outputs:
        res = result["img_bbox"]
        scores = res["scores_3d"]
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy()
            boxes_np = res["boxes_3d"].cpu().numpy()
            labels_np = res["labels_3d"].cpu().numpy()
        else:
            scores_np = np.asarray(scores)
            boxes_np = np.asarray(res["boxes_3d"])
            labels_np = np.asarray(res["labels_3d"])

        mask = scores_np >= score_thr
        boxes_np = boxes_np[mask]
        labels_np = labels_np[mask]
        scores_np = scores_np[mask]

        if nms_iou > 0 and len(boxes_np) > 0:
            keep = nms_3d_bev(boxes_np, scores_np, labels_np, iou_thr=nms_iou)
            boxes_np = boxes_np[keep]
            labels_np = labels_np[keep]
            scores_np = scores_np[keep]

        if len(boxes_np) > max_per_frame:
            topk = np.argsort(scores_np)[::-1][:max_per_frame]
            boxes_np = boxes_np[topk]
            labels_np = labels_np[topk]
            scores_np = scores_np[topk]

        processed.append({
            "img_bbox": {
                "boxes_3d": torch.from_numpy(boxes_np),
                "scores_3d": torch.from_numpy(scores_np),
                "labels_3d": torch.from_numpy(labels_np),
            }
        })
    return processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMDet test (and eval) a model with post-processing"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, e.g., "img_bbox"',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where results will be saved"
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--show_only", action="store_true")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.1,
        help="minimum score to keep a prediction",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="BEV IoU threshold for class-aware NMS; <=0 disables NMS",
    )
    parser.add_argument(
        "--max-per-frame",
        type=int,
        default=1000,
        help="maximum predictions per frame after post-processing",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert (
        args.out or args.eval or args.format_only or args.show or args.show_dir
    ), (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline
            )
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    print("distributed:", distributed)
    if distributed:
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
    else:
        data_loader = build_dataloader_origin(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    if args.result_file is not None:
        outputs = torch.load(args.result_file)
    elif not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, None)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = custom_multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )

    # === Post-processing =====================================================
    print(
        f"\nApplying post-processing: score_thr={args.score_thr}, "
        f"nms_iou={args.nms_iou}, max_per_frame={args.max_per_frame}"
    )
    outputs = apply_postprocess(
        outputs,
        score_thr=args.score_thr,
        nms_iou=args.nms_iou,
        max_per_frame=args.max_per_frame,
    )
    print("Post-processing done.")
    # ========================================================================

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.show_only:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(kwargs)
            dataset.show(outputs, show=True, **eval_kwargs)
        elif args.format_only:
            dataset.format_results(outputs, **kwargs)
        elif args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(eval_kwargs)
            results_dict = dataset.evaluate(outputs, **eval_kwargs)
            print(results_dict)


if __name__ == "__main__":
    import cv2  # imported here so the worker subprocess has it
    torch.multiprocessing.set_start_method("fork")
    main()
