"""
Visualize specific frames from results pickle.
"""
import argparse
import os
import sys

import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("result", help="result pickle file path")
    parser.add_argument("--out-dir", default="./visual_frames")
    parser.add_argument("--frame-idx", type=int, nargs="+", required=True)
    parser.add_argument("--score-thr", type=float, default=0.20)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
        importlib.import_module(_module_path)

    dataset = build_dataset(cfg.data.test)
    dataset.vis_score_threshold = args.score_thr

    results = mmcv.load(args.result, file_format="pkl")
    outputs = [results[i] for i in args.frame_idx]
    print(f"Visualizing frames: {args.frame_idx}")

    vis_pipeline = [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(type="Collect", keys=["img"], meta_keys=["timestamp", "lidar2img"]),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    dataset.show(outputs, show=False, save_dir=args.out_dir, pipeline=vis_pipeline)
    print("Done.")


if __name__ == "__main__":
    main()
