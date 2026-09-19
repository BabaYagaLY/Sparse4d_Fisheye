# 鱼眼投影在训练中的使用位置

## 投影只在模型前向传播中使用

训练中用到鱼眼投影的地方只有一处：

### `DeformableFeatureAggregation.project_points()` → `fisheye_projection.py`

**调用链：**
```
训练: Sparse4D.forward_train()
  → Sparse4DHead.forward()
    → DeformableFeatureAggression.forward()           # projects/.../models/blocks.py
      → project_points()                              # L222
        → fisheye_project()                            # projects/.../models/fisheye_projection.py
          → _fisheye_project_equidistant()             # L46
```

**作用：** 将 3D 参考点投影到 2D 特征图上，用 `grid_sample` 采样图像特征。

```python
# blocks.py L222-232
def project_points(self, key_points, ...):
    if self.projection_mode == "fisheye":
        from .fisheye_projection import fisheye_project
        return fisheye_project(key_points, camera_extrinsic,
                               camera_intrinsic, camera_distortion, image_wh)
```

**投影公式**（`_fisheye_project_equidistant`, fisheye_projection.py）：
```python
pts_cam = R_l2c @ pts_3d + t_l2c          # ego → 相机帧
r = sqrt(x² + y²)
θ = atan2(r, z)                            # z 可为负 ✅
θ_d = θ * (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
u = fx * (θ_d/r) * x + cx
v = fy * (θ_d/r) * y + cy
```

已经修正：`atan2(r, z)` 替代了原来的 `atan(r/z.clamp(min=1e-6))`，正确支持 FOV > 180°。

---

## 训练中**不使用**投影的地方

| 训练组件 | 是否涉及投影 | 说明 |
|---|---|---|
| **Loss 计算** | ❌ | 3D 空间直接算 L1 loss，不投影 |
| **分类 (refine_layer)** | ❌ | 3D 特征直接分类 |
| **Anchor generation** | ❌ | kmeans 3D anchors |
| **Temporal attention** | ❌ | 3D instance features |
| **GT matching** | ❌ | 3D IoU/hungarian |
| **数据加载** | ❌ | 只读 pkl，不投影 |

---

## 总结

```
┌──────────────────────────────────────────────────────┐
│                    训练全链路                          │
│                                                      │
│  预处理             模型前向               Loss       │
│  ───────           ──────────           ───────      │
│  BEVData标注  →    3D参考点 → 2D采样  →  3D Loss    │
│  → pkl (3D)         ↑ 唯一用到投影      (纯3D)       │
│                     fisheye_project                  │
│                     atan2(r,z) ✅                    │
└──────────────────────────────────────────────────────┘
```

投影只在 deformable attention 的 feature sampling 阶段使用一次。Loss、分类、匹配全在 3D 空间完成。
```

---

## 预测结果可视化

验证/测试时的可视化调用链：

```
NuScenes3DDetTrackDataset.show()
  → draw_lidar_bbox3d_on_fisheye_img()   # utils.py (鱼眼版)
  → draw_lidar_bbox3d_on_bev()           # BEV 俯视图 (纯 3D，无需投影)
```

`show()` 检测到 `self._calib_D is not None` 时自动使用鱼眼版：
- 用 `atan2(r,z)` 做鱼眼投影（与训练一致）
- 每条 3D 边密集采样 30 个点 + `cv2.polylines` 画曲线（与 `visualize_dataset.py` 一致）

---

## 代码修改汇总

| 文件 | 修改 |
|---|---|
| `fisheye_projection.py` | `z.clamp` → `atan2(r,z)` |
| `preprocess_all.py` | 过滤改为 visibility 字段 |
| `visualize_dataset.py` | 密集边采样画鱼眼曲线 |
| `utils.py` | 新增 `draw_lidar_bbox3d_on_fisheye_img` |
| `nuscenes_3d_det_track_dataset.py` | `show()` 自动切换鱼眼/针孔画框 |
