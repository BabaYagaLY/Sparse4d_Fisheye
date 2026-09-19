# 鱼眼可视化工具文档

## 概述

`tools/visualize_dataset.py` 将 BEVData 标注框投影到鱼眼图像上，用于验证数据质量和投影正确性。

用法：
```bash
python tools/visualize_dataset.py --dataset Data05 --frames 3 [--camera back] [--cars-only] [--no-ground]
```

## 核心设计

### 为什么需要密集采样画曲线？

**鱼眼相机中，3D 世界的直线投影到图像是曲线，不是直线。**

针孔相机：直线保持直线 → 投影 8 个角点 + `cv2.line` 即可。

鱼眼相机：直线变曲线 → 如果不做密集采样，画出的直线框会偏离真实物体。物体越长、越靠近图像边缘，偏差越大（可达上百像素）。

### 解决方法

对 3D 框的每条边（共 12 条）在 3D 空间**均匀采样 30 个中间点**，逐个投影到鱼眼图像，然后用 `cv2.polylines` 画折线逼近曲线。

```
针孔相机:    8 角点 → cv2.line(直线)       ← 正确
鱼眼相机:    30×12 采样点 → cv2.polylines  ← 正确
```

## 投影流程

### 1. 构建 3D 框角点 (`project_box`, L124-176)

```
输入: pos_xyz, quat, scale_xyz [L,W,H]
输出: 每条边的密集 2D 曲线 + 角点投影
```

**a) 8 个局部角点**（物体坐标系）：
```python
corners_local = [
    [ l/2, w/2,-h/2], [ l/2,-w/2,-h/2], [-l/2,-w/2,-h/2], [-l/2, w/2,-h/2],  # 底面
    [ l/2, w/2, h/2], [ l/2,-w/2, h/2], [-l/2,-w/2, h/2], [-l/2, w/2, h/2],  # 顶面
]
# 其中 l=scale[0], w=scale[1], h=scale[2] (直接使用原始 label，不交换)
```

**b) 局部 → ego 坐标系**：
```python
corners_ego = corners_local @ R_obj.T + pos
```
`R_obj = quat_to_rot(quat)` 将物体朝向转到 ego 帧。

**c) ego → 相机坐标系**：
```python
corners_cam = (R_c2e.T @ (corners_ego - t_c2e)).T
```
`R_c2e`/`t_c2e` 来自 `calibration.yml` 的 extrinsic：相机在 ego 帧中的位姿。

### 2. 鱼眼投影 (`fisheye_project_pts`, L93-111)

**等距投影模型（OpenCV fisheye 兼容）**：

```python
r = sqrt(x² + y²)          # 相机 xy 平面距离
θ = atan2(r, z)            # 入射角 (z 可为负 → θ > 90°, FOV > 180°)
θ_d = θ * (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)  # 畸变后角度
scale = θ_d / r
u = fx * scale * x + cx    # 像素坐标
v = fy * scale * y + cy
```

关键：使用 `atan2(r, z)` 而非 `atan(r/z.clamp(min=ε))`，正确支持 z<0（相机后方）的情况。

### 3. 密集边采样 (L155-170)

对每条边 (i,j)，在 3D 空间线性插值 30 个点：
```python
t = linspace(0, 1, 30)
samples_ego = p0 + t * (p1 - p0)
```
逐点投影后，**只保留落在图像范围内的点**（自动裁剪跨相机平面的边）。

### 4. 绘制

**3D 物体**（车辆、护栏等）：
```python
for curve in edge_curves:
    if curve is not None:
        cv2.polylines(img, [curve], False, color, 2)  # 折线逼近曲线
```

**地面标记**（箭头、车位线等）：
```python
# 用 4 条底面边的密集采样点拼成闭合多边形
cv2.fillPoly(overlay, [bottom_polygon], color)  # 半透明填充
```

## 坐标帧

```
ego 坐标系 → 相机坐标系 → 鱼眼像素
  (x前y左z上)    (x右y下z前)     (u,v)
     │               │              │
  pos_xyz        R_c2e^T @       fisheye_project_pts
  + quat         (p - t_c2e)     (atan2 + inv_poly)
```

## 与预处理逻辑对比

| 方面 | 预处理 | 可视化 |
|---|---|---|
| 坐标帧 | ego | ego |
| pos_xyz | 直接存入 pkl | 直接使用 |
| scale 惯例 | 交换 L↔W (适配 nuScenes) | 不交换 (直接读 label) |
| 投影 | **不投影**（仅 3D） | **鱼眼投影**（atan2 + inv_poly） |
| 过滤 | 中心全在相机后 → 丢弃 | visibility 字段 |
| 边绘制 | 无 | 密集采样曲线 |
```
