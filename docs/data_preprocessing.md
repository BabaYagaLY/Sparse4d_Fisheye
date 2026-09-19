# BEVData → Sparse4D 数据预处理文档

## 概述

`tools/preprocess_all.py` 将 BEVData（4 鱼眼相机停车场景）原始标注转成 Sparse4D 训练格式，复用 nuScenes 预训练权重。

## 输入

```
/mnt/e/BEVData/
├── Data01/ ~ Data05/
│   ├── calibration.yml          # 相机标定（4台 cam_hy_n5_avm_*）
│   ├── pose(ego_center).txt     # 每帧 ego 全局位姿
│   ├── frames_labels/           # *.json 每帧标注
│   └── camera/
│       ├── cam_hy_n5_avm_front/  # 1920×1280 鱼眼图像
│       ├── cam_hy_n5_avm_back/
│       ├── cam_hy_n5_avm_left/
│       └── cam_hy_n5_avm_right/
```

## 输出

| 文件 | 说明 |
|---|---|
| `fisheye_calib_4cam.npz` | K(4,3,3), D(4,4), lidar2cam(4,4,4), image_wh(4,2) |
| `fisheye_infos_train.pkl` | 4771 帧, 106,728 boxes |
| `fisheye_infos_val.pkl` | 1191 帧, 26,417 boxes |

## 处理步骤

### 1. 加载相机标定 (`load_calibration`, L58-81)

从 `calibration.yml` 的 `cam_hy_n5_avm_*` 段提取：

```
每台相机:
  focal (fx, fy)     → K 内参矩阵 (3×3)
  pp (cx, cy)
  inv_poly [k1,k2,k3,k4] → D 畸变系数 (4,)
  image_size [1920,1280] → image_wh
  extrinsic [x,y,z,qx,qy,qz,qw] → lidar2cam (4×4)
```

**lidar2cam 构建**（L73-77）：
```python
R_c2e = quat_to_rot(ext[3],ext[4],ext[5],ext[6])  # 相机→ego 旋转
t_c2e = [ext[0], ext[1], ext[2]]                    # 相机在 ego 中的位置
R_l2c = R_c2e.T                                      # ego→相机 旋转
t_l2c = -R_l2c @ t_c2e                              # ego→相机 平移
```

### 2. 类别映射 (L17-45)

23 种 BEVData 类型 → 10 种 nuScenes 类型（复用预训练权重）：

| nuScenes 类 | BEVData 来源 |
|---|---|
| `car` | passenger_car |
| `truck` | truck_tractor |
| `trailer` | trailer.open_top, trailer.box |
| `bicycle` | bicycle |
| `motorcycle` | motorcycle, tricycle |
| `pedestrian` | pedestrian |
| `barrier` | hard_barrier, soft_barrier, gate_barrier, bollard, box, pole, speed_bump, wheel_stopper, charging_infra, indoor_column, parking_lock, tree, traffic_sign |
| `traffic_cone` | cone |

### 3. 逐帧处理 (`process_dataset`, L104-210)

**a) 位姿匹配**：label 时间戳 → `pose(ego_center).txt`（容差 1ms）

**b) 图像路径**：`{相机目录}/{int(ts*1000)}.jpg`

**c) 标注框转换**：

```
BEVData 原始字段           →    nuScenes gt_boxes
────────────────────────────────────────────────
pos_xyz [x, y, z]               [x, y, z]        不变
quat [qx,qy,qz,qw]              yaw               提取偏航角
scale_xyz [L, W, H]             [W, L, H]         L↔W 交换
obj_track_id                    instance_inds     跟踪ID
obj_type                        gt_names          映射后类名
```

**yaw 提取** (L173-176)：
```python
siny = 2*(qw*qz + qx*qy)
cosy = 1 - 2*(qy*qy + qz*qz)
yaw = arctan2(siny, cosy)
```

**L↔W 交换原因**：nuScenes 格式 `gt_boxes = [x, y, z, w, l, h, yaw]`，其中 w=宽度(y方向), l=长度(x方向)。BEVData `scale_xyz = [长度, 宽度, 高度]`，所以 scale[0]→l, scale[1]→w。

**d) 过滤** (`is_box_behind_all_cameras`, L83-93)：
只过滤 box **中心点**在 4 台相机**全部**后方的情况（z_cam < 0）。保守策略——鱼眼 FOV > 190°，即使中心在后方，边缘可能仍可见。

**e) 附加字段**：`gt_velocity`, `num_lidar_pts`, `valid_flag` 填默认值（全 1/全 True），`lidar2ego` 置为单位阵。

### 4. 训练/验证划分 (L258-269)

**每个数据集独立分割**，最后 20% 作验证集。保证时序不泄露（val 帧全部在 train 帧之后）：

```
Data01: 1192 →  954 train + 238 val
Data05: 1392 → 1114 train + 278 val
总计:   5962 → 4771 train + 1191 val
```

### 5. 时序扫描 (L271-285)

每帧 `sweeps` 指向前一帧 token，`scene_token` = 数据集名。供模型 temporal attention 做帧间特征聚合。

---

## 与可视化逻辑对比

| 方面 | 预处理 | 可视化 |
|---|---|---|
| **标定读取** | `calibration.yml` → s2l_rot/trans, cam_intrinsics | 相同，读取同一文件 |
| **坐标帧** | ego 帧 (x=前,y=左,z=上) | 相同 |
| **pos_xyz** | 直接存入 gt_boxes | 直接用于 box 中心 |
| **scale** | [L,W,H] → 交换为 [W,L,H] 存 pkl | [L,W,H] → l=scale[0], w=scale[1] 直接使用 |
| **投影到像素** | ❌ 不涉及 — 只存 3D 数据 | ✅ `atan2(r,z)` 鱼眼投影 |
| **过滤** | 中心点全在相机后方则丢弃 | 按 visibility 字段过滤 |

两者等价——预处理 L↔W 交换是在 **nuScenes 格式层面的字段重排**，不改变物理含义。可视化不交换是因为直接读原始 label，自己管理 l/w/h 语义。

---

## 坐标系约定

```
Ego 坐标系 (nuScenes 标准):
  x = 前 (forward)
  y = 左 (left)  
  z = 上 (up)

相机 extrinsics: 相机在 ego 中的位姿
  translation = [x, y, z]  → 相机位置
  quaternion  = [qx,qy,qz,qw] → 相机朝向 (xyzw 格式)

gt_boxes: [x, y, z, w, l, h, yaw]
  x,y,z  = box 中心在 ego 帧
  w      = 宽度 (ego y方向)
  l      = 长度 (ego x方向)
  h      = 高度 (ego z方向)
  yaw    = 绕 z 轴旋转角 (弧度)
```
