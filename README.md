# 鱼眼环视 3D 感知模型（Sparse4D 迁移）

将原生面向 nuScenes 6 路针孔相机的 Sparse4D v3 检测/跟踪模型，迁移到自研 4 路鱼眼环视平台（FOV > 180°），保留时序融合、实例记忆库、查询传播等核心跟踪能力；在小数据（约 4800 帧）、小显存（6GB）约束下跑通 3D 检测 + 跟踪闭环。

> 本项目基于上游 [Sparse4D（HorizonRobotics）](https://github.com/HorizonRobotics/Sparse4D) 进行鱼眼环视适配与迁移训练。

## 改造要点

- **投影几何重写**：定位可变形特征聚合为唯一投影耦合点，将针孔透视投影改为等距鱼眼投影 + 4 系数多项式畸变，保留原始 FOV、避免去畸变的边缘分辨率损失。
- **数值稳定**：FP16 下溢改 FP32 中间计算、atan2 替换 atan 消除 z≤0 奇异性、grid_sample 越界掩蔽与坐标钳制，投影层全程零 NaN / Inf。
- **两阶段迁移训练**：「参数冻结 → 渐进解冻」，一阶段冻结约 99% 参数仅训相机编码器与投影融合权重，二阶段解冻 FFN / Refine 等模块，小数据下稳定收敛无过拟合。
- **鱼眼数据管线**：自研数据转 nuScenes 格式、四路标定注入、按鱼眼分布 kmeans 重聚类 900 个 3D anchor、多相机可见性过滤清洗 GT。
- **诊断工具**：投影可视化（3D GT 框反投影验证标定/投影正确性）、训练-真值对比工具。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirement.txt

# 2. 训练 / 测试
bash local_train.sh
bash local_test.sh
```

> 标定 `fisheye_calib_4cam.npz`、anchor `nuscenes_kmeans900.npy` 已随仓库提供；详细配置与口径见 `TRAINING_REPORT.md`。

## 目录结构

```
.
├─ projects/                  # 模型配置与入口
├─ tools/                     # 训练 / 评测 / 可视化脚本
├─ docs/                      # 文档
├─ fisheye_calib_4cam.npz     # 四路鱼眼标定（内参/畸变/外参）
├─ nuscenes_kmeans900.npy     # 按鱼眼分布重聚类的 900 anchor
├─ local_train.sh             # 训练脚本
├─ local_test.sh              # 测试脚本
├─ TRAINING_REPORT.md         # 训练与指标说明
├─ arch_diagram.png           # 架构图
└─ requirement.txt
```

## 效果

- 在约 4800 帧、6GB 显存（RTX 3050）下稳定收敛，一阶段 loss 28.7 → 19.6、无过拟合。
- 跑通检测 + 跟踪闭环；训练仍在持续调优，定量指标以 `TRAINING_REPORT.md` 为准。

## 致谢 / 上游

- [Sparse4D（HorizonRobotics）](https://github.com/HorizonRobotics/Sparse4D)
- Sparse4D v1/v2/v3 论文链接见上游仓库 README。

## License

本项目在上游 Sparse4D 基础上修改，许可证见仓库 `LICENSE` 文件。
