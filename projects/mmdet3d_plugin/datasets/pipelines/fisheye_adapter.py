import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


# 3D box edges: 12 edges connecting 8 corners
_BOX3D_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _box3d_to_corners(bboxes):
    """Convert [x,y,z,w,l,h,yaw] to 8 corners in ego/LiDAR frame."""
    bboxes = np.asarray(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes[None]

    x, y, z = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2]
    w, l, h = bboxes[:, 3], bboxes[:, 4], bboxes[:, 5]
    yaw = bboxes[:, 6]

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
    rot_mat = np.tile(np.eye(3)[None], (len(bboxes), 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos

    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(-1)
    corners += bboxes[:, None, :3]
    return corners


def _fisheye_project_pts(pts_cam, fx, fy, cx, cy, k):
    """Equidistant fisheye projection for numpy arrays."""
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


def _box_visible_in_cameras(bbox, camera_extrinsic, camera_intrinsic,
                            camera_distortion, image_wh, num_edge_samples=10):
    """
    Check whether a 3D box is visible in at least one camera.
    Visibility means at least one point on the box edges projects in front of
    the camera (z > 0) and inside the image bounds.
    """
    corners = _box3d_to_corners(bbox[:7])[0]  # (8, 3)
    for cam_idx in range(len(camera_intrinsic)):
        K = camera_intrinsic[cam_idx]
        D = camera_distortion[cam_idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        W, H = image_wh[cam_idx]
        R_l2c = camera_extrinsic[cam_idx][:3, :3]
        t_l2c = camera_extrinsic[cam_idx][:3, 3]

        for i, j in _BOX3D_EDGES:
            p0, p1 = corners[i], corners[j]
            t = np.linspace(0, 1, num_edge_samples)
            samples = p0 + t[:, None] * (p1 - p0)
            samples_cam = (R_l2c @ samples.T).T + t_l2c
            # Must be in front of camera
            if np.all(samples_cam[:, 2] <= 0):
                continue
            pts_2d = _fisheye_project_pts(samples_cam, fx, fy, cx, cy, D)
            visible = (
                (samples_cam[:, 2] > 0) &
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
            )
            if visible.any():
                return True
    return False


@PIPELINES.register_module()
class FisheyeSparse4DAdaptor(object):
    """
    Adaptor for 4-fisheye-camera input.
    Constructs fisheye-specific keys:
        - camera_intrinsic:  (4, 3, 3) K matrices
        - camera_distortion: (4, 4) [k1, k2, k3, k4]
        - camera_extrinsic:  (4, 4, 4) LiDAR-to-camera transforms
        - projection_mat:    (4, 4, 4) kept for compatibility (identity-like)
        - image_wh:          (4, 2)
        - T_global / T_global_inv: ego-vehicle global pose
    """

    def __init__(self, filter_invisible_gt=True):
        self.filter_invisible_gt = filter_invisible_gt

    def __call__(self, input_dict):
        num_cams = len(input_dict["lidar2img"])

        # Build camera intrinsic K (3x3) from calibration data
        if "cam_intrinsic" in input_dict:
            camera_intrinsic = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )  # (num_cams, 3, 3)
            input_dict["focal"] = camera_intrinsic[..., 0, 0]
        else:
            camera_intrinsic = np.array(
                [
                    [input_dict["fx"], 0, input_dict["cx"]],
                    [0, input_dict["fy"], input_dict["cy"]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            camera_intrinsic = np.tile(
                camera_intrinsic[None], (num_cams, 1, 1)
            )
            input_dict["focal"] = np.array(
                [input_dict["fx"]] * num_cams, dtype=np.float32
            )

        input_dict["camera_intrinsic"] = camera_intrinsic

        # Build camera distortion coefficients
        if "cam_distortion" in input_dict:
            camera_distortion = np.float32(
                np.stack(input_dict["cam_distortion"])
            )  # (num_cams, N)
            if camera_distortion.shape[-1] < 4:
                pad = np.zeros(
                    (num_cams, 4 - camera_distortion.shape[-1]),
                    dtype=np.float32,
                )
                camera_distortion = np.concatenate(
                    [camera_distortion, pad], axis=-1
                )
            elif camera_distortion.shape[-1] > 4:
                camera_distortion = camera_distortion[..., :4]
        else:
            camera_distortion = np.zeros((num_cams, 4), dtype=np.float32)
        input_dict["camera_distortion"] = camera_distortion

        # Build camera extrinsic: LiDAR-to-camera transform
        # lidar2img = K @ [R|t]  (3x4)
        # For fisheye, store lidar2cam as 4x4
        camera_extrinsic = []
        for i in range(num_cams):
            lidar2img = np.array(input_dict["lidar2img"][i])  # (4, 4)
            K = camera_intrinsic[i]  # (3, 3)
            K_44 = np.eye(4, dtype=np.float32)
            K_44[:3, :3] = K
            # Extract [R|t] from lidar2img = K @ [R|t]
            # lidar2img[:3] = K @ lidar2cam[:3]
            lidar2cam = np.linalg.solve(K_44, lidar2img)
            camera_extrinsic.append(lidar2cam)
        input_dict["camera_extrinsic"] = np.float32(
            np.stack(camera_extrinsic)
        )  # (num_cams, 4, 4)

        # Keep projection_mat for compatibility (camera encoder uses K-based encoding)
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]

        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        # Filter out GT boxes that are not visible in any camera.
        # If the annotation already contains per-camera visibility flags, use
        # those directly; otherwise fall back to a runtime projection check on
        # the (possibly augmented) camera parameters.
        if (
            self.filter_invisible_gt
            and "gt_bboxes_3d" in input_dict
            and len(input_dict["gt_bboxes_3d"]) > 0
        ):
            gt_bboxes = input_dict["gt_bboxes_3d"]
            image_wh = input_dict["image_wh"]
            if "gt_visibility" in input_dict:
                visible_mask = np.asarray(input_dict["gt_visibility"]).any(axis=1)
            else:
                visible_mask = np.array([
                    _box_visible_in_cameras(
                        gt_bboxes[i],
                        input_dict["camera_extrinsic"],
                        input_dict["camera_intrinsic"],
                        input_dict["camera_distortion"],
                        image_wh,
                    )
                    for i in range(len(gt_bboxes))
                ])
            if not visible_mask.all():
                input_dict["gt_bboxes_3d"] = gt_bboxes[visible_mask]
                if "gt_labels_3d" in input_dict:
                    input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][visible_mask]
                if "gt_visibility" in input_dict:
                    input_dict["gt_visibility"] = input_dict["gt_visibility"][visible_mask]
                if "instance_inds" in input_dict:
                    input_dict["instance_inds"] = input_dict["instance_inds"][visible_mask]
                if "gt_names" in input_dict:
                    input_dict["gt_names"] = [
                        input_dict["gt_names"][i]
                        for i in range(len(input_dict["gt_names"]))
                        if visible_mask[i]
                    ]

        # Convert img HWC->CHW and wrap in DC (mirrors NuScenesSparse4DAdaptor)
        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        # Wrap gt data in DataContainer (mirrors NuScenesSparse4DAdaptor)
        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = DC(
                to_tensor(input_dict["gt_visibility"]).bool()
            )

        return input_dict

    @staticmethod
    def limit_period(val, offset=0.5, period=np.pi):
        return val - np.floor(val / period + offset) * period

    def __repr__(self):
        return self.__class__.__name__
