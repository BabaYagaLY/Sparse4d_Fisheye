"""
Preprocess all datasets under BEVData into Sparse4D training format.
Logic:
  - pos_xyz = ego frame (x=forward, y=left, z=up)
  - ego2global from pose(ego_center).txt (comma-separated)
  - cam_hy_n5_avm_* calibration from calibration.yml
  - Images: 1920x1280 raw fisheye, filenames = int(ts_sec*1000).jpg
  - Scene-level split per dataset: last 20% = val
  - Visibility from labels, obj_track_id for tracking
"""
import argparse, copy, json, os, pickle, numpy as np, yaml
from collections import OrderedDict, Counter

RAW_CAM_KEYS = ['cam_hy_n5_avm_front','cam_hy_n5_avm_back','cam_hy_n5_avm_left','cam_hy_n5_avm_right']

# Map to nuScenes 10 classes (leveraging pre-trained weights)
INCLUDE_CLASSES = {
    # ── Vehicles ──
    'class.vehicle.passenger_car': 'car',
    'class.vehicle.truck_tractor': 'truck',
    'class.trailer.open_top': 'trailer',
    'class.trailer.box': 'trailer',
    # ── Cycles ──
    'class.cycle.bicycle': 'bicycle',
    'class.cycle.motorcycle': 'motorcycle',
    'class.cycle.tricycle': 'motorcycle',
    # ── Pedestrian ──
    'class.pedestrian.pedestrian': 'pedestrian',
    # ── Barrier (all static obstacles) ──
    'class.traffic_facility.hard_barrier': 'barrier',
    'class.traffic_facility.soft_barrier': 'barrier',
    'class.traffic_facility.gate_barrier': 'barrier',
    'class.traffic_facility.bollard': 'barrier',
    'class.traffic_facility.box': 'barrier',
    'class.traffic_facility.pole': 'barrier',
    'class.traffic_facility.speed_bump': 'barrier',
    'class.traffic_facility.cone': 'traffic_cone',
    'class.parking.wheel_stopper': 'barrier',
    'class.parking.charging_infra': 'barrier',
    'class.parking.indoor_column': 'barrier',
    'class.parking.parking_lock': 'barrier',
    'class.plant.tree': 'barrier',
    'class.sign.traffic_sign.instruction': 'barrier',
    'class.sign.traffic_sign.prohibition_and_limit': 'barrier',
}


def quat_to_rot(qx, qy, qz, qw):
    R = np.zeros((3,3))
    R[0,0]=1-2*qy*qy-2*qz*qz; R[0,1]=2*qx*qy-2*qz*qw; R[0,2]=2*qx*qz+2*qy*qw
    R[1,0]=2*qx*qy+2*qz*qw; R[1,1]=1-2*qx*qx-2*qz*qz; R[1,2]=2*qy*qz-2*qx*qw
    R[2,0]=2*qx*qz-2*qy*qw; R[2,1]=2*qy*qz+2*qx*qw; R[2,2]=1-2*qx*qx-2*qy*qy
    return R

def quat_to_wxyz(qx, qy, qz, qw):
    return [qw, qx, qy, qz]

def load_calibration(calib_path):
    calib = yaml.safe_load(open(calib_path))
    rig = calib['rig']
    K = np.zeros((4,3,3), dtype=np.float32)
    D = np.zeros((4,4), dtype=np.float32)
    lidar2cam = np.zeros((4,4,4), dtype=np.float32)
    image_wh = np.zeros((4,2), dtype=np.float32)
    s2l_rot, s2l_trans, cam_intrinsics = [], [], []
    for i, key in enumerate(RAW_CAM_KEYS):
        cfg = rig[key]
        fx, fy = cfg['focal']; cx, cy = cfg['pp']
        W, H = cfg['image_size']; inv_poly = cfg['inv_poly']; ext = cfg['extrinsic']
        K[i,0,0]=fx; K[i,1,1]=fy; K[i,0,2]=cx; K[i,1,2]=cy; K[i,2,2]=1.0
        for j,v in enumerate(inv_poly[:4]): D[i,j]=v
        image_wh[i] = [W, H]
        R_c2e = quat_to_rot(ext[3],ext[4],ext[5],ext[6])
        t_c2e = np.array([ext[0],ext[1],ext[2]])
        R_l2c = R_c2e.T; t_l2c = -R_l2c @ t_c2e
        l2c = np.eye(4, dtype=np.float32); l2c[:3,:3]=R_l2c; l2c[:3,3]=t_l2c
        lidar2cam[i] = l2c
        s2l_rot.append(R_c2e.astype(np.float32))
        s2l_trans.append(t_c2e.astype(np.float32))
        cam_intrinsics.append(K[i].copy())
    return K, D, lidar2cam, image_wh, s2l_rot, s2l_trans, cam_intrinsics

def load_poses(pose_path):
    poses = {}
    with open(pose_path) as f:
        for line in f:
            parts = [float(x) for x in line.strip().split(',')]
            poses[parts[0]] = parts[1:]
    return poses

def process_dataset(ds_dir, ds_name, s2l_rot, s2l_trans, cam_intrinsics):
    """Process one dataset directory, return list of info dicts."""
    pose_path = os.path.join(ds_dir, 'pose(ego_center).txt')
    if not os.path.exists(pose_path):
        # Try other pose filenames
        for f in os.listdir(ds_dir):
            if 'pose' in f.lower() and f.endswith('.txt'):
                pose_path = os.path.join(ds_dir, f)
                break
    poses = load_poses(pose_path)
    
    label_dir = os.path.join(ds_dir, 'frames_labels')
    label_files = sorted(os.listdir(label_dir))
    
    infos = []
    total_skipped = 0
    for fname in label_files:
        if not fname.endswith('.json'): continue
        ts_sec = float(fname.replace('.json',''))
        
        ts_match = None
        for pk in poses:
            if abs(pk - ts_sec) < 0.001:
                ts_match = pk; break
        if ts_match is None: continue
        
        ego_p = poses[ts_match]
        ego_t = ego_p[:3]
        ego_rot_wxyz = quat_to_wxyz(ego_p[3], ego_p[4], ego_p[5], ego_p[6])
        
        objs = json.load(open(os.path.join(label_dir, fname)))
        
        cams = OrderedDict()
        ts_ms = str(int(ts_sec * 1000))
        for i, key in enumerate(RAW_CAM_KEYS):
            img_path = os.path.join(ds_dir, 'camera', key, f'{ts_ms}.jpg')
            if not os.path.exists(img_path):
                # Try alternate naming (some datasets might have different path)
                alt = os.path.join(ds_dir, 'camera', key, f'{ts_sec:.3f}.jpg')
                if os.path.exists(alt):
                    img_path = alt
            cams[RAW_CAM_KEYS[i]] = {
                'data_path': img_path,
                'sensor2lidar_rotation': s2l_rot[i],
                'sensor2lidar_translation': s2l_trans[i],
                'cam_intrinsic': copy.deepcopy(cam_intrinsics[i]),
            }
        
        gt_boxes, gt_names, gt_visibility, instance_ids = [], [], [], []
        for obj in objs:
            obj_type = obj['obj_type']
            cls_name = INCLUDE_CLASSES.get(obj_type)
            if cls_name is None: continue

            geom = obj['geometry']
            if isinstance(geom, list):
                geom = geom[0] if len(geom)>0 and isinstance(geom[0], dict) else None
                if geom is None: continue

            pos = geom['pos_xyz']    # ego frame
            quat = geom['quat']     # [qx,qy,qz,qw]
            scale = geom['scale_xyz'] # [L,W,H]

            # Parse per-camera visibility. Raw labels use two formats:
            #   - numeric (box3d objects): >0 means visible
            #   - string of 0/1 (polyline objects): any '1' means visible
            # Keep only objects visible in at least one camera.
            vis = obj.get('visibility', {})
            vis_flags = []
            visible_to_any = False
            for key in RAW_CAM_KEYS:
                v = vis.get(key, 0)
                if isinstance(v, str):
                    flag = any(ch == '1' for ch in v)
                else:
                    flag = float(v) > 0
                vis_flags.append(flag)
                visible_to_any = visible_to_any or flag
            if not visible_to_any:
                total_skipped += 1
                continue

            # yaw from quaternion
            qx,qy,qz,qw = quat
            siny = 2*(qw*qz + qx*qy)
            cosy = 1 - 2*(qy*qy + qz*qz)
            yaw = np.arctan2(siny, cosy)

            # [x, y, z, w, l, h, yaw]
            gt_boxes.append([pos[0], pos[1], pos[2], scale[1], scale[0], scale[2], yaw])
            gt_names.append(cls_name)
            gt_visibility.append(vis_flags)

            tid = obj.get('obj_track_id','0')
            if isinstance(tid, str):
                try: tid = int(tid.replace('_',''))
                except: tid = hash(tid) % 1000000
            instance_ids.append(int(tid))
        
        n = len(gt_boxes)
        info = {
            'token': f"{ds_name}_{ts_sec:.3f}",
            'timestamp': ts_sec,
            'lidar2ego_translation': [0.0,0.0,0.0],
            'lidar2ego_rotation': [1.0,0.0,0.0,0.0],
            'ego2global_translation': ego_t,
            'ego2global_rotation': ego_rot_wxyz,
            'cams': cams,
            'lidar_path': '',
            'sweeps': [],
            'gt_boxes': np.array(gt_boxes, dtype=np.float32) if n>0 else np.zeros((0,7), dtype=np.float32),
            'gt_names': np.array(gt_names) if n>0 else np.array([], dtype=str),
            'gt_velocity': np.zeros((max(n,1),2), dtype=np.float32)[:n],
            'num_lidar_pts': np.ones(n, dtype=np.int32) if n>0 else np.zeros(0, dtype=np.int32),
            'valid_flag': np.ones(n, dtype=bool) if n>0 else np.zeros(0, dtype=bool),
            'instance_inds': np.array(instance_ids, dtype=np.int32) if n>0 else np.zeros(0, dtype=np.int32),
            'gt_visibility': np.array(gt_visibility, dtype=bool) if n>0 else np.zeros((0, len(RAW_CAM_KEYS)), dtype=bool),
        }
        infos.append(info)

    if total_skipped:
        print(f"  ⚠ Skipped {total_skipped} objects (not visible in any camera)")
    return infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/mnt/e/BEVData')
    parser.add_argument('--out-dir', default='data/fisheye_v3')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--datasets', type=str, default='',
                       help='Comma-separated dataset names, default=all')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all dataset dirs
    ds_candidates = args.datasets.split(',') if args.datasets else None
    ds_dirs = []
    for name in sorted(os.listdir(args.data_root)):
        d = os.path.join(args.data_root, name)
        if not os.path.isdir(d): continue
        if not os.path.exists(os.path.join(d, 'calibration.yml')): continue
        if not os.path.exists(os.path.join(d, 'frames_labels')): continue
        if ds_candidates and name not in ds_candidates: continue
        ds_dirs.append((name, d))
    
    print(f"Found {len(ds_dirs)} datasets: {[n for n,_ in ds_dirs]}")
    
    # Load calibration from FIRST dataset (all should share same cameras)
    first_calib = os.path.join(ds_dirs[0][1], 'calibration.yml')
    K, D, lidar2cam, image_wh, s2l_rot, s2l_trans, cam_intrinsics = load_calibration(first_calib)
    np.savez(os.path.join(args.out_dir, 'fisheye_calib_4cam.npz'),
             K=K, D=D, lidar2cam=lidar2cam, image_wh=image_wh,
             camera_order=np.array(RAW_CAM_KEYS))
    print(f"Saved calibration: {args.out_dir}/fisheye_calib_4cam.npz")
    
    all_infos = []
    class_counts = Counter()
    
    for ds_name, ds_dir in ds_dirs:
        print(f"\nProcessing {ds_name}...")
        infos = process_dataset(ds_dir, ds_name, s2l_rot, s2l_trans, cam_intrinsics)
        for i in infos: class_counts.update(i['gt_names'])
        all_infos.extend(infos)
        print(f"  → {len(infos)} frames, {sum(len(i['gt_boxes']) for i in infos)} boxes")
    
    all_infos.sort(key=lambda x: x['timestamp'])
    print(f"\nTotal: {len(all_infos)} frames")
    
    # Scene-level split: per dataset, keep last val_ratio% as val
    train_infos, val_infos = [], []
    ds_groups = {}
    for info in all_infos:
        ds = info['token'].split('_')[0]
        ds_groups.setdefault(ds, []).append(info)
    
    for ds, infos in ds_groups.items():
        infos.sort(key=lambda x: x['timestamp'])
        n_val = max(1, int(len(infos) * args.val_ratio))
        train_infos.extend(infos[:-n_val] if n_val > 0 else infos)
        val_infos.extend(infos[-n_val:])
    
    # Build sweeps for temporal training
    for info_list, label in [(train_infos, 'train'), (val_infos, 'val')]:
        info_list.sort(key=lambda x: x['timestamp'])
        # Set scene tokens and sweeps by dataset group
        ds_seqs = {}
        for info in info_list:
            ds = info['token'].split('_')[0]
            ds_seqs.setdefault(ds, []).append(info)
        for ds, seq in ds_seqs.items():
            for i, info in enumerate(seq):
                info['scene_token'] = ds
                if i > 0:
                    info['sweeps'] = [seq[i-1]['token']]
                else:
                    info['sweeps'] = []
    
    class_names = sorted(set(INCLUDE_CLASSES.values()))
    metadata = {
        'version': 'fisheye-v3.0',
        'num_cameras': 4,
        'camera_order': RAW_CAM_KEYS,
        'class_names': class_names,
    }
    
    for split_name, split_data in [('train', train_infos), ('val', val_infos)]:
        pkl_path = os.path.join(args.out_dir, f'fisheye_infos_{split_name}.pkl')
        pickle.dump({'infos': split_data, 'metadata': metadata}, open(pkl_path, 'wb'))
        n_boxes = sum(len(i['gt_boxes']) for i in split_data)
        print(f"{split_name}: {len(split_data)} frames, {n_boxes} boxes → {pkl_path}")
    
    print(f"\nClass distribution: {dict(class_counts.most_common())}")
    print(f"Classes: {class_names}")


if __name__ == '__main__':
    main()
