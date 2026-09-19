"""
Visualize ALL objects from a dataset's labels onto fisheye images.
Usage: python tools/visualize_dataset.py --dataset Data05 [--frames 5|all] [--camera front]

Logic: same projection as training — cam_hy_n5_avm_* params, pos_xyz in ego frame,
visibility filtering, fisheye equidistant projection.

Ground markers (arrows, parking slots, crosswalk, lane lines, etc.) are drawn as
semi-transparent filled polygons on the ground (bottom face of 3D box).
"""
import argparse, cv2, json, numpy as np, os, random, yaml

RAW_CAM_KEYS = ['cam_hy_n5_avm_front','cam_hy_n5_avm_back','cam_hy_n5_avm_left','cam_hy_n5_avm_right']
CAM_LABELS  = ['FRONT','BACK','LEFT','RIGHT']

# ── Color map (BGR) ──
TYPE_COLORS_3D = {
    'passenger_car': (0,255,0),
    'truck_tractor': (0,200,0),
    'trailer': (0,150,0),
    'pedestrian': (0,0,255),
    'bicycle': (255,255,0),
    'motorcycle': (255,180,0),
    'tricycle': (200,150,0),
    'hard_barrier': (0,165,255),
    'soft_barrier': (255,255,100),
    'gate_barrier': (0,255,200),
    'bollard': (255,0,255),
    'box': (255,128,0),
    'pole': (128,128,128),
    'speed_bump': (128,128,0),
    'cone': (200,200,0),
    'wheel_stopper': (128,255,128),
    'charging_infra': (255,255,128),
    'indoor_column': (128,0,128),
    'parking_lock': (100,100,255),
    'tree': (0,128,0),
    'traffic_sign': (0,180,180),
    'drain': (80,80,80),
    'manhole_cover': (100,100,100),
}

TYPE_COLORS_GROUND = {
    'parking_slot': (100,180,255),
    'arrow': (0,100,255),
    'arrow_heading_triangle': (0,80,220),
    'crosswalk': (0,220,220),
    'no_parking_zone': (0,100,200),
    'intersection': (160,160,255),
    'lane_line': (200,200,200),
    'geo_shape': (180,180,180),
    'dont_care_region': (60,60,60),
}

# Marker types (flat on ground, draw filled bottom face)
GROUND_MARKER_TYPES = {
    'class.parking.parking_slot',
    'class.road_marker.arrow',
    'class.road_marker.arrow_heading_triangle',
    'class.road_marker.crosswalk',
    'class.road_marker.no_parking_zone',
    'class.road.intersection',
    'class.road_marker.lane_line',
    'class.road_marker.geo_shape',
    'class.road_marker.dont_care_region',
}

# 3D object types (wireframe boxes)
GEO_OBJECT_TYPES = {
    'class.vehicle.passenger_car', 'class.vehicle.truck_tractor', 'class.trailer.open_top',
    'class.trailer.box', 'class.pedestrian.pedestrian',
    'class.cycle.bicycle', 'class.cycle.motorcycle', 'class.cycle.tricycle',
    'class.traffic_facility.hard_barrier', 'class.traffic_facility.soft_barrier',
    'class.traffic_facility.gate_barrier', 'class.traffic_facility.bollard',
    'class.traffic_facility.box', 'class.traffic_facility.pole',
    'class.traffic_facility.speed_bump', 'class.traffic_facility.cone',
    'class.traffic_facility.drain', 'class.traffic_facility.manhole_cover',
    'class.parking.wheel_stopper', 'class.parking.charging_infra',
    'class.parking.indoor_column', 'class.parking.parking_lock',
    'class.plant.tree',
    'class.sign.traffic_sign.instruction', 'class.sign.traffic_sign.prohibition_and_limit',
}


def quat_to_rot(qx, qy, qz, qw):
    R = np.zeros((3,3))
    R[0,0]=1-2*qy*qy-2*qz*qz; R[0,1]=2*qx*qy-2*qz*qw; R[0,2]=2*qx*qz+2*qy*qw
    R[1,0]=2*qx*qy+2*qz*qw; R[1,1]=1-2*qx*qx-2*qz*qz; R[1,2]=2*qy*qz-2*qx*qw
    R[2,0]=2*qx*qz-2*qy*qw; R[2,1]=2*qy*qz+2*qx*qw; R[2,2]=1-2*qx*qx-2*qy*qy
    return R


def fisheye_project_pts(pts_cam, fx, fy, cx, cy, k):
    """Fisheye equidistant projection.
    For z>0: theta = arctan(r/z) ∈ [0, 90°), point is in front of camera.
    For z<0: theta = arctan(r/z) ∈ [90°, 180°), point is behind camera plane.
    Both are valid for fisheye with FOV > 180° — the image boundary check
    (in_frame) naturally filters out points beyond the lens FOV."""
    z = pts_cam[:, 2]; x, y = pts_cam[:, 0], pts_cam[:, 1]
    r = np.sqrt(x*x + y*y) + 1e-12
    theta = np.arctan2(r, z)  # z can be negative → theta > 90°
    th2,th4,th6,th8 = theta**2, theta**4, theta**6, theta**8
    theta_d = theta * (1.0 + k[0]*th2 + k[1]*th4 + k[2]*th6 + k[3]*th8)
    scale = theta_d / r
    return np.stack([fx*scale*x+cx, fy*scale*y+cy], axis=-1)


def get_short_type(obj_type):
    return obj_type.replace('class.','').rsplit('.',1)[-1] if '.' in obj_type else obj_type


def get_color(obj_type):
    short = get_short_type(obj_type)
    if obj_type in GROUND_MARKER_TYPES:
        return TYPE_COLORS_GROUND.get(short, (180,180,180))
    return TYPE_COLORS_3D.get(short, (255,255,255))


# 8 corners of a 3D box: [l/2, w/2, h/2]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
BOTTOM_FACE = [0,1,2,3]


def project_box(pos, quat, scale, R_c2e, t_c2e, fx, fy, cx, cy, k, W, H,
                dense_edges=True, n_pts_per_edge=30):
    """Project 3D box onto fisheye. Returns (edge_curves, corners_2d, is_visible).

    On a fisheye camera, straight lines in 3D become CURVES in the image.
    Drawing straight lines between projected corners is WRONG — the error grows
    with object length and proximity to image edges (large theta).

    With dense_edges=True, each box edge is densely sampled in 3D space,
    then all sample points are projected and drawn as a polyline, producing
    correct curved edges.

    Returns:
        edge_curves: list of (N,2) int arrays, one per EDGE; None if edge invisible
        corners_2d:  (8,2) int array of corner projections (for label placement)
        is_visible:  bool, False if fewer than 2 corners are in frame
    """
    l, w, h = float(scale[0]), float(scale[1]), float(scale[2])
    R_obj = quat_to_rot(quat[0], quat[1], quat[2], quat[3])
    corners_local = np.array([
        [ l/2, w/2,-h/2],[ l/2,-w/2,-h/2],[-l/2,-w/2,-h/2],[-l/2, w/2,-h/2],
        [ l/2, w/2, h/2],[ l/2,-w/2, h/2],[-l/2,-w/2, h/2],[-l/2, w/2, h/2],
    ], dtype=np.float32)
    corners_ego = corners_local @ R_obj.T + pos

    # Project corners (fast check + label placement)
    corners_cam = (R_c2e.T @ (corners_ego - t_c2e).T).T
    corners_2d = fisheye_project_pts(corners_cam, fx, fy, cx, cy, k)
    in_frame = (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < W) & \
               (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < H)
    if in_frame.sum() < 2:
        return None, None, False

    if dense_edges:
        # Densely sample each 3D edge → project → polyline (fisheye curve).
        # Only keep the in-frame portion: edges that cross behind the camera
        # have samples projecting far outside — we clip to image boundary.
        edge_curves = []
        any_edge_visible = False
        for i, j in EDGES:
            p0 = corners_ego[i]
            p1 = corners_ego[j]
            t = np.linspace(0, 1, n_pts_per_edge)
            samples_ego = p0 + t[:, np.newaxis] * (p1 - p0)
            samples_cam = (R_c2e.T @ (samples_ego - t_c2e).T).T
            pts = fisheye_project_pts(samples_cam, fx, fy, cx, cy, k)
            visible = (pts[:, 0] >= 0) & (pts[:, 0] < W) & \
                      (pts[:, 1] >= 0) & (pts[:, 1] < H)
            if visible.sum() >= 2:
                # Keep only the in-frame segment
                in_frame_pts = pts[visible].astype(np.int32)
                edge_curves.append(in_frame_pts)
                any_edge_visible = True
            else:
                edge_curves.append(None)
        if not any_edge_visible:
            return None, None, False
        return edge_curves, corners_2d.astype(int), True
    else:
        # Legacy: return 8 corners only (straight-line drawing, pinhole-like)
        return None, corners_2d.astype(int), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name, e.g. Data05')
    parser.add_argument('--data-root', default='/mnt/e/BEVData')
    parser.add_argument('--out-dir', default='output/gt_vis')
    parser.add_argument('--frames', type=str, default='5',
                        help='Number of frames, or "all"')
    parser.add_argument('--camera', default='all', choices=['all','front','back','left','right'])
    parser.add_argument('--cars-only', action='store_true')
    parser.add_argument('--no-ground', action='store_true')
    args = parser.parse_args()

    ds_dir = os.path.join(args.data_root, args.dataset)
    calib = yaml.safe_load(open(os.path.join(ds_dir, 'calibration.yml')))
    rig = calib['rig']

    poses = {}
    for fname in os.listdir(ds_dir):
        if 'pose' in fname.lower() and fname.endswith('.txt'):
            with open(os.path.join(ds_dir, fname)) as f:
                for line in f:
                    p = [float(x) for x in line.strip().split(',')]
                    poses[p[0]] = p[1:]
            break

    label_dir = os.path.join(ds_dir, 'frames_labels')
    all_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])

    if args.frames.lower() == 'all':
        selected = all_files
        print(f"Processing ALL {len(selected)} frames")
    else:
        n = int(args.frames)
        random.seed(42)
        scored = []
        for fname in all_files[:min(200, len(all_files))]:
            objs = json.load(open(os.path.join(label_dir, fname)))
            n_vis = sum(1 for obj in objs if any(
                float(obj.get('visibility',{}).get(k,0))>0 for k in RAW_CAM_KEYS))
            scored.append((fname, n_vis))
        scored.sort(key=lambda x: -x[1])
        selected = [s[0] for s in scored[:n]]
        print(f"Selected {len(selected)} / {len(all_files)} frames")

    cam_indices = range(4) if args.camera == 'all' else [CAM_LABELS.index(args.camera.upper())]
    out_root = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_root, exist_ok=True)

    total_geo, total_ground = 0, 0

    for fidx, fname in enumerate(selected):
        ts_sec = float(fname.replace('.json',''))
        ts_ms = str(int(ts_sec*1000))
        objs = json.load(open(os.path.join(label_dir, fname)))

        # Classify
        geo_list, ground_list = [], []
        for obj in objs:
            ot = obj['obj_type']
            if args.cars_only and ot != 'class.vehicle.passenger_car':
                continue
            if ot in GROUND_MARKER_TYPES:
                ground_list.append(obj)
            elif ot in GEO_OBJECT_TYPES:
                geo_list.append(obj)

        if args.no_ground:
            ground_list = []

        if fidx == 0 or len(selected) <= 20 or (fidx+1) % 50 == 0:
            print(f"  [{fidx+1}/{len(selected)}] t={ts_sec:.3f}: {len(geo_list)} geo + {len(ground_list)} ground")

        for cam_idx in cam_indices:
            key = RAW_CAM_KEYS[cam_idx]
            cfg = rig[key]
            fx, fy = cfg['focal']; cx, cy = cfg['pp']
            W, H = int(cfg['image_size'][0]), int(cfg['image_size'][1])
            k = cfg['inv_poly'][:4]
            ext = cfg['extrinsic']
            R_c2e = quat_to_rot(ext[3],ext[4],ext[5],ext[6])
            t_c2e = np.array([ext[0],ext[1],ext[2]], dtype=np.float32)

            # Find image file
            img_path = os.path.join(ds_dir, 'camera', key, f'{ts_ms}.jpg')
            if not os.path.exists(img_path):
                cam_dir = os.path.join(ds_dir, 'camera', key)
                if os.path.isdir(cam_dir):
                    for cand in os.listdir(cam_dir):
                        try:
                            cts = float(os.path.splitext(cand)[0])
                            if abs(cts/1000.0 - ts_sec) < 0.002:
                                img_path = os.path.join(cam_dir, cand)
                                break
                        except: pass
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None: continue

            overlay = img.copy()
            vis_geo, vis_ground = 0, 0

            # ── 1. Ground markers: filled bottom polygon ──
            for obj in ground_list:
                vis = obj.get('visibility',{})
                if float(vis.get(key,0)) <= 0: continue
                geom = obj['geometry']
                if isinstance(geom, list):
                    geom = geom[0] if len(geom)>0 and isinstance(geom[0], dict) else None
                    if geom is None: continue
                pos = np.array(geom['pos_xyz'], dtype=np.float32)
                quat = geom['quat']; scale = geom['scale_xyz']
                if float(scale[0])*float(scale[1]) < 0.01: continue
                edge_curves, corners, ok = project_box(
                    pos, quat, scale, R_c2e, t_c2e, fx, fy, cx, cy, k, W, H)
                if not ok: continue
                color = get_color(obj['obj_type'])
                # Build dense bottom-face polygon from bottom edge curves
                # BOTTOM_FACE = [0,1,2,3] → edges [0,1]=idx0, [1,2]=idx1, [2,3]=idx2, [3,0]=idx3
                bottom_pts = []
                for idx in [0, 1, 2, 3]:
                    curve = edge_curves[idx]
                    if curve is None:
                        continue
                    if len(bottom_pts) == 0:
                        bottom_pts.extend(curve)
                    else:
                        # Skip first point if it matches last point (shared corner)
                        if np.allclose(curve[0], bottom_pts[-1], atol=1.5):
                            bottom_pts.extend(curve[1:])
                        else:
                            bottom_pts.extend(curve)
                if len(bottom_pts) >= 3:
                    poly = np.array(bottom_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [poly], color)
                    vis_ground += 1

            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

            # ── 2. 3D objects: wireframe ──
            for obj in geo_list:
                vis = obj.get('visibility',{})
                if float(vis.get(key,0)) <= 0: continue
                geom = obj['geometry']
                if isinstance(geom, list):
                    geom = geom[0] if len(geom)>0 and isinstance(geom[0], dict) else None
                    if geom is None: continue
                pos = np.array(geom['pos_xyz'], dtype=np.float32)
                quat = geom['quat']; scale = geom['scale_xyz']
                if min(float(scale[0]),float(scale[1]),float(scale[2])) < 0.005: continue
                edge_curves, corners, ok = project_box(
                    pos, quat, scale, R_c2e, t_c2e, fx, fy, cx, cy, k, W, H)
                if not ok: continue
                color = get_color(obj['obj_type'])
                for curve in edge_curves:
                    if curve is not None:
                        cv2.polylines(img, [curve.reshape((-1, 1, 2))], False, color, 2)
                label = get_short_type(obj['obj_type'])[:10]
                cv2.putText(img, label, tuple(corners[5]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                vis_geo += 1

            total_geo += vis_geo
            total_ground += vis_ground

            cv2.putText(img, f"{args.dataset} {CAM_LABELS[cam_idx]} t={ts_sec:.3f} geo={vis_geo} ground={vis_ground}",
                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            out_name = f"{ts_sec:.3f}_{CAM_LABELS[cam_idx]}.jpg"
            cv2.imwrite(os.path.join(out_root, out_name), img)

    print(f"\nDone! {len(selected)} frames × {len(cam_indices)} cameras")
    print(f"  3D objects: {total_geo}  |  Ground markers: {total_ground}")
    print(f"  Output: {out_root}/")


if __name__ == '__main__':
    main()
