#!/usr/bin/env python3
"""Convert an RTAB-Map database to a COLMAP text model for 3D Gaussian Splatting.

Uses rtabmap-export to get the optimized camera poses (after loop closures),
the rectified left images, the ground truth poses (if any) and the assembled
stereo point cloud, then writes:

  OUT/images/<id>.jpg
  OUT/sparse/0/cameras.txt      PINHOLE model of the rectified left camera
  OUT/sparse/0/images.txt       world -> camera poses (COLMAP convention)
  OUT/sparse/0/points3D.txt     initial points (no tracks)
  OUT/sparse/0/points3D.ply     same points as a PLY file
  OUT/gt_camera_poses.txt       ground truth camera poses (stamp x y z qx qy qz qw id)
  OUT/camera_poses.txt          estimated camera poses, same format

With --euroc DIR (the EuRoC-layout recording the map was made from, with the
D435i color stream in mav0/rgb, see realsense_bag_to_euroc.py --rgb), the color
images are exported instead of the infrared ones, and depth maps are added:

  OUT/images/<id>.jpg           color image nearest to each keyframe; its pose is the
                                keyframe pose interpolated at the color stamp, moved to
                                the color camera with the T_BS extrinsics of sensor.yaml
  OUT/depths/<id>.png           16-bit depth (mm) in the color camera, from the infrared
                                stereo pair of the keyframe (OpenCV SGBM), 0 = unknown; from the
                                nearest pair with the emitter on when mav0/cam0_emitter exists
  OUT/sparse/0/points3D.txt     colored points back-projected from the depth maps
  OUT/frames.txt                name, stamp and keyframe flag (1) of each image

--extra-rate HZ adds color images between the keyframes (the sharpest one in each
1/HZ s slot, pose interpolated between the keyframes, depth from its own infrared
pair): more views for the Gaussians. Their names start with x, so train_gsplat.py
still holds out the same keyframe images for evaluation (and also leaves out the
extra images next to them).

Usage: rtabmap_to_colmap.py map.db OUT [--euroc DIR] [--rtabmap-export PATH] [--voxel 0.02] [--max-points 200000]
"""
import argparse, math, os, random, re, shutil, struct, subprocess, sys, tempfile


def quat_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return [[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]]


def rot_to_quat(R):
    """Returns (qw, qx, qy, qz)."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25*s, (R[2][1]-R[1][2])/s, (R[0][2]-R[2][0])/s, (R[1][0]-R[0][1])/s)
    if R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        return ((R[2][1]-R[1][2])/s, 0.25*s, (R[0][1]+R[1][0])/s, (R[0][2]+R[2][0])/s)
    if R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        return ((R[0][2]-R[2][0])/s, (R[0][1]+R[1][0])/s, 0.25*s, (R[1][2]+R[2][1])/s)
    s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
    return ((R[1][0]-R[0][1])/s, (R[0][2]+R[2][0])/s, (R[1][2]+R[2][1])/s, 0.25*s)


def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def read_poses(path):
    """TUM-like poses with id (rtabmap-export --poses_format 11): id -> (stamp, R, t)."""
    poses = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            v = line.split()
            stamp, x, y, z, qx, qy, qz, qw = map(float, v[:8])
            poses[int(v[8])] = (stamp, quat_to_rot(qx, qy, qz, qw), [x, y, z])
    return poses


def write_poses(path, poses):
    with open(path, 'w') as f:
        f.write('#timestamp x y z qx qy qz qw id\n')
        for i in sorted(poses):
            stamp, R, t = poses[i]
            qw, qx, qy, qz = rot_to_quat(R)
            f.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d\n' % (stamp, t[0], t[1], t[2], qx, qy, qz, qw, i))


def yaml_matrix(text, name):
    m = re.search(name + r':\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*data:\s*\[([^\]]*)\]', text)
    if not m:
        return None
    rows, cols = int(m.group(1)), int(m.group(2))
    data = [float(x) for x in m.group(3).replace('\n', ' ').split(',')]
    return [data[r*cols:(r+1)*cols] for r in range(rows)]


def read_ply_vertices(path):
    """Binary little endian or ASCII PLY: list of (x, y, z, r, g, b)."""
    sizes = {'char': 'b', 'uchar': 'B', 'int8': 'b', 'uint8': 'B', 'short': 'h', 'ushort': 'H',
             'int': 'i', 'uint': 'I', 'int32': 'i', 'uint32': 'I', 'float': 'f', 'float32': 'f',
             'double': 'd', 'float64': 'd', 'int16': 'h', 'uint16': 'H'}
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii', 'replace').strip()
            header.append(line)
            if line == 'end_header':
                break
        fmt = [l for l in header if l.startswith('format')][0].split()[1]
        count, props, in_vertex = 0, [], False
        for l in header:
            v = l.split()
            if v[0] == 'element':
                in_vertex = v[1] == 'vertex'
                if in_vertex:
                    count = int(v[2])
            elif v[0] == 'property' and in_vertex:
                props.append((v[2], v[1]))
        names = [p[0] for p in props]
        ix = [names.index(c) for c in ('x', 'y', 'z')]
        ic = [names.index(c) if c in names else -1 for c in ('red', 'green', 'blue')]
        points = []
        if fmt == 'ascii':
            for _ in range(count):
                v = f.readline().split()
                rgb = [int(float(v[i])) if i >= 0 else 128 for i in ic]
                points.append((float(v[ix[0]]), float(v[ix[1]]), float(v[ix[2]]), rgb[0], rgb[1], rgb[2]))
        else:
            endian = '<' if fmt == 'binary_little_endian' else '>'
            st = struct.Struct(endian + ''.join(sizes[p[1]] for p in props))
            data = f.read(st.size * count)
            for v in st.iter_unpack(data):
                rgb = [v[i] if i >= 0 else 128 for i in ic]
                points.append((v[ix[0]], v[ix[1]], v[ix[2]], rgb[0], rgb[1], rgb[2]))
    return points


def read_euroc_camera(folder):
    """EuRoC sensor.yaml + data.csv: (K 3x3, (width, height), T_BS 4x4 camera -> body, [(stamp s, path)])."""
    import numpy as np
    text = open(os.path.join(folder, 'sensor.yaml')).read()
    t_bs = [float(x) for x in re.search(r'T_BS:.*?data:\s*\[([^\]]*)\]', text, re.S).group(1).split(',')]
    fx, fy, cx, cy = [float(x) for x in re.search(r'intrinsics:\s*\[([^\]]*)\]', text).group(1).split(',')]
    w, h = [int(x) for x in re.search(r'resolution:\s*\[([^\]]*)\]', text).group(1).split(',')]
    frames = []
    for line in open(os.path.join(folder, 'data.csv')):
        if line.startswith('#') or not line.strip():
            continue
        stamp, name = line.strip().split(',')[:2]
        frames.append((int(stamp) * 1e-9, os.path.join(folder, 'data', name.strip())))
    frames.sort()
    return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), (w, h),
            np.array(t_bs).reshape(4, 4), frames)


def to_matrix(R, t):
    import numpy as np
    T = np.eye(4)
    T[:3, :3] = np.array(R)
    T[:3, 3] = t
    return T


def interpolate_pose(stamps, poses, t):
    """Pose at time t, linear in translation and slerp in rotation between the two nearest stamps."""
    import numpy as np
    i = int(np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1))
    t0, t1 = stamps[i-1], stamps[i]
    a = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), -0.5, 1.5))
    A, B = poses[i-1], poses[i]
    dR = A[:3, :3].T @ B[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
    if angle < 1e-9:
        R = A[:3, :3]
    else:
        axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * math.sin(angle))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        th = a * angle
        R = A[:3, :3] @ (np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k)
    return to_matrix(R, (1 - a) * A[:3, 3] + a * B[:3, 3])


def stereo_depth(left, right, fx, baseline, max_depth):
    """Depth (m) of the rectified left infrared image, 0 where unknown."""
    import cv2
    import numpy as np
    block = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=block,
                                 P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(left, right).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > 1.0
    depth[valid] = fx * baseline / disp[valid]
    depth[depth > max_depth] = 0
    return depth


def project_depth(depth_ir, K_ir, T_rgb_ir, K_rgb, size_rgb):
    """Moves an infrared depth map to the color camera (z-buffer; each pixel covers the area of an
    infrared pixel in the color image, which has about twice the resolution on the D435i)."""
    import numpy as np
    h, w = depth_ir.shape
    v, u = np.nonzero(depth_ir)
    z = depth_ir[v, u]
    pts = np.stack([(u - K_ir[0, 2]) * z / K_ir[0, 0], (v - K_ir[1, 2]) * z / K_ir[1, 1], z], 1)
    pts = pts @ T_rgb_ir[:3, :3].T + T_rgb_ir[:3, 3]
    pts = pts[pts[:, 2] > 0.05]
    W, H = size_rgb
    out = np.full((H, W), np.inf, np.float32)
    x = pts[:, 0] / pts[:, 2] * K_rgb[0, 0] + K_rgb[0, 2]
    y = pts[:, 1] / pts[:, 2] * K_rgb[1, 1] + K_rgb[1, 2]
    size = max(1, int(math.ceil(K_rgb[0, 0] / K_ir[0, 0])))
    for dx in range(size):
        for dy in range(size):
            xi = np.floor(x - (size - 1) / 2.0).astype(int) + dx
            yi = np.floor(y - (size - 1) / 2.0).astype(int) + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            np.minimum.at(out, (yi[ok], xi[ok]), pts[ok, 2])
    out[np.isinf(out)] = 0
    return out


EXTRA_ID = 1000000  # image ids of the extra images (keyframe images keep their node id)


def export_euroc_color(args, cameras, out_images, out_depths):
    """Color images, their poses and depth maps from the EuRoC recording. Returns (names, poses, K, size, points)."""
    import cv2
    import numpy as np
    mav = os.path.join(args.euroc, 'mav0') if os.path.isdir(os.path.join(args.euroc, 'mav0')) else args.euroc
    if not os.path.isdir(os.path.join(mav, 'rgb')):
        sys.exit('No color stream in %s (convert the recording with realsense_bag_to_euroc.py --rgb)' % mav)
    K_rgb, size_rgb, T_b_rgb, rgb_frames = read_euroc_camera(os.path.join(mav, 'rgb'))
    # Emitter On Off recordings (realsense_bag_to_euroc.py --emitter-split): depth from the
    # frames with the dot pattern, which are not the keyframes' frames
    emitter = os.path.isdir(os.path.join(mav, 'cam0_emitter'))
    suffix = '_emitter' if emitter else ''
    K_ir, _, T_b_c0, left_frames = read_euroc_camera(os.path.join(mav, 'cam0' + suffix))
    _, _, T_b_c1, right_frames = read_euroc_camera(os.path.join(mav, 'cam1' + suffix))
    if emitter:
        print('Depth from the infrared frames with the emitter on (cam0_emitter, cam1_emitter)')
    baseline = float(np.linalg.norm((np.linalg.inv(T_b_c0) @ T_b_c1)[:3, 3]))
    T_c0_rgb = np.linalg.inv(T_b_c0) @ T_b_rgb

    ids = sorted(cameras, key=lambda i: cameras[i][0])
    kf_stamps = np.array([cameras[i][0] for i in ids])
    kf_poses = [to_matrix(cameras[i][1], cameras[i][2]) for i in ids]
    rgb_stamps = np.array([f[0] for f in rgb_frames])
    left_stamps = np.array([f[0] for f in left_frames])
    right_by_stamp = {round(f[0], 4): f[1] for f in right_frames}

    def depth_at(stamp, T_w_rgb):
        """Depth of the infrared pair nearest to stamp, moved to the color camera (None if no pair)."""
        li = int(np.argmin(np.abs(left_stamps - stamp)))
        right_path = right_by_stamp.get(round(left_stamps[li], 4))
        if abs(left_stamps[li] - stamp) > (0.04 if emitter else 0.02) or right_path is None:
            return None
        left = cv2.imread(left_frames[li][1], cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            return None
        depth_ir = stereo_depth(left, right, K_ir[0, 0], baseline, args.max_depth)
        T_rgb_ir = np.linalg.inv(T_w_rgb) @ interpolate_pose(kf_stamps, kf_poses, left_stamps[li])
        return project_depth(depth_ir, K_ir, T_rgb_ir, K_rgb, size_rgb)

    def save(i, name, j, img, T_w_rgb, depth):
        cv2.imwrite(os.path.join(out_images, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names[i] = name
        poses[i] = (rgb_stamps[j], T_w_rgb[:3, :3].tolist(), T_w_rgb[:3, 3].tolist())
        if depth is not None:
            cv2.imwrite(os.path.join(out_depths, os.path.splitext(name)[0] + '.png'), np.round(depth * 1000).astype(np.uint16))

    names, poses, points, colors = {}, {}, [], []
    keyframe_rgb = set()
    rng = np.random.default_rng(0)
    for k, i in enumerate(ids):
        stamp = kf_stamps[k]
        j = int(np.argmin(np.abs(rgb_stamps - stamp)))
        if abs(rgb_stamps[j] - stamp) > args.max_dt:
            print('Keyframe %d: no color image within %.0f ms, skipped' % (i, args.max_dt * 1000))
            continue
        T_w_rgb = interpolate_pose(kf_stamps, kf_poses, rgb_stamps[j]) @ T_c0_rgb
        img = cv2.imread(rgb_frames[j][1], cv2.IMREAD_COLOR)
        if img is None:
            print('Cannot read %s, skipped' % rgb_frames[j][1])
            continue
        # Depth of the keyframe's infrared pair, moved to the color camera at the color stamp
        depth = depth_at(stamp, T_w_rgb)
        save(i, '%d.jpg' % i, j, img, T_w_rgb, depth)
        keyframe_rgb.add(j)
        if depth is None:
            continue

        # Colored initial points from a random subset of the depth pixels
        v, u = np.nonzero(depth)
        if len(v) == 0:
            continue
        sel = rng.choice(len(v), min(len(v), 20000), replace=False)
        v, u = v[sel], u[sel]
        z = depth[v, u]
        pc = np.stack([(u - K_rgb[0, 2]) * z / K_rgb[0, 0], (v - K_rgb[1, 2]) * z / K_rgb[1, 1], z], 1)
        points.append(pc @ T_w_rgb[:3, :3].T + T_w_rgb[:3, 3])
        colors.append(img[v, u, ::-1])
        if k % 20 == 0:
            print('%d/%d keyframes, %.0f%% of the color pixels with depth' % (k + 1, len(ids), 100.0 * np.count_nonzero(depth) / depth.size))

    if args.extra_rate > 0:
        # Sharpest color image (variance of the Laplacian) of each 1/rate slot between the keyframes
        slot = np.floor((rgb_stamps - kf_stamps[0]) * args.extra_rate).astype(int)
        inside = (rgb_stamps > kf_stamps[0]) & (rgb_stamps < kf_stamps[-1])
        kf_slots = {slot[j] for j in keyframe_rgb}
        best = {}
        for j in np.nonzero(inside)[0]:
            if slot[j] in kf_slots:
                continue
            gray = cv2.imread(rgb_frames[j][1], cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            sharp = cv2.Laplacian(cv2.resize(gray, None, fx=0.5, fy=0.5), cv2.CV_32F).var()
            if slot[j] not in best or sharp > best[slot[j]][1]:
                best[slot[j]] = (j, sharp)
        for n, (j, _) in enumerate(sorted(best.values())):
            T_w_rgb = interpolate_pose(kf_stamps, kf_poses, rgb_stamps[j]) @ T_c0_rgb
            img = cv2.imread(rgb_frames[j][1], cv2.IMREAD_COLOR)
            save(EXTRA_ID + n, 'x%05d.jpg' % n, j, img, T_w_rgb, depth_at(rgb_stamps[j], T_w_rgb))
            if n % 50 == 0:
                print('%d/%d extra images' % (n + 1, len(best)))
        print('%d extra images between the keyframes (%.1f per second)' % (len(best), args.extra_rate))

    pts = []
    if points:
        P = np.concatenate(points)
        C = np.concatenate(colors)
        # One point per voxel
        keys = np.floor(P / args.voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        pts = [(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(P[first].tolist(), C[first].tolist())]
    return names, poses, K_rgb, size_rgb, pts


def find_rtabmap_export():
    """rtabmap-export in PATH, else in the default install folders (Windows installer, /usr/local)."""
    found = shutil.which('rtabmap-export')
    if found:
        return found
    candidates = [os.path.join(os.environ.get(v, ''), 'RTABMap', 'bin', 'rtabmap-export.exe')
                  for v in ('ProgramFiles', 'ProgramFiles(x86)') if os.environ.get(v)]
    candidates.append('/usr/local/bin/rtabmap-export')
    for c in candidates:
        if os.path.exists(c):
            return c
    return 'rtabmap-export'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('database')
    parser.add_argument('output')
    parser.add_argument('--rtabmap-export', default=find_rtabmap_export(),
                        help='rtabmap-export executable (e.g., C:/Program Files/RTABMap/bin/rtabmap-export.exe)')
    parser.add_argument('--voxel', type=float, default=0.02, help='Voxel size (m) of the initial point cloud.')
    parser.add_argument('--max-points', type=int, default=200000, help='Random subset of the initial points (0 = all).')
    parser.add_argument('--euroc', help='EuRoC-layout recording of the map with mav0/rgb: export color images and depth maps.')
    parser.add_argument('--max-depth', type=float, default=4.0, help='Maximum stereo depth (m) with --euroc.')
    parser.add_argument('--max-dt', type=float, default=0.02, help='Maximum time (s) between a keyframe and its color image with --euroc.')
    parser.add_argument('--extra-rate', type=float, default=0, help='--euroc: color images added between the keyframes, per second (0 = keyframes only).')
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix='rtabmap_colmap_')
    try:
        cmd = [args.rtabmap_export, '--poses_camera', '--poses_gt', '--poses_format', '11', '--images_id',
               '--cloud', '--voxel', str(args.voxel), '--output', 'map', '--output_dir', tmp, args.database]
        print(' '.join(cmd))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except FileNotFoundError:
            sys.exit('rtabmap-export not found: install RTAB-Map or give its path with --rtabmap-export')

        cameras = read_poses(os.path.join(tmp, 'map_camera_poses.txt'))
        if not cameras:
            sys.exit('No camera poses exported')

        # Rectified intrinsics and base -> camera transform of the left camera
        calib_dir = os.path.join(tmp, 'map_calib')
        first = min(cameras)
        # Stereo: <id>_left.yaml, RGB-D (e.g., RealSense color + depth): <id>.yaml, multi-camera: <id>_0.yaml
        calib_file = next((os.path.join(calib_dir, n % first) for n in ('%d_left.yaml', '%d.yaml', '%d_0.yaml')
                           if os.path.exists(os.path.join(calib_dir, n % first))),
                          os.path.join(calib_dir, sorted(os.listdir(calib_dir))[0]))
        text = open(calib_file).read()
        width = int(re.search(r'image_width:\s*(\d+)', text).group(1))
        height = int(re.search(r'image_height:\s*(\d+)', text).group(1))
        P = yaml_matrix(text, 'projection_matrix')
        K = yaml_matrix(text, 'camera_matrix')
        if P is not None and P[0][0] > 0:
            fx, fy, cx, cy = P[0][0], P[1][1], P[0][2], P[1][2]
        else:
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        L = yaml_matrix(text, 'local_transform')  # base -> camera optical frame (3x4)

        out_images = os.path.join(args.output, 'images')
        out_sparse = os.path.join(args.output, 'sparse', '0')
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_sparse, exist_ok=True)

        euroc_points = None
        if args.euroc:
            out_depths = os.path.join(args.output, 'depths')
            os.makedirs(out_depths, exist_ok=True)
            names, cameras, K, (width, height), euroc_points = export_euroc_color(args, cameras, out_images, out_depths)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            L = None  # ground truth stays in the infrared camera frame: not exported
        else:
            # Left image of a stereo camera, or color image of an RGB-D camera
            image_dir = next((os.path.join(tmp, d) for d in ('map_left', 'map_rgb', 'map')
                              if os.path.isdir(os.path.join(tmp, d))), None)
            if image_dir is None:
                sys.exit('No image exported from the database (were images saved in it?)')
            names = {}
            for fname in os.listdir(image_dir):
                stem = os.path.splitext(fname)[0]
                if stem.isdigit() and int(stem) in cameras:
                    shutil.copy(os.path.join(image_dir, fname), os.path.join(out_images, fname))
                    names[int(stem)] = fname

        with open(os.path.join(out_sparse, 'cameras.txt'), 'w') as f:
            f.write('# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n')
            f.write('1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n' % (width, height, fx, fy, cx, cy))

        with open(os.path.join(out_sparse, 'images.txt'), 'w') as f:
            f.write('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (empty)\n')
            for i in sorted(names):
                _, R, t = cameras[i]
                Rcw = transpose(R)  # world -> camera
                tcw = [-sum(Rcw[r][k]*t[k] for k in range(3)) for r in range(3)]
                qw, qx, qy, qz = rot_to_quat(Rcw)
                f.write('%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 %s\n\n' % (i, qw, qx, qy, qz, tcw[0], tcw[1], tcw[2], names[i]))
        write_poses(os.path.join(args.output, 'camera_poses.txt'), {i: cameras[i] for i in names})
        with open(os.path.join(args.output, 'frames.txt'), 'w') as f:
            f.write('# NAME STAMP KEYFRAME\n')
            for i in sorted(names, key=lambda i: cameras[i][0]):
                f.write('%s %.6f %d\n' % (names[i], cameras[i][0], 0 if i >= EXTRA_ID else 1))

        gt_path = os.path.join(tmp, 'map_gt_poses.txt')
        gt_count = 0
        if os.path.exists(gt_path) and L is not None:
            Rl = [row[:3] for row in L]
            tl = [row[3] for row in L]
            gt = {}
            for i, (stamp, R, t) in read_poses(gt_path).items():
                if i in names:
                    gt[i] = (stamp, matmul(R, Rl), [t[r] + sum(R[r][k]*tl[k] for k in range(3)) for r in range(3)])
            write_poses(os.path.join(args.output, 'gt_camera_poses.txt'), gt)
            gt_count = len(gt)

        points = euroc_points if euroc_points else read_ply_vertices(os.path.join(tmp, 'map_cloud.ply'))
        total = len(points)
        if args.max_points > 0 and len(points) > args.max_points:
            random.seed(0)
            points = random.sample(points, args.max_points)
        with open(os.path.join(out_sparse, 'points3D.txt'), 'w') as f:
            f.write('# POINT3D_ID X Y Z R G B ERROR TRACK[]\n')
            for k, p in enumerate(points):
                f.write('%d %.5f %.5f %.5f %d %d %d 0\n' % (k+1, p[0], p[1], p[2], p[3], p[4], p[5]))
        with open(os.path.join(out_sparse, 'points3D.ply'), 'wb') as f:
            f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
                     'property float x\nproperty float y\nproperty float z\n'
                     'property float nx\nproperty float ny\nproperty float nz\n'
                     'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(points)).encode())
            st = struct.Struct('<ffffffBBB')
            for p in points:
                f.write(st.pack(p[0], p[1], p[2], 0, 0, 0, p[3], p[4], p[5]))

        print('%d images, %d ground truth poses, %d points (of %d), camera PINHOLE %dx%d f=%.1f,%.1f c=%.1f,%.1f -> %s'
              % (len(names), gt_count, len(points), total, width, height, fx, fy, cx, cy, args.output))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
