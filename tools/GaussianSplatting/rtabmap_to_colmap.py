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

Usage: rtabmap_to_colmap.py map.db OUT [--rtabmap-export PATH] [--voxel 0.02] [--max-points 200000]
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
        calib_file = os.path.join(calib_dir, '%d_left.yaml' % first)
        if not os.path.exists(calib_file):
            calib_file = os.path.join(calib_dir, sorted(os.listdir(calib_dir))[0])
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

        image_dir = os.path.join(tmp, 'map_left')
        if not os.path.isdir(image_dir):
            image_dir = os.path.join(tmp, 'map')
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

        points = read_ply_vertices(os.path.join(tmp, 'map_cloud.ply'))
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
