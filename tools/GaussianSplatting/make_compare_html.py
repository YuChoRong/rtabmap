#!/usr/bin/env python3
"""
Writes a single HTML file that compares SLAM trajectories with a COLMAP SfM
reconstruction. It opens in any browser (double-click on Windows), with no
installation: 3D view of both maps, per-frame errors and the keyframe images.

Usage:
  make_compare_html.py SFM_DIR OUT.html [--slam NAME=camera_poses.txt ...]
                       [--slam-map SLAM_COLMAP_DIR] [--max-points 40000]

SFM_DIR is the folder made by run_colmap_sfm.py (sparse/0/, images/,
slam_camera_poses.txt). Without --slam, its slam_camera_poses.txt is used.
--slam-map adds the SLAM point cloud (sparse/0/points3D.txt of the
rtabmap_to_colmap.py export). SfM is aligned to the first SLAM trajectory
with a similarity transform (SfM has no metric scale).
"""
import argparse, base64, json, os, sys
import numpy as np
import cv2


def q2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, D, Vt = np.linalg.svd(B.T @ A / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / A.var(0).sum()
    return s, R, md - s * R @ ms


def frame_id(name):
    return int(os.path.splitext(os.path.basename(name))[0])


def read_tum(path):
    """timestamp x y z qx qy qz qw id -> {id: position}"""
    poses = {}
    for line in open(path):
        v = line.split()
        if not v or v[0].startswith('#'):
            continue
        poses[int(v[8]) if len(v) > 8 else len(poses)] = np.array(list(map(float, v[1:4])))
    return poses


def read_sfm(model_dir):
    cams = {}
    lines = [l for l in open(os.path.join(model_dir, 'images.txt')) if not l.startswith('#')]
    for l in lines[0::2]:
        v = l.split()
        if len(v) < 10:
            continue
        R = q2R(*map(float, v[1:5]))
        cams[frame_id(v[9])] = (-R.T @ np.array(list(map(float, v[5:8]))), v[9])
    return cams


def read_points(path, max_points, seed=1):
    pts = []
    for l in open(path):
        if l.startswith('#'):
            continue
        v = l.split()
        if len(v) >= 7:
            pts.append([float(v[1]), float(v[2]), float(v[3]), int(v[4]), int(v[5]), int(v[6])])
    pts = np.array(pts) if pts else np.zeros((0, 6))
    if max_points and len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sfm')
    ap.add_argument('output')
    ap.add_argument('--slam', action='append', default=[], help='NAME=camera_poses.txt (repeatable)')
    ap.add_argument('--slam-map', help='rtabmap_to_colmap.py export folder for the SLAM point cloud')
    ap.add_argument('--max-points', type=int, default=40000)
    ap.add_argument('--title', default='SLAM vs COLMAP')
    ap.add_argument('--fragment', action='store_true', help='Omit the document header (for embedding)')
    args = ap.parse_args()

    slams = [s.split('=', 1) for s in args.slam] or [['SLAM', os.path.join(args.sfm, 'slam_camera_poses.txt')]]
    sfm = read_sfm(os.path.join(args.sfm, 'sparse', '0'))
    ref = read_tum(slams[0][1])
    ids = sorted(set(ref) & set(sfm))
    if len(ids) < 3:
        sys.exit('Only %d frames are in both SfM and %s' % (len(ids), slams[0][1]))
    s, R, t = umeyama(np.array([sfm[i][0] for i in ids]), np.array([ref[i] for i in ids]))
    align = lambda p: (s * (R @ np.asarray(p).T)).T + t

    trajectories = []
    for name, path in slams:
        poses = read_tum(path)
        common = [i for i in ids if i in poses]
        err = [float(np.linalg.norm(align(sfm[i][0]) - poses[i])) for i in common]
        trajectories.append(dict(
            name=name, ids=sorted(poses), xyz=[poses[i].round(4).tolist() for i in sorted(poses)],
            err_ids=common, err=[round(e, 4) for e in err],
            rmse=float(np.sqrt(np.mean(np.square(err)))), mean=float(np.mean(err)), max=float(np.max(err))))

    sfm_ids = sorted(sfm)
    sfm_xyz = align([sfm[i][0] for i in sfm_ids]).round(4).tolist()
    sfm_pts = read_points(os.path.join(args.sfm, 'sparse', '0', 'points3D.txt'), args.max_points)
    if len(sfm_pts):
        sfm_pts[:, :3] = align(sfm_pts[:, :3])
    slam_pts = np.zeros((0, 6))
    if args.slam_map:
        slam_pts = read_points(os.path.join(args.slam_map, 'sparse', '0', 'points3D.txt'), args.max_points)

    # Keyframe thumbnails
    images = {}
    img_dir = os.path.join(args.sfm, 'images')
    if os.path.isdir(img_dir):
        for f in sorted(os.listdir(img_dir)):
            img = cv2.imread(os.path.join(img_dir, f))
            if img is None:
                continue
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                images[frame_id(f)] = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()

    def flat(p):
        return dict(xyz=p[:, :3].round(3).ravel().tolist(), rgb=p[:, 3:6].astype(int).ravel().tolist())

    data = dict(title=args.title, scale=float(s), n_images=len(images) or len(sfm),
                n_sfm=len(sfm), trajectories=trajectories,
                sfm=dict(ids=sfm_ids, xyz=sfm_xyz), sfm_points=flat(sfm_pts), slam_points=flat(slam_pts),
                images={str(k): v for k, v in images.items()})
    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_viewer_template.html'),
                    encoding='utf-8').read()
    html = template.replace('__TITLE__', args.title).replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    if not args.fragment:
        html = ('<!doctype html>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n') + html
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    for tr in trajectories:
        print('%s: %d frames, RMSE %.1f cm, max %.1f cm' % (tr['name'], len(tr['err']), tr['rmse'] * 100, tr['max'] * 100))
    print('Wrote %s (%.1f MB)' % (args.output, os.path.getsize(args.output) / 1e6))


if __name__ == '__main__':
    main()
