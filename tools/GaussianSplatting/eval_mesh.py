#!/usr/bin/env python3
"""Compares meshes with the stereo depth of the images held out from training and fusion.

  python eval_mesh.py DATA MESH.ply [MESH2.ply ...] [--test-every 8] [--json out.json]

For every held-out image (every --test-every image of DATA, sorted by name, as in
train_gsplat.py and extract_mesh.py) that has a depth map in DATA/depths, the mesh
is ray cast from the image's camera and its depth compared with the stereo depth
on the pixels where the stereo depth is known:

  hit %        stereo pixels where the mesh is seen at all
  within 2/5cm stereo pixels where the mesh depth is within 2 / 5 cm (completeness)
  median err   median |mesh - stereo| depth on the hit pixels (accuracy)

Stereo depth is noisy (about 1 % of the depth at 1 m for the D435i), so these numbers
compare meshes with each other; they are not absolute errors.
Dependencies: open3d (MIT), numpy, Pillow.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gsplat import read_colmap_text, load_images  # noqa: E402


def mesh_depth(scene, im, width, height):
    """Z-depth of the mesh in the image (inf where no hit)."""
    import open3d as o3d
    K, viewmat = im['K'], im['viewmat']
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1
    R_wc = viewmat[:3, :3].T
    center = -R_wc @ viewmat[:3, 3]
    d_world = d_cam.reshape(-1, 3) @ R_wc.T
    rays = np.concatenate([np.broadcast_to(center, d_world.shape), d_world], 1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()  # in units of the direction: z-depth
    return hit.reshape(height, width)


def evaluate(path, images, width, height):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(path)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    n = hit = w2 = w5 = 0
    errors = []
    for im in images:
        stereo = im['depth']
        valid = stereo > 0
        z = mesh_depth(scene, im, width, height)[valid]
        s = stereo[valid]
        seen = np.isfinite(z)
        err = np.abs(z[seen] - s[seen])
        n += len(s)
        hit += int(seen.sum())
        w2 += int((err < 0.02).sum())
        w5 += int((err < 0.05).sum())
        errors.append(err)
    errors = np.concatenate(errors) if errors else np.zeros(0)
    return dict(mesh=path, triangles=len(mesh.triangles), images=len(images), pixels=n,
                hit=100.0 * hit / max(n, 1), within_2cm=100.0 * w2 / max(n, 1), within_5cm=100.0 * w5 / max(n, 1),
                median_error_cm=100.0 * float(np.median(errors)) if len(errors) else float('nan'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data')
    parser.add_argument('meshes', nargs='+')
    parser.add_argument('--test-every', type=int, default=8)
    parser.add_argument('--factor', type=int, default=2, help='Image downscale factor for the comparison')
    parser.add_argument('--json', help='Write the results to this JSON file')
    args = parser.parse_args()

    cameras, images, _, _ = read_colmap_text(args.data)
    test = [im for k, im in enumerate(images) if args.test_every > 0 and k % args.test_every == 0]
    load_images(args.data, cameras, test, args.factor)
    test = [im for im in test if im['depth'] is not None]
    if not test:
        sys.exit('No held-out image with a depth map in %s/depths' % args.data)
    height, width = test[0]['image'].shape[:2]

    results = []
    print('%-48s %10s %6s %8s %8s %10s' % ('mesh', 'triangles', 'hit%', '<2cm%', '<5cm%', 'median cm'))
    for path in args.meshes:
        r = evaluate(path, test, width, height)
        results.append(r)
        print('%-48s %10d %6.1f %8.1f %8.1f %10.2f' % (path[-48:], r['triangles'], r['hit'], r['within_2cm'],
                                                         r['within_5cm'], r['median_error_cm']))
    print('%d held-out images, %d stereo depth pixels' % (len(test), results[0]['pixels']))
    if args.json:
        json.dump(results, open(args.json, 'w'), indent=2)


if __name__ == '__main__':
    main()
