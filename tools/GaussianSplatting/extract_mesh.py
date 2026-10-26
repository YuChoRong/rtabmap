#!/usr/bin/env python3
"""Extracts a triangle mesh from a Gaussian model (or from the stereo depth maps) by TSDF fusion.

  python extract_mesh.py DATA                       # model DATA/3dgs (made by train_gsplat.py) -> DATA/3dgs/mesh.ply
  python extract_mesh.py DATA --model DATA/2dgs     # another model folder
  python extract_mesh.py DATA --source stereo       # baseline without Gaussians: DATA/depths -> DATA/stereo_mesh.ply

DATA is a COLMAP folder made by rtabmap_to_colmap.py (sparse/0, images/, depths/).
With a model, the depth and color of every training image are rendered with gsplat
(median depth for 2dgs models by default, expected depth for 3dgs models, pixels with an
opacity under --min-alpha are left out) and fused in an Open3D TSDF volume; the mesh
is the zero level set (marching cubes). Small disconnected pieces are removed.
The images held out by train_gsplat.py (--test-every) are not fused, so
eval_mesh.py can check the mesh on them.

Dependencies: open3d (MIT), numpy, Pillow; with a model also torch and gsplat (Apache-2.0).
"""
import argparse, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gsplat import read_colmap_text, load_images  # noqa: E402


def read_gaussians(path):
    """3DGS PLY of train_gsplat.py -> dict of numpy arrays (activations not applied)."""
    with open(path, 'rb') as f:
        names, count = [], 0
        while True:
            line = f.readline().decode('ascii').strip()
            v = line.split()
            if v[0] == 'element':
                count = int(v[2])
            elif v[0] == 'property':
                names.append(v[2])
            elif v[0] == 'end_header':
                break
        data = np.frombuffer(f.read(4 * len(names) * count), '<f4').reshape(count, len(names))
    col = {n: i for i, n in enumerate(names)}
    pick = lambda prefix: data[:, [col[n] for n in names if n.startswith(prefix)]]
    rest = pick('f_rest_')
    return dict(means=data[:, [col['x'], col['y'], col['z']]],
                sh0=pick('f_dc_').reshape(count, 1, 3),
                shN=rest.reshape(count, 3, -1).transpose(0, 2, 1),
                opacities=data[:, col['opacity']],
                scales=pick('scale_'), quats=pick('rot_'))


def render_views(model_dir, images, width, height, min_alpha, depth_mode):
    """Yields (image index, rgb uint8 [H, W, 3], depth float32 [H, W]) rendered from the model."""
    import torch
    from gsplat import rasterization, rasterization_2dgs
    meta = json.load(open(os.path.join(model_dir, 'model.json')))
    g = read_gaussians(os.path.join(model_dir, 'point_cloud.ply'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in g.items()}
    colors = torch.cat([t['sh0'], t['shN']], 1)
    degree = int(round(np.sqrt(colors.shape[1]))) - 1
    print('%s model: %d Gaussians, SH degree %d' % (meta['mode'], t['means'].shape[0], degree))
    with torch.no_grad():
        for k, im in enumerate(images):
            viewmat = torch.tensor(im['viewmat'], dtype=torch.float32, device=device)[None]
            K = torch.tensor(im['K'], dtype=torch.float32, device=device)[None]
            a = (t['means'], torch.nn.functional.normalize(t['quats'], dim=-1), torch.exp(t['scales']),
                 torch.sigmoid(t['opacities']), colors, viewmat, K, width, height)
            if meta['mode'] == '2dgs':
                out, alphas, _, _, _, median, _ = rasterization_2dgs(*a, sh_degree=degree, packed=False, render_mode='RGB+ED')
                depth = median[0, ..., 0] if depth_mode == 'median' else out[0, ..., 3]
            else:
                out, alphas, _ = rasterization(*a, sh_degree=degree, packed=False, render_mode='RGB+ED')
                depth = out[0, ..., 3]
            depth = torch.where(alphas[0, ..., 0] >= min_alpha, depth, torch.zeros_like(depth))
            rgb = (out[0, ..., :3].clamp(0, 1) * 255).byte()
            yield k, rgb.cpu().numpy(), depth.float().cpu().numpy()


def stereo_views(images):
    for k, im in enumerate(images):
        if im['depth'] is not None:
            yield k, im['image'], im['depth']


def fuse(views, images, voxel, max_depth, device, min_weight):
    """TSDF fusion in an Open3D VoxelBlockGrid (sparse voxel blocks, CPU or CUDA)."""
    import open3d as o3d
    import open3d.core as o3c
    dev = o3c.Device(device)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=100000, device=dev)
    count = 0
    for k, rgb, depth in views:
        intrinsic = o3c.Tensor(images[k]['K'], o3c.float64)
        extrinsic = o3c.Tensor(images[k]['viewmat'], o3c.float64)
        depth_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth, np.float32))).to(dev)
        color_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb, np.float32) / 255.0)).to(dev)
        blocks = vbg.compute_unique_block_coordinates(depth_img, intrinsic, extrinsic, depth_scale=1.0,
                                                      depth_max=max_depth, trunc_voxel_multiplier=4.0)
        vbg.integrate(blocks, depth_img, color_img, intrinsic, extrinsic, depth_scale=1.0,
                      depth_max=max_depth, trunc_voxel_multiplier=4.0)
        count += 1
    return vbg.extract_triangle_mesh(weight_threshold=min_weight).to_legacy(), count


def clean(mesh, min_triangles):
    """Removes the connected pieces with fewer than min_triangles triangles."""
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[clusters] < min_triangles)
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help='COLMAP folder of rtabmap_to_colmap.py')
    parser.add_argument('--source', choices=['model', 'stereo'], default='model')
    parser.add_argument('--model', help='Model folder (default DATA/3dgs)')
    parser.add_argument('--output', help='Mesh PLY (default MODEL/mesh.ply, or DATA/stereo_mesh.ply)')
    parser.add_argument('--voxel', type=float, default=0.01, help='TSDF voxel size (m)')
    parser.add_argument('--max-depth', type=float, default=4.0, help='Depth truncation (m)')
    parser.add_argument('--depth', choices=['median', 'expected'], default='median', help='2dgs: rendered depth fused')
    parser.add_argument('--min-alpha', type=float, default=0.5, help='Rendered pixels under this opacity are not fused')
    parser.add_argument('--factor', type=int, default=0, help='Image downscale factor (default: the model\'s, else 1)')
    parser.add_argument('--test-every', type=int, default=-1, help='Held-out images not fused (default: the model\'s, else 8)')
    parser.add_argument('--min-weight', type=float, default=2.0, help='Voxels seen by fewer views are left out of the mesh')
    parser.add_argument('--device', default='CPU:0', help='Open3D device of the TSDF volume (CPU:0, or CUDA:0 with a CUDA build of Open3D)')
    parser.add_argument('--min-triangles', type=int, default=500, help='Smaller disconnected pieces are removed')
    args = parser.parse_args()

    model_dir = args.model or os.path.join(args.data, '3dgs')
    meta = {}
    if args.source == 'model':
        if not os.path.exists(os.path.join(model_dir, 'model.json')):
            sys.exit('No model in %s: train one with train_gsplat.py, or use --source stereo' % model_dir)
        meta = json.load(open(os.path.join(model_dir, 'model.json')))
    factor = args.factor or meta.get('factor', 1)
    test_every = args.test_every if args.test_every >= 0 else meta.get('test_every', 8)
    output = args.output or (os.path.join(model_dir, 'mesh.ply') if args.source == 'model'
                             else os.path.join(args.data, 'stereo_mesh.ply'))

    cameras, images, _, _ = read_colmap_text(args.data)
    images = [im for k, im in enumerate(images) if not (test_every > 0 and k % test_every == 0)]
    load_images(args.data, cameras, images, factor)
    height, width = images[0]['image'].shape[:2]

    start = time.time()
    if args.source == 'model':
        views = render_views(model_dir, images, width, height, args.min_alpha, args.depth)
    else:
        if not any(im['depth'] is not None for im in images):
            sys.exit('No depth map in %s/depths (export with rtabmap_to_colmap.py --euroc)' % args.data)
        views = stereo_views(images)
    mesh, count = fuse(views, images, args.voxel, args.max_depth, args.device, args.min_weight)
    raw = len(mesh.triangles)
    mesh = clean(mesh, args.min_triangles)
    import open3d as o3d
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output, mesh)
    print('%d views fused, %d triangles (%d before cleaning), %.0f s -> %s'
          % (count, len(mesh.triangles), raw, time.time() - start, output))


if __name__ == '__main__':
    main()
