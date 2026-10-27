#!/usr/bin/env python3
"""Extracts a triangle mesh from a Gaussian model (or from the stereo depth maps) by TSDF fusion.

  python extract_mesh.py DATA                       # model DATA/3dgs (made by train_gsplat.py) -> DATA/3dgs/mesh.ply
  python extract_mesh.py DATA --model DATA/2dgs     # another model folder
  python extract_mesh.py DATA --source stereo       # baseline without Gaussians: DATA/depths -> DATA/stereo_mesh.ply
  python extract_mesh.py DATA --model DATA/2dgs --method poisson   # holes filled -> DATA/2dgs/mesh_poisson.ply

DATA is a COLMAP folder made by rtabmap_to_colmap.py (sparse/0, images/, depths/).
With a model, the depth and color of every training image are rendered with gsplat
(median depth for 2dgs models by default, expected depth for 3dgs models, pixels with an
opacity under --min-alpha are left out) and fused in an Open3D TSDF volume; the mesh
is the zero level set (marching cubes). Small disconnected pieces are removed.
With --method poisson, the rendered depth pixels become an oriented point cloud
(normals rendered by 2dgs models, estimated otherwise) and a screened Poisson
surface is fitted: it fills the holes left where no depth is known (shiny floor,
object edges). The surface is cut where it is farther than --poisson-trim from the
points, so only holes up to about twice that size are filled.
With --method hybrid (default for models), the TSDF mesh is kept as it is and only
the parts of the Poisson surface farther than --hybrid-gap from it (the holes) are
added: the accuracy of TSDF with the coverage of Poisson. Small holes of the TSDF
mesh (boundary loops under --fill-holes, e.g. specks on a shiny floor) are first
closed directly.
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
    """Yields (image index, rgb uint8 [H, W, 3], depth float32 [H, W], world normals [H, W, 3] or None)
    rendered from the model."""
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
            normals = None
            if meta['mode'] == '2dgs':
                out, alphas, normals, _, _, median, _ = rasterization_2dgs(*a, sh_degree=degree, packed=False, render_mode='RGB+ED')
                depth = median[0, ..., 0] if depth_mode == 'median' else out[0, ..., 3]
                normals = normals[0].float().cpu().numpy()
            else:
                out, alphas, _ = rasterization(*a, sh_degree=degree, packed=False, render_mode='RGB+ED')
                depth = out[0, ..., 3]
            depth = torch.where(alphas[0, ..., 0] >= min_alpha, depth, torch.zeros_like(depth))
            rgb = (out[0, ..., :3].clamp(0, 1) * 255).byte()
            yield k, rgb.cpu().numpy(), depth.float().cpu().numpy(), normals


def stereo_views(images):
    for k, im in enumerate(images):
        if im['depth'] is not None:
            yield k, im['image'], im['depth'], None


def fuse(views, images, voxel, max_depth, device, min_weight):
    """TSDF fusion in an Open3D VoxelBlockGrid (sparse voxel blocks, CPU or CUDA)."""
    import open3d as o3d
    import open3d.core as o3c
    dev = o3c.Device(device)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=100000, device=dev)
    count = 0
    for k, rgb, depth, _ in views:
        if not np.any((depth > 0) & (depth < max_depth)):
            continue  # Open3D aborts on a view that touches no voxel
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


def poisson(views, images, voxel, max_depth, trim, depth_level):
    """Screened Poisson surface of the depth pixels of all views (Open3D), cut far from the points."""
    import open3d as o3d
    points, colors, normals, count = [], [], [], 0
    for k, rgb, depth, nrm in views:
        valid = (depth > 0) & (depth < max_depth)
        if not np.any(valid):
            continue
        K, viewmat = images[k]['K'], images[k]['viewmat']
        v, u = np.nonzero(valid)
        z = depth[v, u]
        cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
        R_wc = viewmat[:3, :3].T
        points.append((cam - viewmat[:3, 3]) @ R_wc.T)
        colors.append(rgb[v, u] / 255.0)
        normals.append(nrm[v, u] if nrm is not None else np.zeros((len(z), 3)))
        count += 1
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    cloud.normals = o3d.utility.Vector3dVector(np.concatenate(normals))
    cloud = cloud.voxel_down_sample(voxel / 2)
    n = np.asarray(cloud.normals)
    if not np.any(n):
        # No rendered normals (stereo depth, 3dgs model): estimate them, facing the cameras
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
        centers = np.array([-im['viewmat'][:3, :3].T @ im['viewmat'][:3, 3] for im in images])
        p = np.asarray(cloud.points)
        to_cam = centers.mean(0) - p
        n = np.asarray(cloud.normals)
        flip = np.sum(n * to_cam, 1) < 0
        n[flip] *= -1
        cloud.normals = o3d.utility.Vector3dVector(n)
    else:
        cloud.normals = o3d.utility.Vector3dVector(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
    print('%d points from %d views' % (len(cloud.points), count))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth_level)
    # Cut the surface far from the points (Poisson closes the scene into a watertight blob)
    dist = np.asarray(o3d.geometry.PointCloud(mesh.vertices).compute_point_cloud_distance(cloud))
    mesh.remove_vertices_by_mask(dist > trim)
    return mesh, count


def hybrid(tsdf, filler, gap, hole_size):
    """TSDF mesh plus the triangles of the Poisson surface that lie in its holes."""
    import open3d as o3d
    if hole_size > 0:
        n = len(tsdf.triangles)
        tsdf = o3d.t.geometry.TriangleMesh.from_legacy(tsdf).fill_holes(hole_size=hole_size).to_legacy()
        print('%d triangles close the small holes of the TSDF mesh' % (len(tsdf.triangles) - n))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tsdf))
    vertices = np.asarray(filler.vertices, np.float32)
    far = scene.compute_distance(o3d.core.Tensor(vertices)).numpy() > gap
    triangles = np.asarray(filler.triangles)
    filler.remove_triangles_by_mask(~far[triangles].any(1))   # keep the triangles touching a hole
    filler.remove_unreferenced_vertices()
    print('%d triangles added in the holes of the TSDF mesh' % len(filler.triangles))
    return tsdf + filler


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
    parser.add_argument('--method', choices=['tsdf', 'poisson', 'hybrid'], default='tsdf',
                        help='tsdf: TSDF fusion; poisson: screened Poisson surface, fills holes; '
                             'hybrid: TSDF mesh with its holes filled by the Poisson surface')
    parser.add_argument('--poisson-depth', type=int, default=9, help='Octree depth of the Poisson surface (9: about 2 cm cells in a 10 m scene)')
    parser.add_argument('--poisson-trim', type=float, default=0.03,
                        help='Poisson surface farther than this from the depth points is removed (m)')
    parser.add_argument('--fill-holes', type=float, default=0.1,
                        help='hybrid: holes of the TSDF mesh smaller than this are closed first (m, 0 = off)')
    parser.add_argument('--hybrid-gap', type=float, default=0.02,
                        help='hybrid: Poisson triangles farther than this from the TSDF mesh fill its holes (m)')
    parser.add_argument('--model', help='Model folder (default DATA/3dgs)')
    parser.add_argument('--output', help='Mesh PLY (default MODEL/mesh.ply, or DATA/stereo_mesh.ply)')
    parser.add_argument('--voxel', type=float, default=0.02, help='TSDF voxel size (m)')
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
    suffix = '' if args.method == 'tsdf' else '_' + args.method
    output = args.output or (os.path.join(model_dir, 'mesh%s.ply' % suffix) if args.source == 'model'
                             else os.path.join(args.data, 'stereo_mesh%s.ply' % suffix))

    cameras, images, _, _ = read_colmap_text(args.data)
    images = [im for k, im in enumerate(images) if not (test_every > 0 and k % test_every == 0)]
    load_images(args.data, cameras, images, factor)
    height, width = images[0]['image'].shape[:2]
    refined = os.path.join(model_dir, 'refined_viewmats.json')
    if args.source == 'model' and os.path.exists(refined):
        # Camera poses refined by the training
        viewmats = json.load(open(refined))
        for im in images:
            if im['name'] in viewmats:
                im['viewmat'] = np.array(viewmats[im['name']])
        print('Refined camera poses of %s' % refined)

    start = time.time()
    if args.source == 'model':
        views = render_views(model_dir, images, width, height, args.min_alpha, args.depth)
    else:
        if not any(im['depth'] is not None for im in images):
            sys.exit('No depth map in %s/depths (export with rtabmap_to_colmap.py --euroc)' % args.data)
        views = stereo_views(images)
    if args.method == 'hybrid':
        views = list(views)   # used twice
    if args.method != 'tsdf':
        mesh, count = poisson(views, images, args.voxel, args.max_depth, args.poisson_trim, args.poisson_depth)
    if args.method != 'poisson':
        tsdf, count = fuse(views, images, args.voxel, args.max_depth, args.device, args.min_weight)
        mesh = hybrid(clean(tsdf, args.min_triangles), mesh, args.hybrid_gap, args.fill_holes) if args.method == 'hybrid' else tsdf
    raw = len(mesh.triangles)
    mesh = clean(mesh, args.min_triangles)
    import open3d as o3d
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output, mesh)
    print('%d views fused, %d triangles (%d before cleaning), %.0f s -> %s'
          % (count, len(mesh.triangles), raw, time.time() - start, output))


if __name__ == '__main__':
    main()
