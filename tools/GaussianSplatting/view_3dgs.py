#!/usr/bin/env python3
"""Shows a Gaussian Splatting model (and a mesh) in the web browser, with viser (Apache-2.0).

  python view_3dgs.py MODEL/point_cloud.ply [--mesh mesh.ply] [--port 8080] [--no-browser]

MODEL/point_cloud.ply is a model of train_gsplat.py (3dgs or 2dgs, usual 3DGS PLY layout).
The page http://localhost:PORT opens in the default browser; drag to turn, scroll to zoom.
The world frame is the RTAB-Map map frame (z up), so a mesh of extract_mesh.py overlays it.
Stop with Ctrl+C (or close the console).

Dependencies: viser, numpy; trimesh (MIT, installed with viser) for --mesh.
"""
import argparse, os, sys, time, webbrowser

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_mesh import read_gaussians  # noqa: E402


def splats(path, max_count, max_scale):
    """(centers, covariances, rgbs, opacities) of the PLY model, the most opaque max_count ones,
    without the Gaussians larger than max_scale (m; floaters around the scene)."""
    g = read_gaussians(path)
    opacities = 1.0 / (1.0 + np.exp(-g['opacities']))
    small = np.exp(g['scales']).max(1) <= max_scale if max_scale > 0 else np.ones(len(opacities), bool)
    order = np.argsort(-np.where(small, opacities, -1.0))[:int(small.sum())]
    keep = order[:max_count] if 0 < max_count < len(order) else order
    q = g['quats'][keep]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                  np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                  np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)
    s = np.exp(g['scales'][keep])
    cov = np.einsum('nij,nj,nkj->nik', R, s * s, R)
    rgbs = np.clip(0.5 + 0.28209479177387814 * g['sh0'][keep, 0], 0, 1)   # SH degree 0 (view independent)
    return (g['means'][keep].astype(np.float32), cov.astype(np.float32), rgbs.astype(np.float32),
            opacities[keep, None].astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='point_cloud.ply of train_gsplat.py (or its folder)')
    parser.add_argument('--mesh', help='Mesh PLY to show too (e.g. mesh_hybrid.ply)')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--max-gaussians', type=int, default=0, help='Show only the most opaque ones (0 = all)')
    parser.add_argument('--max-scale', type=float, default=0.3, help='Leave out the Gaussians larger than this (m, 0 = none)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser')
    args = parser.parse_args()
    import viser

    path = os.path.join(args.model, 'point_cloud.ply') if os.path.isdir(args.model) else args.model
    centers, cov, rgbs, opacities = splats(path, args.max_gaussians, args.max_scale)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction('+z')
    splat = server.scene.add_gaussian_splats('/gaussians', centers=centers, covariances=cov, rgbs=rgbs, opacities=opacities)
    show_splats = server.gui.add_checkbox('Gaussians', True)
    show_splats.on_update(lambda _: setattr(splat, 'visible', show_splats.value))
    if args.mesh:
        import trimesh
        mesh = server.scene.add_mesh_trimesh('/mesh', trimesh.load(args.mesh, process=False), visible=False)
        show_mesh = server.gui.add_checkbox('Mesh', False)
        show_mesh.on_update(lambda _: setattr(mesh, 'visible', show_mesh.value))

    # Start at the first training camera (refined_viewmats.json of train_gsplat.py), else above the model
    center = np.median(centers, 0)
    extent = np.percentile(np.linalg.norm(centers - center, axis=1), 90)
    position, look_at = center + np.array([-extent, -extent, extent]), center
    views = os.path.join(os.path.dirname(path), 'refined_viewmats.json')
    if os.path.exists(views):
        import json
        viewmats = json.load(open(views))
        if viewmats:
            V = np.array(viewmats[sorted(viewmats)[0]])
            R_wc = V[:3, :3].T
            position = -R_wc @ V[:3, 3]
            look_at = position + R_wc[:, 2]

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(position)
        client.camera.look_at = tuple(look_at)
        client.camera.up_direction = (0.0, 0.0, 1.0)  # map frame: z up

    url = 'http://localhost:%d' % args.port
    print('%d Gaussians of %s -> %s (Ctrl+C to stop)' % (len(centers), path, url), flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
