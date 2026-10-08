#!/usr/bin/env python3
"""Trains a 3D Gaussian Splatting model on a COLMAP folder made by rtabmap_to_colmap.py.

Self-contained trainer built on the gsplat library (Apache-2.0) only: no Inria
code, no gsplat example scripts. Dependencies: torch (CUDA), gsplat, numpy, Pillow.

  pip install torch gsplat numpy pillow
  python train_gsplat.py COLMAP_DIR [--output DIR] [--steps 30000] [--factor 1]

- Reads sparse/0/{cameras,images,points3D}.txt (PINHOLE or SIMPLE_PINHOLE) and images/.
- Gaussians start at the points of points3D.txt (the RTAB-Map stereo cloud).
- Loss 0.8 L1 + 0.2 (1 - SSIM); densification and pruning by gsplat's DefaultStrategy;
  spherical harmonics up to degree 3.
- Every --test-every image is held out: PSNR / SSIM on them are written to
  OUTPUT/stats.json and the renders to OUTPUT/renders.
- The model is saved as OUTPUT/point_cloud.ply in the usual 3DGS PLY layout
  (readable by 3DGS viewers such as SuperSplat).

--check loads the data and prints a summary without torch (no GPU needed).
"""
import argparse, json, math, os, struct, sys, time

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- data

def qvec_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_colmap_text(root):
    sparse = None
    for d in (os.path.join(root, 'sparse', '0'), os.path.join(root, 'sparse'), root):
        if os.path.exists(os.path.join(d, 'images.txt')):
            sparse = d
            break
    if sparse is None:
        sys.exit('No sparse/0/images.txt in %s (COLMAP text model expected)' % root)

    cameras = {}
    for line in open(os.path.join(sparse, 'cameras.txt')):
        v = line.split()
        if not v or v[0].startswith('#'):
            continue
        model, w, h, p = v[1], int(v[2]), int(v[3]), [float(x) for x in v[4:]]
        if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif model in ('PINHOLE', 'OPENCV', 'FULL_OPENCV'):
            fx, fy, cx, cy = p[:4]
        else:
            sys.exit('Unsupported camera model %s' % model)
        if model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            print('Warning: camera %s distortion is ignored, use undistorted images' % v[0])
        cameras[int(v[0])] = dict(width=w, height=h, K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

    images = []
    lines = open(os.path.join(sparse, 'images.txt')).read().splitlines()
    i = 0
    while i < len(lines):
        v = lines[i].split()
        i += 1
        if not v or v[0].startswith('#'):
            continue
        q = np.array([float(x) for x in v[1:5]])
        t = np.array([float(x) for x in v[5:8]])
        viewmat = np.eye(4)
        viewmat[:3, :3] = qvec_to_rot(q)
        viewmat[:3, 3] = t
        images.append(dict(id=int(v[0]), camera=int(v[8]), name=' '.join(v[9:]), viewmat=viewmat))
        i += 1  # 2D points line
    images.sort(key=lambda im: im['name'])

    xyz, rgb = [], []
    for line in open(os.path.join(sparse, 'points3D.txt')):
        v = line.split()
        if not v or v[0].startswith('#'):
            continue
        xyz.append([float(x) for x in v[1:4]])
        rgb.append([int(x) for x in v[4:7]])
    if not xyz:
        sys.exit('No point in points3D.txt: Gaussians need initial points')
    return cameras, images, np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


def load_images(root, cameras, images, factor):
    """RGB images as uint8 arrays (grayscale images are expanded), with intrinsics scaled by factor."""
    for im in images:
        path = os.path.join(root, 'images', im['name'])
        img = Image.open(path).convert('RGB')
        cam = cameras[im['camera']]
        w, h = cam['width'] // factor, cam['height'] // factor
        if img.size != (w, h):
            img = img.resize((w, h), Image.BICUBIC)
        im['image'] = np.asarray(img)
        K = cam['K'].copy()
        K[:2] /= factor
        im['K'] = K


def scene_scale_of(images):
    centers = np.array([-im['viewmat'][:3, :3].T @ im['viewmat'][:3, 3] for im in images])
    return float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1


# ---------------------------------------------------------------- export

def save_ply(path, means, scales, quats, opacities, sh0, shN):
    """3DGS PLY layout: log scales, opacity logits, wxyz quaternions, SH coefficients per channel."""
    n = means.shape[0]
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)  # [N, K, 3] -> per channel
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + ['f_dc_%d' % i for i in range(3)] + \
            ['f_rest_%d' % i for i in range(f_rest.shape[1])] + ['opacity'] + \
            ['scale_%d' % i for i in range(3)] + ['rot_%d' % i for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), sh0.reshape(n, 3), f_rest,
                           opacities.reshape(n, 1), scales, quats], axis=1).astype('<f4')
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n).encode())
        for name in names:
            f.write(('property float %s\n' % name).encode())
        f.write(b'end_header\n')
        f.write(data.tobytes())


# ---------------------------------------------------------------- training

def ssim(a, b, window=11, sigma=1.5):
    """SSIM of images [B, 3, H, W] in [0, 1] (Gaussian window)."""
    import torch
    import torch.nn.functional as F
    coords = torch.arange(window, dtype=a.dtype, device=a.device) - window // 2
    g = torch.exp(-coords**2 / (2*sigma*sigma))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()
    blur = lambda x: F.conv2d(x, kernel, padding=window // 2, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a*a) - mu_a**2
    var_b = blur(b*b) - mu_b**2
    cov = blur(a*b) - mu_a*mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2*mu_a*mu_b + c1) * (2*cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.mean()


def knn_distance(points, k=3, chunk=2048):
    """Mean distance to the k nearest neighbors of each point (brute force by chunks, on the GPU)."""
    import torch
    out = torch.empty(points.shape[0], device=points.device)
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i:i+chunk], points)
        out[i:i+chunk] = d.topk(k+1, largest=False).values[:, 1:].mean(1)
    return out


def train(args, cameras, images, xyz, rgb):
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    device = torch.device('cuda')
    torch.manual_seed(0)
    test = [im for k, im in enumerate(images) if args.test_every > 0 and k % args.test_every == 0]
    trainset = [im for k, im in enumerate(images) if not (args.test_every > 0 and k % args.test_every == 0)]
    scene_scale = scene_scale_of(images)
    print('%d training images, %d test images, %d initial Gaussians, scene scale %.2f m'
          % (len(trainset), len(test), len(xyz), scene_scale))

    def to_gpu(ims):
        return (torch.from_numpy(np.stack([im['image'] for im in ims])).to(device),
                torch.tensor(np.stack([im['viewmat'] for im in ims]), dtype=torch.float32, device=device),
                torch.tensor(np.stack([im['K'] for im in ims]), dtype=torch.float32, device=device))
    train_images, train_viewmats, train_Ks = to_gpu(trainset)
    height, width = train_images.shape[1:3]

    # Gaussians
    means = torch.tensor(xyz, device=device)
    dist = knn_distance(means).clamp(min=1e-4)
    n = means.shape[0]
    sh_degree = args.sh_degree
    colors = torch.zeros(n, (sh_degree+1)**2, 3, device=device)
    colors[:, 0] = (torch.tensor(rgb, device=device) - 0.5) / 0.28209479177387814  # RGB -> SH DC
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(means),
        'scales': torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        'quats': torch.nn.Parameter(torch.rand(n, 4, device=device)),
        'opacities': torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=device))),
        'sh0': torch.nn.Parameter(colors[:, :1]),
        'shN': torch.nn.Parameter(colors[:, 1:]),
    })
    lrs = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2,
           'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{'params': params[k], 'lr': lr, 'name': k}], eps=1e-15)
                  for k, lr in lrs.items()}
    # Position learning rate decays to 1% over the training
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=0.01 ** (1.0 / args.steps))
    strategy = DefaultStrategy(refine_stop_iter=min(15000, args.steps // 2), verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(viewmats, Ks, degree):
        out, _, info = rasterization(
            params['means'], params['quats'], torch.exp(params['scales']), torch.sigmoid(params['opacities']),
            torch.cat([params['sh0'], params['shN']], 1), viewmats, Ks, width, height,
            sh_degree=degree, packed=False)
        return out.clamp(0, 1), info

    start = time.time()
    for step in range(args.steps):
        k = torch.randint(len(trainset), (1,)).item()
        gt = train_images[k:k+1].float() / 255.0
        degree = min(step // 1000, sh_degree)
        image, info = render(train_viewmats[k:k+1], train_Ks[k:k+1], degree)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (image - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(image.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 500 == 0 or step == args.steps - 1:
            print('step %d/%d loss %.4f, %d Gaussians, %.0f s' % (step, args.steps, loss.item(), params['means'].shape[0], time.time() - start))

    os.makedirs(args.output, exist_ok=True)
    with torch.no_grad():
        stats = {'steps': args.steps, 'gaussians': int(params['means'].shape[0]), 'train_seconds': time.time() - start}
        if test:
            test_images, test_viewmats, test_Ks = to_gpu(test)
            os.makedirs(os.path.join(args.output, 'renders'), exist_ok=True)
            psnrs, ssims = [], []
            for k, im in enumerate(test):
                gt = test_images[k:k+1].float() / 255.0
                image, _ = render(test_viewmats[k:k+1], test_Ks[k:k+1], sh_degree)
                psnrs.append((-10.0 * torch.log10(((image - gt)**2).mean())).item())
                ssims.append(ssim(image.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
                side = torch.cat([gt[0], image[0]], 1)  # ground truth | render
                Image.fromarray((side.cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(args.output, 'renders', os.path.splitext(im['name'])[0] + '.png'))
            stats.update(test_images=len(test), psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)))
            print('Test images: PSNR %.2f dB, SSIM %.4f' % (stats['psnr'], stats['ssim']))
        json.dump(stats, open(os.path.join(args.output, 'stats.json'), 'w'), indent=2)
        q = torch.nn.functional.normalize(params['quats'], dim=-1)
        save_ply(os.path.join(args.output, 'point_cloud.ply'), params['means'].cpu().numpy(),
                 params['scales'].cpu().numpy(), q.cpu().numpy(), params['opacities'].cpu().numpy(),
                 params['sh0'].cpu().numpy(), params['shN'].cpu().numpy())
    print('Model: %s' % os.path.join(args.output, 'point_cloud.ply'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help='COLMAP folder (sparse/0 and images/)')
    parser.add_argument('--output', help='Output folder (default DATA/3dgs)')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--factor', type=int, default=1, help='Image downscale factor')
    parser.add_argument('--sh-degree', type=int, default=3)
    parser.add_argument('--test-every', type=int, default=8, help='Hold out every Nth image for evaluation (0 = none)')
    parser.add_argument('--check', action='store_true', help='Load the data, write a test PLY of the initial points, and exit (no torch)')
    args = parser.parse_args()
    args.output = args.output or os.path.join(args.data, '3dgs')

    cameras, images, xyz, rgb = read_colmap_text(args.data)
    load_images(args.data, cameras, images, args.factor)
    print('%d images %dx%d, %d points, scene scale %.2f m'
          % (len(images), images[0]['image'].shape[1], images[0]['image'].shape[0], len(xyz), scene_scale_of(images)))
    if args.check:
        # Projection sanity check: share of points in front of each camera that fall in its image
        inside = []
        for im in images[::max(1, len(images)//20)]:
            pc = xyz @ im['viewmat'][:3, :3].T + im['viewmat'][:3, 3]
            front = pc[:, 2] > 0.1
            uv = pc[front] @ im['K'].T
            uv = uv[:, :2] / uv[:, 2:]
            h, w = im['image'].shape[:2]
            inside.append(np.mean((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)) if len(uv) else 0)
        print('Points in front of a camera falling in its image: %.0f%% on average' % (100 * np.mean(inside)))
        os.makedirs(args.output, exist_ok=True)
        n = len(xyz)
        sh0 = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)
        save_ply(os.path.join(args.output, 'initial_points.ply'), xyz, np.full((n, 3), math.log(0.01), np.float32),
                 np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32), np.zeros(n, np.float32), sh0,
                 np.zeros((n, 15, 3), np.float32))
        print('Check done, wrote %s' % os.path.join(args.output, 'initial_points.ply'))
        return
    train(args, cameras, images, xyz, rgb)


if __name__ == '__main__':
    main()
