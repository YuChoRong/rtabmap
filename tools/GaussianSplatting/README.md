# Dense mapping with 3D Gaussian Splatting

Post-processing of an RTAB-Map session: the optimized camera poses (after loop
closures), the rectified images and the stereo point cloud are exported as a
COLMAP model, a 3D Gaussian Splatting model is trained on it, and
`colmap_viewer` checks the model and compares the poses with the ground truth.

## 1. Export the map to COLMAP

```bash
python3 rtabmap_to_colmap.py map.db colmap_out [--voxel 0.02] [--max-points 200000]
```

Writes `colmap_out/images/`, `colmap_out/sparse/0/{cameras,images,points3D}.txt`,
`points3D.ply` (initial Gaussians), and `camera_poses.txt` /
`gt_camera_poses.txt` (estimated and ground truth camera poses, `stamp x y z qx qy qz qw id`).
The `rtabmap-export` tool must be in `PATH` (or use `--rtabmap-export PATH`).

More images give a better 3DGS model: with `rtabmap-euroc_dataset`, a higher
`--Rtabmap/DetectionRate` (e.g., 4) keeps more keyframes in the map.

## 2. Check the model and compare with the ground truth

```bash
cd colmap_viewer && mkdir build && cd build && cmake .. && make
./colmap_viewer colmap_out            # gt_camera_poses.txt is loaded when present
./colmap_viewer colmap_out --gt other_gt.txt
```

Dependencies: Qt5 Widgets, Eigen3.

- Left: top or side view of the points with the estimated trajectory (blue)
  and the ground truth aligned on it (orange, rigid or similarity alignment),
  with error segments. Wheel to zoom, drag to pan, click a camera to select it.
- Right: the selected image with the 3D points reprojected by its estimated
  (or ground truth) pose, colored by depth. Points falling on the right image
  structures confirm the poses and intrinsics given to 3DGS.
- Position error over time (click to select a frame) and statistics: RMSE,
  mean, median, max, scale, rotation error, and the constant rotation between
  the estimated and ground truth camera frames.
- `--screenshot out.png [--frame N] [--plane xy|xz|yz]` renders the window and
  prints the statistics without a display (`QT_QPA_PLATFORM=offscreen`).

Ground truth files from other tools: TUM format `stamp x y z qx qy qz qw [id]`
of the camera optical frame, matched by id (image name) or by stamp (needs
`camera_poses.txt`).

## 3. Train 3D Gaussian Splatting (CUDA GPU)

```bash
pip install torch gsplat numpy pillow   # once (train_3dgs.sh installs them when missing)
./train_3dgs.sh colmap_out [colmap_out/3dgs] [--steps 30000] [--factor 2]
```

`train_gsplat.py` is a self-contained trainer built only on the
[gsplat](https://github.com/nerfstudio-project/gsplat) rasterizer and
densification strategy (Apache-2.0); it does not use the Inria reference code,
whose license forbids commercial use and redistribution outside research.
All dependencies (torch BSD-3, gsplat Apache-2.0, numpy BSD-3, Pillow MIT-CMU)
can be redistributed.

- Gaussians start at the RTAB-Map stereo points, images are loaded as RGB
  (grayscale is fine), loss 0.8 L1 + 0.2 (1 - SSIM), gsplat `DefaultStrategy`
  densification, spherical harmonics up to degree 3.
- Every 8th image is held out: PSNR / SSIM in `3dgs/stats.json`, ground truth |
  render images in `3dgs/renders`.
- The model is `3dgs/point_cloud.ply` (standard 3DGS PLY, e.g., SuperSplat).
- `python3 train_gsplat.py colmap_out --check` checks the data without a GPU.
