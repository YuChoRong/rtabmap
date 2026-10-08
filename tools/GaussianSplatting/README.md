# Dense mapping with 3D Gaussian Splatting

Post-processing of an RTAB-Map session: the optimized camera poses (after loop
closures), the rectified images and the stereo point cloud are exported as a
COLMAP model, a 3D Gaussian Splatting model is trained on it, and
`colmap_viewer` checks the model and compares the poses with the ground truth.

Linux and Windows are supported; Windows steps are in [Windows](#windows).

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

Python version, no compilation (PySide6 LGPL-3.0 + numpy):

```bash
pip install pyside6 numpy
python3 colmap_viewer.py colmap_out            # gt_camera_poses.txt is loaded when present
python3 colmap_viewer.py colmap_out --gt other_gt.txt
```

The C++ version in `colmap_viewer/` has the same features (Qt5 or Qt6 Widgets
and Eigen3, built with CMake):

```bash
cmake -S colmap_viewer -B build && cmake --build build && ./build/colmap_viewer colmap_out
```

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

## Windows

Tested parts: the Python scripts are platform independent, the C++ viewer
builds with Qt5 and Qt6, and the PowerShell script was checked with PowerShell 7.
Nothing needs CMake or a compiler except the optional C++ viewer.

1. **Export** (only if you have your own RTAB-Map database; skip with a
   provided `colmap_*.zip`, extract it with `Expand-Archive colmap_v101.zip .`):
   ```powershell
   py rtabmap_to_colmap.py map.db colmap_out   # finds C:\Program Files\RTABMap\bin\rtabmap-export.exe
   ```
2. **Train** (NVIDIA GPU, Python 3.10-3.12 from python.org):
   ```powershell
   .\train_3dgs.ps1 -Data colmap_v101 -Cuda cu124          # PowerShell
   train_3dgs.bat colmap_v101 -Cuda cu124 -Factor 2         # or cmd.exe
   ```
   The first run creates `.gs_env`, installs PyTorch for `-Cuda` (cu118,
   cu121, cu124, cu126: pick one supported by your driver, see `nvidia-smi`)
   and a pre-compiled gsplat wheel from https://docs.gsplat.studio/whl. When
   no wheel matches, gsplat compiles its CUDA code at the first run: install
   Visual Studio Build Tools (Desktop development with C++) and the CUDA
   Toolkit of the same version, and run from the "x64 Native Tools Command
   Prompt". If PowerShell blocks the script, use `train_3dgs.bat` or
   `powershell -ExecutionPolicy Bypass -File train_3dgs.ps1 ...`.
3. **Viewer**, no compilation: `colmap_viewer.bat colmap_v101` (creates or
   reuses `.gs_env` and installs PySide6 and numpy at the first run).

   Or the C++ viewer (Visual Studio 2019/2022 with C++, CMake):
   - with [vcpkg](https://github.com/microsoft/vcpkg): `vcpkg install qtbase eigen3 --triplet x64-windows`,
     `set VCPKG_ROOT=C:\vcpkg`, then `build_viewer_windows.bat`;
   - or with the Qt online installer (LGPL, MSVC 64-bit kit) and the Eigen
     sources: `set QT_DIR=C:\Qt\6.8.0\msvc2022_64`, `set EIGEN_DIR=C:\eigen-3.4.0`,
     then `build_viewer_windows.bat`.

   Run `colmap_viewer\build\Release\colmap_viewer.exe colmap_v101`. Qt is
   linked dynamically (LGPL), its DLLs are copied next to the executable.
