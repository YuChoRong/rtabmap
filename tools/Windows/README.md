# Windows build (VIO + ALIKED/LightGlue + live D435i)

`build_windows.ps1` builds this fork with our VIO (`--Odom/Strategy 15`), the Python
feature plugins (ALIKED + LightGlue, on the GPU when PyTorch sees CUDA), the RealSense2
driver and the GUI (`-NoGui` for the command-line tools only).

1. Install Visual Studio 2022 (Desktop development with C++), Git and Python 3.12
   (python.org, "Add python.exe to PATH"), and the NVIDIA driver.
2. In cmd.exe:
   ```
   git clone -b claude/project-thread-vev8qn https://github.com/yuchorong/rtabmap C:\slam\src
   C:\slam\src\tools\Windows\build_windows.bat -Root C:\slam
   ```
   The PyTorch CUDA build is chosen from `nvidia-smi` (`-Cuda cu126` etc. to force it,
   `-Cuda cpu` without NVIDIA GPU). The first run builds OpenCV, PCL, Qt, VTK, GTSAM and
   librealsense with vcpkg (2-3 hours); running it again only rebuilds RTAB-Map.
   It ends by printing the RTAB-Map version and the GPU seen by PyTorch.

## Live D435i

```
call C:\slam\install\rtabmap_env.bat
C:\slam\src\tools\Windows\run_d435i_live.bat          (ORB loop closures)
C:\slam\src\tools\Windows\run_d435i_live.bat aliked   (ALIKED + LightGlue on the GPU)
```
It starts the GUI with our VIO and the D435i camera-IMU time offset (-37 ms). Once,
in Preferences (Advanced mode), set the source:
- Source > RGB-D > Driver **RealSense2**; in the RealSense2 group uncheck **IR emitter
  enabled**, check **IR mode**, uncheck **IR depth** (left + right IR as stereo),
  resolution 848 x 480 at 30 Hz.
- Source > IMU filtering: check **Publish inter IMU**.
- Odometry > Strategy: **VIO (stereo + IMU, GTSAM)** (set by the script).

Hold the camera still for 1-2 s at the start, or move it smoothly (dynamic
initialization). Odometry and loop closure times are in the Statistics panel.

## Recordings

Record with Intel RealSense Viewer (see `tools/RealSense/README.md`), then in cmd:
```
call C:\slam\install\rtabmap_env.bat
C:\slam\src\tools\Windows\process_d435i.bat C:\data\room.db3 C:\data\room [C:\COLMAP\COLMAP.bat]
```
It converts the recording, runs VIO + SLAM with ORB and with ALIKED+LightGlue loop
closures (`orb.db`, `aliked.db`), exports to COLMAP and optionally runs the COLMAP SfM
comparison.

GTSAM comes from a vcpkg overlay port (`vcpkg-overlay/gtsam`) that also builds
`gtsam_unstable` (fixed-lag smoother and smart stereo factors of the VIO).
Disabled for licensing: OpenCV nonfree (SURF), OpenVINS and ORB-OCTREE (GPL-3.0).
