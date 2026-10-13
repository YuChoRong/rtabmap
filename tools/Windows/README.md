# Windows build (VIO + ALIKED/LightGlue)

`build_windows.ps1` builds this fork with our VIO (`--Odom/Strategy 15`) and the
Python feature plugins, without the GUI by default (`-Gui` adds Qt/VTK).

1. Install Visual Studio 2022 (Desktop development with C++), Git and Python 3.12
   (python.org, "Add python.exe to PATH").
2. From the repository root: `tools\Windows\build_windows.bat -Root C:\slam -Cuda cu124`
   (`-Cuda cpu` without NVIDIA GPU; pick the CUDA version supported by `nvidia-smi`).
   The first run builds OpenCV, PCL and GTSAM with vcpkg (1-2 hours).
3. Record with Intel RealSense Viewer (see `tools/RealSense/README.md`), then in cmd:
   ```
   C:\slam\install\rtabmap_env.bat
   tools\Windows\process_d435i.bat C:\data\room.bag C:\data\room [C:\COLMAP\COLMAP.bat]
   ```
   It converts the bag, runs VIO + SLAM with ORB and with ALIKED+LightGlue loop
   closures (`orb.db`, `aliked.db`), exports to COLMAP and optionally runs the
   COLMAP SfM comparison.

GTSAM comes from a vcpkg overlay port (`vcpkg-overlay/gtsam`) that also builds
`gtsam_unstable` (fixed-lag smoother and smart stereo factors of the VIO).
Disabled for licensing: OpenCV nonfree (SURF), OpenVINS and ORB-OCTREE (GPL-3.0).
