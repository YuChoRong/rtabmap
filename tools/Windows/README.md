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

## Live viewer (mapping, localization, map viewing)

```
C:\slam\src\tools\Windows\live_viewer.bat [--open MAP.db]
```
`rtabmap-live_viewer.exe` (tools/LiveViewer, built with RTAB-Map; Qt window with RTAB-Map's 3D
view), run with the environment of `rtabmap_env.bat`:
- **1. 실시간 매핑**: D435i (IR stereo, emitter off, + IMU; nothing to set in Preferences) or a
  recording replayed in real time (`.bag`/`.db3`, converted once next to it, or a EuRoC
  folder) -> VIO + SLAM. Point cloud map, keyframe graph, loop closures, camera pose and
  image are shown as they are built; the map is saved as `map_<date>.db` in the chosen folder
  when stopped (or when the recording ends).
- **2. 실시간 로컬라이제이션**: shows a saved map, then localizes the live camera in it (the map
  file is not modified): current pose, live path, map matches in green. The features (ORB or
  ALIKED + LightGlue) follow the map.
- **파일 > 맵 불러오기**: any RTAB-Map `.db`, offline, with the stereo point clouds;
  **파일 > 녹화 파일 열기**: a recording to replay.
- When a task ends (recording finished, camera stopped, map saved or opened) the viewer goes back
  to standby ("대기 중") and waits for the next one (live camera, recording or map). It closes only
  when the user quits (파일 > 종료 or the window's close button).

`--auto map|loc|view --source realsense|REC --db OUT_DIR_OR_MAP [--screenshot out.png] [--exit]` starts
one module at startup, then waits in standby; `--exit` closes the viewer at the end (unattended tests).

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
