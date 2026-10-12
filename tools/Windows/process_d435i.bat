@echo off
rem Processes a RealSense Viewer recording of a D435i (IR stereo + IMU) with our VIO,
rem SLAM with ORB and with ALIKED+LightGlue loop closures, then exports to COLMAP.
rem
rem   call C:\slam\install\rtabmap_env.bat
rem   tools\Windows\process_d435i.bat recording.bag [out_dir] [C:\COLMAP\COLMAP.bat]
rem
rem The COLMAP SfM comparison runs only when the COLMAP path is given.
setlocal
if "%~1"=="" (
    echo Usage: %~nx0 recording.bag [out_dir] [COLMAP.bat]
    exit /b 1
)
if not defined RTABMAP_TOOLS (
    echo Run rtabmap_env.bat of the build first, e.g. C:\slam\install\rtabmap_env.bat
    exit /b 1
)
set BAG=%~f1
set OUT=%~2
if "%OUT%"=="" set OUT=%~dpn1
set COLMAP=%~3
set PY=%RTABMAP_PYTHON_SCRIPTS%

echo === 1/4 Convert the bag to the EuRoC layout ===
python "%RTABMAP_TOOLS%\RealSense\realsense_bag_to_euroc.py" "%BAG%" "%OUT%\euroc" || exit /b 1

echo === 2/4 VIO + SLAM, ORB loop closures ===
rtabmap-euroc_dataset --Odom/Strategy 15 --Rtabmap/ImagesAlreadyRectified true --output "%OUT%" --output_name orb "%OUT%\euroc" || exit /b 1

echo === 3/4 VIO + SLAM, ALIKED + LightGlue loop closures ===
rtabmap-euroc_dataset --Odom/Strategy 15 --Rtabmap/ImagesAlreadyRectified true ^
  --Vis/FeatureType 15 --Kp/DetectorStrategy 15 --PyDetector/Path "%PY%\rtabmap_aliked.py" ^
  --Vis/CorNNType 6 --PyMatcher/Path "%PY%\rtabmap_lightglue.py" --PyMatcher/Model aliked --PyMatcher/Threshold 0.1 ^
  --output "%OUT%" --output_name aliked "%OUT%\euroc" || exit /b 1

echo === 4/4 Export to COLMAP ===
python "%RTABMAP_TOOLS%\GaussianSplatting\rtabmap_to_colmap.py" "%OUT%\aliked.db" "%OUT%\colmap" || exit /b 1
if not "%COLMAP%"=="" (
    python "%RTABMAP_TOOLS%\GaussianSplatting\run_colmap_sfm.py" "%OUT%\colmap" --colmap "%COLMAP%" || exit /b 1
    echo Compare: "%RTABMAP_TOOLS%\GaussianSplatting\colmap_viewer.bat" "%OUT%\colmap_sfm" --gt "%OUT%\colmap_sfm\slam_camera_poses.txt" --ref-name SLAM --scale
)
echo Done: %OUT%
