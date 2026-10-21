@echo off
rem Live D435i in the RTAB-Map GUI with our VIO (stereo IR + IMU).
rem
rem   call C:\slam\install\rtabmap_env.bat
rem   tools\Windows\run_d435i_live.bat [aliked]
rem
rem "aliked" uses ALIKED + LightGlue on the GPU for loop closures and localization
rem (ORB otherwise). The source is set once in the GUI, see README.md.
setlocal
if not defined RTABMAP_PYTHON_SCRIPTS (
    echo Run rtabmap_env.bat of the build first, e.g. C:\slam\install\rtabmap_env.bat
    exit /b 1
)
rem Camera-IMU time offset estimated on two D435i recordings: -37 ms
set ARGS=--Odom/Strategy 15 --OdomVIO/TimeOffset -0.037
if /i "%~1"=="aliked" set ARGS=%ARGS% --Vis/FeatureType 15 --Kp/DetectorStrategy 15 --PyDetector/Path "%RTABMAP_PYTHON_SCRIPTS%\rtabmap_aliked.py" --PyDetector/Cuda true --Vis/CorNNType 6 --PyMatcher/Path "%RTABMAP_PYTHON_SCRIPTS%\rtabmap_lightglue.py" --PyMatcher/Model aliked --PyMatcher/Threshold 0.1 --PyMatcher/Cuda true
rtabmap %ARGS%
