# Intel RealSense D435i recordings with our VIO

`realsense_bag_to_euroc.py` converts a RealSense Viewer recording (`.bag`) to
the EuRoC layout, so it can be processed by `rtabmap-euroc_dataset` with our
VIO (`--Odom/Strategy 15`, stereo IR + IMU), with or without ALIKED/LightGlue.

## Recording (Intel RealSense Viewer)

- Stereo Module: **Infrared 1 and Infrared 2**, Y8, same resolution (e.g. 640x480 or 848x480) and 30 fps.
- Stereo Module > Controls: **Emitter Enabled = Off** (the dot pattern breaks the tracking).
- Motion Module: Accel and Gyro on, highest rates.
- RGB Camera (optional, for COLMAP / 3DGS).
- Hold the camera **still for 2-3 s** at the start (VIO static initialization),
  then move slowly, and come back to the start to close a loop. 60-90 s is enough.

## Processing

```bash
pip install rosbags numpy opencv-python-headless       # rosbags: Apache-2.0
python realsense_bag_to_euroc.py recording.bag rec [--rgb]
rtabmap-euroc_dataset --Odom/Strategy 15 --Rtabmap/ImagesAlreadyRectified true --output out --output_name rec rec
```

The converter checks the stereo baseline (~50 mm on a D435i) and prints the
mean acceleration at the start, which should be about `[9.8, 0, 0]` with the
camera level (IMU rotated to the EuRoC axes expected by `rtabmap-euroc_dataset`).

`make_test_bag_from_euroc.py` writes a bag with the same topics from a EuRoC
sequence (rectified stereo, IMU in camera axes), used to test the converter:
on the first 60 s of V1_01 the VIO gives 3.9 cm RMSE after conversion.
