vcpkg overlay port of gtsam (from microsoft/vcpkg e456309, MIT license),
changed only to build gtsam_unstable (-DGTSAM_BUILD_UNSTABLE=ON), needed by OdometryVIO,
and without precompiled headers (MSVC 14.39 hangs on geometry/Unit3.cpp with them).
