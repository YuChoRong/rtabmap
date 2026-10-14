vcpkg overlay port of gtsam (from microsoft/vcpkg e456309, MIT license),
changed only to build gtsam_unstable (-DGTSAM_BUILD_UNSTABLE=ON), needed by OdometryVIO,
and without precompiled headers (MSVC 14.39 hangs on geometry/Unit3.cpp with them),
as a static library (the DLL exports std::vector<Key> through Ordering, LNK2005 in rtabmap_core).

qtdeclarative: the port of microsoft/vcpkg e456309 (MIT license) with one comment-only patch,
so that the source folder is named ...-dc3513ca88 instead of ...-f94468adb5. On this test PC an
endpoint security policy silently kills any process whose command line contains "adb"
(Android Debug Bridge), which made CMake exit with code 0 in the middle of the configure.
If another package fails the same way (configure stops right after "The C compiler
identification is ...", no CMakeCache.txt), check its buildtrees folder name for "adb".
