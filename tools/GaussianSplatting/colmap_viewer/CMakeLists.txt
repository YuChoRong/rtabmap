cmake_minimum_required(VERSION 3.16)
project(colmap_viewer CXX)

# Standalone viewer of a COLMAP text model against ground truth (Qt6 or Qt5 Widgets + Eigen).
# Linux:   cmake -B build && cmake --build build
# Windows: cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.8.0/msvc2022_64" -DEIGEN3_INCLUDE_DIR="C:/eigen-3.4.0"
#          cmake --build build --config Release   (or the vcpkg toolchain, see ../README.md)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Qt6 QUIET COMPONENTS Widgets)
if(Qt6_FOUND)
  set(QT_WIDGETS Qt6::Widgets)
else()
  find_package(Qt5 5.14 REQUIRED COMPONENTS Widgets)
  set(QT_WIDGETS Qt5::Widgets)
endif()

# Eigen is header-only: a CMake package (apt, vcpkg) or just its folder in EIGEN3_INCLUDE_DIR
if(NOT EIGEN3_INCLUDE_DIR)
  find_package(Eigen3 QUIET NO_MODULE)
endif()
if(NOT TARGET Eigen3::Eigen AND NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 not found: install it (apt install libeigen3-dev / vcpkg install eigen3) or set -DEIGEN3_INCLUDE_DIR=<eigen folder>")
endif()

add_executable(colmap_viewer main.cpp ColmapModel.cpp Views.cpp)
target_link_libraries(colmap_viewer PRIVATE ${QT_WIDGETS})
if(TARGET Eigen3::Eigen AND NOT EIGEN3_INCLUDE_DIR)
  target_link_libraries(colmap_viewer PRIVATE Eigen3::Eigen)
else()
  target_include_directories(colmap_viewer PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(MSVC)
  target_compile_definitions(colmap_viewer PRIVATE _USE_MATH_DEFINES NOMINMAX)
  target_compile_options(colmap_viewer PRIVATE /utf-8 /bigobj)
endif()

if(WIN32)
  # Copy the Qt DLLs next to the executable so that it runs from the build folder
  find_program(WINDEPLOYQT windeployqt HINTS "${Qt6_DIR}/../../../bin" "${Qt5_DIR}/../../../bin" "${Qt6_DIR}/../../tools/Qt6/bin" "${Qt5_DIR}/../../tools/qt5/bin")
  if(WINDEPLOYQT)
    add_custom_command(TARGET colmap_viewer POST_BUILD
      COMMAND "${WINDEPLOYQT}" --no-translations --no-compiler-runtime "$<TARGET_FILE:colmap_viewer>"
      COMMENT "Deploying Qt DLLs")
  endif()
endif()

install(TARGETS colmap_viewer RUNTIME DESTINATION bin)
