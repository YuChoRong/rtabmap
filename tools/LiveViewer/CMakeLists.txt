FIND_PACKAGE(yaml-cpp QUIET)

IF(yaml-cpp_FOUND)
  IF (TARGET yaml-cpp::yaml-cpp)
    SET(YAML_CPP_LIBRARIES yaml-cpp::yaml-cpp)
  ELSEIF (TARGET yaml-cpp)
    SET(YAML_CPP_LIBRARIES yaml-cpp)
  ENDIF()
ELSE()
   find_package(PkgConfig QUIET)
   IF(PKG_CONFIG_FOUND)
      pkg_check_modules(yaml_cpp QUIET yaml-cpp)
      IF(yaml_cpp_FOUND)
         SET(YAML_CPP_LIBRARIES ${yaml_cpp_LIBRARIES})
         SET(YAML_CPP_INCLUDE_DIR ${yaml_cpp_INCLUDEDIR})
         SET(yaml-cpp_FOUND ${yaml_cpp_FOUND})
      ENDIF(yaml_cpp_FOUND)
   ENDIF(PKG_CONFIG_FOUND)
ENDIF(yaml-cpp_FOUND)

IF(yaml-cpp_FOUND)

INCLUDE_DIRECTORIES(${YAML_CPP_INCLUDE_DIR})

# console application: the console keeps RTAB-Map's own log
ADD_EXECUTABLE(live_viewer main.cpp)
TARGET_LINK_LIBRARIES(live_viewer rtabmap_gui ${YAML_CPP_LIBRARIES})
IF(MSVC)
  # Korean UI strings in main.cpp (UTF-8)
  TARGET_COMPILE_OPTIONS(live_viewer PRIVATE /utf-8)
ENDIF()

SET_TARGET_PROPERTIES( live_viewer
  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-live_viewer)

INSTALL(TARGETS live_viewer
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
		BUNDLE DESTINATION "${CMAKE_BUNDLE_LOCATION}" COMPONENT runtime)

ELSE()
  MESSAGE(STATUS "yaml-cpp not found, live_viewer won't be built...")
ENDIF()
