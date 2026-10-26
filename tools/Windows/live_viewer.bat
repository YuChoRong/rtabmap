@echo off
rem Real-time mapping / localization viewer (rtabmap-live_viewer.exe, built with RTAB-Map):
rem   tools\Windows\live_viewer.bat [--open MAP.db]
rem Sets the environment of the build (DLLs, Qt plugins, Python plugins for ALIKED/LightGlue,
rem converter of recordings) with rtabmap_env.bat of C:\slam\install, or RTABMAP_INSTALL.
setlocal
if not defined RTABMAP_INSTALL set RTABMAP_INSTALL=%~dp0..\..\..\install
if not exist "%RTABMAP_INSTALL%\rtabmap_env.bat" (
    echo rtabmap_env.bat not found in %RTABMAP_INSTALL%: build with build_windows.bat first, or set RTABMAP_INSTALL
    exit /b 1
)
call "%RTABMAP_INSTALL%\rtabmap_env.bat"
if not exist "%RTABMAP_INSTALL%\bin\rtabmap-live_viewer.exe" (
    echo rtabmap-live_viewer.exe not found: run build_windows.bat again to rebuild RTAB-Map
    exit /b 1
)
"%RTABMAP_INSTALL%\bin\rtabmap-live_viewer.exe" %*
