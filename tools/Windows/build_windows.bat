@echo off
rem cmd.exe wrapper of build_windows.ps1, e.g.: tools\Windows\build_windows.bat -Root C:\slam
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
exit /b %errorlevel%
