@echo off
rem Windows cmd.exe wrapper of train_3dgs.ps1: train_3dgs.bat COLMAP_DIR [-Cuda cu124] [-Steps 30000] [-Factor 2]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_3dgs.ps1" -Data %*
