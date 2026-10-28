@echo off
rem Windows cmd.exe wrapper of mesh_3dgs.ps1: mesh_3dgs.bat -Euroc C:\slam\data\REC\euroc_rgb [-Out DIR] [-Map map.db] [-Steps 30000]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0mesh_3dgs.ps1" %*
