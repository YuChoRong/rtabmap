@echo off
rem cmd.exe wrapper of make_portable.ps1, e.g.: tools\Windows\make_portable.bat -Root C:\slam -Out D:\rtabmap_portable
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_portable.ps1" %*
exit /b %errorlevel%
