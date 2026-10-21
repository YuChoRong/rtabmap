@echo off
rem Starts result_viewer.py (no compilation): result_viewer.bat [RESULT_DIR, default C:\slam\out]
rem Uses the Python environment of build_windows.ps1 (C:\slam\venv) when present, else creates
rem .viewer_env here, and installs PySide6 and numpy once.
setlocal
set PYEXE=%~dp0..\..\..\venv\Scripts\python.exe
if not exist "%PYEXE%" (
  set PYEXE=%~dp0.viewer_env\Scripts\python.exe
  if not exist "%~dp0.viewer_env\Scripts\python.exe" (
    echo Creating the Python environment %~dp0.viewer_env
    py -3 -m venv "%~dp0.viewer_env" 2>nul || python -m venv "%~dp0.viewer_env" || (echo Python 3 not found: install it from https://www.python.org/downloads/windows/ & exit /b 1)
  )
)
"%PYEXE%" -c "import PySide6.QtCharts, numpy" 2>nul || "%PYEXE%" -m pip install pyside6 numpy || (echo Installing PySide6 failed & exit /b 1)
"%PYEXE%" "%~dp0result_viewer.py" %*
