@echo off
rem Starts colmap_viewer.py on Windows (no compilation): colmap_viewer.bat [COLMAP_DIR] [--gt FILE]
rem Uses the Python environment .gs_env of train_3dgs (created here if missing) and installs PySide6 and numpy once.
setlocal
set ENV=%~dp0.gs_env
if not exist "%ENV%\Scripts\python.exe" (
  echo Creating the Python environment %ENV%
  py -3 -m venv "%ENV%" 2>nul || python -m venv "%ENV%" || (echo Python 3 not found: install it from https://www.python.org/downloads/windows/ & exit /b 1)
)
"%ENV%\Scripts\python.exe" -c "import PySide6, numpy" 2>nul || "%ENV%\Scripts\python.exe" -m pip install pyside6 numpy || exit /b 1
"%ENV%\Scripts\python.exe" "%~dp0colmap_viewer.py" %*
