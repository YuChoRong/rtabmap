# Self-contained Windows package of a build_windows.ps1 build: unzip it on another PC and
# run live_viewer.bat, no Visual Studio, vcpkg or Python install needed (only an NVIDIA
# driver for ALIKED/LightGlue on the GPU; the CPU path works without).
#
# usage (cmd.exe): tools\Windows\make_portable.bat [-Root C:\slam] [-Out D:\rtabmap_portable] [-NoZip]
#
# Layout of the package:
#   bin\        RTAB-Map executables, its DLLs and the vcpkg DLLs (Qt, OpenCV, PCL, VTK,
#               GTSAM, librealsense), MSVC and OpenMP runtimes
#   plugins\    Qt plugins
#   python\     Python (standard library) with the packages of the venv (torch, LightGlue...)
#   torch\      ALIKED / LightGlue weights (offline)
#   scripts\    Python feature plugins (rtabmap_aliked.py, rtabmap_lightglue.py...)
#   tools\      recording converter (RealSense\realsense_bag_to_euroc.py)
#   maps\       default folder of the maps made by the viewer
#   env.bat, live_viewer.bat, cmd.bat
param(
    [string]$Root = "C:\slam",
    [string]$Out = "",
    [switch]$NoZip
)
# see build_windows.ps1: native programs' stderr must not stop the script
$ErrorActionPreference = "Continue"
$Source = (Resolve-Path (Join-Path $PSScriptRoot "..\..") -ErrorAction Stop).Path
$Install = Join-Path $Root "install"
$VcpkgBin = Join-Path $Root "vcpkg\installed\x64-windows-release"
$Venv = Join-Path $Root "venv"
if (-not $Out) { $Out = Join-Path $Root "portable\rtabmap_portable" }

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
# robocopy: exit codes 0-7 are success
function Copy-Dir($from, $to, [string[]]$extra) {
    robocopy $from $to /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1 @extra | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copy failed: $from -> $to (robocopy $LASTEXITCODE)" }
}

foreach ($p in @("$Install\bin\rtabmap-live_viewer.exe", "$VcpkgBin\bin", "$Venv\pyvenv.cfg")) {
    if (-not (Test-Path $p)) { throw "$p not found: build first with build_windows.bat -Root $Root" }
}
$pyHome = ((Get-Content "$Venv\pyvenv.cfg" | Where-Object { $_ -match '^home\s*=' }) -replace '^home\s*=\s*', '').Trim()
if (-not (Test-Path "$pyHome\python.exe")) { throw "Python of the venv not found: $pyHome" }

if (Test-Path $Out) {
    Write-Host "Removing the previous package $Out"
    Remove-Item -Recurse -Force $Out -ErrorAction Stop
}
New-Item -ItemType Directory -Force -Path $Out -ErrorAction Stop | Out-Null

Step "1/6 RTAB-Map and its DLLs"
Copy-Dir "$Install\bin" "$Out\bin" @("/XF", "gmock*.dll", "gtest*.dll")
# only the DLLs: the .pdb debug files are 90% of the vcpkg bin folder
Copy-Dir "$VcpkgBin\bin" "$Out\bin" @("*.dll")
Copy-Dir "$VcpkgBin\Qt6\plugins" "$Out\plugins"
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$redist = Get-ChildItem "$vs\VC\Redist\MSVC" -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path "$($_.FullName)\x64" } | Sort-Object Name | Select-Object -Last 1
if ($redist) {
    foreach ($crt in @("Microsoft.VC143.CRT", "Microsoft.VC143.OpenMP")) {
        if (Test-Path "$($redist.FullName)\x64\$crt") { Copy-Dir "$($redist.FullName)\x64\$crt" "$Out\bin" }
    }
} else {
    Write-Warning "MSVC redistributable not found: the target PC needs the Visual C++ 2015-2022 x64 runtime"
}

Step "2/6 Python ($pyHome) and the venv packages"
Copy-Dir $pyHome "$Out\python" @("/XD", "$pyHome\Lib\site-packages", "$pyHome\Lib\test", "$pyHome\Doc", "$pyHome\Scripts", "$pyHome\include", "$pyHome\libs", "$pyHome\tcl", "__pycache__")
Copy-Dir "$Venv\Lib\site-packages" "$Out\python\Lib\site-packages" @("/XD", "__pycache__", "/XF", "*.pdb")

Step "3/6 ALIKED / LightGlue weights"
$hub = Join-Path $env:USERPROFILE ".cache\torch\hub\checkpoints"
New-Item -ItemType Directory -Force -Path "$Out\torch\hub\checkpoints" | Out-Null
foreach ($w in @("aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth", "superpoint_v1.pth", "superpoint_lightglue_v0-1_arxiv.pth")) {
    if (Test-Path "$hub\$w") { Copy-Item "$hub\$w" "$Out\torch\hub\checkpoints\" -ErrorAction Stop }
}
if (-not (Test-Path "$Out\torch\hub\checkpoints\aliked-n16.pth")) {
    Write-Warning "ALIKED weights not cached in ${hub}: they are downloaded at the first ALIKED run on the target PC"
}

Step "4/6 Python plugins and tools"
New-Item -ItemType Directory -Force -Path "$Out\scripts", "$Out\tools\RealSense", "$Out\maps" | Out-Null
Copy-Item "$Source\corelib\src\python\*.py" "$Out\scripts\" -ErrorAction Stop
Copy-Item "$Source\tools\RealSense\*.py", "$Source\tools\RealSense\README.md" "$Out\tools\RealSense\" -ErrorAction Stop

Step "5/6 Launchers"
# all paths relative to the package folder (%~dp0), whatever its location
@"
@echo off
rem Environment of the portable RTAB-Map package (made by make_portable.ps1)
set "RTABMAP_ROOT=%~dp0"
if "%RTABMAP_ROOT:~-1%"=="\" set "RTABMAP_ROOT=%RTABMAP_ROOT:~0,-1%"
set "PATH=%RTABMAP_ROOT%\bin;%RTABMAP_ROOT%\python;%RTABMAP_ROOT%\python\Lib\site-packages\torch\lib;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem"
set "PYTHONHOME=%RTABMAP_ROOT%\python"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "QT_PLUGIN_PATH=%RTABMAP_ROOT%\plugins"
set "TORCH_HOME=%RTABMAP_ROOT%\torch"
set "RTABMAP_TOOLS=%RTABMAP_ROOT%\tools"
set "RTABMAP_PYTHON_SCRIPTS=%RTABMAP_ROOT%\scripts"
"@ | Set-Content -Encoding ASCII "$Out\env.bat"
@"
@echo off
rem Real-time mapping / localization viewer: live_viewer.bat [--open MAP.db]
setlocal
call "%~dp0env.bat"
start "" "%RTABMAP_ROOT%\bin\rtabmap-live_viewer.exe" %*
"@ | Set-Content -Encoding ASCII "$Out\live_viewer.bat"
@"
@echo off
rem Command prompt with the RTAB-Map environment (rtabmap-*.exe, python)
call "%~dp0env.bat"
cd /d "%RTABMAP_ROOT%"
cmd /k echo RTAB-Map portable: %RTABMAP_ROOT%
"@ | Set-Content -Encoding ASCII "$Out\cmd.bat"
$commit = (git -C $Source rev-parse --short HEAD 2>$null)
@"
RTAB-Map portable package (commit $commit, made $(Get-Date -Format 'yyyy-MM-dd'))

Run:  live_viewer.bat          real-time mapping / localization viewer (D435i or recordings)
      cmd.bat                  command prompt with rtabmap-*.exe and python in the PATH

Needs: Windows 10/11 x64. ALIKED + LightGlue on the GPU need an NVIDIA driver (CUDA is
included with PyTorch); ORB works on any PC. D435i: plug it in a USB 3 port.
Maps are saved in maps\ by default. Unzip to a short path without spaces (e.g. C:\rtabmap).
"@ | Set-Content -Encoding UTF8 "$Out\README.txt"

Step "6/6 Check (clean PATH, this package only)"
$check = Join-Path $env:TEMP "rtabmap_portable_check.bat"
@"
@echo off
call "$Out\env.bat"
python -c "import torch, lightglue, cv2, rosbags; print('python ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 exit /b 1
"$Out\bin\rtabmap-console.exe" --version
"@ | Set-Content -Encoding ASCII $check
cmd /c $check
if ($LASTEXITCODE -ne 0) { throw "The package does not run (see above)" }

$size = (Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("Package: {0} ({1:N1} GB)" -f $Out, $size)
if (-not $NoZip) {
    $zip = "$Out.zip"
    Write-Host "Zipping to $zip (several minutes)..."
    if (Test-Path $zip) { Remove-Item -Force $zip -ErrorAction Stop }
    # Python's zipfile writes ZIP64 (> 4 GB) archives, unlike Compress-Archive of PowerShell 5
    & "$Out\python\python.exe" -I -c "import shutil, sys; shutil.make_archive(sys.argv[1][:-4], 'zip', sys.argv[2], sys.argv[3])" $zip (Split-Path $Out) (Split-Path $Out -Leaf)
    if ($LASTEXITCODE -ne 0) { throw "zip failed" }
    Write-Host ("Zip: {0} ({1:N1} GB)" -f $zip, ((Get-Item $zip).Length / 1GB))
}
