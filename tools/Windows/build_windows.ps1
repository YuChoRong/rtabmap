# Builds this fork on Windows with our VIO (--Odom/Strategy 15), the Python feature
# plugins (ALIKED + LightGlue on the GPU) and the RealSense2 driver for live D435i use.
#
# usage (PowerShell, from the repository root):
#   .\tools\Windows\build_windows.ps1 [-Root C:\slam] [-Cuda auto|cu118|cu126|cu128|cpu] [-NoGui] [-Jobs N]
# or from cmd.exe: tools\Windows\build_windows.bat -Root C:\slam
#
# Needs Visual Studio 2022 (Desktop development with C++), Git and Python 3.10-3.13
# from python.org. The first run builds OpenCV, PCL, Qt, VTK, GTSAM and librealsense
# with vcpkg (2-3 hours); later runs only rebuild RTAB-Map.
#
# Licenses: everything built here is BSD/MIT/Apache/MPL/LGPL (Qt, dynamically linked).
# OpenCV nonfree, OpenVINS and ORB-OCTREE (GPL-3.0) are disabled.
param(
    [string]$Root = "C:\slam",
    [string]$Cuda = "auto",
    [switch]$NoGui,
    [int]$Jobs = 0
)
# Windows PowerShell 5 turns any stderr line of a native program (pip warnings, a failed
# import probe) into a terminating error under "Stop": keep "Continue" and check
# $LASTEXITCODE after each native command; cmdlets use -ErrorAction Stop.
$ErrorActionPreference = "Continue"
$VcpkgCommit = "e456309491fd875487bf02b8a0dca76c1b1b00cc"  # gtsam 4.3.0, the version the VIO was validated with
$Triplet = "x64-windows-release"
$Source = (Resolve-Path (Join-Path $PSScriptRoot "..\..") -ErrorAction Stop).Path
$Vcpkg = Join-Path $Root "vcpkg"
$Venv = Join-Path $Root "venv"
$Build = Join-Path $Root "build"
$Install = Join-Path $Root "install"
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

if ($Root.Length -gt 12) { Write-Warning "Long -Root paths can break the vcpkg builds (Windows path limit), prefer e.g. C:\slam" }
New-Item -ItemType Directory -Force -Path $Root -ErrorAction Stop | Out-Null

Step "1/6 Check Visual Studio, Git, Python"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git not found: install it from https://git-scm.com/download/win" }
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio 2022 not found: install it with 'Desktop development with C++'" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ tools not found: add 'Desktop development with C++' in the Visual Studio Installer" }
Write-Host "Visual Studio: $vs"
# x64 compiler, CMake and Ninja of Visual Studio in this shell
Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll") -ErrorAction Stop
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" -ErrorAction Stop | Out-Null
foreach ($t in @("cl", "cmake", "ninja")) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not found: add 'C++ CMake tools for Windows' in the Visual Studio Installer" }
}

function Find-Python {
    foreach ($cmd in @("py -3.12", "py -3.11", "py -3.13", "py -3.10", "python")) {
        $exe, $arg = $cmd.Split(" ", 2)
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            try { & $exe $arg -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)" 2>$null; if ($LASTEXITCODE -eq 0) { return @($exe, $arg) } } catch {}
        }
    }
    throw "Python 3.10-3.13 not found: install it from https://www.python.org/downloads/windows/"
}

if ($Cuda -eq "auto") {
    $Cuda = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $m = (nvidia-smi | Select-String "CUDA Version:\s*(\d+)\.(\d+)").Matches
        if ($m) {
            $v = [int]$m[0].Groups[1].Value * 10 + [int]$m[0].Groups[2].Value
            if ($v -ge 128) { $Cuda = "cu128" } elseif ($v -ge 126) { $Cuda = "cu126" } elseif ($v -ge 118) { $Cuda = "cu118" }
        }
    }
    Write-Host "PyTorch: $Cuda (from nvidia-smi)"
}

Step "2/6 Python environment ($Venv): PyTorch, LightGlue/ALIKED"
$python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
    $exe, $arg = Find-Python
    if ($arg) { & $exe $arg -m venv $Venv } else { & $exe -m venv $Venv }
    Check "venv creation"
}
& $python -c "import torch, lightglue, pybind11, numpy, rosbags, cv2" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install --upgrade pip
    # torchvision from the same index: kornia/LightGlue would otherwise pull it from PyPI
    # and replace torch with the CPU-only wheel
    & $python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    Check "PyTorch installation (try another -Cuda: cu118, cu126, cu128 or cpu)"
    # LightGlue (Apache-2.0) also provides ALIKED (BSD-3)
    & $python -m pip install numpy pybind11 rosbags opencv-python-headless "git+https://github.com/cvg/LightGlue.git"
    Check "Python packages installation"
}
& $python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
$pyBase = (& $python -c "import sys; print(sys.base_prefix)").Trim()
$pybind11Dir = (& $python -m pybind11 --cmakedir).Trim()

Step "3/6 vcpkg $($VcpkgCommit.Substring(0,7)) ($Vcpkg)"
if (-not (Test-Path (Join-Path $Vcpkg ".git"))) {
    git clone https://github.com/microsoft/vcpkg.git $Vcpkg; Check "vcpkg clone"
}
git -C $Vcpkg cat-file -e "$VcpkgCommit^{commit}" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { git -C $Vcpkg fetch origin; Check "vcpkg fetch" }
git -C $Vcpkg checkout -q $VcpkgCommit; Check "vcpkg checkout"
& (Join-Path $Vcpkg "bootstrap-vcpkg.bat") -disableMetrics; Check "vcpkg bootstrap"

Step "4/6 Dependencies with vcpkg (first run: 2-3 hours)"
# Release-only triplet (half the build time); no default features to keep nonfree/GPL code out.
$packages = @(
    "eigen3", "zlib",
    "opencv4[core,calib3d,contrib,fs,highgui,intrinsics,jpeg,png,thread,win32ui]",
    "gtsam[core,boost]",           # overlay port with gtsam_unstable (fixed-lag smoother of the VIO)
    "realsense2[core]"             # live D435i (Apache-2.0)
)
if ($NoGui) {
    $packages += "pcl[core]"
} else {
    $packages += @("pcl[core,qt,visualization]", "qtbase[core,gui,opengl,widgets,png]", "vtk[core]")
}
# From $Root (classic mode): in the repository root vcpkg would use the upstream vcpkg.json manifest
Push-Location $Root
& (Join-Path $Vcpkg "vcpkg.exe") install --triplet $Triplet --host-triplet $Triplet `
    --overlay-ports (Join-Path $PSScriptRoot "vcpkg-overlay") --clean-after-build @packages
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "vcpkg install failed (exit code $code), see the log named above" }

Step "5/6 Configure and build RTAB-Map ($Build)"
$gui = if ($NoGui) { "OFF" } else { "ON" }
cmake -S $Source -B $Build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$Vcpkg\scripts\buildsystems\vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=$Triplet" `
    -DVCPKG_MANIFEST_MODE=OFF `
    "-DCMAKE_INSTALL_PREFIX=$Install" `
    "-DWITH_QT=$gui" -DRTABMAP_QT_VERSION=6 `
    -DWITH_PYTHON=ON "-DPython3_EXECUTABLE=$python" "-Dpybind11_DIR=$pybind11Dir" `
    -DWITH_GTSAM=ON -DWITH_REALSENSE2=ON `
    -DWITH_OPENVINS=OFF -DWITH_ORB_OCTREE=OFF -DWITH_TORCH=OFF `
    -DWITH_PDAL=OFF -DWITH_OCTOMAP=OFF -DWITH_G2O=OFF -DWITH_CERES=OFF `
    -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=ON -DBUILD_APP=ON
Check "CMake configuration"
cmake --build $Build --parallel $Jobs; Check "Build"
cmake --install $Build; Check "Install"

Step "6/6 Environment script"
$qtPlugins = Join-Path $Vcpkg "installed\$Triplet\Qt6\plugins"
$envBat = Join-Path $Install "rtabmap_env.bat"
@"
@echo off
rem Generated by build_windows.ps1: RTAB-Map, its DLLs, Python plugins and tools in this cmd.exe
call "$Venv\Scripts\activate.bat"
set "PATH=$Install\bin;$Vcpkg\installed\$Triplet\bin;$pyBase;%PATH%"
set "PYTHONPATH=$Venv\Lib\site-packages"
set "QT_PLUGIN_PATH=$qtPlugins"
set "RTABMAP_TOOLS=$Source\tools"
set "RTABMAP_PYTHON_SCRIPTS=$Source\corelib\src\python"
"@ | Set-Content -Encoding ASCII $envBat

# Smoke test with the same environment
$env:PATH = "$Install\bin;$Vcpkg\installed\$Triplet\bin;$pyBase;$env:PATH"
& (Join-Path $Install "bin\rtabmap-console.exe") -v; Check "rtabmap-console"
& $python -c "import torch, lightglue; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Check "Python plugins"
Write-Host "`nDone. In cmd.exe:  call $envBat" -ForegroundColor Green
if (-not $NoGui) { Write-Host "Live D435i: $Source\tools\Windows\run_d435i_live.bat" -ForegroundColor Green }
