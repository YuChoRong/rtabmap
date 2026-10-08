# Trains 3D Gaussian Splatting on Windows with train_gsplat.py (gsplat library, Apache-2.0).
# Needs an NVIDIA GPU with its driver, and Python 3.10-3.12 (python.org or conda).
#
# usage (PowerShell):
#   .\train_3dgs.ps1 -Data colmap_v101 [-Output colmap_v101\3dgs] [-Cuda cu124] [-Steps 30000] [-Factor 1] [-Venv .gs_env]
# or from cmd.exe: train_3dgs.bat colmap_v101
#
# On the first run it creates a Python virtual environment, installs PyTorch for
# the given CUDA version and a pre-compiled gsplat wheel matching that PyTorch
# (https://docs.gsplat.studio/whl). If no pre-compiled wheel exists, gsplat is
# installed from PyPI and compiles its CUDA code at the first run, which needs
# Visual Studio Build Tools (C++) and the CUDA Toolkit (see README.md).
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [string]$Output = "",
    [string]$Cuda = "cu124",
    [int]$Steps = 30000,
    [int]$Factor = 1,
    [string]$Venv = (Join-Path $PSScriptRoot ".gs_env"),
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
$ErrorActionPreference = "Stop"
if (-not $Output) { $Output = Join-Path $Data "3dgs" }

function Find-Python {
    foreach ($cmd in @("py -3.12", "py -3.11", "py -3.10", "python")) {
        $exe, $arg = $cmd.Split(" ", 2)
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            try { & $exe $arg -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" 2>$null; if ($LASTEXITCODE -eq 0) { return @($exe, $arg) } } catch {}
        }
    }
    throw "Python 3.10-3.12 not found: install it from https://www.python.org/downloads/windows/"
}

$python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
    $exe, $arg = Find-Python
    Write-Host "Creating the Python environment $Venv"
    if ($arg) { & $exe $arg -m venv $Venv } else { & $exe -m venv $Venv }
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

& $python -c "import torch, gsplat, numpy, PIL" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install --upgrade pip
    & $python -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed (check -Cuda, e.g. cu118, cu121, cu124, cu126)" }
    & $python -m pip install numpy pillow
    # Pre-compiled gsplat for this PyTorch and CUDA, e.g. pt24cu124
    $tag = & $python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print('pt%s%scu%s' % (v[0], v[1], torch.version.cuda.replace('.', '')))"
    # Dependencies from PyPI, then gsplat only from its wheel index (a newer PyPI version would have no binaries)
    & $python -m pip install ninja jaxtyping rich typing_extensions
    & $python -m pip install gsplat --no-deps --index-url "https://docs.gsplat.studio/whl/$tag"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "No pre-compiled gsplat for ${tag}: installing from PyPI (CUDA code compiled at first run, needs Visual Studio Build Tools and the CUDA Toolkit)"
        & $python -m pip install gsplat
        if ($LASTEXITCODE -ne 0) { throw "gsplat installation failed" }
    }
}

& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not available to PyTorch'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot use the GPU (driver or -Cuda version)" }

& $python (Join-Path $PSScriptRoot "train_gsplat.py") $Data --output $Output --steps $Steps --factor $Factor @Extra
if ($LASTEXITCODE -ne 0) { throw "Training failed" }
