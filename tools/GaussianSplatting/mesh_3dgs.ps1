# Mesh (2D Gaussian Splatting) and 3DGS model of a recording, on Windows (NVIDIA GPU).
#
# usage (PowerShell):
#   .\mesh_3dgs.ps1 -Euroc C:\slam\data\20261013_105416\euroc_rgb [-Out C:\slam\out\mesh_20261013] [-Map map.db]
#   .\mesh_3dgs.ps1 -Bag C:\slam\data\rec.bag [-EmitterSplit]   (converted first to OUT\euroc_rgb;
#     -EmitterSplit for "Emitter On Off" recordings, see tools\RealSense\README.md)
# or from cmd.exe: mesh_3dgs.bat -Euroc C:\slam\data\20261013_105416\euroc_rgb
#
# Steps (each one is skipped when its result already exists, delete the file to redo it):
#   1. SLAM with more keyframes (-DetectionRate per second) -> OUT\map.db       (skipped with -Map)
#   2. rtabmap_to_colmap.py --euroc: color images, poses, stereo depth -> OUT\colmap
#   3. train_gsplat.py --mode 2dgs (environment of train_3dgs.ps1)    -> OUT\colmap\2dgs
#   4. extract_mesh.py: TSDF fusion of the rendered depth              -> OUT\colmap\2dgs\mesh.ply
#      and the same with holes filled by a Poisson surface            -> OUT\colmap\2dgs\mesh_hybrid.ply
#   5. Baselines: TSDF of the stereo depth (OUT\colmap\stereo_mesh.ply) and
#      RTAB-Map's Poisson mesh (OUT\poisson_mesh.ply)
#   6. eval_mesh.py on the held-out images                            -> OUT\mesh_eval.json
#   7. Model for viewing (skipped with -NoRender): export with extra color images between
#      the keyframes (-ExtraRate per second) and 3DGS training with the MCMC strategy,
#      blurred images left out                                         -> OUT\colmap_x8\3dgs
param(
    [string]$Euroc = "",
    [string]$Bag = "",
    [switch]$EmitterSplit,
    [string]$ConvertPython = "C:\slam\venv\Scripts\python.exe",
    [string]$Out = "",
    [string]$Map = "",
    [string]$Install = "C:\slam\install",
    [double]$DetectionRate = 4,
    [string]$SlamArgs = "--Odom/Strategy 15 --OdomVIO/TimeOffset -0.037 --Rtabmap/ImagesAlreadyRectified true",
    [int]$Steps = 30000,
    [int]$MaxGaussians = 1500000,   # more did not improve the mesh (20261027 recording), only the time
    [int]$Factor = 2,          # half resolution: 3x faster than 1, mesh as good (20261027 emitter recording)
    [switch]$NoRender,
    [double]$ExtraRate = 8,
    [string]$Cuda = "cu124",
    [string]$Venv = ""
)
$ErrorActionPreference = "Stop"
# Progress lines of the Python steps show up as they are printed, also when the output is piped
$env:PYTHONUNBUFFERED = "1"
# PowerShell 5: $PSScriptRoot is empty in param() defaults
if (-not $Venv) { $Venv = Join-Path $PSScriptRoot ".gs_env" }
if (-not $Euroc -and -not $Bag) { throw "Give -Euroc (converted recording) or -Bag (RealSense Viewer recording)" }
if (-not $Out) {
    $name = if ($Bag) { [IO.Path]::GetFileNameWithoutExtension($Bag) } else { Split-Path (Split-Path $Euroc -Parent) -Leaf }
    $Out = Join-Path "C:\slam\out" ("mesh_" + $name)
}
New-Item -ItemType Directory -Force $Out | Out-Null
$colmap = Join-Path $Out "colmap"
$model = Join-Path $colmap "2dgs"
$python = Join-Path $Venv "Scripts\python.exe"

# RTAB-Map tools and DLLs: PATH of rtabmap_env.bat only (its Python environment is not used here)
$envBat = Join-Path $Install "rtabmap_env.bat"
if (-not (Test-Path $envBat)) { throw "$envBat not found (build RTAB-Map with tools\Windows\build_windows.ps1, or give -Install)" }
# Delayed expansion: %PATH% would be expanded before the call
$rtabPath = (cmd /v:on /c "call `"$envBat`" >nul && echo !PATH!").Trim()
$qtPlugins = (cmd /v:on /c "call `"$envBat`" >nul && echo !QT_PLUGIN_PATH!").Trim()

function Run-Rtabmap([string]$exe, [string[]]$arguments, [string]$log) {
    # Native tools write to stderr: do not let PowerShell 5 stop on it, check the exit code instead
    $saved = $env:PATH, $env:QT_PLUGIN_PATH
    $env:PATH = $rtabPath; $env:QT_PLUGIN_PATH = $qtPlugins
    $ErrorActionPreference = "Continue"
    & (Join-Path $Install "bin\$exe") @arguments *> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $env:PATH, $env:QT_PLUGIN_PATH = $saved
    if ($code -ne 0) { throw "$exe failed ($code), see $log" }
}

function Run-Python([string[]]$arguments, [switch]$WithRtabmap) {
    $saved = $env:PATH
    if ($WithRtabmap) { $env:PATH = $rtabPath }   # rtabmap-export called by rtabmap_to_colmap.py
    $ErrorActionPreference = "Continue"
    & $python @arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $env:PATH = $saved
    if ($code -ne 0) { throw "python $($arguments[0]) failed ($code)" }
}

$sw = [Diagnostics.Stopwatch]::StartNew()
function Step([string]$name) { Write-Host ("[{0:hh\:mm\:ss}] {1}" -f $sw.Elapsed, $name) -ForegroundColor Cyan }

# 0. Conversion of the recording (python of rtabmap_env.bat's environment, with rosbags)
if ($Bag) {
    $Euroc = Join-Path $Out "euroc_rgb"
    if (-not (Test-Path (Join-Path $Euroc "mav0\cam0\data.csv"))) {
        Step "0/7 Convert $Bag -> $Euroc"
        $a = @((Join-Path $PSScriptRoot "..\RealSense\realsense_bag_to_euroc.py"), $Bag, $Euroc, "--rgb")
        if ($EmitterSplit) { $a += "--emitter-split" }
        $ErrorActionPreference = "Continue"
        & $ConvertPython @a
        $code = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($code -ne 0) { throw "Conversion failed ($code)" }
    }
}

# 1. SLAM
if (-not $Map) {
    $Map = Join-Path $Out "map.db"
    if (-not (Test-Path $Map)) {
        Step "1/7 SLAM ($DetectionRate keyframes/s) -> $Map"
        $a = $SlamArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries) + @("--Rtabmap/DetectionRate", "$DetectionRate", "--output", $Out, "--output_name", "map", $Euroc)
        Run-Rtabmap "rtabmap-euroc_dataset.exe" $a (Join-Path $Out "slam.log")
    }
}

# Python environment (PyTorch, gsplat, Open3D, OpenCV): created by train_3dgs.ps1 with 0 steps
$ready = $false
if (Test-Path $python) {
    $ErrorActionPreference = "Continue"
    & $python -c "import torch, gsplat, cv2, open3d" 2>$null
    $ready = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
}
if (-not $ready) {
    Step "Python environment $Venv"
    & (Join-Path $PSScriptRoot "train_3dgs.ps1") -Data $colmap -Cuda $Cuda -Venv $Venv -Steps 0 -SetupOnly
}

# 2. Export
if (-not (Test-Path (Join-Path $colmap "sparse\0\images.txt"))) {
    Step "2/7 Export color images, poses and stereo depth -> $colmap"
    Run-Python @((Join-Path $PSScriptRoot "rtabmap_to_colmap.py"), $Map, $colmap, "--euroc", $Euroc,
                 "--rtabmap-export", (Join-Path $Install "bin\rtabmap-export.exe")) -WithRtabmap
}

# 3. Training
if (-not (Test-Path (Join-Path $model "point_cloud.ply"))) {
    Step "3/7 2DGS training ($Steps steps)"
    Run-Python @((Join-Path $PSScriptRoot "train_gsplat.py"), $colmap, "--output", $model, "--mode", "2dgs",
                 "--steps", "$Steps", "--factor", "$Factor", "--max-gaussians", "$MaxGaussians")
}

# 4. Mesh
if (-not (Test-Path (Join-Path $model "mesh.ply"))) {
    Step "4/7 Mesh of the 2DGS model"
    Run-Python @((Join-Path $PSScriptRoot "extract_mesh.py"), $colmap, "--model", $model)
}
if (-not (Test-Path (Join-Path $model "mesh_hybrid.ply"))) {
    Step "4/7 Mesh of the 2DGS model, holes of the TSDF mesh filled by a Poisson surface"
    Run-Python @((Join-Path $PSScriptRoot "extract_mesh.py"), $colmap, "--model", $model, "--method", "hybrid")
}

# 5. Baselines
if (-not (Test-Path (Join-Path $colmap "stereo_mesh.ply"))) {
    Step "5/7 Baseline: TSDF of the stereo depth"
    Run-Python @((Join-Path $PSScriptRoot "extract_mesh.py"), $colmap, "--source", "stereo", "--factor", "$Factor")
}
$poisson = Join-Path $Out "poisson_mesh.ply"
if (-not (Test-Path $poisson)) {
    Step "5/7 Baseline: RTAB-Map Poisson mesh"
    Run-Rtabmap "rtabmap-export.exe" @("--mesh", "--output", "poisson", "--output_dir", $Out, $Map) (Join-Path $Out "poisson.log")
}

# 6. Evaluation
Step "6/7 Comparison on the held-out images"
$meshes = @((Join-Path $model "mesh.ply"), (Join-Path $model "mesh_hybrid.ply"), (Join-Path $colmap "stereo_mesh.ply"))
if (Test-Path $poisson) { $meshes += $poisson }
Run-Python (@((Join-Path $PSScriptRoot "eval_mesh.py"), $colmap) + $meshes + @("--json", (Join-Path $Out "mesh_eval.json")))

# 7. Model for viewing: best render quality of the 20261027 tests (aligned PSNR 28.9 dB vs 26.0 dB
#    for 3DGS on the keyframes only, same time; stronger pose refinement removed ghost edges);
#    full resolution did not add detail there
if (-not $NoRender) {
    $colmapX = Join-Path $Out "colmap_x8"
    $render = Join-Path $colmapX "3dgs"
    if (-not (Test-Path (Join-Path $colmapX "sparse\0\images.txt"))) {
        Step "7/7 Export with $ExtraRate extra color images per second -> $colmapX"
        Run-Python @((Join-Path $PSScriptRoot "rtabmap_to_colmap.py"), $Map, $colmapX, "--euroc", $Euroc, "--extra-rate", "$ExtraRate",
                     "--rtabmap-export", (Join-Path $Install "bin\rtabmap-export.exe")) -WithRtabmap
    }
    if (-not (Test-Path (Join-Path $render "point_cloud.ply"))) {
        Step "7/7 3DGS training for viewing ($Steps steps, MCMC)"
        Run-Python @((Join-Path $PSScriptRoot "train_gsplat.py"), $colmapX, "--output", $render, "--mode", "3dgs", "--strategy", "mcmc",
                     "--min-sharpness", "0.5", "--pose-lr", "1e-4", "--steps", "$Steps", "--factor", "$Factor", "--max-gaussians", "$MaxGaussians")
    }
}
Step "Done: $(Join-Path $model 'mesh_hybrid.ply')$(if (-not $NoRender) { ', ' + (Join-Path $Out 'colmap_x8\3dgs\point_cloud.ply') })"
