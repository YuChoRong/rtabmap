#!/usr/bin/env bash
# Trains 3D Gaussian Splatting (graphdeco-inria/gaussian-splatting) on a COLMAP
# folder made by rtabmap_to_colmap.py. Needs a CUDA GPU and the environment of
# that repository (see its README: conda env create --file environment.yml).
#
# usage: train_3dgs.sh COLMAP_DIR [OUTPUT_DIR] [extra train.py options...]
#   GS_REPO: gaussian-splatting checkout (default ~/gaussian-splatting, cloned if missing)
#   ITERATIONS: training iterations (default 30000)
set -e
DATA=$(realpath "$1"); shift || { echo "usage: $0 COLMAP_DIR [OUTPUT_DIR]"; exit 1; }
OUT=${1:-$DATA/3dgs}; [ $# -gt 0 ] && shift
OUT=$(realpath -m "$OUT")
GS_REPO=${GS_REPO:-$HOME/gaussian-splatting}
ITERATIONS=${ITERATIONS:-30000}

if [ ! -d "$GS_REPO" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting "$GS_REPO"
fi

# The RTAB-Map export may be grayscale (e.g., EuRoC): 3DGS expects RGB images
python - "$DATA/images" <<'PY'
import os, sys
from PIL import Image
d = sys.argv[1]
n = 0
for f in os.listdir(d):
    p = os.path.join(d, f)
    im = Image.open(p)
    if im.mode != 'RGB':
        im.convert('RGB').save(p, quality=95)
        n += 1
print('%d images converted to RGB' % n)
PY

cd "$GS_REPO"
# --eval keeps every 8th image for testing, so render.py/metrics.py report PSNR/SSIM/LPIPS
python train.py -s "$DATA" -m "$OUT" --eval --iterations "$ITERATIONS" "$@"
python render.py -m "$OUT" --skip_train
python metrics.py -m "$OUT"
echo "Model: $OUT/point_cloud/iteration_$ITERATIONS/point_cloud.ply (open it with SIBR_gaussianViewer, supersplat, ...)"
