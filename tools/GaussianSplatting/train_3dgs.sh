#!/usr/bin/env bash
# Trains 3D Gaussian Splatting with gsplat (nerfstudio-project/gsplat,
# Apache-2.0) on a COLMAP folder made by rtabmap_to_colmap.py. Needs a CUDA
# GPU, PyTorch and gsplat with the requirements of its examples:
#   pip install gsplat && pip install -r GSPLAT_REPO/examples/requirements.txt
#
# The Inria reference code (graphdeco-inria/gaussian-splatting) is not used:
# its license allows research and evaluation only, without commercial use.
#
# usage: train_3dgs.sh COLMAP_DIR [OUTPUT_DIR] [extra simple_trainer.py options...]
#   GSPLAT_REPO: gsplat checkout for its examples (default ~/gsplat, cloned if missing)
#   MAX_STEPS: training steps (default 30000)
set -e
[ $# -ge 1 ] || { echo "usage: $0 COLMAP_DIR [OUTPUT_DIR] [simple_trainer.py options]"; exit 1; }
DATA=$(realpath "$1"); shift
OUT=${1:-$DATA/3dgs}; [ $# -gt 0 ] && shift
OUT=$(realpath -m "$OUT")
GSPLAT_REPO=${GSPLAT_REPO:-$HOME/gsplat}
MAX_STEPS=${MAX_STEPS:-30000}

if [ ! -d "$GSPLAT_REPO" ]; then
  git clone https://github.com/nerfstudio-project/gsplat "$GSPLAT_REPO"
fi

# The RTAB-Map export may be grayscale (e.g., EuRoC): the trainer expects RGB images
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

cd "$GSPLAT_REPO/examples"
# Every 8th image is kept for evaluation (PSNR/SSIM/LPIPS in OUT/stats)
python simple_trainer.py default --data_dir "$DATA" --data_factor 1 --test_every 8 \
  --max_steps "$MAX_STEPS" --save_ply --result_dir "$OUT" "$@"
echo "Model and statistics: $OUT/ply, $OUT/stats (open the .ply with supersplat or any 3DGS viewer)"
