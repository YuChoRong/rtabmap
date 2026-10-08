#!/usr/bin/env bash
# Trains 3D Gaussian Splatting on a COLMAP folder made by rtabmap_to_colmap.py,
# with train_gsplat.py (built on the gsplat library, Apache-2.0). Needs a CUDA GPU.
#
# usage: train_3dgs.sh COLMAP_DIR [OUTPUT_DIR] [train_gsplat.py options, e.g. --steps 30000 --factor 2]
#   Installs the Python dependencies (torch, gsplat, numpy, pillow) when missing.
set -e
[ $# -ge 1 ] || { echo "usage: $0 COLMAP_DIR [OUTPUT_DIR] [train_gsplat.py options]"; exit 1; }
DATA=$1; shift
OUT=${DATA%/}/3dgs
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then OUT=$1; shift; fi
HERE=$(cd "$(dirname "$0")" && pwd)

python -c "import torch, gsplat, numpy, PIL" 2>/dev/null || pip install torch gsplat numpy pillow
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'"
python "$HERE/train_gsplat.py" "$DATA" --output "$OUT" "$@"
