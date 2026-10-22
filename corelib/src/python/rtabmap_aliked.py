#! /usr/bin/env python3
#
# ALIKED keypoints and descriptors (BSD-3, https://github.com/Shiaoming/ALIKED)
# through the LightGlue package (Apache-2.0, https://github.com/cvg/LightGlue):
#   pip install torch torchvision kornia
#   pip install git+https://github.com/cvg/LightGlue.git
# The weights are downloaded at the first run (ALIKED BSD-3).
#
# To use with rtabmap (built with WITH_PYTHON=ON):
#   --Vis/FeatureType 15 --Kp/DetectorStrategy 15 --PyDetector/Path "~/rtabmap_aliked.py"
# Matching with LightGlue: see rtabmap_lightglue.py.
#
# Options (environment variables):
#   RTABMAP_ALIKED_MODEL      aliked-n16 (default), aliked-n16rot, aliked-n32, aliked-t16
#   RTABMAP_ALIKED_MAX_KPTS   maximum keypoints per image (default 1024), Kp/MaxFeatures still applies
#   RTABMAP_ALIKED_THRESHOLD  detection score threshold (default 0.2)
#

import os
import numpy as np
import torch

from lightglue import ALIKED

torch.set_grad_enabled(False)

device = 'cpu'
aliked = None

def init(cuda):
    global device, aliked
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    aliked = ALIKED(model_name=os.environ.get('RTABMAP_ALIKED_MODEL', 'aliked-n16'),
                    max_num_keypoints=int(os.environ.get('RTABMAP_ALIKED_MAX_KPTS', '1024')),
                    detection_threshold=float(os.environ.get('RTABMAP_ALIKED_THRESHOLD', '0.2'))).eval().to(device)
    print('ALIKED initialized on %s' % device)

@torch.no_grad()  # grad mode is per thread: set_grad_enabled at import does not cover the caller thread
def detect(imageBuffer):
    image = np.asarray(imageBuffer)  # HxW uint8
    tensor = torch.from_numpy(image.astype(np.float32) / 255.)[None, None].to(device)
    # Full resolution (no resize): keypoints are in pixels of the given image
    feats = aliked({'image': tensor})
    kpts = feats['keypoints'][0].cpu().numpy()
    scores = feats['keypoint_scores'][0].cpu().numpy()
    desc = feats['descriptors'][0].cpu().numpy()
    # rtabmap expects format:
    #   pts:         array Nx3 (x, y, score) float32
    #   descriptors: array NxDIM (128) float32
    pts = np.ascontiguousarray(np.column_stack((kpts, scores)), dtype=np.float32)
    return pts, np.ascontiguousarray(desc, dtype=np.float32)


if __name__ == '__main__':
    init(True)
    pts, desc = detect((np.random.rand(480, 640) * 255).astype(np.uint8))
    print(pts.shape, desc.shape)
