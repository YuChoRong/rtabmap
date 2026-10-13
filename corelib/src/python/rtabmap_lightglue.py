#! /usr/bin/env python3
#
# LightGlue matcher (code and weights Apache-2.0, https://github.com/cvg/LightGlue):
#   pip install torch torchvision kornia
#   pip install git+https://github.com/cvg/LightGlue.git
#
# To use with rtabmap (built with WITH_PYTHON=ON), with ALIKED features (rtabmap_aliked.py):
#   --Vis/CorNNType 6 --PyMatcher/Path "~/rtabmap_lightglue.py" --PyMatcher/Model aliked
#   --PyMatcher/Threshold 0.1
# PyMatcher/Model is the feature type the LightGlue weights were trained for:
# aliked (default when "indoor"/"outdoor" is given) or disk. The superpoint
# weights are not offered: SuperPoint itself is under a non-commercial license.
# PyMatcher/Threshold is the LightGlue match confidence threshold (filter_threshold).
# PyMatcher/Iterations is not used.
#

import numpy as np
import torch

from lightglue import LightGlue

torch.set_grad_enabled(False)

device = 'cpu'
matcher = None

def init(descriptorDim, matchThreshold, iterations, cuda, model):
    global device, matcher
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    features = model if model in ('aliked', 'disk') else 'aliked'
    expected = {'aliked': 128, 'disk': 128}[features]
    assert descriptorDim == expected, \
        'LightGlue %s weights expect %d-dim descriptors, got %d (use rtabmap_aliked.py features)' % (features, expected, descriptorDim)
    matcher = LightGlue(features=features, filter_threshold=matchThreshold).eval().to(device)
    print('LightGlue (%s) initialized on %s' % (features, device))

def match(kptsFrom, kptsTo, scoresFrom, scoresTo, descriptorsFrom, descriptorsTo, imageWidth, imageHeight):
    size = torch.tensor([[imageWidth, imageHeight]], dtype=torch.float32, device=device)
    def view(kpts, desc):
        return {'keypoints': torch.from_numpy(np.asarray(kpts, dtype=np.float32))[None].to(device),
                'descriptors': torch.from_numpy(np.asarray(desc, dtype=np.float32))[None].to(device),
                'image_size': size}
    result = matcher({'image0': view(kptsFrom, descriptorsFrom), 'image1': view(kptsTo, descriptorsTo)})
    matches = result['matches'][0].cpu().numpy()  # Kx2: index in From, index in To
    # rtabmap expects format:
    #   matches: array Nx2 (type=9 or uint64)
    return np.ascontiguousarray(matches, dtype=np.uint64)


if __name__ == '__main__':
    init(128, 0.1, 20, True, 'aliked')
    k = np.random.rand(200, 2).astype(np.float32) * [640, 480]
    d = np.random.randn(200, 128).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    print(match(k, k, np.ones(200), np.ones(200), d, d, 640, 480).shape)
