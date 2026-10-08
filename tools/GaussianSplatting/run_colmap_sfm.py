#!/usr/bin/env python3
"""Runs a COLMAP Structure-from-Motion reconstruction on the images of a folder
made by rtabmap_to_colmap.py, as an independent reference for the SLAM poses.

  python run_colmap_sfm.py colmap_v101 [--output colmap_v101_sfm] [--matcher sequential|exhaustive] [--colmap PATH]
  python colmap_viewer.py colmap_v101_sfm --gt colmap_v101_sfm/slam_camera_poses.txt --ref-name SLAM --scale

- The intrinsics of the rectified images are known (cameras.txt), so they are
  fixed during the reconstruction (PINHOLE model, single camera).
- Images are copied with zero padded names (000123.jpg) so that the sequential
  matcher follows the capture order; the number is still the RTAB-Map node id,
  which the viewer uses to pair the SfM and SLAM poses.
- The SfM model has an arbitrary scale: compare with a similarity alignment
  (--scale in the viewer), errors are then given in meters of the SLAM poses.

License: COLMAP itself is BSD-3, but its GPU SIFT (SiftGPU) is restricted to
non-profit use and its LSD module is AGPL. This script always uses the CPU
SIFT (VLFeat, BSD) and no line detection; COLMAP is not redistributed. For a
COLMAP binary without those modules, build it with
-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DLSD_ENABLED=OFF -DCGAL_ENABLED=OFF.
"""
import argparse, os, shutil, subprocess, sys, time


def find_colmap():
    found = shutil.which('colmap') or shutil.which('COLMAP.bat') or shutil.which('colmap.exe')
    return found or 'colmap'


def run(cmd):
    print('>', ' '.join('"%s"' % c if ' ' in c else c for c in cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit('COLMAP failed: %s' % cmd[1])
    print('  done in %.0f s' % (time.time() - t), flush=True)
    return r.stdout


def option(colmap, command, *names):
    """First option name the installed COLMAP knows (names changed between versions)."""
    text = subprocess.run([colmap, command, '-h'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True).stdout
    for n in names:
        if '--' + n in text:
            return '--' + n
    return None


def read_camera(path):
    for line in open(path):
        v = line.split()
        if v and not v[0].startswith('#'):
            if v[1] == 'PINHOLE':
                return 'PINHOLE', ','.join(v[4:8])
            if v[1] == 'SIMPLE_PINHOLE':
                return 'SIMPLE_PINHOLE', ','.join(v[4:7])
            sys.exit('Camera model %s not supported (rectified PINHOLE expected)' % v[1])
    sys.exit('No camera in %s' % path)


def count_images(model_dir):
    path = os.path.join(model_dir, 'images.txt')
    lines = [l for l in open(path) if l.strip() and not l.startswith('#')]
    return len(lines) // 2 if lines else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help='Folder made by rtabmap_to_colmap.py (images/, sparse/0/, camera_poses.txt)')
    parser.add_argument('--output', help='Output folder (default DATA_sfm)')
    parser.add_argument('--matcher', choices=['sequential', 'exhaustive'], default='sequential',
                        help='sequential: neighbors in capture order (fast); exhaustive: all pairs (slow, finds loops)')
    parser.add_argument('--overlap', type=int, default=15, help='Sequential matcher: number of following images to match')
    parser.add_argument('--colmap', default=find_colmap(), help='COLMAP executable (Windows: COLMAP.bat of the release)')
    args = parser.parse_args()
    data = os.path.abspath(args.data)
    out = os.path.abspath(args.output or data.rstrip('/\\') + '_sfm')
    colmap = args.colmap
    if shutil.which(colmap) is None and not os.path.exists(colmap):
        sys.exit('COLMAP not found: install it (https://colmap.github.io/install.html) or give --colmap PATH')

    model, params = read_camera(os.path.join(data, 'sparse', '0', 'cameras.txt'))
    images = os.path.join(out, 'images')
    os.makedirs(images, exist_ok=True)
    names = sorted(f for f in os.listdir(os.path.join(data, 'images')) if os.path.splitext(f)[0].isdigit())
    if not names:
        sys.exit('No image named by node id in %s/images' % data)
    for f in names:
        stem, ext = os.path.splitext(f)
        shutil.copy(os.path.join(data, 'images', f), os.path.join(images, '%06d%s' % (int(stem), ext)))
    for f in ('camera_poses.txt', 'gt_camera_poses.txt'):
        if os.path.exists(os.path.join(data, f)):
            shutil.copy(os.path.join(data, f), os.path.join(out, 'slam_' + f if f == 'camera_poses.txt' else f))
    print('%d images, camera %s %s' % (len(names), model, params))

    database = os.path.join(out, 'database.db')
    if os.path.exists(database):
        os.remove(database)
    gpu_extract = option(colmap, 'feature_extractor', 'FeatureExtraction.use_gpu', 'SiftExtraction.use_gpu')
    run([colmap, 'feature_extractor', '--database_path', database, '--image_path', images,
         '--ImageReader.single_camera', '1', '--ImageReader.camera_model', model,
         '--ImageReader.camera_params', params] + ([gpu_extract, '0'] if gpu_extract else []))

    gpu_match = option(colmap, args.matcher + '_matcher', 'FeatureMatching.use_gpu', 'SiftMatching.use_gpu')
    cmd = [colmap, args.matcher + '_matcher', '--database_path', database] + ([gpu_match, '0'] if gpu_match else [])
    if args.matcher == 'sequential':
        cmd += ['--SequentialMatching.overlap', str(args.overlap)]
    run(cmd)

    sparse = os.path.join(out, 'sparse_bin')
    if os.path.exists(sparse):
        shutil.rmtree(sparse)
    os.makedirs(sparse)
    # Known intrinsics of rectified images: not refined
    run([colmap, 'mapper', '--database_path', database, '--image_path', images, '--output_path', sparse,
         '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_principal_point', '0',
         '--Mapper.ba_refine_extra_params', '0'])

    # Largest model as text in OUT/sparse/0 (the layout colmap_viewer.py and train_gsplat.py read)
    best, best_count = None, 0
    for m in sorted(os.listdir(sparse)):
        txt = os.path.join(out, 'tmp_txt_' + m)
        os.makedirs(txt, exist_ok=True)
        run([colmap, 'model_converter', '--input_path', os.path.join(sparse, m), '--output_path', txt, '--output_type', 'TXT'])
        n = count_images(txt)
        print('  model %s: %d images' % (m, n))
        if n > best_count:
            if best:
                shutil.rmtree(best)
            best, best_count = txt, n
        else:
            shutil.rmtree(txt)
    if not best:
        sys.exit('No SfM model was reconstructed')
    final = os.path.join(out, 'sparse', '0')
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    shutil.move(best, final)
    print('SfM model: %d of %d images registered -> %s' % (best_count, len(names), final))
    print('Compare: python colmap_viewer.py "%s" --gt "%s" --ref-name SLAM --scale'
          % (out, os.path.join(out, 'slam_camera_poses.txt')))


if __name__ == '__main__':
    main()
