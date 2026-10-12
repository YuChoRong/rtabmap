#!/usr/bin/env python3
"""Converts an Intel RealSense Viewer recording (.bag or .db3) of a D435i to the EuRoC
folder layout read by rtabmap-euroc_dataset (stereo IR + IMU, our VIO).

  pip install rosbags numpy opencv-python-headless      (rosbags: Apache-2.0)
  python realsense_bag_to_euroc.py recording.bag out_dir [--rgb]
  rtabmap-euroc_dataset --Odom/Strategy 15 --output OUT --output_name NAME out_dir

Recording (RealSense Viewer): Infrared 1 + Infrared 2 (Y8, same resolution and
fps), Emitter Enabled Off, Accel + Gyro on, optional RGB. Keep the camera still
for 2-3 s at the start (static IMU initialization of the VIO).

Written:
  out_dir/mav0/cam0/data/<ns>.png, cam0/sensor.yaml   (Infrared 1, left)
  out_dir/mav0/cam1/data/<ns>.png, cam1/sensor.yaml   (Infrared 2, right)
  out_dir/mav0/imu0/data.csv, imu0/sensor.yaml        (gyro rate, accel interpolated)
  out_dir/mav0/rgb/data/<ns>.png, rgb/sensor.yaml     (with --rgb)

Frames: librealsense gives IMU samples and extrinsics in camera axes (x right,
y down, z forward). rtabmap-euroc_dataset assumes the EuRoC IMU axes (x up,
y right, z forward), so IMU samples and camera poses are rotated into those
axes; the map then comes out with z up.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import cv2
import sqlite3
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# EuRoC-like IMU axes from RealSense camera axes: x_up = -y, y_right = x, z_forward = z
R_EUROC_FROM_RS = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def stamp_ns(header):
    return int(header.stamp.sec) * 1000000000 + int(header.stamp.nanosec)


def quat2rot_rs(q):
    """Rotation matrix of a librealsense extrinsic: its quat2rot() fills the
    column-major rs2_extrinsics.rotation with the standard matrix of q."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])


class BagSource:
    """Messages of a RealSense recording: ROS1 .bag, rosbag2 folder, or a bare
    rosbag2 .db3 file (newer RealSense Viewer versions, no metadata.yaml)."""

    def __init__(self, path):
        self.path = Path(path)
        self.db = None
        self.reader = None

    def __enter__(self):
        if self.path.suffix == '.db3':
            self.store = get_typestore(Stores.ROS2_HUMBLE)
            self.db = sqlite3.connect(str(self.path))
            rows = self.db.execute('SELECT id, name, type FROM topics').fetchall()
            self.ids = {name: (i, typ) for i, name, typ in rows}
            self.topics = set(self.ids)
        else:
            store = get_typestore(Stores.ROS1_NOETIC if self.path.suffix == '.bag' else Stores.ROS2_HUMBLE)
            self.reader = AnyReader([self.path], default_typestore=store).__enter__()
            self.conns = {c.topic: c for c in self.reader.connections}
            self.topics = set(self.conns)
        return self

    def __exit__(self, *exc):
        if self.db:
            self.db.close()
        if self.reader:
            self.reader.__exit__(*exc)

    def messages(self, topic):
        """Deserialized messages of a topic, in recording order."""
        if self.db:
            tid, typ = self.ids[topic]
            for (data,) in self.db.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,)):
                yield self.store.deserialize_cdr(bytes(data), typ)
        else:
            for conn, _, raw in self.reader.messages(connections=[self.conns[topic]]):
                yield self.reader.deserialize(raw, conn.msgtype)


def image_to_numpy(msg):
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    data = np.frombuffer(bytes(msg.data), dtype=np.uint16 if enc in ('mono16', '16uc1', 'y16') else np.uint8)
    if enc in ('mono8', '8uc1', 'y8'):
        return data.reshape(h, msg.step)[:, :w]
    if enc in ('rgb8', 'bgr8'):
        img = data.reshape(h, msg.step)[:, :w*3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == 'rgb8' else img
    if enc in ('mono16', '16uc1', 'y16'):
        return (data.reshape(h, msg.step // 2)[:, :w] >> 8).astype(np.uint8)
    raise ValueError('Unsupported image encoding %s' % msg.encoding)


def write_camera_yaml(path, K, D, size, rate, T):
    with open(path, 'w') as f:
        f.write('sensor_type: camera\ncomment: converted from a RealSense bag\n')
        f.write('T_BS:\n  cols: 4\n  rows: 4\n  data: [%s]\n' % ', '.join('%.9f' % v for v in T.ravel()))
        f.write('rate_hz: %d\nresolution: [%d, %d]\ncamera_model: pinhole\n' % (round(rate), size[0], size[1]))
        f.write('intrinsics: [%.6f, %.6f, %.6f, %.6f]\n' % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        d = list(D[:5]) + [0.0] * max(0, 4 - len(D))
        f.write('distortion_model: radial-tangential\ndistortion_coefficients: [%s]\n' % ', '.join('%.9f' % v for v in d))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag')
    parser.add_argument('output')
    parser.add_argument('--rgb', action='store_true', help='Also export the color stream (for COLMAP/3DGS)')
    parser.add_argument('--max-dt', type=float, default=0.002, help='Max left/right stamp difference (s)')
    args = parser.parse_args()

    out = Path(args.output) / 'mav0'
    with BagSource(args.bag) as reader:
        topics = reader.topics
        def find(suffix, contains):
            hits = [t for t in topics if t.endswith(suffix) and contains in t]
            return hits[0] if hits else None
        ir1, ir2 = find('/image/data', 'Infrared_1'), find('/image/data', 'Infrared_2')
        gyro, accel = find('/imu/data', 'Gyro'), find('/imu/data', 'Accel')
        rgb = find('/image/data', 'Color') if args.rgb else None
        if not (ir1 and ir2 and gyro and accel):
            sys.exit('The bag needs Infrared 1, Infrared 2, Gyro and Accel streams. Topics found:\n  ' +
                     '\n  '.join(sorted(t for t in topics if t.endswith('/data'))))

        def first(topic):
            if topic not in topics:
                return None
            for m in reader.messages(topic):
                return m

        def info(img_topic):
            m = first(img_topic.replace('/image/data', '/info/camera_info'))
            K = np.array(m.k if hasattr(m, 'k') else m.K, dtype=float).reshape(3, 3)
            D = np.array(m.d if hasattr(m, 'd') else m.D, dtype=float)
            return K, D, (m.width, m.height)

        def extrinsic(stream_topic):
            """Pose of the stream in the reference frame (4x4), None if not recorded."""
            prefix = stream_topic.rsplit('/', 2)[0]  # .../Infrared_1
            tf = [t for t in topics if t.startswith(prefix + '/tf/')]
            if not tf:
                return None
            m = first(tf[0])
            T = np.eye(4)
            T[:3, :3] = quat2rot_rs(m.rotation)
            T[:3, 3] = [m.translation.x, m.translation.y, m.translation.z]
            return T

        # Extrinsics in the common reference (depth / Infrared 1 origin)
        T_ir1, T_ir2, T_gyro = extrinsic(ir1), extrinsic(ir2), extrinsic(gyro)
        if T_ir2 is None:
            sys.exit('No extrinsics for Infrared 2 in the bag')
        T_ir1 = np.eye(4) if T_ir1 is None else T_ir1
        T_gyro = np.eye(4) if T_gyro is None else T_gyro
        T_ir2_in_ir1 = np.linalg.inv(T_ir1) @ T_ir2
        if T_ir2_in_ir1[0, 3] < 0:
            # Infrared 2 is the right imager (+x); the bag stores the inverse convention
            print('Extrinsics stored as reference->stream, inverting them')
            T_ir1, T_ir2, T_gyro = [np.linalg.inv(T) for T in (T_ir1, T_ir2, T_gyro)]
            T_ir2_in_ir1 = np.linalg.inv(T_ir1) @ T_ir2
        print('Baseline %.1f mm' % (T_ir2_in_ir1[0, 3] * 1000))
        R_imu = T_gyro[:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(R_imu) - 1) / 2, -1, 1)))
        if angle > 5:
            print('Warning: IMU to camera rotation is %.1f deg (expected ~0 for a D435i)' % angle)

        # Camera poses in the EuRoC-like IMU frame: T_imu_cam = R_euroc * inv(T_gyro) * T_cam
        A = np.eye(4); A[:3, :3] = R_EUROC_FROM_RS
        def T_BS(T_cam):
            return A @ np.linalg.inv(T_gyro) @ T_cam

        # Stereo pairs
        def stamps_and_msgs(topic):
            for m in reader.messages(topic):
                yield stamp_ns(m.header), m
        right = {s: m for s, m in stamps_and_msgs(ir2)}
        right_stamps = np.array(sorted(right))
        for d in ('cam0/data', 'cam1/data', 'imu0'):
            (out / d).mkdir(parents=True, exist_ok=True)
        n, first_stamp, last_stamp = 0, None, None
        with open(out / 'cam0/data.csv', 'w') as f0, open(out / 'cam1/data.csv', 'w') as f1:
            f0.write('#timestamp [ns],filename\n'); f1.write('#timestamp [ns],filename\n')
            for s, m in stamps_and_msgs(ir1):
                i = np.searchsorted(right_stamps, s)
                cand = [right_stamps[j] for j in (i - 1, i) if 0 <= j < len(right_stamps)]
                if not cand:
                    continue
                r = min(cand, key=lambda x: abs(int(x) - s))
                if abs(int(r) - s) > args.max_dt * 1e9:
                    continue
                name = '%d.png' % s
                cv2.imwrite(str(out / 'cam0/data' / name), image_to_numpy(m))
                cv2.imwrite(str(out / 'cam1/data' / name), image_to_numpy(right[r]))
                f0.write('%d,%s\n' % (s, name)); f1.write('%d,%s\n' % (s, name))
                first_stamp = s if first_stamp is None else first_stamp
                last_stamp = s
                n += 1
        if n < 2:
            sys.exit('No synchronized Infrared 1 / Infrared 2 pairs found')
        rate = (n - 1) / ((last_stamp - first_stamp) * 1e-9)
        K1, D1, size1 = info(ir1)
        K2, D2, size2 = info(ir2)
        write_camera_yaml(out / 'cam0/sensor.yaml', K1, D1, size1, rate, T_BS(T_ir1))
        write_camera_yaml(out / 'cam1/sensor.yaml', K2, D2, size2, rate, T_BS(T_ir2))
        print('%d stereo pairs, %.1f Hz, %dx%d, fx=%.1f' % (n, rate, size1[0], size1[1], K1[0, 0]))

        # IMU: gyro timestamps, accel linearly interpolated, rotated to EuRoC-like axes
        g = np.array([(s, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
                      for s, m in stamps_and_msgs(gyro)], dtype=float)
        a = np.array([(s, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z)
                      for s, m in stamps_and_msgs(accel)], dtype=float)
        g = g[np.argsort(g[:, 0])]; a = a[np.argsort(a[:, 0])]
        g = g[(g[:, 0] >= a[0, 0]) & (g[:, 0] <= a[-1, 0])]
        acc = np.column_stack([np.interp(g[:, 0], a[:, 0], a[:, k]) for k in (1, 2, 3)])
        gyr = g[:, 1:4] @ R_EUROC_FROM_RS.T
        acc = acc @ R_EUROC_FROM_RS.T
        with open(out / 'imu0/data.csv', 'w') as f:
            f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                    'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
            for s, w, ac in zip(g[:, 0].astype(np.int64), gyr, acc):
                f.write('%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n' % (s, *w, *ac))
        imu_rate = (len(g) - 1) / ((g[-1, 0] - g[0, 0]) * 1e-9)
        with open(out / 'imu0/sensor.yaml', 'w') as f:
            f.write('sensor_type: imu\ncomment: RealSense IMU in EuRoC-like axes\n'
                    'T_BS:\n  cols: 4\n  rows: 4\n  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n'
                    'rate_hz: %d\n' % round(imu_rate))
        print('%d IMU samples, %.0f Hz; mean accel at start (should be ~[9.8,0,0] when level): %s'
              % (len(g), imu_rate, np.round(acc[:50].mean(0), 2)))
        if g[0, 0] > first_stamp or g[-1, 0] < last_stamp:
            print('Note: IMU covers %.2f..%.2f s of the images %.2f..%.2f s' % (
                g[0, 0] * 1e-9, g[-1, 0] * 1e-9, first_stamp * 1e-9, last_stamp * 1e-9))

        if rgb:
            (out / 'rgb/data').mkdir(parents=True, exist_ok=True)
            k = 0
            with open(out / 'rgb/data.csv', 'w') as f:
                f.write('#timestamp [ns],filename\n')
                for s, m in stamps_and_msgs(rgb):
                    cv2.imwrite(str(out / 'rgb/data' / ('%d.png' % s)), image_to_numpy(m))
                    f.write('%d,%d.png\n' % (s, s)); k += 1
            Kc, Dc, sizec = info(rgb)
            T_rgb = extrinsic(rgb)
            write_camera_yaml(out / 'rgb/sensor.yaml', Kc, Dc, sizec, 30, T_BS(np.eye(4) if T_rgb is None else T_rgb))
            print('%d color images' % k)
    print('Done: %s' % out.parent)


if __name__ == '__main__':
    main()
