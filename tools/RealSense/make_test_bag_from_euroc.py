# Test only: writes a RealSense-Viewer-like bag from EuRoC (rectified stereo + IMU in camera-like axes)
import sys, numpy as np, cv2, yaml
from pathlib import Path
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as Rot

seq, bag, seconds = Path(sys.argv[1]) / 'mav0', Path(sys.argv[2]), float(sys.argv[3])
ros2 = bag.suffix != '.bag'  # otherwise a rosbag2 folder (sqlite3 .db3 inside), like newer RealSense Viewers
ts = get_typestore(Stores.ROS2_HUMBLE if ros2 else Stores.ROS1_NOETIC)
Image, Imu, CameraInfo = ts.types['sensor_msgs/msg/Image'], ts.types['sensor_msgs/msg/Imu'], ts.types['sensor_msgs/msg/CameraInfo']
Header, Time = ts.types['std_msgs/msg/Header'], ts.types['builtin_interfaces/msg/Time']
Transform, V3, Q = ts.types['geometry_msgs/msg/Transform'], ts.types['geometry_msgs/msg/Vector3'], ts.types['geometry_msgs/msg/Quaternion']
RoiT = ts.types['sensor_msgs/msg/RegionOfInterest']

def cal(c):
    y = yaml.safe_load(open(seq / c / 'sensor.yaml'))
    fu, fv, cu, cv = y['intrinsics']
    return np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]]), np.array(y['distortion_coefficients']), np.array(y['T_BS']['data']).reshape(4, 4)
K0, D0, T0 = cal('cam0'); K1, D1, T1 = cal('cam1')
T10 = np.linalg.inv(T1) @ T0  # cam0 -> cam1
size = (752, 480)
R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K0, D0, K1, D1, size, T10[:3, :3], T10[:3, 3], alpha=0)
m0 = cv2.initUndistortRectifyMap(K0, D0, R1, P1, size, cv2.CV_32FC1)
m1 = cv2.initUndistortRectifyMap(K1, D1, R2, P2, size, cv2.CV_32FC1)
B = -P2[0, 3] / P2[0, 0]
K = P1[:3, :3]
R_EU_RS = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
# Poses in the reference frame (rectified cam0)
T_imu_c0r = T0 @ np.block([[R1.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
A = np.eye(4); A[:3, :3] = R_EU_RS
T_ref_rsimu = np.linalg.inv(T_imu_c0r) @ A
T_ref_ir2 = np.eye(4); T_ref_ir2[0, 3] = B
print('baseline', B, 'K', K[0, 0], K[0, 2], K[1, 2])

def tf(T):
    q = Rot.from_matrix(T[:3, :3]).as_quat()
    return Transform(translation=V3(x=T[0, 3], y=T[1, 3], z=T[2, 3]), rotation=Q(x=q[0], y=q[1], z=q[2], w=q[3]))
def hdr(ns, i=0):
    t = Time(sec=int(ns // 1000000000), nanosec=int(ns % 1000000000))
    return Header(stamp=t, frame_id='') if ros2 else Header(seq=i, stamp=t, frame_id='')
def caminfo(ns):
    m = dict(D=np.zeros(5), K=K.ravel().astype(float), R=np.eye(3).ravel(), P=np.hstack([K, np.zeros((3, 1))]).ravel())
    if ros2:
        m = {k.lower(): v for k, v in m.items()}
    return CameraInfo(header=hdr(ns), height=480, width=752, distortion_model='Brown Conrady', binning_x=0, binning_y=0,
                      roi=RoiT(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False), **m)

stamps = [int(l.split(',')[0]) for l in open(seq / 'cam0/data.csv') if not l.startswith('#')]
t0 = stamps[0]; stamps = [s for s in stamps if s - t0 <= seconds * 1e9]
P = '/device_0/sensor_0/'; M = '/device_0/sensor_2/'
if ros2:
    import shutil; shutil.rmtree(bag, ignore_errors=True)
else:
    bag.unlink(missing_ok=True)
with (Writer2(bag, version=8) if ros2 else Writer1(bag)) as w:
    c = {}
    def conn(topic, typ):
        if topic not in c:
            c[topic] = w.add_connection(topic, typ, typestore=ts)
        return c[topic]
    def put(topic, typ, ns, msg):
        w.write(conn(topic, typ), ns, ts.serialize_cdr(msg, typ) if ros2 else ts.serialize_ros1(msg, typ))
    put(P + 'Infrared_1/info/camera_info', 'sensor_msgs/msg/CameraInfo', t0, caminfo(t0))
    put(P + 'Infrared_2/info/camera_info', 'sensor_msgs/msg/CameraInfo', t0, caminfo(t0))
    put(P + 'Infrared_1/tf/0', 'geometry_msgs/msg/Transform', t0, tf(np.eye(4)))
    put(P + 'Infrared_2/tf/0', 'geometry_msgs/msg/Transform', t0, tf(T_ref_ir2))
    put(M + 'Gyro_0/tf/0', 'geometry_msgs/msg/Transform', t0, tf(T_ref_rsimu))
    put(M + 'Accel_0/tf/0', 'geometry_msgs/msg/Transform', t0, tf(T_ref_rsimu))
    for l in open(seq / 'imu0/data.csv'):
        if l.startswith('#'): continue
        v = l.split(','); s = int(v[0])
        if s < t0 or s > stamps[-1]: continue
        g = R_EU_RS.T @ np.array(v[1:4], float); a = R_EU_RS.T @ np.array(v[4:7], float)
        put(M + 'Gyro_0/imu/data', 'sensor_msgs/msg/Imu', s, Imu(header=hdr(s), orientation=Q(x=0, y=0, z=0, w=1), orientation_covariance=np.zeros(9),
            angular_velocity=V3(x=g[0], y=g[1], z=g[2]), angular_velocity_covariance=np.zeros(9), linear_acceleration=V3(x=0, y=0, z=0), linear_acceleration_covariance=np.zeros(9)))
        put(M + 'Accel_0/imu/data', 'sensor_msgs/msg/Imu', s + 1000, Imu(header=hdr(s + 1000), orientation=Q(x=0, y=0, z=0, w=1), orientation_covariance=np.zeros(9),
            angular_velocity=V3(x=0, y=0, z=0), angular_velocity_covariance=np.zeros(9), linear_acceleration=V3(x=a[0], y=a[1], z=a[2]), linear_acceleration_covariance=np.zeros(9)))
    for i, s in enumerate(stamps):
        for cam, mp, name in (('cam0', m0, 'Infrared_1'), ('cam1', m1, 'Infrared_2')):
            im = cv2.remap(cv2.imread(str(seq / cam / 'data' / ('%d.png' % s)), 0), mp[0], mp[1], cv2.INTER_LINEAR)
            put(P + name + '/image/data', 'sensor_msgs/msg/Image', s, Image(header=hdr(s, i), height=480, width=752, encoding='mono8', is_bigendian=0, step=752, data=im.ravel()))
print('wrote', bag, len(stamps), 'frames')
