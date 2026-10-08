#!/usr/bin/env python3
"""COLMAP model viewer (Python + Qt): checks a COLMAP text model (e.g., made by
rtabmap_to_colmap.py) and compares its camera poses with the ground truth.
Same features as colmap_viewer/ (C++), without any compilation.

  pip install pyside6 numpy pillow
  python colmap_viewer.py [DIR] [--gt FILE] [--scale] [--frame N] [--plane xy|xz|yz] [--screenshot OUT.png]

DIR: COLMAP folder with sparse/0/{cameras,images,points3D}.txt and images/
     (camera_poses.txt and gt_camera_poses.txt are read when present).
--gt: ground truth camera poses, TUM format "stamp x y z qx qy qz qw [id]".
--screenshot: renders the window to a PNG, prints the statistics and exits.

Dependencies: PySide6 (LGPL-3.0, PySide2 also works), numpy, Pillow (not needed by the viewer itself).
"""
import argparse, math, os, re, sys

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # Qt5 bindings, same API for what is used here
    from PySide2 import QtCore, QtGui, QtWidgets
Qt = QtCore.Qt

EST_COLOR = QtGui.QColor(30, 110, 230)
GT_COLOR = QtGui.QColor(240, 140, 20)
ERR_COLOR = QtGui.QColor(220, 40, 40, 150)


# ---------------------------------------------------------------- model

def quat_to_rot(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rot_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def data_lines(path):
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                yield line.split()


def read_tum(path):
    """List of (stamp, R, t, id or -1), camera -> world."""
    poses = []
    for v in data_lines(path):
        if len(v) < 8:
            continue
        x, y, z, qx, qy, qz, qw = map(float, v[1:8])
        poses.append((float(v[0]), quat_to_rot(qw, qx, qy, qz), np.array([x, y, z]), int(v[8]) if len(v) >= 9 else -1))
    return poses


def umeyama(src, dst, with_scale):
    """dst ~ s * R @ src + t for 3xN arrays."""
    mu_s, mu_d = src.mean(1, keepdims=True), dst.mean(1, keepdims=True)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd @ xs.T / src.shape[1])
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / xs.var(1).sum() if with_scale else 1.0
    return s, R, (mu_d - s * R @ mu_s).ravel()


class Model:
    def __init__(self):
        self.dir = ''
        self.frames = []  # dicts: name, key, stamp, R, t (camera -> world), gt_R, gt_t, error, rot_error
        self.points = np.zeros((0, 3), np.float32)
        self.intrinsics = None
        self.raw_gt = {}  # frame index -> (R, t)
        self.gt_path = ''
        self.stats = {}

    def load(self, directory):
        self.__init__()
        self.dir = directory
        sparse = next((d for d in (os.path.join(directory, 'sparse', '0'), os.path.join(directory, 'sparse'), directory)
                       if os.path.exists(os.path.join(d, 'images.txt'))), None)
        if sparse is None:
            raise RuntimeError('No images.txt in %s (text model expected: sparse/0/{cameras,images,points3D}.txt)' % directory)
        for v in data_lines(os.path.join(sparse, 'cameras.txt')):
            model, w, h, p = v[1], int(v[2]), int(v[3]), [float(x) for x in v[4:]]
            if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[:4]
            self.intrinsics = dict(model=model, width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
            break
        lines = open(os.path.join(sparse, 'images.txt')).read().splitlines()
        i = 0
        while i < len(lines):
            v = lines[i].split()
            i += 1
            if not v or v[0].startswith('#') or len(v) < 10:
                continue
            Rcw = quat_to_rot(*map(float, v[1:5]))
            tcw = np.array([float(x) for x in v[5:8]])
            name = ' '.join(v[9:])
            stem = os.path.splitext(name)[0]
            self.frames.append(dict(name=name, key=int(stem) if stem.isdigit() else -1, stamp=-1.0,
                                    R=Rcw.T, t=-Rcw.T @ tcw, gt_R=None, gt_t=None, error=0.0, rot_error=0.0))
            i += 1  # 2D points line
        if not self.frames:
            raise RuntimeError('No image in images.txt')
        self.frames.sort(key=lambda f: (f['key'], f['name']) if f['key'] >= 0 else (1 << 30, f['name']))
        poses_file = os.path.join(directory, 'camera_poses.txt')
        if os.path.exists(poses_file):
            stamps = {p[3]: p[0] for p in read_tum(poses_file)}
            for f in self.frames:
                f['stamp'] = stamps.get(f['key'], -1.0)
        pts = [[float(x) for x in v[1:4]] for v in data_lines(os.path.join(sparse, 'points3D.txt')) if len(v) >= 7]
        self.points = np.array(pts, np.float32).reshape(-1, 3)
        centers = np.array([f['t'] for f in self.frames])
        self.path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        gt_file = os.path.join(directory, 'gt_camera_poses.txt')
        if os.path.exists(gt_file):
            self.load_gt(gt_file)

    def load_gt(self, path):
        gt = read_tum(path)
        raw = {}
        if gt and gt[0][3] >= 0:  # by id (image name)
            by_id = {p[3]: p for p in gt}
            for k, f in enumerate(self.frames):
                if f['key'] in by_id:
                    raw[k] = (by_id[f['key']][1], by_id[f['key']][2])
        else:  # by nearest stamp
            stamps = np.array([p[0] for p in gt])
            for k, f in enumerate(self.frames):
                if f['stamp'] >= 0 and len(stamps):
                    j = int(np.argmin(np.abs(stamps - f['stamp'])))
                    if abs(stamps[j] - f['stamp']) < 0.02:
                        raw[k] = (gt[j][1], gt[j][2])
        if len(raw) < 3:
            raise RuntimeError('Only %d ground truth poses match the images' % len(raw))
        self.raw_gt, self.gt_path = raw, path
        self.align(False)

    def align(self, with_scale):
        for f in self.frames:
            f['gt_R'] = f['gt_t'] = None
        self.stats = {}
        if len(self.raw_gt) < 3:
            return
        idx = sorted(self.raw_gt)
        src = np.array([self.raw_gt[k][1] for k in idx]).T
        dst = np.array([self.frames[k]['t'] for k in idx]).T
        s_sim, _, _ = umeyama(src, dst, True)
        s, R, t = umeyama(src, dst, with_scale)
        errors, rot_errors, D_sum = [], [], np.zeros((3, 3))
        for k in idx:
            f = self.frames[k]
            f['gt_R'] = R @ self.raw_gt[k][0]
            f['gt_t'] = s * R @ self.raw_gt[k][1] + t
            f['error'] = float(np.linalg.norm(f['gt_t'] - f['t']))
            f['rot_error'] = rot_angle_deg(f['gt_R'].T @ f['R'])
            errors.append(f['error']); rot_errors.append(f['rot_error'])
            D_sum += f['gt_R'].T @ f['R']
        # Constant rotation between the camera frames (chordal mean), and the rotation error without it
        U, _, Vt = np.linalg.svd(D_sum)
        S = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            S[2, 2] = -1
        D = U @ S @ Vt
        rot_no_offset = [rot_angle_deg((self.frames[k]['gt_R'] @ D).T @ self.frames[k]['R']) for k in idx]
        e = np.array(errors)
        self.stats = dict(matched=len(idx), rmse=float(np.sqrt((e**2).mean())), mean=float(e.mean()),
                          median=float(np.median(e)), max=float(e.max()), scale=1.0 / s_sim,
                          rot_rmse=float(np.sqrt(np.mean(np.square(rot_errors)))), rot_offset=rot_angle_deg(D),
                          rot_rmse_no_offset=float(np.sqrt(np.mean(np.square(rot_no_offset)))))

    def image_path(self, frame):
        for d in ('images', 'input', ''):
            p = os.path.join(self.dir, d, frame['name'])
            if os.path.exists(p):
                return p
        return ''


# ---------------------------------------------------------------- views

def colormap(t):
    t = min(1.0, max(0.0, t))
    c = [min(1.0, max(0.0, 1.5 - abs(4.0*t - k))) for k in (3.0, 2.0, 1.0)]
    return QtGui.QColor.fromRgbF(*c)


class TrajectoryView(QtWidgets.QWidget):
    frameClicked = QtCore.Signal(int)
    LABELS = ['top view (X right, Y up)', 'side view (X right, Z up)', 'side view (Y right, Z up)']
    AXES = [(0, 1), (0, 2), (1, 2)]

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.model, self.plane, self.selected = None, 0, -1
        self.show_points = self.show_errors = True
        self.zoom, self.center, self.last = 1.0, np.zeros(2), None

    def set_model(self, model):
        self.model, self.selected = model, -1
        self.fit()
        self.update()

    def ax(self, p):
        a, b = self.AXES[self.plane]
        return np.asarray(p)[..., [a, b]]

    def screen(self, p):
        a = self.ax(p)
        return np.stack([self.width()/2 + (a[..., 0]-self.center[0])*self.zoom,
                         self.height()/2 - (a[..., 1]-self.center[1])*self.zoom], -1)

    def fit(self):
        if not self.model or not self.model.frames:
            return
        pts = [f['t'] for f in self.model.frames] + [f['gt_t'] for f in self.model.frames if f['gt_t'] is not None]
        a = self.ax(np.array(pts))
        lo, hi = a.min(0), a.max(0)
        span = max(float((hi - lo).max()), 0.5) * 1.4
        self.zoom = min(self.width(), self.height()) / span
        self.center = (lo + hi) / 2

    def resizeEvent(self, e):
        self.fit()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        if not self.model or not self.model.frames:
            p.drawText(self.rect(), Qt.AlignCenter, 'Open a COLMAP folder')
            return
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        text = self.palette().text().color()
        if self.show_points and len(self.model.points):
            s = self.screen(self.model.points).astype(int)
            ok = (s[:, 0] >= 0) & (s[:, 1] >= 0) & (s[:, 0] < w) & (s[:, 1] < h)
            counts = np.bincount(s[ok, 1] * w + s[ok, 0], minlength=w*h).reshape(h, w)
            norm = math.log(1 + min(max(counts.max(), 1), 20))
            alpha = np.where(counts > 0, 40 + 150 * np.minimum(1.0, np.log1p(counts) / norm), 0).astype(np.uint8)
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = text.red(), text.green(), text.blue(), alpha
            img = QtGui.QImage(rgba.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888)
            p.drawImage(0, 0, img)
        meters = 10 ** math.floor(math.log10(150.0 / self.zoom))
        p.setPen(QtGui.QPen(text, 2))
        p.drawLine(QtCore.QPointF(15, h-15), QtCore.QPointF(15 + meters*self.zoom, h-15))
        p.drawText(QtCore.QPointF(15, h-20), ('%g m' % meters) if meters >= 1 else ('%g cm' % (meters*100)))
        p.drawText(QtCore.QPointF(10, 18), self.LABELS[self.plane])
        frames = self.model.frames
        est = self.screen(np.array([f['t'] for f in frames]))
        gt_frames = [f for f in frames if f['gt_t'] is not None]
        gt = self.screen(np.array([f['gt_t'] for f in gt_frames])) if gt_frames else np.zeros((0, 2))
        if self.show_errors and gt_frames:
            p.setPen(QtGui.QPen(ERR_COLOR, 1))
            for f, g in zip(gt_frames, gt):
                a = self.screen(f['t'])
                p.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*g))
        for pts, color in ((gt, GT_COLOR), (est, EST_COLOR)):
            p.setPen(QtGui.QPen(color, 2))
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*q) for q in pts]))
        if 0 <= self.selected < len(frames):
            f = frames[self.selected]
            c, d = self.screen(f['t']), self.screen(f['t'] + f['R'][:, 2] * (40.0 / self.zoom))
            p.setPen(QtGui.QPen(self.palette().highlight().color(), 2))
            p.drawLine(QtCore.QPointF(*c), QtCore.QPointF(*d))
            p.setBrush(self.palette().highlight())
            p.drawEllipse(QtCore.QPointF(*c), 5, 5)
            if f['gt_t'] is not None:
                p.setBrush(GT_COLOR)
                p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(*self.screen(f['gt_t'])), 4, 4)
        labels = ['estimate (COLMAP)', 'ground truth (aligned)' if self.model.raw_gt else 'no ground truth']
        x0 = w - max(p.fontMetrics().horizontalAdvance(l) for l in labels) - 45
        for k, (label, color) in enumerate(zip(labels, (EST_COLOR, GT_COLOR))):
            p.setPen(QtGui.QPen(color, 3))
            p.drawLine(x0, 15 + 17*k, x0 + 25, 15 + 17*k)
            p.setPen(text)
            p.drawText(x0 + 30, 20 + 17*k, label)

    def wheelEvent(self, e):
        pos = e.position() if hasattr(e, 'position') else QtCore.QPointF(e.pos())
        before = np.array([(pos.x()-self.width()/2)/self.zoom, -(pos.y()-self.height()/2)/self.zoom]) + self.center
        self.zoom *= 1.0015 ** e.angleDelta().y()
        after = np.array([(pos.x()-self.width()/2)/self.zoom, -(pos.y()-self.height()/2)/self.zoom]) + self.center
        self.center += before - after
        self.update()

    def mousePressEvent(self, e):
        self.last = e.pos()
        if e.button() == Qt.LeftButton and self.model and self.model.frames:
            d = np.linalg.norm(self.screen(np.array([f['t'] for f in self.model.frames])) - [e.pos().x(), e.pos().y()], axis=1)
            k = int(np.argmin(d))
            if d[k] < 15:
                self.frameClicked.emit(k)

    def mouseMoveEvent(self, e):
        if self.last is not None:
            delta = e.pos() - self.last
            self.center -= np.array([delta.x(), -delta.y()]) / self.zoom
            self.last = e.pos()
            self.update()


class ErrorPlot(QtWidgets.QWidget):
    frameClicked = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.model, self.selected = None, -1

    def rect_(self):
        return QtCore.QRectF(55, 10, self.width()-70, self.height()-40)

    def xs(self):
        frames = self.model.frames
        stamps = np.array([f['stamp'] for f in frames])
        v = stamps - stamps[0] if stamps[0] >= 0 and stamps[-1] > stamps[0] else np.arange(len(frames), dtype=float)
        r = self.rect_()
        return r.left() + r.width() * v / max(v[-1], 1e-9)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        m = self.model
        if not m or not m.frames or not m.stats:
            p.drawText(self.rect(), Qt.AlignCenter, 'Position error over time (needs ground truth)')
            return
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r, s, xs = self.rect_(), m.stats, self.xs()
        max_e = max(s['max'] * 1.1, 1e-3)
        y = lambda err: r.bottom() - r.height() * err / max_e
        p.setPen(self.palette().mid().color())
        p.drawRect(r)
        p.setPen(self.palette().text().color())
        for k in range(5):
            err = max_e * k / 4
            p.drawText(QtCore.QRectF(0, y(err)-8, 50, 16), Qt.AlignRight | Qt.AlignVCenter, '%.1f' % (err*100))
        p.drawText(QtCore.QRectF(r.left(), r.bottom()+4, r.width(), 20), Qt.AlignCenter,
                   'time (s)        error (cm)' if m.frames[0]['stamp'] >= 0 else 'frame        error (cm)')
        p.setPen(QtGui.QPen(ERR_COLOR, 1, Qt.DashLine))
        p.drawLine(QtCore.QPointF(r.left(), y(s['rmse'])), QtCore.QPointF(r.right(), y(s['rmse'])))
        p.drawText(QtCore.QPointF(r.right()-120, y(s['rmse'])-4), 'RMSE %.2f cm' % (s['rmse']*100))
        p.setPen(QtGui.QPen(EST_COLOR, 1.5))
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(xs[k], y(f['error'])) for k, f in enumerate(m.frames) if f['gt_t'] is not None]))
        if 0 <= self.selected < len(m.frames):
            p.setPen(QtGui.QPen(self.palette().highlight().color(), 1))
            p.drawLine(QtCore.QPointF(xs[self.selected], r.top()), QtCore.QPointF(xs[self.selected], r.bottom()))

    def mousePressEvent(self, e):
        if self.model and self.model.frames:
            self.frameClicked.emit(int(np.argmin(np.abs(self.xs() - e.pos().x()))))


class ImageView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 200)
        self.model, self.index, self.image = None, -1, QtGui.QImage()
        self.show_points, self.use_gt = True, False
        self.proj = np.zeros((0, 3))

    def set_frame(self, index):
        self.index, self.image, self.proj = index, QtGui.QImage(), np.zeros((0, 3))
        m = self.model
        if m and 0 <= index < len(m.frames):
            f = m.frames[index]
            self.image.load(m.image_path(f))
            K = m.intrinsics
            R, t = (f['gt_R'], f['gt_t']) if self.use_gt and f['gt_t'] is not None else (f['R'], f['t'])
            pc = (m.points.astype(np.float64) - t) @ R  # world -> camera
            z = pc[:, 2]
            ok = (z > 0.1) & (z < 8.0)
            pc = pc[ok]
            u = K['fx'] * pc[:, 0] / pc[:, 2] + K['cx']
            v = K['fy'] * pc[:, 1] / pc[:, 2] + K['cy']
            w, h = K['width'] or self.image.width(), K['height'] or self.image.height()
            inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
            self.proj = np.stack([u[inside], v[inside], pc[inside, 2]], 1)
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        m = self.model
        if not m or self.index < 0:
            p.drawText(self.rect(), Qt.AlignCenter, 'Select a frame')
            return
        K = m.intrinsics
        iw, ih = (self.image.width(), self.image.height()) if not self.image.isNull() else (K['width'], K['height'])
        s = min(self.width() / iw, (self.height()-40) / ih)
        target = QtCore.QRectF(0, 0, iw*s, ih*s)
        target.moveCenter(QtCore.QPointF(self.width()/2, (self.height()-40)/2))
        if not self.image.isNull():
            p.drawImage(target, self.image)
        else:
            p.drawText(target, Qt.AlignCenter, 'image not found')
        if self.show_points and len(self.proj):
            p.setPen(Qt.NoPen)
            order = np.argsort(-self.proj[:, 2])[::max(1, len(self.proj) // 6000)]  # far first
            for u, v, z in self.proj[order]:
                p.setBrush(colormap(1.0 - (z - 0.5) / 5.0))
                p.drawRect(QtCore.QRectF(target.left() + u*s - 1, target.top() + v*s - 1, 2, 2))
        f = m.frames[self.index]
        info = '%s: %d points reprojected with the %s pose (red near, blue far)' % (
            f['name'], len(self.proj), 'ground truth' if self.use_gt and f['gt_t'] is not None else 'estimated')
        if f['gt_t'] is not None:
            info += '   error %.1f cm, %.2f deg' % (f['error']*100, f['rot_error'])
        p.setPen(self.palette().text().color())
        p.drawText(QtCore.QRectF(4, self.height()-40, self.width()-8, 40), Qt.AlignCenter | Qt.TextWordWrap, info)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('COLMAP viewer')
        self.model = Model()
        self.trajectory, self.image, self.plot = TrajectoryView(), ImageView(), ErrorPlot()
        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plane = QtWidgets.QComboBox()
        self.plane.addItems(['Top (XY)', 'Side (XZ)', 'Side (YZ)'])
        points, errors = QtWidgets.QCheckBox('Points'), QtWidgets.QCheckBox('Error lines')
        points.setChecked(True); errors.setChecked(True)
        self.scale = QtWidgets.QCheckBox('Align with scale')
        reproject, gt_pose = QtWidgets.QCheckBox('Reprojected points'), QtWidgets.QCheckBox('Reproject with ground truth pose')
        reproject.setChecked(True)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)

        controls = QtWidgets.QHBoxLayout()
        for wdg in (QtWidgets.QLabel('View'), self.plane, points, errors, self.scale):
            controls.addWidget(wdg)
        controls.addSpacing(20)
        controls.addWidget(reproject)
        controls.addWidget(gt_pose)
        controls.addStretch()
        right = QtWidgets.QSplitter(Qt.Vertical)
        right.addWidget(self.image)
        right.addWidget(self.plot)
        right.addWidget(self.stats)
        right.setStretchFactor(0, 3); right.setStretchFactor(1, 2); right.setStretchFactor(2, 0)
        split = QtWidgets.QSplitter(Qt.Horizontal)
        split.addWidget(self.trajectory)
        split.addWidget(right)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(split, 1)
        layout.addWidget(self.slider)
        self.setCentralWidget(central)
        self.resize(1500, 900)

        menu = self.menuBar().addMenu('&File')
        for text, slot, key in (('Open COLMAP folder...', self.open_dialog, QtGui.QKeySequence.Open),
                                ('Load ground truth...', self.gt_dialog, None),
                                ('Save screenshot...', self.screenshot_dialog, None),
                                ('Quit', self.close, QtGui.QKeySequence.Quit)):
            action = menu.addAction(text)
            if key is not None:
                action.setShortcut(key)
            action.triggered.connect(slot)

        self.plane.currentIndexChanged.connect(self.set_plane)
        points.toggled.connect(lambda on: (setattr(self.trajectory, 'show_points', on), self.trajectory.update()))
        errors.toggled.connect(lambda on: (setattr(self.trajectory, 'show_errors', on), self.trajectory.update()))
        reproject.toggled.connect(lambda on: (setattr(self.image, 'show_points', on), self.image.update()))
        gt_pose.toggled.connect(lambda on: (setattr(self.image, 'use_gt', on), self.image.set_frame(self.image.index)))
        self.scale.toggled.connect(lambda on: (self.model.align(on), self.refresh()))
        self.slider.valueChanged.connect(self.select)
        self.trajectory.frameClicked.connect(self.slider.setValue)
        self.plot.frameClicked.connect(self.slider.setValue)

    def set_plane(self, i):
        self.trajectory.plane = i
        self.trajectory.fit()
        self.trajectory.update()

    def error(self, message):
        print(message, file=sys.stderr)
        if self.isVisible():
            QtWidgets.QMessageBox.warning(self, 'COLMAP viewer', message)

    def open_folder(self, directory):
        try:
            self.model.load(directory)
        except Exception as e:
            self.error(str(e))
            return False
        self.model.align(self.scale.isChecked())
        for view in (self.trajectory, self.image, self.plot):
            view.model = self.model
        self.trajectory.set_model(self.model)
        self.slider.setEnabled(True)
        self.slider.setRange(0, len(self.model.frames) - 1)
        self.refresh()
        self.select(0)
        return True

    def open_gt(self, path):
        try:
            self.model.load_gt(path)
        except Exception as e:
            self.error(str(e))
            return False
        self.model.align(self.scale.isChecked())
        self.refresh()
        return True

    def open_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'COLMAP folder (with sparse/0 and images)')
        if d:
            self.open_folder(d)

    def gt_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Ground truth camera poses (stamp x y z qx qy qz qw [id])',
                                                        self.model.dir, 'Poses (*.txt *.csv);;All (*)')
        if path:
            self.open_gt(path)

    def screenshot_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Screenshot', 'colmap_viewer.png', 'PNG (*.png)')
        if path:
            self.grab().save(path)

    def select(self, index):
        self.trajectory.selected = self.plot.selected = index
        self.trajectory.update()
        self.plot.update()
        self.image.set_frame(index)

    def refresh(self):
        self.trajectory.fit()
        self.trajectory.update()
        self.plot.update()
        self.image.set_frame(self.slider.value())
        m, K, s = self.model, self.model.intrinsics, self.model.stats
        text = '<b>%s</b><br>%d images, %d points, camera %s %dx%d f=%.1f c=(%.1f, %.1f), path %.2f m' % (
            m.dir, len(m.frames), len(m.points), K['model'], K['width'], K['height'], K['fx'], K['cx'], K['cy'], m.path_length)
        if s:
            text += ('<br>Ground truth: %s (%d matched)<br><b>Position error RMSE %.2f cm</b>, mean %.2f, median %.2f, '
                     'max %.2f cm; scale estimate/GT %.4f (%s alignment)<br>Rotation error RMSE %.3f deg; '
                     'constant camera frame offset %.2f deg, RMSE without it %.3f deg') % (
                m.gt_path, s['matched'], s['rmse']*100, s['mean']*100, s['median']*100, s['max']*100, s['scale'],
                'similarity' if self.scale.isChecked() else 'rigid', s['rot_rmse'], s['rot_offset'], s['rot_rmse_no_offset'])
        else:
            text += '<br>No ground truth (File > Load ground truth, or gt_camera_poses.txt in the folder)'
        self.stats.setText(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir', nargs='?')
    parser.add_argument('--gt')
    parser.add_argument('--scale', action='store_true', help='Similarity alignment (with scale)')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--plane', choices=['xy', 'xz', 'yz'], default='xy')
    parser.add_argument('--screenshot')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv[:1])
    window = MainWindow()
    window.scale.setChecked(args.scale)
    window.plane.setCurrentIndex(['xy', 'xz', 'yz'].index(args.plane))
    if args.dir and not window.open_folder(args.dir) and args.screenshot:
        return 1
    if args.gt and not window.open_gt(args.gt) and args.screenshot:
        return 1
    if args.dir:
        window.slider.setValue(args.frame)
    if args.screenshot:
        window.show()
        app.processEvents()
        window.trajectory.fit()
        window.select(args.frame)
        app.processEvents()
        window.grab().save(args.screenshot)
        print(re.sub('<[^>]*>', '', window.stats.text().replace('<br>', '\n')))
        return 0
    window.show()
    return app.exec() if hasattr(app, 'exec') else app.exec_()


if __name__ == '__main__':
    sys.exit(main())
