#!/usr/bin/env python3
"""Result viewer (Python + Qt, no compilation) for the outputs of process_d435i.bat and
rtabmap-euroc_dataset: compares runs (e.g., ORB vs ALIKED+LightGlue) of each recording.

  pip install pyside6 numpy
  python result_viewer.py [DIR] [--screenshot OUT_PREFIX]

DIR (default C:\\slam\\out, else the current directory) is searched recursively for:
  NAME.db                  RTAB-Map database: optimized graph, loop closures, per keyframe statistics
  NAME_localization.txt    localization log of --map runs: stamp, odometry pose, map pose, localized id
  NAME.log / NAME.txt      console output of rtabmap-euroc_dataset: per frame camera/odom/slam times
Each sub-folder of DIR is one recording; files of the same NAME are one run.

Tabs: trajectories with loop closures (top view, drag to pan, wheel to zoom), processing times
against the real-time budgets (33 ms per frame at 30 Hz, 1 s per keyframe at Rtabmap/DetectionRate 1)
and a summary table, with the trajectory difference of each run against a reference run.
--screenshot: renders each tab to OUT_PREFIX_<tab>.png, prints the summary and exits.

Dependencies: PySide6 (LGPL-3.0, Qt Charts included), numpy.
"""
import argparse, math, os, re, sqlite3, sys, zlib

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtCharts
import shiboken6
Qt = QtCore.Qt

FRAME_BUDGET_MS = 1000.0 / 30.0
KEYFRAME_BUDGET_MS = 1000.0
COLORS = [QtGui.QColor(c) for c in ('#1f6fe6', '#e6731f', '#2a9d3a', '#c43c9a', '#7a5af0', '#b8860b', '#16a3b8', '#d23c3c')]
LINK_NAMES = {1: 'global', 2: 'proximity', 3: 'time'}


# ---------------------------------------------------------------- loading

def uncompress(blob, dtype):
    """rtabmap compressData2(): zlib data followed by rows, cols and type (3 int32)."""
    if not blob or len(blob) <= 12:
        return None
    return np.frombuffer(zlib.decompress(bytes(blob[:-12])), dtype=dtype)


def quat_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    x, y, z, w = qx/n, qy/n, qz/n, qw/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class Run:
    def __init__(self, recording, name):
        self.recording, self.name = recording, name
        self.files = []
        self.ids = np.zeros(0, int)          # keyframes of the graph, sorted by stamp
        self.stamps = np.zeros(0)
        self.xyz = np.zeros((0, 3))
        self.optimized = False
        self.links = []                      # (from_id, to_id, type) loop closures
        self.stats = {}                      # keyframe id -> {key: value}
        self.frames = None                   # per frame console times: dict of arrays (ms)
        self.total_time = None               # s, "Total time=" of the console output
        self.loc = None                      # localization log: stamps, odom xyz, map xyz, localized ids
        self.error = None

    @property
    def label(self):
        return '%s / %s' % (self.recording, self.name)

    def position(self, node_id):
        i = self.index.get(node_id)
        return None if i is None else self.xyz[i]

    def load_db(self, path):
        self.files.append(path)
        con = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
        try:
            nodes = {}
            for nid, stamp, pose in con.execute('SELECT id, stamp, pose FROM Node'):
                if pose and len(pose) == 48:
                    nodes[nid] = (stamp or 0.0, np.frombuffer(pose, np.float32).reshape(3, 4)[:, 3].astype(float))
            poses = {}
            try:
                row = con.execute('SELECT opt_ids, opt_poses FROM Admin').fetchone()
            except sqlite3.Error:
                row = None
            if row and row[0] and row[1]:
                ids = uncompress(row[0], np.int32)
                mats = uncompress(row[1], np.float32)
                if ids is not None and mats is not None and len(mats) == 12 * len(ids):
                    mats = mats.reshape(-1, 3, 4)
                    poses = {int(i): mats[k][:, 3].astype(float) for k, i in enumerate(ids)}
                    self.optimized = True
            if not poses:  # odometry poses of the database
                poses = {i: v[1] for i, v in nodes.items()}
            ids = sorted(poses, key=lambda i: (nodes.get(i, (0.0,))[0], i))
            self.ids = np.array(ids, int)
            self.stamps = np.array([nodes.get(i, (0.0,))[0] for i in ids])
            self.xyz = np.array([poses[i] for i in ids]).reshape(-1, 3)
            self.links = [(a, b, t) for a, b, t in con.execute(
                'SELECT from_id, to_id, type FROM Link WHERE type IN (1, 2, 3) AND from_id > to_id')]
            for nid, blob in con.execute('SELECT id, data FROM Statistics'):
                try:
                    data = uncompress(blob, np.uint8)
                    text = data.tobytes().split(b'\0')[0].decode('utf-8', 'replace')
                except Exception:
                    continue
                d = {}
                for item in text.split(';'):
                    k, _, v = item.rpartition(':')
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
                self.stats[nid] = d
        finally:
            con.close()

    def load_log(self, path):
        cols = {'camera': [], 'odom': [], 'slam': [], 'loop': []}
        pattern = re.compile(r'Iteration \d+/\d+: camera=(\d+)ms, odom\(.*\)=(\d+)ms, slam=(\d+)ms')
        with open(path, errors='replace') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    cols['camera'].append(float(m.group(1)))
                    cols['odom'].append(float(m.group(2)))
                    cols['slam'].append(float(m.group(3)))
                    cols['loop'].append(line.rstrip().endswith('*'))
                elif line.startswith('Total time='):
                    self.total_time = float(line.split('=')[1].rstrip('s\n'))
        if cols['odom']:
            self.files.append(path)
            self.frames = {k: np.array(v) for k, v in cols.items()}

    def load_localization(self, path):
        rows = []
        with open(path) as f:
            for line in f:
                v = line.split()
                if len(v) >= 16:
                    rows.append([float(x) for x in v[:16]])
        if rows:
            self.files.append(path)
            a = np.array(rows)
            self.loc = {'stamps': a[:, 0], 'odom': a[:, 1:4], 'map': a[:, 8:11], 'ids': a[:, 15].astype(int)}

    def finish(self):
        self.index = {int(i): k for k, i in enumerate(self.ids)}

    def stat(self, key):
        return np.array([d[key] for d in self.stats.values() if key in d])


def run_name(filename):
    base = os.path.basename(filename)
    for suffix in ('_localization.txt', '.db', '.log', '.txt'):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return None


def find_runs(root):
    runs = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d not in ('euroc', 'images', 'colmap', 'colmap_sfm', 'sparse'))
        rel = os.path.relpath(dirpath, root)
        recording = os.path.basename(os.path.abspath(root)) if rel == '.' else rel.replace(os.sep, '/')
        for fn in sorted(filenames):
            name = run_name(fn)
            if name is None or fn.endswith(('_poses.txt', '_rmse.txt', '_calib.txt')):
                continue
            path = os.path.join(dirpath, fn)
            run = runs.setdefault((recording, name), Run(recording, name))
            try:
                if fn.endswith('_localization.txt'):
                    run.load_localization(path)
                elif fn.endswith('.db'):
                    run.load_db(path)
                else:
                    run.load_log(path)
            except Exception as e:
                run.error = '%s: %s' % (fn, e)
                print('Cannot read %s: %s' % (path, e), file=sys.stderr)
    out = []
    for run in runs.values():
        run.finish()
        if len(run.ids) or run.frames is not None or run.loc is not None:
            out.append(run)
    return sorted(out, key=lambda r: (r.recording, r.name))


# ---------------------------------------------------------------- metrics

def percentile(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


def path_length(xyz):
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


def rigid_align(src, dst):
    """R, t minimizing |R src + t - dst| (Kabsch, no scale)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def trajectory_difference(run, ref, max_dt=0.02):
    """RMSE and max (m) of the positions of run vs ref at the same stamps, after rigid alignment."""
    if run is ref or len(run.xyz) < 3 or len(ref.xyz) < 3:
        return None
    order = np.argsort(ref.stamps)
    rs = ref.stamps[order]
    k = np.clip(np.searchsorted(rs, run.stamps), 1, len(rs) - 1)
    k = np.where(np.abs(rs[k - 1] - run.stamps) < np.abs(rs[k] - run.stamps), k - 1, k)
    ok = np.abs(rs[k] - run.stamps) < max_dt
    if ok.sum() < 3:
        return None
    a, b = run.xyz[ok], ref.xyz[order][k[ok]]
    R, t = rigid_align(a, b)
    e = np.linalg.norm(a @ R.T + t - b, axis=1)
    return float(np.sqrt((e**2).mean())), float(e.max()), int(ok.sum())


def summary_rows(runs, refs):
    rows = []
    for run in runs:
        r = {'run': run.label}
        if len(run.ids):
            r['keyframes'] = len(run.ids)
            r['duration_s'] = float(run.stamps[-1] - run.stamps[0]) if len(run.stamps) > 1 else 0.0
            r['path_m'] = path_length(run.xyz)
            for t, n in LINK_NAMES.items():
                r['loops_' + n] = sum(1 for l in run.links if l[2] == t)
            r['rejected'] = int((run.stat('Loop/RejectedHypothesis/') > 0).sum())
            tt = run.stat('Timing/Total/ms')
            if len(tt):
                r['kf_mean_ms'], r['kf_median_ms'], r['kf_p95_ms'], r['kf_max_ms'] = tt.mean(), np.median(tt), percentile(tt, 95), tt.max()
                r['kf_over_budget'] = int((tt > KEYFRAME_BUDGET_MS).sum())
        if run.frames is not None:
            f = run.frames
            per_frame = f['odom'] + f['slam']
            r['frames'] = len(per_frame)
            r['odom_mean_ms'], r['odom_p95_ms'] = f['odom'].mean(), percentile(f['odom'], 95)
            r['frame_mean_ms'], r['frame_p95_ms'] = per_frame.mean(), percentile(per_frame, 95)
            r['frames_over_33ms'] = int((per_frame > FRAME_BUDGET_MS).sum())
            if run.total_time is not None:
                r['total_s'] = run.total_time
        if run.loc is not None:
            ids = run.loc['ids']
            r['loc_frames'] = len(ids)
            r['loc_matched_pct'] = 100.0 * (ids > 0).sum() / len(ids)
            first = np.nonzero(ids > 0)[0]
            if len(first):
                r['loc_first_s'] = float(run.loc['stamps'][first[0]] - run.loc['stamps'][0])
        ref = refs.get(run.recording)
        d = trajectory_difference(run, ref) if ref is not None else None
        if d:
            r['ref'] = ref.name
            r['diff_rmse_cm'], r['diff_max_cm'] = 100 * d[0], 100 * d[1]
        rows.append(r)
    return rows


COLUMNS = [
    ('run', '실행'), ('keyframes', '키프레임'), ('duration_s', '길이 (s)'), ('path_m', '경로 (m)'),
    ('loops_global', '루프 클로저'), ('loops_proximity', '근접 링크'), ('rejected', '기각'),
    ('kf_mean_ms', '키프레임 평균 (ms)'), ('kf_median_ms', '키프레임 중앙값 (ms)'), ('kf_p95_ms', '키프레임 p95 (ms)'),
    ('kf_max_ms', '키프레임 최대 (ms)'), ('kf_over_budget', '1 s 초과'),
    ('frames', '프레임'), ('odom_mean_ms', '오도메트리 평균 (ms)'), ('odom_p95_ms', '오도메트리 p95 (ms)'),
    ('frame_mean_ms', '프레임 평균 (ms)'), ('frame_p95_ms', '프레임 p95 (ms)'), ('frames_over_33ms', '33 ms 초과'),
    ('total_s', '전체 시간 (s)'),
    ('loc_frames', '위치추정 프레임'), ('loc_matched_pct', '지도 매칭 (%)'), ('loc_first_s', '첫 위치인식 (s)'),
    ('ref', '기준'), ('diff_rmse_cm', '기준 대비 RMSE (cm)'), ('diff_max_cm', '기준 대비 최대 (cm)'),
]


def fmt(v):
    if isinstance(v, float):
        return '-' if math.isnan(v) else ('%.1f' % v if abs(v) >= 10 else '%.2f' % v)
    return str(v)


# ---------------------------------------------------------------- views

PLANES = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}


class TrajectoryView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.runs, self.colors = [], {}
        self.plane, self.show_loops, self.show_loc = 'xy', True, True
        self.zoom, self.offset, self.drag = 1.0, QtCore.QPointF(), None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_runs(self, runs, colors):
        self.runs, self.colors = runs, colors
        self.zoom, self.offset = 1.0, QtCore.QPointF()
        self.update()

    def curves(self):
        i, j = PLANES[self.plane]
        out = []
        for run in self.runs:
            if len(run.xyz):
                out.append((run, run.xyz[:, [i, j]], 'graph'))
            if self.show_loc and run.loc is not None:
                out.append((run, run.loc['map'][:, [i, j]], 'loc'))
        return out

    def transform(self, curves):
        pts = np.vstack([c[1] for c in curves])
        lo, hi = pts.min(0), pts.max(0)
        span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-3)
        margin = 40
        s = self.zoom * min(self.width() - 2 * margin, self.height() - 2 * margin - 30) / span
        c = (lo + hi) / 2
        cx, cy = self.width() / 2 + self.offset.x(), (self.height() + 30) / 2 + self.offset.y()
        return lambda p: QtCore.QPointF(cx + (p[0] - c[0]) * s, cy - (p[1] - c[1]) * s), s

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), self.palette().base())
        curves = self.curves()
        if not curves:
            p.drawText(self.rect(), Qt.AlignCenter, '선택한 실행에 궤적이 없어요 (.db 또는 _localization.txt)')
            return
        to_px, s = self.transform(curves)
        # 1 m grid scale bar
        p.setPen(QtGui.QPen(self.palette().text().color(), 2))
        bar = s * 1.0
        p.drawLine(QtCore.QPointF(20, self.height() - 20), QtCore.QPointF(20 + bar, self.height() - 20))
        p.drawText(QtCore.QPointF(24, self.height() - 26), '1 m')
        for run, xy, kind in curves:
            color = QtGui.QColor(self.colors[run.label])
            if kind == 'loc':
                ids = run.loc['ids']
                for k, q in enumerate(xy):
                    c = QtGui.QColor(color) if ids[k] > 0 else QtGui.QColor(150, 150, 150)
                    p.setPen(Qt.NoPen)
                    p.setBrush(c)
                    p.drawEllipse(to_px(q), 2.5, 2.5)
                continue
            p.setPen(QtGui.QPen(color, 2))
            p.drawPolyline(QtGui.QPolygonF([to_px(q) for q in xy]))
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(to_px(xy[0]), 5, 5)
            if self.show_loops:
                i, j = PLANES[self.plane]
                pen = QtGui.QPen(color.darker(140), 1.2, Qt.DashLine)
                p.setPen(pen)
                for a, b, t in run.links:
                    pa, pb = run.position(a), run.position(b)
                    if pa is not None and pb is not None:
                        p.drawLine(to_px(pa[[i, j]]), to_px(pb[[i, j]]))
        # legend
        y = 18
        p.setPen(self.palette().text().color())
        for run in self.runs:
            p.setPen(QtGui.QPen(self.colors[run.label], 4))
            p.drawLine(12, y - 4, 32, y - 4)
            p.setPen(self.palette().text().color())
            extra = ' (최적화)' if run.optimized else (' (오도메트리)' if len(run.ids) else '')
            if run.loc is not None:
                extra += ', 점: 위치추정 (회색: 매칭 없음)'
            p.drawText(38, y, run.label + extra)
            y += 18
        if self.show_loops:
            p.drawText(38, y, '점선: 루프 클로저 / 근접 링크')

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        center = QtCore.QPointF(self.width() / 2, (self.height() + 30) / 2) + self.offset
        pos = e.position()
        self.offset += (pos - center) * (1 - f)
        self.zoom *= f
        self.update()

    def mousePressEvent(self, e):
        self.drag = e.position()

    def mouseMoveEvent(self, e):
        if self.drag is not None and e.buttons():
            self.offset += e.position() - self.drag
            self.drag = e.position()
            self.update()

    def mouseReleaseEvent(self, e):
        self.drag = None

    def mouseDoubleClickEvent(self, e):
        self.zoom, self.offset = 1.0, QtCore.QPointF()
        self.update()


class TimingView(QtWidgets.QWidget):
    METRICS = [
        ('frame', '프레임당 시간: 오도메트리 + SLAM (ms, 콘솔 로그)'),
        ('odom', '오도메트리 시간 (ms, 콘솔 로그)'),
        ('kf', '키프레임 SLAM 시간 Timing/Total (ms, DB)'),
        ('kf_odom', '키프레임의 오도메트리 시간 (ms, DB)'),
    ]

    def __init__(self):
        super().__init__()
        self.runs, self.colors = [], {}
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.metric_box = QtWidgets.QComboBox()
        for k, label in self.METRICS:
            self.metric_box.addItem(label, k)
        self.metric_box.currentIndexChanged.connect(self.refresh)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(['시간 순서', '누적 분포 (CDF)'])
        self.kind.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.metric_box, 1)
        top.addWidget(self.kind)
        lay.addLayout(top)
        # the chart of the view, owned by it: a QChart made in Python and given to the view
        # would be deleted twice when the window closes (heap corruption at exit)
        self.chart_view = QtCharts.QChartView()
        self.chart = self.chart_view.chart()
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.ax_x, self.ax_y = QtCharts.QValueAxis(), QtCharts.QValueAxis()
        self.chart.addAxis(self.ax_x, Qt.AlignBottom)
        self.chart.addAxis(self.ax_y, Qt.AlignLeft)
        self.chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.chart_view.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        lay.addWidget(self.chart_view)
        hint = QtWidgets.QLabel('드래그: 확대, 오른쪽 클릭: 축소. 빨간 선: 실시간 예산 (프레임 33 ms, 키프레임 1 s)')
        hint.setStyleSheet('color: gray')
        lay.addWidget(hint)

    def set_runs(self, runs, colors):
        self.runs, self.colors = runs, colors
        self.refresh()

    def series_of(self, run, metric):
        if metric in ('frame', 'odom'):
            if run.frames is None:
                return None
            y = run.frames['odom'] + (run.frames['slam'] if metric == 'frame' else 0)
            return np.arange(len(y)) / 30.0, y
        key = 'Timing/Total/ms' if metric == 'kf' else 'Odometry/TotalTime/ms'
        x, y = [], []
        t0 = run.stamps[0] if len(run.stamps) else 0.0
        stamp = dict(zip(run.ids.tolist(), run.stamps.tolist()))
        for nid in sorted(run.stats):
            d = run.stats[nid]
            if key in d:
                x.append(stamp[nid] - t0 if nid in stamp else float(nid))
                y.append(d[key])
        if not y:
            return None
        return np.array(x), np.array(y)

    def refresh(self):
        self.chart.removeAllSeries()
        metric = self.metric_box.currentData()
        cdf = self.kind.currentIndex() == 1
        budget = KEYFRAME_BUDGET_MS if metric == 'kf' else FRAME_BUDGET_MS
        ax_x, ax_y = self.ax_x, self.ax_y
        ax_x.setTitleText('시간 (ms)' if cdf else '녹화 시간 (s)')
        ax_y.setTitleText('비율 (%)' if cdf else 'ms')
        xmax, ymax, any_series = 1.0, budget * 1.2, False
        for run in self.runs:
            s = self.series_of(run, metric)
            if s is None:
                continue
            x, y = s
            if cdf:
                y = np.sort(y)
                x, y = y, 100.0 * np.arange(1, len(y) + 1) / len(y)
            line = QtCharts.QLineSeries()
            name = run.name if len({r.recording for r in self.runs}) == 1 else run.label
            line.setName('%s (평균 %.1f ms, p95 %.1f ms)' % (name, s[1].mean(), percentile(s[1], 95)))
            step = max(1, len(x) // 4000)
            line.append([QtCore.QPointF(float(a), float(b)) for a, b in zip(x[::step], y[::step])])
            line.setPen(QtGui.QPen(self.colors[run.label], 1.5))
            self.chart.addSeries(line)
            line.attachAxis(ax_x)
            line.attachAxis(ax_y)
            xmax, ymax = max(xmax, float(x.max())), max(ymax, float(y.max()))
            any_series = True
        b = QtCharts.QLineSeries()
        b.setName('실시간 예산 %.0f ms' % budget)
        b.setPen(QtGui.QPen(QtGui.QColor(220, 40, 40), 1.5, Qt.DashLine))
        if cdf:
            b.append(budget, 0)
            b.append(budget, 100)
            xmax = max(xmax, budget * 1.2)
        else:
            b.append(0, budget)
            b.append(xmax, budget)
        self.chart.addSeries(b)
        b.attachAxis(ax_x)
        b.attachAxis(ax_y)
        ax_x.setRange(0, xmax)
        ax_y.setRange(0, 100 if cdf else min(ymax, budget * 6) * 1.05)
        self.chart.setTitle('' if any_series else '선택한 실행에 이 데이터가 없어요')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.setWindowTitle('RTAB-Map 결과 뷰어')
        self.resize(1400, 850)
        self.runs, self.colors = [], {}
        split = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(root)
        browse = QtWidgets.QPushButton('폴더...')
        browse.clicked.connect(self.browse)
        reload_ = QtWidgets.QPushButton('다시 읽기')
        reload_.clicked.connect(lambda: self.load(self.folder.text()))
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        row.addWidget(reload_)
        ll.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['녹화 / 실행', '파일'])
        self.tree.itemChanged.connect(self.selection_changed)
        ll.addWidget(self.tree, 1)
        opts = QtWidgets.QFormLayout()
        self.plane = QtWidgets.QComboBox()
        self.plane.addItems(list(PLANES))
        self.plane.currentTextChanged.connect(self.plane_changed)
        opts.addRow('궤적 평면', self.plane)
        self.loops = QtWidgets.QCheckBox('루프 클로저 표시')
        self.loops.setChecked(True)
        self.loops.toggled.connect(self.options_changed)
        self.locs = QtWidgets.QCheckBox('위치추정 결과 표시')
        self.locs.setChecked(True)
        self.locs.toggled.connect(self.options_changed)
        opts.addRow(self.loops)
        opts.addRow(self.locs)
        self.reference = QtWidgets.QComboBox()
        self.reference.addItems(['orb', 'aliked'])
        self.reference.setEditable(True)
        self.reference.currentTextChanged.connect(self.selection_changed)
        opts.addRow('궤적 비교 기준', self.reference)
        ll.addLayout(opts)
        split.addWidget(left)
        self.tabs = QtWidgets.QTabWidget()
        self.traj = TrajectoryView()
        self.timing = TimingView()
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.traj, '궤적')
        self.tabs.addTab(self.timing, '처리 시간')
        self.tabs.addTab(self.table, '요약')
        split.addWidget(self.tabs)
        split.setSizes([380, 1020])
        self.setCentralWidget(split)
        self.load(root)

    def browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, '결과 폴더', self.folder.text())
        if d:
            self.folder.setText(d)
            self.load(d)

    def load(self, root):
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.runs = find_runs(root) if os.path.isdir(root) else []
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.colors = {}
        self.tree.blockSignals(True)
        self.tree.clear()
        recs = {}
        names = sorted({r.name for r in self.runs})  # same color for the same run name in every recording
        for run in self.runs:
            parent = recs.get(run.recording)
            if parent is None:
                parent = recs[run.recording] = QtWidgets.QTreeWidgetItem([run.recording, ''])
                self.tree.addTopLevelItem(parent)
                parent.setExpanded(True)
            self.colors[run.label] = COLORS[names.index(run.name) % len(COLORS)]
            item = QtWidgets.QTreeWidgetItem([run.name, ', '.join(os.path.basename(f) for f in run.files)])
            item.setToolTip(1, '\n'.join(run.files) + ('\n' + run.error if run.error else ''))
            item.setForeground(0, QtGui.QBrush(self.colors[run.label]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if len(recs) == 1 else Qt.Unchecked)
            item.setData(0, Qt.UserRole, run.label)
            parent.addChild(item)
        self.tree.resizeColumnToContents(0)
        self.tree.blockSignals(False)
        if not self.runs:
            self.statusBar().showMessage('%s 에서 결과 파일(.db, _localization.txt, .log)을 찾지 못했어요' % root)
        else:
            self.statusBar().showMessage('녹화 %d개, 실행 %d개' % (len(recs), len(self.runs)))
        self.selection_changed()

    def selected(self):
        labels = set()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                c = parent.child(j)
                if c.checkState(0) == Qt.Checked:
                    labels.add(c.data(0, Qt.UserRole))
        return [r for r in self.runs if r.label in labels]

    def references(self):
        """Reference run of each recording: the run named like the reference box, else the first with a graph."""
        want = self.reference.currentText().strip().lower()
        refs = {}
        for run in self.runs:
            if not len(run.xyz):
                continue
            cur = refs.get(run.recording)
            if cur is None or (want and want in run.name.lower() and not (want in cur.name.lower())):
                refs[run.recording] = run
        return refs

    def selection_changed(self, *args):
        runs = self.selected()
        self.traj.set_runs(runs, self.colors)
        self.timing.set_runs(runs, self.colors)
        rows = summary_rows(runs, self.references())
        cols = [c for c in COLUMNS if any(c[0] in r for r in rows)]
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels([c[1] for c in cols])
        for i, r in enumerate(rows):
            for j, (k, _) in enumerate(cols):
                it = QtWidgets.QTableWidgetItem(fmt(r[k]) if k in r else '')
                if k != 'run':
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.last_rows = rows

    def plane_changed(self, plane):
        self.traj.plane = plane
        self.traj.update()

    def options_changed(self):
        self.traj.show_loops = self.loops.isChecked()
        self.traj.show_loc = self.locs.isChecked()
        self.traj.update()

    def check_all(self):
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                parent.child(j).setCheckState(0, Qt.Checked)


def main():
    default = r'C:\slam\out' if os.path.isdir(r'C:\slam\out') else os.getcwd()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('folder', nargs='?', default=default)
    ap.add_argument('--plane', choices=list(PLANES), default='xy')
    ap.add_argument('--reference', default='orb', help='name (part) of the reference run of each recording')
    ap.add_argument('--screenshot', help='renders each tab to SCREENSHOT_<tab>.png and exits')
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    w = MainWindow(os.path.abspath(args.folder))
    w.reference.setCurrentText(args.reference)
    w.plane.setCurrentText(args.plane)
    if args.screenshot:
        w.check_all()
        w.show()
        for i, name in enumerate(['trajectory', 'timing', 'summary']):
            w.tabs.setCurrentIndex(i)
            if name == 'timing':
                for k in range(w.timing.metric_box.count()):
                    w.timing.metric_box.setCurrentIndex(k)
                    app.processEvents()
                    w.grab().save('%s_%s_%s.png' % (args.screenshot, name, w.timing.metric_box.currentData()))
                continue
            app.processEvents()
            w.grab().save('%s_%s.png' % (args.screenshot, name))
        for r in w.last_rows:
            print('; '.join('%s=%s' % (k, fmt(v)) for k, v in r.items()))
        code = 0
    else:
        w.show()
        code = app.exec()
    # delete the window before the QApplication: left to the garbage collector, the
    # QApplication can go first and the window's destructor then corrupts the heap at exit
    shiboken6.delete(w)
    return code

if __name__ == '__main__':
    sys.exit(main())
