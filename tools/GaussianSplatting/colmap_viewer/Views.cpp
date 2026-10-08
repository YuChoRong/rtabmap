#include "Views.h"
#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
const QColor kEstimateColor(30, 110, 230);
const QColor kGtColor(240, 140, 20);
const QColor kErrorColor(220, 40, 40, 150);

// Turbo-like colormap for t in [0,1]
QColor colormap(double t)
{
	t = std::min(1.0, std::max(0.0, t));
	const double r = std::min(1.0, std::max(0.0, 1.5 - std::fabs(4.0*t - 3.0)));
	const double g = std::min(1.0, std::max(0.0, 1.5 - std::fabs(4.0*t - 2.0)));
	const double b = std::min(1.0, std::max(0.0, 1.5 - std::fabs(4.0*t - 1.0)));
	return QColor::fromRgbF(r, g, b);
}
} // namespace

// ---------------------------------------------------------------- TrajectoryView

TrajectoryView::TrajectoryView(QWidget * parent) : QWidget(parent)
{
	setMinimumSize(300, 300);
	setMouseTracking(false);
}

void TrajectoryView::setModel(const ColmapModel * model)
{
	model_ = model;
	selected_ = -1;
	fit();
	update();
}

QPointF TrajectoryView::axes(const Eigen::Vector3d & p) const
{
	switch(plane_)
	{
	case kXZ: return QPointF(p.x(), p.z());
	case kYZ: return QPointF(p.y(), p.z());
	default: return QPointF(p.x(), p.y());
	}
}

QPointF TrajectoryView::toScreen(const Eigen::Vector3d & p) const
{
	QPointF a = axes(p);
	return QPointF(width()/2.0 + (a.x()-center_.x())*zoom_, height()/2.0 - (a.y()-center_.y())*zoom_);
}

void TrajectoryView::fit()
{
	if(!model_ || model_->frames().empty())
	{
		return;
	}
	double minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
	for(const Frame & f : model_->frames())
	{
		for(const Eigen::Vector3d & p : {f.estimate.t, f.hasGt ? f.gt.t : f.estimate.t})
		{
			QPointF a = axes(p);
			minX = std::min(minX, a.x()); maxX = std::max(maxX, a.x());
			minY = std::min(minY, a.y()); maxY = std::max(maxY, a.y());
		}
	}
	// Some margin for the points around the trajectory
	double span = std::max(std::max(maxX-minX, maxY-minY), 0.5) * 1.4;
	zoom_ = std::min(width(), height()) / span;
	center_ = QPointF((minX+maxX)/2.0, (minY+maxY)/2.0);
}

void TrajectoryView::paintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.fillRect(rect(), palette().base());
	if(!model_ || model_->frames().empty())
	{
		painter.drawText(rect(), Qt::AlignCenter, "Open a COLMAP folder");
		return;
	}
	painter.setRenderHint(QPainter::Antialiasing, true);
	const std::vector<Frame> & frames = model_->frames();

	// Points: density image, cheaper than drawing each point
	if(showPoints_ && !model_->points().empty())
	{
		QImage density(size(), QImage::Format_ARGB32_Premultiplied);
		density.fill(Qt::transparent);
		std::vector<int> counts(width()*height(), 0);
		int maxCount = 1;
		for(const Point3D & p : model_->points())
		{
			QPointF s = toScreen(p.xyz.cast<double>());
			int x = (int)s.x(), y = (int)s.y();
			if(x >= 0 && y >= 0 && x < width() && y < height())
			{
				maxCount = std::max(maxCount, ++counts[y*width()+x]);
			}
		}
		const double norm = std::log(1.0 + std::min(maxCount, 20));
		QColor text = palette().text().color();
		for(int y=0; y<height(); ++y)
		{
			QRgb * line = (QRgb*)density.scanLine(y);
			for(int x=0; x<width(); ++x)
			{
				int c = counts[y*width()+x];
				if(c)
				{
					int a = (int)(40 + 150 * std::min(1.0, std::log(1.0 + c) / norm));
					line[x] = qPremultiply(qRgba(text.red(), text.green(), text.blue(), a));
				}
			}
		}
		painter.drawImage(0, 0, density);
	}

	// Scale bar
	double meters = std::pow(10.0, std::floor(std::log10(150.0 / zoom_)));
	painter.setPen(QPen(palette().text().color(), 2));
	painter.drawLine(QPointF(15, height()-15), QPointF(15 + meters*zoom_, height()-15));
	painter.drawText(QPointF(15, height()-20), meters >= 1 ? QString("%1 m").arg(meters) : QString("%1 cm").arg(meters*100));
	const char * labels[] = {"top view (X right, Y up)", "side view (X right, Z up)", "side view (Y right, Z up)"};
	painter.drawText(QPointF(10, 18), labels[plane_]);

	// Error segments
	if(showErrors_)
	{
		painter.setPen(QPen(kErrorColor, 1));
		for(const Frame & f : frames)
		{
			if(f.hasGt) painter.drawLine(toScreen(f.estimate.t), toScreen(f.gt.t));
		}
	}
	// Trajectories
	QPolygonF est, gt;
	for(const Frame & f : frames)
	{
		est << toScreen(f.estimate.t);
		if(f.hasGt) gt << toScreen(f.gt.t);
	}
	painter.setPen(QPen(kGtColor, 2));
	painter.drawPolyline(gt);
	painter.setPen(QPen(kEstimateColor, 2));
	painter.drawPolyline(est);

	// Selected camera with its viewing direction
	if(selected_ >= 0 && selected_ < (int)frames.size())
	{
		const Frame & f = frames[selected_];
		QPointF c = toScreen(f.estimate.t);
		QPointF d = toScreen(f.estimate.t + f.estimate.R.col(2) * (40.0/zoom_));
		painter.setPen(QPen(palette().highlight().color(), 2));
		painter.drawLine(c, d);
		painter.setBrush(palette().highlight());
		painter.drawEllipse(c, 5, 5);
		if(f.hasGt)
		{
			painter.setBrush(kGtColor);
			painter.setPen(Qt::NoPen);
			painter.drawEllipse(toScreen(f.gt.t), 4, 4);
		}
	}

	// Legend
	const QString estLabel = "estimate (COLMAP)";
	const QString gtLabel = model_->hasGroundTruth() ? "ground truth (aligned)" : "no ground truth";
	const int textWidth = std::max(fontMetrics().horizontalAdvance(estLabel), fontMetrics().horizontalAdvance(gtLabel));
	const int x0 = width() - textWidth - 45;
	painter.setPen(QPen(kEstimateColor, 3));
	painter.drawLine(x0, 15, x0+25, 15);
	painter.setPen(QPen(kGtColor, 3));
	painter.drawLine(x0, 32, x0+25, 32);
	painter.setPen(palette().text().color());
	painter.drawText(x0+30, 20, estLabel);
	painter.drawText(x0+30, 37, gtLabel);
}

void TrajectoryView::wheelEvent(QWheelEvent * event)
{
	double factor = std::pow(1.0015, event->angleDelta().y());
	QPointF pos = event->position();
	// Keep the point under the cursor fixed
	QPointF before((pos.x()-width()/2.0)/zoom_ + center_.x(), -(pos.y()-height()/2.0)/zoom_ + center_.y());
	zoom_ *= factor;
	QPointF after((pos.x()-width()/2.0)/zoom_ + center_.x(), -(pos.y()-height()/2.0)/zoom_ + center_.y());
	center_ += before - after;
	update();
}

void TrajectoryView::mousePressEvent(QMouseEvent * event)
{
	lastMouse_ = event->pos();
	dragged_ = false;
	if(event->button() == Qt::LeftButton && model_)
	{
		// Nearest camera within 15 pixels
		int best = -1;
		double bestDistance = 15.0;
		for(size_t i=0; i<model_->frames().size(); ++i)
		{
			QPointF d = toScreen(model_->frames()[i].estimate.t) - event->pos();
			double distance = std::hypot(d.x(), d.y());
			if(distance < bestDistance) {bestDistance = distance; best = (int)i;}
		}
		if(best >= 0) emit frameClicked(best);
	}
}

void TrajectoryView::mouseMoveEvent(QMouseEvent * event)
{
	QPoint delta = event->pos() - lastMouse_;
	lastMouse_ = event->pos();
	center_ -= QPointF(delta.x()/zoom_, -delta.y()/zoom_);
	dragged_ = true;
	update();
}

// ---------------------------------------------------------------- ErrorPlot

ErrorPlot::ErrorPlot(QWidget * parent) : QWidget(parent)
{
	setMinimumHeight(150);
}

QRectF ErrorPlot::plotRect() const
{
	return QRectF(55, 10, width()-70, height()-40);
}

double ErrorPlot::xOf(size_t i) const
{
	const std::vector<Frame> & frames = model_->frames();
	bool stamps = frames.front().stamp >= 0 && frames.back().stamp > frames.front().stamp;
	double v = stamps ? frames[i].stamp - frames.front().stamp : (double)i;
	double maxV = stamps ? frames.back().stamp - frames.front().stamp : (double)std::max<size_t>(1, frames.size()-1);
	QRectF r = plotRect();
	return r.left() + r.width() * v / maxV;
}

void ErrorPlot::paintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.fillRect(rect(), palette().base());
	if(!model_ || model_->frames().empty() || !model_->hasGroundTruth())
	{
		painter.drawText(rect(), Qt::AlignCenter, "Position error over time (needs ground truth)");
		return;
	}
	painter.setRenderHint(QPainter::Antialiasing, true);
	const std::vector<Frame> & frames = model_->frames();
	const Stats & s = model_->stats();
	QRectF r = plotRect();
	double maxE = std::max(s.max * 1.1, 1e-3);
	auto yOf = [&](double e) {return r.bottom() - r.height() * e / maxE;};

	painter.setPen(palette().mid().color());
	painter.drawRect(r);
	painter.setPen(palette().text().color());
	for(int k=0; k<=4; ++k)
	{
		double e = maxE * k / 4.0;
		painter.drawText(QRectF(0, yOf(e)-8, 50, 16), Qt::AlignRight|Qt::AlignVCenter, QString::number(e*100, 'f', 1));
	}
	painter.drawText(QRectF(r.left(), r.bottom()+4, r.width(), 20), Qt::AlignCenter,
			frames.front().stamp >= 0 ? "time (s)        error (cm)" : "frame        error (cm)");

	// RMSE line
	painter.setPen(QPen(kErrorColor, 1, Qt::DashLine));
	painter.drawLine(QPointF(r.left(), yOf(s.rmse)), QPointF(r.right(), yOf(s.rmse)));
	painter.drawText(QPointF(r.right()-120, yOf(s.rmse)-4), QString("RMSE %1 cm").arg(s.rmse*100, 0, 'f', 2));

	QPolygonF line;
	for(size_t i=0; i<frames.size(); ++i)
	{
		if(frames[i].hasGt) line << QPointF(xOf(i), yOf(frames[i].error));
	}
	painter.setPen(QPen(kEstimateColor, 1.5));
	painter.drawPolyline(line);

	if(selected_ >= 0 && selected_ < (int)frames.size())
	{
		painter.setPen(QPen(palette().highlight().color(), 1));
		painter.drawLine(QPointF(xOf(selected_), r.top()), QPointF(xOf(selected_), r.bottom()));
	}
}

void ErrorPlot::mousePressEvent(QMouseEvent * event)
{
	if(!model_ || model_->frames().empty()) return;
	int best = 0;
	double bestDistance = 1e9;
	for(size_t i=0; i<model_->frames().size(); ++i)
	{
		double d = std::fabs(xOf(i) - event->pos().x());
		if(d < bestDistance) {bestDistance = d; best = (int)i;}
	}
	emit frameClicked(best);
}

// ---------------------------------------------------------------- ImageView

ImageView::ImageView(QWidget * parent) : QWidget(parent)
{
	setMinimumSize(320, 200);
}

void ImageView::setFrame(int index)
{
	index_ = index;
	image_ = QImage();
	projections_.clear();
	projected_ = 0;
	if(model_ && index >= 0 && index < (int)model_->frames().size())
	{
		const Frame & f = model_->frames()[index];
		image_.load(model_->imagePath(f));
		const Intrinsics & K = model_->intrinsics();
		const Pose & pose = useGt_ && f.hasGt ? f.gt : f.estimate;
		const Eigen::Matrix3d Rcw = pose.R.transpose();
		const Eigen::Vector3d tcw = -Rcw * pose.t;
		const int w = K.width > 0 ? K.width : image_.width();
		const int h = K.height > 0 ? K.height : image_.height();
		for(const Point3D & p : model_->points())
		{
			Eigen::Vector3d pc = Rcw * p.xyz.cast<double>() + tcw;
			if(pc.z() < 0.1 || pc.z() > 8.0) continue;
			double u = K.fx * pc.x() / pc.z() + K.cx;
			double v = K.fy * pc.y() / pc.z() + K.cy;
			if(u >= 0 && v >= 0 && u < w && v < h)
			{
				projections_.push_back(std::make_pair(QPointF(u, v), (float)pc.z()));
			}
		}
		projected_ = (int)projections_.size();
	}
	update();
}

void ImageView::paintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.fillRect(rect(), palette().base());
	if(!model_ || index_ < 0)
	{
		painter.drawText(rect(), Qt::AlignCenter, "Select a frame");
		return;
	}
	const Intrinsics & K = model_->intrinsics();
	QSizeF imageSize = !image_.isNull() ? QSizeF(image_.size()) : QSizeF(K.width, K.height);
	if(imageSize.isEmpty()) return;
	double s = std::min(width() / imageSize.width(), (height()-40) / imageSize.height());
	QRectF target(0, 0, imageSize.width()*s, imageSize.height()*s);
	target.moveCenter(QPointF(width()/2.0, (height()-40)/2.0));
	if(!image_.isNull())
	{
		painter.drawImage(target, image_);
	}
	else
	{
		painter.drawText(target, Qt::AlignCenter, "image not found");
	}
	if(showPoints_)
	{
		painter.setRenderHint(QPainter::Antialiasing, false);
		painter.setPen(Qt::NoPen);
		// Far points first so that close ones stay visible
		std::vector<std::pair<QPointF, float> > sorted = projections_;
		std::sort(sorted.begin(), sorted.end(), [](const std::pair<QPointF, float> & a, const std::pair<QPointF, float> & b) {return a.second > b.second;});
		const int step = std::max<int>(1, (int)sorted.size() / 6000);
		for(size_t i=0; i<sorted.size(); i+=step)
		{
			painter.setBrush(colormap(1.0 - (sorted[i].second - 0.5) / 5.0));
			painter.drawRect(QRectF(target.left() + sorted[i].first.x()*s - 1, target.top() + sorted[i].first.y()*s - 1, 2, 2));
		}
	}
	const Frame & f = model_->frames()[index_];
	painter.setPen(palette().text().color());
	QString info = QString("%1: %2 points reprojected with the %3 pose (red near, blue far)")
			.arg(QString::fromStdString(f.name)).arg(projected_).arg(useGt_ && f.hasGt ? "ground truth" : "estimated");
	if(f.hasGt)
	{
		info += QString("   error %1 cm, %2 deg").arg(f.error*100, 0, 'f', 1).arg(f.rotError, 0, 'f', 2);
	}
	painter.drawText(QRectF(4, height()-40, width()-8, 40), Qt::AlignCenter|Qt::TextWordWrap, info);
}
