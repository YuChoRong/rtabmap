#ifndef VIEWS_H
#define VIEWS_H

#include "ColmapModel.h"
#include <QImage>
#include <QWidget>

/** Orthographic view of the points and trajectories on a plane (XY top, XZ / YZ sides). */
class TrajectoryView : public QWidget
{
	Q_OBJECT
public:
	enum Plane {kXY, kXZ, kYZ};
	explicit TrajectoryView(QWidget * parent = 0);
	void setModel(const ColmapModel * model);
	void setPlane(Plane plane) {plane_ = plane; fit(); update();}
	void setSelected(int index) {selected_ = index; update();}
	void setShowPoints(bool show) {showPoints_ = show; update();}
	void setShowErrors(bool show) {showErrors_ = show; update();}
	void fit();
signals:
	void frameClicked(int index);
protected:
	void paintEvent(QPaintEvent *) override;
	void wheelEvent(QWheelEvent *) override;
	void mousePressEvent(QMouseEvent *) override;
	void mouseMoveEvent(QMouseEvent *) override;
	void resizeEvent(QResizeEvent *) override {fit();}
private:
	QPointF toScreen(const Eigen::Vector3d & p) const;
	QPointF axes(const Eigen::Vector3d & p) const;
	const ColmapModel * model_ = 0;
	Plane plane_ = kXY;
	int selected_ = -1;
	bool showPoints_ = true;
	bool showErrors_ = true;
	double zoom_ = 1.0;      // pixels per meter
	QPointF center_;         // world (plane coordinates) at the widget center
	QPoint lastMouse_;
	bool dragged_ = false;
};

/** Position error of each frame over time (or frame index). */
class ErrorPlot : public QWidget
{
	Q_OBJECT
public:
	explicit ErrorPlot(QWidget * parent = 0);
	void setModel(const ColmapModel * model) {model_ = model; update();}
	void setSelected(int index) {selected_ = index; update();}
signals:
	void frameClicked(int index);
protected:
	void paintEvent(QPaintEvent *) override;
	void mousePressEvent(QMouseEvent *) override;
private:
	double xOf(size_t i) const;
	QRectF plotRect() const;
	const ColmapModel * model_ = 0;
	int selected_ = -1;
};

/**
 * Image of a frame with the 3D points reprojected by the estimated pose
 * (colored by depth), to check the poses and intrinsics of the COLMAP model.
 */
class ImageView : public QWidget
{
	Q_OBJECT
public:
	explicit ImageView(QWidget * parent = 0);
	void setModel(const ColmapModel * model) {model_ = model; setFrame(-1);}
	void setFrame(int index);
	void setShowPoints(bool show) {showPoints_ = show; update();}
	void setUseGroundTruth(bool gt) {useGt_ = gt; setFrame(index_);}
	int projectedPoints() const {return projected_;}
protected:
	void paintEvent(QPaintEvent *) override;
private:
	const ColmapModel * model_ = 0;
	int index_ = -1;
	QImage image_;
	bool showPoints_ = true;
	bool useGt_ = false;
	std::vector<std::pair<QPointF, float> > projections_; // pixel, depth
	int projected_ = 0;
};

#endif
