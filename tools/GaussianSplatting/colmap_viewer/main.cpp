// COLMAP model viewer: checks a COLMAP text model (e.g., exported from
// RTAB-Map by rtabmap_to_colmap.py) and compares its camera poses with the
// ground truth.
//
// colmap_viewer [DIR] [--gt FILE] [--scale] [--frame N] [--plane xy|xz|yz] [--screenshot OUT.png]

#include "ColmapModel.h"
#include "Views.h"
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QMessageBox>
#include <QSlider>
#include <QSplitter>
#include <QStatusBar>
#include <QRegularExpression>
#include <QVBoxLayout>
#include <functional>
#include <iostream>

class MainWindow : public QMainWindow
{
public:
	MainWindow()
	{
		setWindowTitle("COLMAP viewer");
		trajectory_ = new TrajectoryView;
		image_ = new ImageView;
		plot_ = new ErrorPlot;
		stats_ = new QLabel;
		stats_->setTextInteractionFlags(Qt::TextSelectableByMouse);
		stats_->setWordWrap(true);

		plane_ = new QComboBox;
		plane_->addItems({"Top (XY)", "Side (XZ)", "Side (YZ)"});
		QCheckBox * points = new QCheckBox("Points");
		points->setChecked(true);
		QCheckBox * errors = new QCheckBox("Error lines");
		errors->setChecked(true);
		scale_ = new QCheckBox("Align with scale");
		QCheckBox * gtPose = new QCheckBox("Reproject with ground truth pose");
		QCheckBox * reproject = new QCheckBox("Reprojected points");
		reproject->setChecked(true);
		slider_ = new QSlider(Qt::Horizontal);
		slider_->setEnabled(false);

		QHBoxLayout * controls = new QHBoxLayout;
		controls->addWidget(new QLabel("View"));
		controls->addWidget(plane_);
		controls->addWidget(points);
		controls->addWidget(errors);
		controls->addWidget(scale_);
		controls->addSpacing(20);
		controls->addWidget(reproject);
		controls->addWidget(gtPose);
		controls->addStretch();

		QSplitter * right = new QSplitter(Qt::Vertical);
		right->addWidget(image_);
		right->addWidget(plot_);
		QWidget * statsBox = new QWidget;
		QVBoxLayout * statsLayout = new QVBoxLayout(statsBox);
		statsLayout->addWidget(stats_);
		statsLayout->addStretch();
		right->addWidget(statsBox);
		right->setStretchFactor(0, 3);
		right->setStretchFactor(1, 2);
		right->setStretchFactor(2, 0);
		QSplitter * split = new QSplitter(Qt::Horizontal);
		split->addWidget(trajectory_);
		split->addWidget(right);
		split->setStretchFactor(0, 1);
		split->setStretchFactor(1, 1);

		QWidget * central = new QWidget;
		QVBoxLayout * layout = new QVBoxLayout(central);
		layout->addLayout(controls);
		layout->addWidget(split, 1);
		layout->addWidget(slider_);
		setCentralWidget(central);
		resize(1500, 900);

		QMenu * file = menuBar()->addMenu("&File");
		// Same menu API in Qt5 and Qt6
		auto addAction = [this](QMenu * menu, const QString & text, std::function<void()> slot, QKeySequence shortcut = QKeySequence()) {
			QAction * action = menu->addAction(text);
			action->setShortcut(shortcut);
			connect(action, &QAction::triggered, this, slot);
		};
		addAction(file, "Open COLMAP folder...", [this]() {
			QString dir = QFileDialog::getExistingDirectory(this, "COLMAP folder (with sparse/0 and images)");
			if(!dir.isEmpty()) openFolder(dir);
		}, QKeySequence::Open);
		addAction(file, "Load ground truth...", [this]() {
			QString path = QFileDialog::getOpenFileName(this, "Ground truth camera poses (stamp x y z qx qy qz qw [id])",
					model_.directory(), "Poses (*.txt *.csv);;All (*)");
			if(!path.isEmpty()) openGroundTruth(path);
		});
		addAction(file, "Save screenshot...", [this]() {
			QString path = QFileDialog::getSaveFileName(this, "Screenshot", "colmap_viewer.png", "PNG (*.png)");
			if(!path.isEmpty()) grab().save(path);
		});
		file->addSeparator();
		addAction(file, "Quit", [this]() {close();}, QKeySequence::Quit);

		connect(plane_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int i) {
			trajectory_->setPlane((TrajectoryView::Plane)i);
		});
		connect(points, &QCheckBox::toggled, trajectory_, &TrajectoryView::setShowPoints);
		connect(errors, &QCheckBox::toggled, trajectory_, &TrajectoryView::setShowErrors);
		connect(reproject, &QCheckBox::toggled, image_, &ImageView::setShowPoints);
		connect(gtPose, &QCheckBox::toggled, image_, &ImageView::setUseGroundTruth);
		connect(scale_, &QCheckBox::toggled, this, [this](bool on) {
			model_.align(on);
			refresh();
		});
		connect(slider_, &QSlider::valueChanged, this, &MainWindow::select);
		connect(trajectory_, &TrajectoryView::frameClicked, slider_, &QSlider::setValue);
		connect(plot_, &ErrorPlot::frameClicked, slider_, &QSlider::setValue);
	}

	bool openFolder(const QString & dir)
	{
		QString error;
		if(!model_.load(dir, &error))
		{
			showError(error);
			return false;
		}
		model_.align(scale_->isChecked());
		trajectory_->setModel(&model_);
		image_->setModel(&model_);
		plot_->setModel(&model_);
		slider_->setEnabled(true);
		slider_->setRange(0, (int)model_.frames().size()-1);
		refresh();
		select(0);
		slider_->setValue(0);
		return true;
	}

	bool openGroundTruth(const QString & path)
	{
		QString error;
		if(!model_.loadGroundTruth(path, &error))
		{
			showError(error);
			return false;
		}
		model_.align(scale_->isChecked());
		refresh();
		return true;
	}

	void setPlane(int i) {plane_->setCurrentIndex(i);}
	void setScale(bool on) {scale_->setChecked(on);}
	void setFrame(int i) {slider_->setValue(i); select(i);}
	QString statsText() const {return stats_->text();}

private:
	void showError(const QString & error)
	{
		std::cerr << error.toStdString() << std::endl;
		if(isVisible()) QMessageBox::warning(this, "COLMAP viewer", error);
	}

	void select(int index)
	{
		trajectory_->setSelected(index);
		plot_->setSelected(index);
		image_->setFrame(index);
	}

	void refresh()
	{
		trajectory_->fit();
		trajectory_->update();
		plot_->update();
		image_->setFrame(slider_->value());
		const Stats & s = model_.stats();
		const Intrinsics & K = model_.intrinsics();
		QString text = QString("<b>%1</b><br>%2 images, %3 points, camera %4 %5x%6 f=%7 c=(%8, %9), path %10 m")
				.arg(model_.directory()).arg(model_.frames().size()).arg(model_.points().size())
				.arg(QString::fromStdString(K.model)).arg(K.width).arg(K.height)
				.arg(K.fx, 0, 'f', 1).arg(K.cx, 0, 'f', 1).arg(K.cy, 0, 'f', 1).arg(s.pathLength, 0, 'f', 2);
		if(model_.hasGroundTruth())
		{
			text += QString("<br>Ground truth: %1 (%2 matched)<br>"
					"<b>Position error RMSE %3 cm</b>, mean %4, median %5, max %6 cm; scale estimate/GT %8 (%9 alignment)<br>"
					"Rotation error RMSE %7 deg; constant camera frame offset %10 deg, RMSE without it %11 deg")
					.arg(model_.groundTruthPath()).arg(s.matched)
					.arg(s.rmse*100, 0, 'f', 2).arg(s.mean*100, 0, 'f', 2).arg(s.median*100, 0, 'f', 2).arg(s.max*100, 0, 'f', 2)
					.arg(s.rotRmse, 0, 'f', 3).arg(s.scale, 0, 'f', 4).arg(scale_->isChecked() ? "similarity" : "rigid")
					.arg(s.rotOffset, 0, 'f', 2).arg(s.rotRmseNoOffset, 0, 'f', 3);
		}
		else
		{
			text += "<br>No ground truth (File > Load ground truth, or gt_camera_poses.txt in the folder)";
		}
		stats_->setText(text);
	}

	ColmapModel model_;
	TrajectoryView * trajectory_;
	ImageView * image_;
	ErrorPlot * plot_;
	QLabel * stats_;
	QComboBox * plane_;
	QCheckBox * scale_;
	QSlider * slider_;
};

int main(int argc, char ** argv)
{
	QApplication app(argc, argv);
	MainWindow window;
	QString dir, gt, screenshot;
	int frame = 0;
	for(int i=1; i<argc; ++i)
	{
		QString a = argv[i];
		if(a == "--gt" && i+1 < argc) gt = argv[++i];
		else if(a == "--screenshot" && i+1 < argc) screenshot = argv[++i];
		else if(a == "--frame" && i+1 < argc) frame = QString(argv[++i]).toInt();
		else if(a == "--scale") window.setScale(true);
		else if(a == "--plane" && i+1 < argc)
		{
			QString p = QString(argv[++i]).toLower();
			window.setPlane(p == "xz" ? 1 : p == "yz" ? 2 : 0);
		}
		else if(a == "-h" || a == "--help")
		{
			std::cout << "Usage: colmap_viewer [DIR] [--gt FILE] [--scale] [--frame N] [--plane xy|xz|yz] [--screenshot OUT.png]\n"
					"  DIR: COLMAP folder with sparse/0/{cameras,images,points3D}.txt and images/\n"
					"       (camera_poses.txt and gt_camera_poses.txt are read when present)\n"
					"  --gt: ground truth camera poses, TUM format \"stamp x y z qx qy qz qw [id]\"\n"
					"  --screenshot: render the window to a PNG and exit (works with QT_QPA_PLATFORM=offscreen)\n";
			return 0;
		}
		else dir = a;
	}
	if(!dir.isEmpty() && !window.openFolder(dir) && !screenshot.isEmpty()) return 1;
	if(!gt.isEmpty() && !window.openGroundTruth(gt) && !screenshot.isEmpty()) return 1;
	if(!dir.isEmpty()) window.setFrame(frame);
	if(!screenshot.isEmpty())
	{
		window.show();
		app.processEvents();
		window.setFrame(frame);
		app.processEvents();
		window.grab().save(screenshot);
		QString stats = window.statsText();
		stats.replace("<br>", "\n").remove(QRegularExpression("<[^>]*>"));
		std::cout << stats.toStdString() << std::endl;
		return 0;
	}
	window.show();
	return app.exec();
}
