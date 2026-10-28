/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// rtabmap-live_viewer: one Qt window for
//  1. real-time mapping: D435i (IR stereo + IMU) or a recording replayed in real time ->
//     our VIO + RTAB-Map SLAM, with the point cloud map, keyframe graph, loop closures,
//     camera pose and image shown while they are built; the map is saved when stopped.
//  2. real-time localization in a saved map, with the current pose and the map matches.
//  3. offline viewing of a map built before (File > Open map).
//  4. 3DGS / mesh of a map made from a recording (menu 3DGS·메쉬): runs
//     tools/GaussianSplatting/mesh_3dgs.ps1 in the background, shows the mesh over the map,
//     and opens the 3DGS model in the web browser (view_3dgs.py, viser).
// The processing is the usual RTAB-Map pipeline (SensorCaptureThread -> OdometryThread ->
// RtabmapThread) in this process; the 3D view is RTAB-Map's CloudViewer.
// Korean UI strings: this file is UTF-8 (built with /utf-8 on MSVC).

#include <rtabmap/core/Version.h>
#include <rtabmap/core/Rtabmap.h>
#ifdef RTABMAP_PYTHON
#include <rtabmap/core/PythonInterface.h>
#endif
#include <rtabmap/core/RtabmapThread.h>
#include <rtabmap/core/RtabmapEvent.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryThread.h>
#include <rtabmap/core/OdometryEvent.h>
#include <rtabmap/core/SensorCaptureThread.h>
#include <rtabmap/core/SensorEvent.h>
#include <rtabmap/core/CameraStereo.h>
#include <rtabmap/core/camera/CameraRealSense2.h>
#include <rtabmap/core/IMU.h>
#include <rtabmap/core/IMUFilter.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/gui/CloudViewer.h>
#include <rtabmap/utilite/UCv2Qt.h>
#include <rtabmap/utilite/UEventsHandler.h>
#include <rtabmap/utilite/UEventsManager.h>
#include <rtabmap/utilite/UEventsSender.h>
#include <rtabmap/utilite/UThread.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UStl.h>
#include <pcl/io/ply_io.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <QApplication>
#include <QMainWindow>
#include <QDockWidget>
#include <QGroupBox>
#include <QGridLayout>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QRadioButton>
#include <QLineEdit>
#include <QPushButton>
#include <QToolButton>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QLabel>
#include <QPlainTextEdit>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QTimer>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QCloseEvent>
#include <QElapsedTimer>
#include <QScreen>
#include <QPainter>
#include <QLocale>

#include <vtkVersionMacros.h>
#if VTK_MAJOR_VERSION > 9 || (VTK_MAJOR_VERSION==9 && VTK_MINOR_VERSION >= 1)
#include <QVTKRenderWidget.h>
#endif
#include <QSurfaceFormat>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <stdio.h>

using namespace rtabmap;

static QString tr8(const char * s) {return QString::fromUtf8(s);}

// ================================================================ processing

struct CloudParams
{
	int decimation = 4;
	float maxDepth = 4.0f;
	float minDepth = 0.2f;
	float voxel = 0.03f;
	ParametersMap stereo;
	CameraModel color;   // color camera of the keyframes' user data (D435i RGB), if any
};

// Color image attached as user data (JPEG bytes in a CV_8SC1 row, see CameraRealSense2::setColorWithIR()).
static cv::Mat colorImage(const SensorData & data)
{
	cv::Mat ud = data.userDataRaw();
	if(ud.empty() && !data.userDataCompressed().empty())
	{
		data.uncompressDataConst(0, 0, 0, &ud);
	}
	if(ud.depth() == CV_8S && ud.channels() == 1 && ud.rows == 1 && ud.cols > 4 && ud.data[0] == 0xFF && ud.data[1] == 0xD8)
	{
		return cv::imdecode(cv::Mat(1, ud.cols, CV_8UC1, ud.data), cv::IMREAD_COLOR);
	}
	return cv::Mat();
}

// Colors the points (base frame) seen by the color camera.
static void colorize(pcl::PointCloud<pcl::PointXYZRGB> & cloud, const cv::Mat & bgr, const CameraModel & model)
{
	if(bgr.empty() || !model.isValidForProjection() || model.localTransform().isNull())
	{
		return;
	}
	Eigen::Affine3f toCam = model.localTransform().inverse().toEigen3f();
	float sx = model.imageWidth()>0?float(bgr.cols)/float(model.imageWidth()):1.0f;
	float sy = model.imageHeight()>0?float(bgr.rows)/float(model.imageHeight()):1.0f;
	float fx = model.fx()*sx, fy = model.fy()*sy, cx = model.cx()*sx, cy = model.cy()*sy;
	for(pcl::PointXYZRGB & p : cloud.points)
	{
		Eigen::Vector3f c = toCam * Eigen::Vector3f(p.x, p.y, p.z);
		if(c[2] > 0.05f)
		{
			int u = int(fx*c[0]/c[2] + cx + 0.5f);
			int v = int(fy*c[1]/c[2] + cy + 0.5f);
			if(u >= 0 && v >= 0 && u < bgr.cols && v < bgr.rows)
			{
				const cv::Vec3b & q = bgr.at<cv::Vec3b>(v, u);
				p.b = q[0]; p.g = q[1]; p.r = q[2];
			}
		}
	}
}

// Point cloud of a keyframe in its base frame: from stereo (or depth) when the images are
// there, else from its 3D visual words.
static pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyframeCloud(const Signature & s, const CloudParams & p)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	SensorData data = s.sensorData();
	if((data.imageRaw().empty() && !data.imageCompressed().empty()) ||
	   (data.userDataRaw().empty() && !data.userDataCompressed().empty()))
	{
		data.uncompressData();
	}
	if(!data.imageRaw().empty() && !data.depthOrRightRaw().empty())
	{
		cloud = util3d::cloudRGBFromSensorData(data, p.decimation, p.maxDepth, p.minDepth, 0, p.stereo);
		cloud = util3d::removeNaNFromPointCloud(cloud); // organized cloud with NaNs
		colorize(*cloud, colorImage(data), p.color);
		if(cloud->size() && p.voxel > 0.0f)
		{
			cloud = util3d::voxelize(cloud, p.voxel);
		}
	}
	if(cloud->empty())
	{
		for(std::multimap<int, int>::const_iterator iter=s.getWords().begin(); iter!=s.getWords().end(); ++iter)
		{
			if(iter->second < (int)s.getWords3().size())
			{
				const cv::Point3f & pt = s.getWords3()[iter->second];
				if(std::isfinite(pt.x) && (pt.x != 0 || pt.y != 0 || pt.z != 0))
				{
					pcl::PointXYZRGB q;
					q.x = pt.x; q.y = pt.y; q.z = pt.z;
					q.r = q.g = q.b = 220;
					cloud->push_back(q);
				}
			}
		}
	}
	return cloud;
}

// What the engine reports, called from its threads.
class EngineListener
{
public:
	virtual ~EngineListener() {}
	virtual void engineStatus(const QString & text, int level) = 0; // 0 info, 1 warning, 2 error
	virtual void engineOdometry(const Transform & odomPose, const Transform & mapPose, const OdometryInfo & info, const cv::Mat & image, const cv::Mat & color, double stamp) = 0;
	virtual void engineGraph(const std::map<int, Transform> & poses, const std::multimap<int, Link> & links,
			bool fullMap, int loopId, int proxId, float slamMs, const Transform & localizedPose) = 0;
	virtual void engineCloud(int id, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud) = 0;
};

// Computes keyframe clouds in its own thread, each keyframe once.
class CloudWorker
{
public:
	CloudWorker(EngineListener * listener, const CloudParams & params) :
		listener_(listener), params_(params), quit_(false), busy_(0)
	{
		thread_ = std::thread(&CloudWorker::loop, this);
	}
	~CloudWorker() {quit();}

	void add(const Signature & s)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if(s.id() <= 0 || done_.find(s.id()) != done_.end())
		{
			return;
		}
		done_.insert(s.id());
		queue_.push_back(s);
		cond_.notify_one();
	}

	void waitIdle()
	{
		while(true)
		{
			{
				std::lock_guard<std::mutex> lock(mutex_);
				if((queue_.empty() && busy_ == 0) || quit_)
				{
					return;
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}
	}

	void quit()
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			quit_ = true;
			cond_.notify_one();
		}
		if(thread_.joinable())
		{
			thread_.join();
		}
	}

private:
	void loop()
	{
		while(true)
		{
			Signature s;
			{
				std::unique_lock<std::mutex> lock(mutex_);
				cond_.wait(lock, [this]{return quit_ || !queue_.empty();});
				if(quit_)
				{
					return;
				}
				s = queue_.front();
				queue_.pop_front();
				++busy_;
			}
			listener_->engineCloud(s.id(), keyframeCloud(s, params_));
			std::lock_guard<std::mutex> lock(mutex_);
			--busy_;
		}
	}

	EngineListener * listener_;
	CloudParams params_;
	std::mutex mutex_;
	std::condition_variable cond_;
	std::deque<Signature> queue_;
	std::set<int> done_;
	bool quit_;
	int busy_;
	std::thread thread_;
};

// Replays a EuRoC-layout folder (mav0/cam0, cam1, imu0, e.g. from realsense_bag_to_euroc.py)
// in real time: IMU samples then the stereo frame are posted in stamp order, like a camera.
class EurocReplay : public UThread, public UEventsSender
{
public:
	EurocReplay(const std::string & path, const ParametersMap & parameters, double rate) :
		path_(path), parameters_(parameters), rate_(rate>0?rate:1.0), capture_(0), finished_(false)
	{}
	virtual ~EurocReplay()
	{
		join(true);
		delete capture_;
	}

	// EuRoC sensor.yaml: intrinsics, distortion and T_BS (camera in the body frame) as local transform
	static bool loadCamera(const std::string & calibPath, const std::string & name, CameraModel & model, std::string & error)
	{
		if(!UFile::exists(calibPath))
		{
			error = "missing " + calibPath;
			return false;
		}
		YAML::Node config = YAML::LoadFile(calibPath);
		YAML::Node data = config["T_BS"]["data"];
		YAML::Node resolution = config["resolution"];
		YAML::Node intrinsics = config["intrinsics"];
		YAML::Node dist = config["distortion_coefficients"];
		if(data.size() != 16 || resolution.size() != 2 || intrinsics.size() != 4)
		{
			error = "invalid calibration " + calibPath;
			return false;
		}
		cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
		K.at<double>(0,0) = intrinsics[0].as<double>();
		K.at<double>(1,1) = intrinsics[1].as<double>();
		K.at<double>(0,2) = intrinsics[2].as<double>();
		K.at<double>(1,2) = intrinsics[3].as<double>();
		cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);
		cv::Mat P = cv::Mat::zeros(3, 4, CV_64FC1);
		K.copyTo(cv::Mat(P, cv::Range(0,3), cv::Range(0,3)));
		cv::Mat D = cv::Mat::zeros(1, dist.size()?dist.size():4, CV_64FC1);
		for(unsigned int i=0; i<dist.size(); ++i)
		{
			D.at<double>(i) = dist[i].as<double>();
		}
		Transform t(data[0].as<float>(), data[1].as<float>(), data[2].as<float>(), data[3].as<float>(),
					data[4].as<float>(), data[5].as<float>(), data[6].as<float>(), data[7].as<float>(),
					data[8].as<float>(), data[9].as<float>(), data[10].as<float>(), data[11].as<float>());
		model = CameraModel(name, cv::Size(resolution[0].as<int>(),resolution[1].as<int>()), K, D, R, P, t);
		return true;
	}

	bool init(const std::string & calibDir, std::string & error)
	{
		std::vector<CameraModel> models(2);
		for(int k=0; k<2; ++k)
		{
			if(!loadCamera(path_ + (k==0?"/mav0/cam0/sensor.yaml":"/mav0/cam1/sensor.yaml"), "live_replay_calib", models[k], error))
			{
				return false;
			}
		}
		StereoCameraModel model("live_replay_calib", models[0], models[1], models[1].localTransform().inverse() * models[0].localTransform());
		if(!model.save(calibDir, false))
		{
			error = "cannot save the calibration to " + calibDir;
			return false;
		}
		// same frames as rtabmap-euroc_dataset
		baseToImu_ = Transform(0,0,1,0, 0,-1,0,0, 1,0,0,0);
		CameraStereoImages * camera = new CameraStereoImages(
				path_+"/mav0/cam0/data",
				path_+"/mav0/cam1/data",
				true,
				0.0f,
				baseToImu_*models[0].localTransform()*CameraModel::opticalRotation().inverse());
		camera->setTimestamps(true, "", false);
		capture_ = new SensorCaptureThread(camera, parameters_);
		capture_->enableIMUFiltering(1, parameters_);
		if(!camera->init(calibDir, "live_replay_calib"))
		{
			error = "cannot read the images of " + path_;
			return false;
		}
		totalFrames_ = (int)camera->filenames().size();
		imu_.open((path_+"/mav0/imu0/data.csv").c_str());
		if(!imu_.good())
		{
			error = "missing " + path_ + "/mav0/imu0/data.csv";
			return false;
		}
		std::string header;
		std::getline(imu_, header);

		// color stream (realsense_bag_to_euroc.py --rgb)
		std::string rgbCsv = path_+"/mav0/rgb/data.csv";
		std::string err;
		if(UFile::exists(rgbCsv) && loadCamera(path_+"/mav0/rgb/sensor.yaml", "live_replay_color", colorModel_, err))
		{
			colorModel_.setLocalTransform(baseToImu_*colorModel_.localTransform());
			std::ifstream f(rgbCsv.c_str());
			std::string line;
			while(std::getline(f, line))
			{
				std::string ns = uSplit(line, ',').front();
				if(ns.size() > 9 && ns[0] != '#')
				{
					double stamp = double(uStr2Int(ns.substr(0, ns.size()-9))) + double(uStr2Int(ns.substr(ns.size()-9, 9)))*1e-9;
					colorFiles_.insert(std::make_pair(stamp, path_+"/mav0/rgb/data/"+ns+".png"));
				}
			}
		}
		if(colorFiles_.empty())
		{
			colorModel_ = CameraModel();
		}
		return true;
	}

	const CameraModel & colorModel() const {return colorModel_;}
	bool finished() const {return finished_;}
	int totalFrames() const {return totalFrames_;}

private:
	virtual void mainLoopBegin()
	{
		timer_.start();
		firstStamp_ = -1.0;
		pendingImu_ = false;
	}

	virtual void mainLoop()
	{
		SensorCaptureInfo info;
		SensorData data = capture_->camera()->takeData(&info);
		if(!data.isValid())
		{
			finished_ = true;
			this->kill();
			return;
		}
		if(firstStamp_ < 0)
		{
			firstStamp_ = data.stamp();
		}
		// real time pacing
		double target = (data.stamp() - firstStamp_) / rate_;
		double now = timer_.elapsed();
		if(target > now)
		{
			uSleep(int((target-now)*1000.0));
		}
		// IMU samples up to a bit after the frame: OdometryThread waits for an IMU sample
		// newer than the frame before processing it, as with a live IMU
		while(true)
		{
			if(!pendingImu_)
			{
				std::string line;
				if(!std::getline(imu_, line))
				{
					break;
				}
				std::stringstream stream(line);
				std::string s;
				std::getline(stream, s, ',');
				if(s.size() < 10)
				{
					continue;
				}
				double stamp = double(uStr2Int(s.substr(0, s.size()-9))) + double(uStr2Int(s.substr(s.size()-9, 9)))*1e-9;
				cv::Vec3d gyr, acc;
				for(int j=0; j<3; ++j) {std::getline(stream, s, ','); gyr[j] = uStr2Double(s);}
				for(int j=0; j<3; ++j) {std::getline(stream, s, ','); acc[j] = uStr2Double(s);}
				imuData_ = SensorData(IMU(gyr, cv::Mat::eye(3,3,CV_64FC1)*0.0001, acc, cv::Mat::eye(3,3,CV_64FC1)*0.0001, baseToImu_), 0, stamp);
				pendingImu_ = true;
			}
			if(imuData_.stamp() > data.stamp() + 0.02)
			{
				break;
			}
			pendingImu_ = false;
			if(imuData_.stamp() - firstStamp_ + 1.0 > 0.0)
			{
				capture_->postUpdate(&imuData_);
				this->post(new IMUEvent(imuData_.imu(), imuData_.stamp()));
			}
		}
		if(!colorFiles_.empty())
		{
			// closest color frame, as user data (same as CameraRealSense2::setColorWithIR())
			std::map<double, std::string>::iterator iter = colorFiles_.lower_bound(data.stamp());
			if(iter != colorFiles_.begin() && (iter == colorFiles_.end() || fabs(std::prev(iter)->first - data.stamp()) < fabs(iter->first - data.stamp())))
			{
				--iter;
			}
			if(iter != colorFiles_.end() && fabs(iter->first - data.stamp()) < 0.05)
			{
				cv::Mat color = cv::imread(iter->second, cv::IMREAD_COLOR);
				std::vector<unsigned char> jpg;
				if(!color.empty() && cv::imencode(".jpg", color, jpg, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 90}))
				{
					data.setUserData(cv::Mat(1, (int)jpg.size(), CV_8SC1, jpg.data()).clone()); // CV_8SC1: a CV_8UC1 row would be taken as already zlib-compressed
				}
			}
		}
		capture_->postUpdate(&data, &info);
		this->post(new SensorEvent(data, info));
	}

	std::string path_;
	ParametersMap parameters_;
	double rate_;
	SensorCaptureThread * capture_;
	Transform baseToImu_;
	std::ifstream imu_;
	SensorData imuData_;
	bool pendingImu_ = false;
	UTimer timer_;
	double firstStamp_ = -1.0;
	std::atomic<bool> finished_;
	int totalFrames_ = 0;
	CameraModel colorModel_;
	std::map<double, std::string> colorFiles_;
};

struct EngineSettings
{
	enum Mode {kMapping, kLocalization, kView};
	Mode mode = kMapping;
	std::string db;          // output map (mapping), map (localization and view)
	bool realsense = true;
	std::string euroc;       // recording (EuRoC layout) when !realsense
	double rate = 1.0;
	int width = 848, height = 480, fps = 30;
	ParametersMap parameters;
	CloudParams cloud;
};

// Runs the pipeline. start() and stop() block (camera, map loading, map saving): call them
// from a worker thread.
class LiveEngine : public UEventsHandler
{
public:
	LiveEngine(EngineListener * listener) :
		listener_(listener), localization_(false), clouds_(0), capture_(0), replay_(0),
		odomThread_(0), rtabmapThread_(0), running_(false), mapCorrection_(Transform::getIdentity())
	{}
	virtual ~LiveEngine() {stop();}

	bool running() const {return running_;}
	bool sourceEnded() const
	{
		return running_ && ((replay_ && replay_->finished()) || (capture_ && !capture_->isRunning()));
	}
	const EngineSettings & settings() const {return settings_;}

	static std::string tmpDir()
	{
		std::string d = QDir::tempPath().toStdString() + "/rtabmap_live";
		if(!UDirectory::exists(d))
		{
			UDirectory::makeDir(d);
		}
		return d;
	}

	// Offline map: graph and clouds, then returns.
	bool view(const EngineSettings & settings, std::string & error)
	{
		if(!UFile::exists(settings.db))
		{
			error = "map not found";
			return false;
		}
		ParametersMap params = settings.parameters;
		CloudParams cloudParams = settings.cloud;
		cloudParams.color = loadColorModel(settings.db);
		uInsert(params, ParametersPair(Parameters::kMemIncrementalMemory(), "false"));
		uInsert(params, ParametersPair(Parameters::kMemInitWMWithAllNodes(), "true"));
		uInsert(params, ParametersPair(Parameters::kRtabmapWorkingDirectory(), tmpDir()));
		std::string tmp = tmpDir() + "/view.db";
		UFile::erase(tmp);
		UFile::copy(settings.db, tmp);
		{
			CloudWorker clouds(listener_, cloudParams);
			Rtabmap rtabmap;
			rtabmap.init(params, tmp);
			sendMap(rtabmap, clouds);
			clouds.waitIdle();
			rtabmap.close(false);
		}
		UFile::erase(tmp);
		return true;
	}

	// color camera calibration saved next to a map: MAP_color.yaml
	static std::string colorModelName(const std::string & db)
	{
		std::string name = UFile::getName(db);
		return name.substr(0, name.find_last_of('.')) + "_color";
	}
	static CameraModel loadColorModel(const std::string & db)
	{
		CameraModel model;
		std::string dir = UDirectory::getDir(db);
		if(UFile::exists(dir + "/" + colorModelName(db) + ".yaml"))
		{
			model.load(dir, colorModelName(db));
		}
		return model;
	}
	const CameraModel & colorModel() const {return colorModel_;}

	bool start(const EngineSettings & settings, std::string & error)
	{
		stop();
		settings_ = settings;
		localization_ = settings.mode == EngineSettings::kLocalization;
		ParametersMap parameters = settings.parameters;
		workingDb_ = settings.db;
		if(localization_)
		{
			if(!UFile::exists(settings.db))
			{
				error = "map not found: " + settings.db;
				return false;
			}
			uInsert(parameters, ParametersPair(Parameters::kMemIncrementalMemory(), "false"));
			uInsert(parameters, ParametersPair(Parameters::kMemInitWMWithAllNodes(), "true"));
			// localize in a copy so that the map stays untouched
			workingDb_ = tmpDir() + "/localization.db";
			UFile::erase(workingDb_);
			UFile::copy(settings.db, workingDb_);
		}
		else
		{
			std::string dir = UDirectory::getDir(settings.db);
			if(!dir.empty() && !UDirectory::exists(dir))
			{
				UDirectory::makeDir(dir);
			}
			UFile::erase(settings.db);
		}
		uInsert(parameters, ParametersPair(Parameters::kRtabmapWorkingDirectory(), tmpDir()));
		std::string calibDir = tmpDir();

		// sensor
		if(!settings.realsense)
		{
			replay_ = new EurocReplay(settings.euroc, parameters, settings.rate);
			if(!replay_->init(calibDir, error))
			{
				delete replay_;
				replay_ = 0;
				return false;
			}
			listener_->engineStatus(tr8("녹화 재생: %1 (%2 프레임, %3배속, %4)")
					.arg(QString::fromStdString(settings.euroc)).arg(replay_->totalFrames()).arg(settings.rate)
					.arg(replay_->colorModel().isValidForProjection()?tr8("컬러 있음"):tr8("컬러 없음: --rgb로 변환한 녹화가 아님")), 0);
			colorModel_ = replay_->colorModel();
		}
		else
		{
			if(!CameraRealSense2::available())
			{
				error = "this RTAB-Map build has no RealSense2 driver";
				return false;
			}
			CameraRealSense2 * camera = new CameraRealSense2("", 0, Transform::getIdentity());
			camera->setEmitterEnabled(false);
			camera->setIRFormat(true, false);
			camera->setResolution(settings.width, settings.height, settings.fps);
			camera->setInterIMUPublishing(true, IMUFilter::create(IMUFilter::kComplementaryFilter, parameters), false);
			camera->setColorWithIR(true, 640, 480);
			listener_->engineStatus(tr8("D435i 연결 중..."), 0);
			if(!camera->init(calibDir, ""))
			{
				error = "cannot open the D435i (USB 3 port? used by another program?)";
				delete camera;
				return false;
			}
			listener_->engineStatus(tr8("D435i 시작: IR 스테레오 %1x%2 %3 Hz + IMU (시리얼 %4)")
					.arg(settings.width).arg(settings.height).arg(settings.fps).arg(QString::fromStdString(camera->getSerial())), 0);
			colorModel_ = camera->colorModel();
			if(!colorModel_.isValidForProjection())
			{
				listener_->engineStatus(tr8("컬러 스트림 없이 진행합니다 (IR만)."), 1);
			}
			capture_ = new SensorCaptureThread(camera, parameters);
		}

		// SLAM
		CloudParams cloudParams = settings.cloud;
		cloudParams.color = colorModel_;
		if(localization_)
		{
			// map clouds: color camera of the map
			cloudParams.color = loadColorModel(settings.db);
		}
		else if(colorModel_.isValidForProjection())
		{
			CameraModel saved = colorModel_;
			saved.setName(colorModelName(settings.db));
			saved.save(UDirectory::getDir(settings.db));
		}
		clouds_ = new CloudWorker(listener_, cloudParams);
		Rtabmap * rtabmap = new Rtabmap();
		listener_->engineStatus(localization_?tr8("맵 불러오는 중..."):tr8("새 맵: %1").arg(QString::fromStdString(settings.db)), 0);
		rtabmap->init(parameters, workingDb_);
		rtabmapRef_ = rtabmap;
		if(localization_)
		{
			sendMap(*rtabmap, *clouds_);
			listener_->engineStatus(tr8("맵 불러옴 (키프레임 %1개). 로컬라이제이션 시작").arg(rtabmap->getWMSize()), 0);
		}
		mapCorrection_ = Transform::getIdentity();
		rtabmapThread_ = new RtabmapThread(rtabmap);
		{
			float detectionRate = Parameters::defaultRtabmapDetectionRate();
			Parameters::parse(parameters, Parameters::kRtabmapDetectionRate(), detectionRate);
			rtabmapThread_->setDetectorRate(detectionRate);
		}
		ParametersMap odomParameters = parameters;
		odomParameters.erase(Parameters::kRtabmapPublishRAMUsage());
		odomThread_ = new OdometryThread(Odometry::create(odomParameters));

		UEventsManager::addHandler(odomThread_);
		UEventsManager::addHandler(rtabmapThread_);
		UEventsManager::addHandler(this);
		// frames go to odometry only (RtabmapThread would take them as frames without odometry)
		if(replay_)
		{
			UEventsManager::createPipe(replay_, odomThread_, "SensorEvent");
			UEventsManager::createPipe(replay_, odomThread_, "IMUEvent");
		}
		else
		{
			UEventsManager::createPipe(capture_, odomThread_, "SensorEvent");
		}
		rtabmapThread_->start();
		odomThread_->start();
		if(replay_)
		{
			replay_->start();
		}
		else
		{
			capture_->start();
		}
		running_ = true;
		return true;
	}

	// Stops and saves the map (mapping). Returns the number of keyframes of the saved map.
	int stop()
	{
		if(!rtabmapThread_)
		{
			return 0;
		}
		running_ = false;
		if(replay_)
		{
			replay_->join(true);
		}
		if(capture_)
		{
			capture_->join(true);
		}
		odomThread_->join(true);
		UEventsManager::removeHandler(odomThread_);
		UEventsManager::removeHandler(this);
		UEventsManager::removeHandler(rtabmapThread_);
		if(replay_) UEventsManager::removeAllPipes(replay_);
		if(capture_) UEventsManager::removeAllPipes(capture_);
		rtabmapThread_->join(true);
		clouds_->waitIdle();
		int nodes = 0;
		if(!localization_)
		{
			// rtabmap is deleted by the thread's close()
			Rtabmap * rtabmap = rtabmapRef_;
			if(rtabmap)
			{
				std::map<int, Transform> poses;
				std::multimap<int, Link> links;
				rtabmap->getGraph(poses, links, true, true);
				nodes = (int)poses.size();
			}
		}
		rtabmapThread_->close(!localization_);
		delete rtabmapThread_;
		rtabmapThread_ = 0;
		rtabmapRef_ = 0;
		delete odomThread_;
		odomThread_ = 0;
		delete replay_;
		replay_ = 0;
		delete capture_;
		capture_ = 0;
		clouds_->quit();
		delete clouds_;
		clouds_ = 0;
		if(localization_)
		{
			UFile::erase(workingDb_);
		}
		return nodes;
	}

protected:
	virtual bool handleEvent(UEvent * event)
	{
		if(event->getClassName().compare("OdometryEvent") == 0)
		{
			const OdometryEvent & e = *(OdometryEvent*)event;
			Transform mapPose;
			{
				std::lock_guard<std::mutex> lock(mutex_);
				if(!e.pose().isNull())
				{
					mapPose = mapCorrection_ * e.pose();
				}
			}
			// color image ~10 Hz (JPEG decoding)
			cv::Mat color;
			if(colorTimer_.elapsed() > 0.1)
			{
				color = colorImage(e.data());
				if(!color.empty())
				{
					colorTimer_.restart();
				}
			}
			listener_->engineOdometry(e.pose(), mapPose, e.info(), e.data().imageRaw(), color, e.data().stamp());
		}
		else if(event->getClassName().compare("RtabmapEvent") == 0)
		{
			const Statistics & stats = ((RtabmapEvent*)event)->getStats();
			{
				std::lock_guard<std::mutex> lock(mutex_);
				if(!stats.mapCorrection().isNull())
				{
					mapCorrection_ = stats.mapCorrection();
				}
			}
			if(!localization_ && clouds_)
			{
				// in localization mode the new nodes are temporary, only the map is shown
				for(std::map<int, Signature>::const_iterator iter=stats.getSignaturesData().begin(); iter!=stats.getSignaturesData().end(); ++iter)
				{
					clouds_->add(iter->second);
				}
			}
			std::map<int, Transform> poses = stats.poses();
			Transform localized;
			if(localization_)
			{
				std::map<int, Transform>::iterator iter = poses.find(stats.refImageId());
				if(iter != poses.end())
				{
					localized = iter->second;
					poses.erase(iter);
				}
			}
			listener_->engineGraph(poses, stats.constraints(), false, stats.loopClosureId(), stats.proximityDetectionId(),
					uValue(stats.data(), Statistics::kTimingTotal(), 0.0f), localized);
		}
		return false;
	}

private:
	void sendMap(Rtabmap & rtabmap, CloudWorker & clouds)
	{
		std::map<int, Transform> poses;
		std::multimap<int, Link> links;
		std::map<int, Signature> signatures;
		rtabmap.getGraph(poses, links, true, true, &signatures, true, false, true, false, true, false);
		listener_->engineGraph(poses, links, true, 0, 0, 0.0f, Transform());
		for(std::map<int, Signature>::iterator iter=signatures.begin(); iter!=signatures.end(); ++iter)
		{
			if(poses.find(iter->first) != poses.end())
			{
				clouds.add(iter->second);
			}
		}
	}

public:
	// set by start(), the thread owns it
	Rtabmap * rtabmapRef_ = 0;

private:
	EngineListener * listener_;
	EngineSettings settings_;
	bool localization_;
	std::string workingDb_;
	CloudWorker * clouds_;
	SensorCaptureThread * capture_;
	EurocReplay * replay_;
	OdometryThread * odomThread_;
	RtabmapThread * rtabmapThread_;
	std::atomic<bool> running_;
	std::mutex mutex_;
	Transform mapCorrection_;
	CameraModel colorModel_;
	UTimer colorTimer_;
};

// ================================================================ window

struct OdomSample
{
	Transform mapPose;
	bool lost = false;
	int inliers = 0;
	int features = 0;
	float ms = 0.0f;
	int frames = 0;
	QImage image;
	bool updated = false;
};

class LiveWindow : public QMainWindow, public EngineListener
{
public:
	LiveWindow(const QStringList & args) :
		engine_(this),
		settings_("rtabmap", "live_viewer"),
		busy_(false),
		mode_(-1)
	{
		setWindowTitle(tr8("RTAB-Map 실시간 매핑 / 로컬라이제이션"));
		resize(1500, 900);
		buildUi();
		resetLive();
		imageTimer_.start();
		QTimer * timer = new QTimer(this);
		connect(timer, &QTimer::timeout, this, [this]{poll();});
		timer->start(200);
		QTimer * render = new QTimer(this);
		connect(render, &QTimer::timeout, this, [this]{this->render();});
		render->start(40);
		log(tr8("RTAB-Map %1").arg(RTABMAP_VERSION));
		if(!CameraRealSense2::available())
		{
			log(tr8("이 빌드에는 RealSense2 드라이버가 없습니다: 녹화 파일만 쓸 수 있습니다."), 1);
		}
		standby();
		autoArgs_ = args;
	}

	void runAutoArgs()
	{
		// --auto map|loc|view --source realsense|REC --db OUT_DIR_OR_MAP [--rate F] [--screenshot PNG] [--exit]
		// the window then waits in standby for the next task; only --exit (scripts) closes it at the end
		QString mode, source, db, open;
		for(int i=1; i<autoArgs_.size(); ++i)
		{
			QString a = autoArgs_[i];
			bool v = i+1 < autoArgs_.size();
			if(a == "--auto" && v) mode = autoArgs_[++i];
			else if(a == "--source" && v) source = autoArgs_[++i];
			else if(a == "--db" && v) db = autoArgs_[++i];
			else if(a == "--rate" && v) rate_->setValue(autoArgs_[++i].toDouble());
			else if(a == "--features" && v) features_->setCurrentIndex(autoArgs_[++i] == "aliked"?1:0);
			else if(a == "--screenshot" && v) screenshot_ = autoArgs_[++i];
			else if(a == "--exit") exitWhenDone_ = true;
			else if(a == "--open" && v) open = autoArgs_[++i];
			else if(a == "--mesh" && v) mesh_ = autoArgs_[++i];
			else if(a == "--splats" && v) splats_ = autoArgs_[++i];
		}
		if(!open.isEmpty())
		{
			openMap(open);
		}
		if(!mesh_.isEmpty() || !splats_.isEmpty())
		{
			// after the map (--open) is loaded
			QTimer * wait = new QTimer(this);
			connect(wait, &QTimer::timeout, this, [this, wait]{
				if(busy_) return;
				wait->deleteLater();
				if(!mesh_.isEmpty()) {loadMesh(mesh_); fitView();}
				if(!splats_.isEmpty()) openSplats(splats_);
				if(!screenshot_.isEmpty())
				{
					QString shot = screenshot_;
					screenshot_.clear();
					QTimer::singleShot(1500, this, [this, shot]{saveScreenshot(shot); if(exitWhenDone_) close();});
				}
			});
			wait->start(500);
			return;
		}
		if(!open.isEmpty())
		{
			return;
		}
		if(mode.isEmpty())
		{
			return;
		}
		autoMode_ = true;
		if(!source.isEmpty())
		{
			srcCamera_->setChecked(source == "realsense");
			srcFile_->setChecked(source != "realsense");
			if(source != "realsense") recPath_->setText(source);
		}
		if(mode == "view") openMap(db);
		else if(mode == "loc") {locMap_->setText(db); start(EngineSettings::kLocalization);}
		else {if(!db.isEmpty()) outDir_->setText(db); start(EngineSettings::kMapping);}
	}

protected:
	virtual void closeEvent(QCloseEvent * event)
	{
		saveSettings();
		if(engine_.running() || busy_)
		{
			if(!exitWhenDone_ && QMessageBox::question(this, tr8("종료"),
					tr8("실행 중입니다. 정지하고 종료할까요? (매핑 중이면 맵을 저장합니다)")) != QMessageBox::Yes)
			{
				event->ignore();
				return;
			}
			if(worker_.joinable()) worker_.join();
			engine_.stop();
		}
		if(gsProc_ && QMessageBox::question(this, tr8("종료"),
				tr8("3DGS·메쉬를 만드는 중입니다. 중단하고 종료할까요?")) != QMessageBox::Yes)
		{
			event->ignore();
			return;
		}
		if(worker_.joinable()) worker_.join();
		killTree(gsProc_);
		killTree(splatProc_);
		event->accept();
	}

	// ------------------------------------------------ engine callbacks (engine threads)

	virtual void engineStatus(const QString & text, int level)
	{
		gui([this, text, level]{log(text, level);});
	}

	virtual void engineOdometry(const Transform & odomPose, const Transform & mapPose, const OdometryInfo & info, const cv::Mat & image, const cv::Mat & color, double stamp)
	{
		// only the latest frame is drawn (by the render timer), so that a slow display
		// never queues up behind the camera
		QImage img;
		if(!color.empty())
		{
			lastColor_ = color;
			lastColorStamp_ = stamp;
		}
		if(!image.empty() && imageTimer_.elapsed() > 100)
		{
			imageTimer_.restart();
			// show the color camera when it is there (decoded ~10 Hz), else the IR image
			cv::Mat thumb = !lastColor_.empty() && fabs(stamp - lastColorStamp_) < 0.5?lastColor_:image;
			if(thumb.cols > 640)
			{
				cv::resize(thumb, thumb, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
			}
			img = uCvMat2QImage(thumb, true).copy();
		}
		std::lock_guard<std::mutex> lock(odomMutex_);
		odom_.mapPose = mapPose;
		odom_.lost = odomPose.isNull();
		odom_.inliers = info.reg.inliers;
		odom_.features = info.features;
		odom_.ms = info.timeEstimation*1000.0f;
		if(!img.isNull())
		{
			odom_.image = img;
		}
		++odom_.frames;
		odom_.updated = true;
	}

	virtual void engineGraph(const std::map<int, Transform> & poses, const std::multimap<int, Link> & links,
			bool fullMap, int loopId, int proxId, float slamMs, const Transform & localizedPose)
	{
		std::map<int, Transform> p;
		for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
		{
			if(iter->first > 0 && !iter->second.isNull()) p.insert(*iter);
		}
		std::vector<std::pair<std::pair<int, int>, int> > l;
		std::set<std::pair<int, int> > seen;
		for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
		{
			const Link & link = iter->second;
			if(link.type() == Link::kGlobalClosure || link.type() == Link::kLocalSpaceClosure ||
			   link.type() == Link::kLocalTimeClosure || link.type() == Link::kUserClosure)
			{
				std::pair<int, int> key(std::min(link.from(), link.to()), std::max(link.from(), link.to()));
				if(key.first > 0 && key.first != key.second && seen.insert(key).second)
				{
					l.push_back(std::make_pair(key, (int)link.type()));
				}
			}
		}
		gui([this, p, l, fullMap, loopId, proxId, slamMs, localizedPose]{onGraph(p, l, fullMap, loopId, proxId, slamMs, localizedPose);});
	}

	virtual void engineCloud(int id, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud)
	{
		gui([this, id, cloud]{onCloud(id, cloud);});
	}

private:
	template<typename F> void gui(F f)
	{
		QMetaObject::invokeMethod(this, f, Qt::QueuedConnection);
	}

	// ------------------------------------------------ UI

	void buildUi()
	{
		viewer_ = new CloudViewer(this);
		viewer_->setBackgroundColor(QColor(22, 24, 28));
		viewer_->setGridShown(true);
		viewer_->setTrajectoryShown(true);
		viewer_->setCameraTargetFollow(true);
		setCentralWidget(viewer_);

		// left: controls
		QWidget * left = new QWidget;
		QVBoxLayout * lay = new QVBoxLayout(left);

		QGroupBox * srcBox = new QGroupBox(tr8("입력"));
		QGridLayout * sl = new QGridLayout(srcBox);
		srcCamera_ = new QRadioButton(tr8("D435i 카메라 (실시간)"));
		srcFile_ = new QRadioButton(tr8("녹화 파일 재생"));
		bool camera = settings_.value("source", "camera").toString() == "camera";
		srcCamera_->setChecked(camera);
		srcFile_->setChecked(!camera);
		recPath_ = new QLineEdit(settings_.value("recording").toString());
		recPath_->setPlaceholderText(tr8("녹화 .bag / .db3 또는 EuRoC 폴더"));
		QToolButton * recBtn = new QToolButton;
		recBtn->setText("...");
		connect(recBtn, &QToolButton::clicked, this, [this]{browseRecording();});
		rate_ = new QDoubleSpinBox;
		rate_->setRange(0.1, 4.0);
		rate_->setSingleStep(0.25);
		rate_->setValue(settings_.value("rate", 1.0).toDouble());
		rate_->setSuffix(tr8(" 배속"));
		sl->addWidget(srcCamera_, 0, 0, 1, 3);
		sl->addWidget(srcFile_, 1, 0, 1, 3);
		sl->addWidget(recPath_, 2, 0, 1, 2);
		sl->addWidget(recBtn, 2, 2);
		sl->addWidget(new QLabel(tr8("재생 속도")), 3, 0);
		sl->addWidget(rate_, 3, 1, 1, 2);
		lay->addWidget(srcBox);

		QGroupBox * optBox = new QGroupBox(tr8("처리 설정"));
		QFormLayout * ol = new QFormLayout(optBox);
		features_ = new QComboBox;
		features_->addItem(tr8("ORB (CPU)"));
		features_->addItem(tr8("ALIKED + LightGlue (GPU)"));
		features_->setCurrentIndex(settings_.value("features", 0).toInt());
		timeOffset_ = new QDoubleSpinBox;
		timeOffset_->setRange(-0.2, 0.2);
		timeOffset_->setDecimals(3);
		timeOffset_->setSingleStep(0.001);
		timeOffset_->setValue(settings_.value("time_offset", -0.037).toDouble());
		timeOffset_->setSuffix(" s");
		voxel_ = new QDoubleSpinBox;
		voxel_->setRange(0.01, 0.2);
		voxel_->setSingleStep(0.01);
		voxel_->setValue(settings_.value("voxel", 0.03).toDouble());
		voxel_->setSuffix(" m");
		maxDepth_ = new QDoubleSpinBox;
		maxDepth_->setRange(0.5, 10.0);
		maxDepth_->setSingleStep(0.5);
		maxDepth_->setValue(settings_.value("max_depth", 4.0).toDouble());
		maxDepth_->setSuffix(" m");
		extraArgs_ = new QLineEdit(settings_.value("extra_args").toString());
		extraArgs_->setPlaceholderText(tr8("예: --Rtabmap/DetectionRate 2"));
		ol->addRow(tr8("특징점 (루프/로컬라이제이션)"), features_);
		ol->addRow(tr8("카메라-IMU 시간 오프셋"), timeOffset_);
		ol->addRow(tr8("클라우드 해상도"), voxel_);
		ol->addRow(tr8("클라우드 최대 거리"), maxDepth_);
		ol->addRow(tr8("추가 파라미터"), extraArgs_);
		lay->addWidget(optBox);

		QGroupBox * mapBox = new QGroupBox(tr8("1. 실시간 매핑"));
		QGridLayout * ml = new QGridLayout(mapBox);
		outDir_ = new QLineEdit(settings_.value("out_dir", defaultMapsDir()).toString());
		QToolButton * outBtn = new QToolButton;
		outBtn->setText("...");
		connect(outBtn, &QToolButton::clicked, this, [this]{
			QString d = QFileDialog::getExistingDirectory(this, tr8("맵 저장 폴더"), outDir_->text());
			if(!d.isEmpty()) outDir_->setText(QDir::toNativeSeparators(d));
		});
		btnMap_ = new QPushButton(tr8("매핑 시작"));
		connect(btnMap_, &QPushButton::clicked, this, [this]{start(EngineSettings::kMapping);});
		ml->addWidget(new QLabel(tr8("맵 저장 폴더 (map_날짜.db)")), 0, 0, 1, 3);
		ml->addWidget(outDir_, 1, 0, 1, 2);
		ml->addWidget(outBtn, 1, 2);
		ml->addWidget(btnMap_, 2, 0, 1, 3);
		lay->addWidget(mapBox);

		QGroupBox * locBox = new QGroupBox(tr8("2. 실시간 로컬라이제이션"));
		QGridLayout * ll = new QGridLayout(locBox);
		locMap_ = new QLineEdit(settings_.value("loc_map").toString());
		locMap_->setPlaceholderText(tr8("저장된 맵 .db"));
		QToolButton * locBtn = new QToolButton;
		locBtn->setText("...");
		connect(locBtn, &QToolButton::clicked, this, [this]{
			QString f = QFileDialog::getOpenFileName(this, tr8("로컬라이제이션에 쓸 맵"), locMap_->text().isEmpty()?outDir_->text():locMap_->text(), "RTAB-Map (*.db)");
			if(!f.isEmpty()) locMap_->setText(QDir::toNativeSeparators(f));
		});
		btnLoc_ = new QPushButton(tr8("로컬라이제이션 시작"));
		connect(btnLoc_, &QPushButton::clicked, this, [this]{start(EngineSettings::kLocalization);});
		ll->addWidget(locMap_, 0, 0, 1, 2);
		ll->addWidget(locBtn, 0, 2);
		ll->addWidget(btnLoc_, 1, 0, 1, 3);
		lay->addWidget(locBox);

		btnStop_ = new QPushButton(tr8("정지"));
		btnStop_->setEnabled(false);
		btnStop_->setStyleSheet("QPushButton:enabled { background: #b33; color: white; font-weight: bold; }");
		connect(btnStop_, &QPushButton::clicked, this, [this]{stop();});
		lay->addWidget(btnStop_);

		QGroupBox * dispBox = new QGroupBox(tr8("표시"));
		QFormLayout * dl = new QFormLayout(dispBox);
		colorMode_ = new QComboBox;
		colorMode_->addItem(tr8("실제 색 (RGB, 없으면 IR 밝기)"));
		colorMode_->addItem(tr8("키프레임별 색"));
		connect(colorMode_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){applyCloudStyle();});
		pointSize_ = new QSpinBox;
		pointSize_->setRange(1, 10);
		pointSize_->setValue(2);
		connect(pointSize_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int){applyCloudStyle();});
		follow_ = new QCheckBox(tr8("카메라 따라가기"));
		follow_->setChecked(true);
		connect(follow_, &QCheckBox::toggled, this, [this](bool on){viewer_->setCameraTargetFollow(on); if(!on) viewer_->setCameraFree();});
		showClouds_ = new QCheckBox(tr8("포인트 클라우드"));
		showClouds_->setChecked(true);
		connect(showClouds_, &QCheckBox::toggled, this, [this](bool){applyCloudStyle();});
		dl->addRow(tr8("색상"), colorMode_);
		dl->addRow(tr8("점 크기"), pointSize_);
		dl->addRow(follow_);
		dl->addRow(showClouds_);
		showMesh_ = new QCheckBox(tr8("메쉬 (3DGS·메쉬 메뉴)"));
		showMesh_->setEnabled(false);
		connect(showMesh_, &QCheckBox::toggled, this, [this](bool on){viewer_->setCloudVisibility("mesh", on); viewer_->refreshView();});
		dl->addRow(showMesh_);
		lay->addWidget(dispBox);
		lay->addStretch(1);

		QDockWidget * dock = new QDockWidget(tr8("설정"), this);
		dock->setObjectName("controls");
		dock->setFeatures(QDockWidget::DockWidgetMovable);
		dock->setWidget(left);
		addDockWidget(Qt::LeftDockWidgetArea, dock);

		// right: camera image + state
		QWidget * right = new QWidget;
		QVBoxLayout * rl = new QVBoxLayout(right);
		image_ = new QLabel(tr8("카메라 영상"));
		image_->setAlignment(Qt::AlignCenter);
		image_->setMinimumSize(360, 210);
		image_->setStyleSheet("background: #111; color: #777;");
		rl->addWidget(image_);
		state_ = new QLabel(tr8("대기"));
		state_->setAlignment(Qt::AlignCenter);
		rl->addWidget(state_);
		setState(tr8("대기 중"), "#444");
		QFormLayout * form = new QFormLayout;
		const char * keys[][2] = {
				{"mode", "모드"}, {"fps", "카메라 FPS"}, {"odom", "VIO 처리 시간"}, {"inliers", "VIO 특징점 (인라이어/전체)"},
				{"frames", "처리 프레임"}, {"keyframes", "키프레임"}, {"loops", "루프 클로저 / 맵 매칭"},
				{"slam", "SLAM 처리 시간"}, {"points", "맵 포인트"}, {"pos", "현재 위치 (m)"}, {"dist", "이동 거리"}};
		for(auto & k : keys)
		{
			QLabel * label = new QLabel("-");
			label->setTextInteractionFlags(Qt::TextSelectableByMouse);
			form->addRow(tr8(k[1]), label);
			stats_[k[0]] = label;
		}
		rl->addLayout(form);
		rl->addStretch(1);
		QDockWidget * rdock = new QDockWidget(tr8("카메라 / 상태"), this);
		rdock->setObjectName("state");
		rdock->setFeatures(QDockWidget::DockWidgetMovable);
		rdock->setWidget(right);
		addDockWidget(Qt::RightDockWidgetArea, rdock);

		logView_ = new QPlainTextEdit;
		logView_->setReadOnly(true);
		logView_->setMaximumBlockCount(2000);
		QDockWidget * bdock = new QDockWidget(tr8("로그"), this);
		bdock->setObjectName("log");
		bdock->setFeatures(QDockWidget::DockWidgetMovable);
		bdock->setWidget(logView_);
		addDockWidget(Qt::BottomDockWidgetArea, bdock);
		resizeDocks({bdock}, {130}, Qt::Vertical);

		// menus
		QMenu * m = menuBar()->addMenu(tr8("파일"));
		QAction * a = m->addAction(tr8("맵 불러오기 (오프라인)..."));
		a->setShortcut(QKeySequence::Open);
		connect(a, &QAction::triggered, this, [this]{
			QString f = QFileDialog::getOpenFileName(this, tr8("맵 불러오기"), locMap_->text().isEmpty()?outDir_->text():locMap_->text(), "RTAB-Map (*.db)");
			if(!f.isEmpty()) openMap(QDir::toNativeSeparators(f));
		});
		connect(m->addAction(tr8("녹화 파일 열기...")), &QAction::triggered, this, [this]{browseRecording();});
		recentMenu_ = m->addMenu(tr8("최근 맵"));
		updateRecent();
		m->addSeparator();
		connect(m->addAction(tr8("화면 저장 (PNG)...")), &QAction::triggered, this, [this]{
			QString f = QFileDialog::getSaveFileName(this, tr8("화면 저장"), "live_viewer.png", "PNG (*.png)");
			if(!f.isEmpty()) saveScreenshot(f);
		});
		m->addSeparator();
		connect(m->addAction(tr8("종료")), &QAction::triggered, this, [this]{close();});
		m = menuBar()->addMenu(tr8("실행"));
		connect(m->addAction(tr8("실시간 매핑 시작")), &QAction::triggered, this, [this]{start(EngineSettings::kMapping);});
		connect(m->addAction(tr8("실시간 로컬라이제이션 시작")), &QAction::triggered, this, [this]{start(EngineSettings::kLocalization);});
		connect(m->addAction(tr8("정지")), &QAction::triggered, this, [this]{stop();});
		m = menuBar()->addMenu(tr8("보기"));
		connect(m->addAction(tr8("맵 전체 보기")), &QAction::triggered, this, [this]{fitView();});
		connect(m->addAction(tr8("위에서 보기")), &QAction::triggered, this, [this]{fitView(true);});
		connect(m->addAction(tr8("카메라 시점")), &QAction::triggered, this, [this]{viewer_->resetCamera(); viewer_->refreshView();});
		connect(m->addAction(tr8("화면 지우기")), &QAction::triggered, this, [this]{clearAll();});
		m = menuBar()->addMenu(tr8("3DGS·메쉬"));
		connect(m->addAction(tr8("현재 맵으로 3DGS·메쉬 만들기...")), &QAction::triggered, this, [this]{makeGaussians();});
		connect(m->addAction(tr8("만들기 중단")), &QAction::triggered, this, [this]{
			if(gsProc_) {log(tr8("3DGS·메쉬 만들기를 중단합니다."), 1); killTree(gsProc_);}
		});
		m->addSeparator();
		connect(m->addAction(tr8("메쉬 열기 (PLY)...")), &QAction::triggered, this, [this]{
			QString f = QFileDialog::getOpenFileName(this, tr8("메쉬 열기"), gsResult("colmap/2dgs/mesh_hybrid.ply"), tr8("메쉬 (*.ply)"));
			if(!f.isEmpty()) loadMesh(QDir::toNativeSeparators(f));
		});
		connect(m->addAction(tr8("3DGS 보기 (웹 브라우저)...")), &QAction::triggered, this, [this]{
			QString f = QFileDialog::getOpenFileName(this, tr8("3DGS 모델 (point_cloud.ply)"), gsResult("colmap_x8/3dgs/point_cloud.ply"), tr8("3DGS (*.ply)"));
			if(!f.isEmpty()) openSplats(QDir::toNativeSeparators(f));
		});
	}

	// ------------------------------------------------ 3DGS / mesh

	// results of mesh_3dgs.ps1 for the current map: <map folder>/<map name>_3dgs
	QString gsDir(const QString & map) const
	{
		QFileInfo fi(map);
		return QDir::toNativeSeparators(fi.dir().absoluteFilePath(fi.completeBaseName() + "_3dgs"));
	}

	QString gsResult(const QString & file) const
	{
		QString dir = locMap_->text().isEmpty()?outDir_->text():gsDir(locMap_->text());
		QString path = QDir(dir).absoluteFilePath(file);
		return QFileInfo(path).exists()?QDir::toNativeSeparators(path):dir;
	}

	static QString toolsDir()
	{
		QString tools = qEnvironmentVariable("RTABMAP_TOOLS");
		return QDir(tools).exists("GaussianSplatting")?tools:QString();
	}

	// environment of the Gaussian tools: without the PYTHONPATH / PYTHONHOME of rtabmap_env.bat,
	// which point to RTAB-Map's Python (another version) and break the imports of .gs_env
	static QProcessEnvironment gsEnvironment()
	{
		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.remove("PYTHONPATH");
		env.remove("PYTHONHOME");
		return env;
	}

	// stops a process and its children (mesh_3dgs.ps1 -> python)
	static void killTree(QProcess * & p)
	{
		if(!p)
		{
			return;
		}
#ifdef _WIN32
		QProcess::execute("taskkill", QStringList() << "/T" << "/F" << "/PID" << QString::number(p->processId()));
#endif
		p->kill();
		p->waitForFinished(3000);
		p = 0; // deleted by its finished handler (deleteLater)
	}

	void makeGaussians()
	{
		if(gsProc_)
		{
			QMessageBox::information(this, tr8("3DGS·메쉬"), tr8("이미 만들고 있습니다 (로그 참고)."));
			return;
		}
		if(engine_.running() || busy_)
		{
			QMessageBox::information(this, tr8("실행 중"), tr8("먼저 정지하세요."));
			return;
		}
		QString tools = toolsDir();
		if(tools.isEmpty())
		{
			QMessageBox::warning(this, tr8("3DGS·메쉬"), tr8("tools\\GaussianSplatting을 찾을 수 없습니다: live_viewer.bat (rtabmap_env.bat)로 실행하세요."));
			return;
		}
		QString map = locMap_->text();
		if(!QFileInfo(map).isFile())
		{
			map = QFileDialog::getOpenFileName(this, tr8("3DGS·메쉬를 만들 맵"), outDir_->text(), "RTAB-Map (*.db)");
			if(map.isEmpty()) return;
			map = QDir::toNativeSeparators(map);
		}
		// the color images and stereo pairs come from the recording the map was made from
		QString euroc = findEuroc(recPath_->text());
		if(euroc.isEmpty() || !QDir(euroc).exists("mav0/rgb"))
		{
			QMessageBox::information(this, tr8("3DGS·메쉬"),
					tr8("이 맵을 만든 녹화(컬러 포함 EuRoC 폴더, euroc_rgb)를 선택하세요.\n"
					    "카메라 실시간 매핑으로 만든 맵은 녹화가 없어 쓸 수 없습니다: 녹화한 뒤 녹화 파일로 매핑하세요."));
			QString d = QFileDialog::getExistingDirectory(this, tr8("녹화 (euroc_rgb 폴더)"), recPath_->text());
			if(d.isEmpty()) return;
			euroc = findEuroc(d);
			if(euroc.isEmpty() || !QDir(euroc).exists("mav0/rgb"))
			{
				log(tr8("컬러 스트림이 있는 EuRoC 폴더가 아닙니다: %1 (realsense_bag_to_euroc.py --rgb로 변환)").arg(d), 2);
				return;
			}
		}
		QString out = gsDir(map);
		if(QMessageBox::question(this, tr8("3DGS·메쉬 만들기"),
				tr8("맵: %1\n녹화: %2\n결과: %3\n\nGPU로 약 50분 걸립니다 (백그라운드). 시작할까요?").arg(map, euroc, out)) != QMessageBox::Yes)
		{
			return;
		}
		QDir install(QCoreApplication::applicationDirPath());
		install.cdUp();
		QStringList args;
		args << "-NoProfile" << "-ExecutionPolicy" << "Bypass" << "-File" << QDir(tools).absoluteFilePath("GaussianSplatting/mesh_3dgs.ps1")
		     << "-Euroc" << euroc << "-Map" << map << "-Out" << out << "-Install" << QDir::toNativeSeparators(install.absolutePath());
		log(tr8("3DGS·메쉬 만들기 시작: %1").arg(out));
		gsProc_ = new QProcess(this);
		gsProc_->setProcessEnvironment(gsEnvironment());
		gsProc_->setProcessChannelMode(QProcess::MergedChannels);
		QProcess * proc = gsProc_;
		connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc]{
			for(const QString & l : QString::fromLocal8Bit(proc->readAllStandardOutput()).split('\n', Qt::SkipEmptyParts))
			{
				QString line = l.trimmed();
				if(line.isEmpty()) continue;
				log("[3DGS] " + line);
				if(line.startsWith("[") && line.contains("/7 "))
				{
					// step lines of mesh_3dgs.ps1: "[00:12:34] 3/7 2DGS training ..."
					gsStep_ = line;
					if(!engine_.running() && !busy_) setState(tr8("3DGS·메쉬: ") + line.mid(line.indexOf(']') + 2, 40), "#6a3d9a");
				}
			}
		});
		connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, proc, out](int code, QProcess::ExitStatus status){
			proc->deleteLater();
			if(gsProc_ == proc) gsProc_ = 0;
			QString mesh = QDir(out).absoluteFilePath("colmap/2dgs/mesh_hybrid.ply");
			if(code == 0 && status == QProcess::NormalExit && QFileInfo(mesh).isFile())
			{
				log(tr8("3DGS·메쉬 완료: %1").arg(out));
				if(!engine_.running() && !busy_) standby(tr8("3DGS·메쉬 완료"));
				loadMesh(QDir::toNativeSeparators(mesh));
				log(tr8("3DGS 모델: 3DGS·메쉬 > 3DGS 보기 (%1)").arg(QDir::toNativeSeparators(QDir(out).absoluteFilePath("colmap_x8/3dgs/point_cloud.ply"))));
			}
			else
			{
				log(tr8("3DGS·메쉬 만들기 실패 또는 중단 (코드 %1, %2)").arg(code).arg(gsStep_), 2);
				if(!engine_.running() && !busy_) standby(tr8("3DGS·메쉬 실패"), "#a33");
			}
		});
		gsStep_.clear();
		gsProc_->start("powershell", args);
	}

	void loadMesh(const QString & path)
	{
		QApplication::setOverrideCursor(Qt::WaitCursor);
		pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
		bool ok = pcl::io::loadPLYFile(path.toStdString(), *mesh) >= 0 && !mesh->polygons.empty();
		viewer_->removeCloud("mesh");
		if(ok)
		{
			// meshes of extract_mesh.py are in the map frame
			ok = viewer_->addCloudMesh("mesh", mesh);
		}
		QApplication::restoreOverrideCursor();
		if(!ok)
		{
			log(tr8("메쉬를 열 수 없습니다: %1").arg(path), 2);
			return;
		}
		log(tr8("메쉬: %1 (삼각형 %2개)").arg(path).arg(QLocale().toString((qlonglong)mesh->polygons.size())));
		showMesh_->setEnabled(true);
		showMesh_->blockSignals(true);
		showMesh_->setChecked(true);
		showMesh_->blockSignals(false);
		// the mesh hides the point clouds
		showClouds_->setChecked(false);
		viewer_->refreshView();
	}

	void openSplats(const QString & ply)
	{
		QString tools = toolsDir();
		if(tools.isEmpty())
		{
			QMessageBox::warning(this, tr8("3DGS 보기"), tr8("tools\\GaussianSplatting을 찾을 수 없습니다: live_viewer.bat (rtabmap_env.bat)로 실행하세요."));
			return;
		}
		killTree(splatProc_);
		// Python environment of the Gaussian tools (train_3dgs.ps1), with viser
		QString python = QDir(tools).absoluteFilePath("GaussianSplatting/.gs_env/Scripts/python.exe");
		if(!QFileInfo(python).isFile()) python = "python";
		QStringList args;
		args << QDir(tools).absoluteFilePath("GaussianSplatting/view_3dgs.py") << ply;
		// mesh of the same result folder, as a layer that can be shown in the browser
		QDir result = QFileInfo(ply).dir();
		result.cdUp();
		result.cdUp();
		QString mesh = result.absoluteFilePath("colmap/2dgs/mesh_hybrid.ply");
		if(QFileInfo(mesh).isFile()) args << "--mesh" << mesh;
		log(tr8("3DGS 보기: %1 -> 웹 브라우저 http://localhost:8080").arg(ply));
		splatProc_ = new QProcess(this);
		splatProc_->setProcessEnvironment(gsEnvironment());
		splatProc_->setProcessChannelMode(QProcess::MergedChannels);
		QProcess * proc = splatProc_;
		connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc]{
			for(const QString & l : QString::fromLocal8Bit(proc->readAllStandardOutput()).split('\n', Qt::SkipEmptyParts))
			{
				if(l.contains("Gaussians") || l.contains("rror")) log("[3DGS] " + l.trimmed());
			}
		});
		connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, proc](int code, QProcess::ExitStatus){
			proc->deleteLater();
			if(splatProc_ == proc) splatProc_ = 0;
			if(code != 0) log(tr8("3DGS 보기가 종료되었습니다 (코드 %1): viser 설치 확인 (pip install viser)").arg(code), 1);
		});
		splatProc_->start(python, args);
	}

	static QString defaultMapsDir()
	{
		// C:\slam\install\bin -> C:\slam\maps
		QDir d(QCoreApplication::applicationDirPath());
		d.cdUp();
		d.cdUp();
		return QDir::toNativeSeparators(d.absoluteFilePath("maps"));
	}

	void log(const QString & text, int level = 0)
	{
		QString prefix = level==2?tr8("[오류] "):level==1?tr8("[경고] "):QString();
		QString line = QDateTime::currentDateTime().toString("HH:mm:ss ") + prefix + text;
		logView_->appendPlainText(line);
		printf("%s\n", line.toUtf8().constData());
		fflush(stdout);
	}

	// nothing running: waits for the next task (live camera, recording or map), until the user quits
	void standby(const QString & done = QString(), const QString & color = "#2a5d9f")
	{
		QString wait = tr8("대기 중 - 카메라/녹화 파일로 매핑·로컬라이제이션을 시작하거나 맵을 여세요");
		if(done.isEmpty())
		{
			setState(tr8("대기 중"), "#444");
		}
		else
		{
			setState(done + tr8(" · 대기 중"), color);
		}
		state_->setToolTip(wait);
		if(!logView_->toPlainText().endsWith(wait))
		{
			log(wait);
		}
	}

	void setState(const QString & text, const QString & color)
	{
		state_->setText(text);
		state_->setStyleSheet(QString("font-size: 16px; font-weight: bold; padding: 6px; background: %1; color: white;").arg(color));
	}

	void browseRecording()
	{
		QString start = recPath_->text();
		QString f = QFileDialog::getOpenFileName(this, tr8("녹화 파일 (EuRoC 폴더는 안의 아무 파일이나 선택)"), start,
				tr8("녹화 (*.bag *.db3 *.yaml *.csv);;모든 파일 (*)"));
		if(f.isEmpty())
		{
			return;
		}
		if(f.endsWith(".yaml") || f.endsWith(".csv"))
		{
			QDir d = QFileInfo(f).dir();
			while(!d.exists("mav0") && d.cdUp()) {}
			f = d.absolutePath();
		}
		recPath_->setText(QDir::toNativeSeparators(f));
		srcFile_->setChecked(true);
	}

	void saveSettings()
	{
		settings_.setValue("source", srcCamera_->isChecked()?"camera":"file");
		settings_.setValue("recording", recPath_->text());
		settings_.setValue("rate", rate_->value());
		settings_.setValue("features", features_->currentIndex());
		settings_.setValue("time_offset", timeOffset_->value());
		settings_.setValue("voxel", voxel_->value());
		settings_.setValue("max_depth", maxDepth_->value());
		settings_.setValue("extra_args", extraArgs_->text());
		settings_.setValue("out_dir", outDir_->text());
		settings_.setValue("loc_map", locMap_->text());
	}

	void addRecent(const QString & path)
	{
		QStringList recent = settings_.value("recent_maps").toStringList();
		recent.removeAll(path);
		recent.prepend(path);
		while(recent.size() > 10) recent.removeLast();
		settings_.setValue("recent_maps", recent);
		updateRecent();
	}

	void updateRecent()
	{
		recentMenu_->clear();
		QStringList recent = settings_.value("recent_maps").toStringList();
		for(const QString & p : recent)
		{
			connect(recentMenu_->addAction(p), &QAction::triggered, this, [this, p]{openMap(p);});
		}
		recentMenu_->setEnabled(!recent.isEmpty());
	}

	void setRunning(bool running)
	{
		btnMap_->setEnabled(!running);
		btnLoc_->setEnabled(!running);
		btnStop_->setEnabled(running);
	}

	void resetLive()
	{
		frames_ = 0;
		keyframes_ = 0;
		loops_ = 0;
		locHits_ = 0;
		locUpdates_ = 0;
		matches_ = 0;
		distance_ = 0.0;
		lastPos_ = Transform();
		curPose_ = Transform();
		lastMatch_.invalidate();
		fpsTimer_.restart();
		fpsFrames_ = 0;
		{
			std::lock_guard<std::mutex> lock(odomMutex_);
			odom_ = OdomSample();
		}
		for(auto & s : stats_) s.second->setText("-");
	}

	void clearAll()
	{
		viewer_->removeAllClouds(); // and the mesh
		showMesh_->setEnabled(false);
		showMesh_->setChecked(false);
		viewer_->removeAllGraphs();
		viewer_->removeAllLines();
		viewer_->removeAllSpheres();
		viewer_->removeAllFrustums();
		viewer_->clearTrajectory();
		clouds_.clear();
		addedClouds_.clear();
		poses_.clear();
		resetLive();
		viewer_->refreshView();
	}

	// ------------------------------------------------ run control

	QString findEuroc(const QString & path) const
	{
		QFileInfo fi(path);
		if(fi.isDir())
		{
			if(QDir(path).exists("mav0")) return path;
			if(QDir(path + "/euroc_rgb").exists("mav0")) return path + "/euroc_rgb";
			if(QDir(path + "/euroc").exists("mav0")) return path + "/euroc";
			return QString();
		}
		if(fi.isFile())
		{
			// prefer conversions that include the color stream (euroc_rgb)
			QString stem = fi.completeBaseName();
			QDir root(QCoreApplication::applicationDirPath());
			root.cdUp();
			root.cdUp();
			QStringList candidates;
			candidates << fi.dir().absoluteFilePath(stem + "/euroc_rgb");
			candidates << root.absoluteFilePath("out/" + stem + "/euroc_rgb");
			candidates << fi.dir().absoluteFilePath(stem + "/euroc");
			candidates << root.absoluteFilePath("out/" + stem + "/euroc");
			for(const QString & c : candidates)
			{
				if(QDir(c).exists("mav0")) return QDir::toNativeSeparators(c);
			}
		}
		return QString();
	}

	ParametersMap buildParameters(EngineSettings::Mode mode, const std::string & db)
	{
		ParametersMap params;
		params.insert(ParametersPair(Parameters::kOdomStrategy(), "15"));
		params.insert(ParametersPair("OdomVIO/TimeOffset", uNumber2Str(timeOffset_->value())));
		bool aliked = features_->currentIndex() == 1;
		if(mode != EngineSettings::kMapping && UFile::exists(db))
		{
			// features of the map
			ParametersMap mapParams = readMapParameters(db);
			if(mapParams.find(Parameters::kKpDetectorStrategy()) != mapParams.end())
			{
				bool mapAliked = mapParams.at(Parameters::kKpDetectorStrategy()) == "15";
				if(mapAliked != aliked)
				{
					aliked = mapAliked;
					features_->setCurrentIndex(aliked?1:0);
					log(tr8("맵이 %1 특징점으로 만들어져 같은 특징점을 씁니다.").arg(aliked?"ALIKED":"ORB"), 1);
				}
			}
		}
		if(aliked)
		{
			QString py = qEnvironmentVariable("RTABMAP_PYTHON_SCRIPTS");
			if(py.isEmpty())
			{
				log(tr8("RTABMAP_PYTHON_SCRIPTS가 없습니다: live_viewer.bat (rtabmap_env.bat)로 실행하세요."), 1);
			}
			py.replace('\\', '/');
			params.insert(ParametersPair(Parameters::kVisFeatureType(), "15"));
			params.insert(ParametersPair(Parameters::kKpDetectorStrategy(), "15"));
			params.insert(ParametersPair(Parameters::kPyDetectorPath(), (py + "/rtabmap_aliked.py").toStdString()));
			params.insert(ParametersPair(Parameters::kPyDetectorCuda(), "true"));
			params.insert(ParametersPair(Parameters::kVisCorNNType(), "6"));
			params.insert(ParametersPair(Parameters::kPyMatcherPath(), (py + "/rtabmap_lightglue.py").toStdString()));
			params.insert(ParametersPair(Parameters::kPyMatcherModel(), "aliked"));
			params.insert(ParametersPair(Parameters::kPyMatcherThreshold(), "0.1"));
			params.insert(ParametersPair(Parameters::kPyMatcherCuda(), "true"));
		}
		// extra parameters: --Name value
		QStringList extra = extraArgs_->text().split(' ', Qt::SkipEmptyParts);
		for(int i=0; i+1<extra.size(); ++i)
		{
			if(extra[i].startsWith("--"))
			{
				uInsert(params, ParametersPair(extra[i].mid(2).toStdString(), extra[i+1].toStdString()));
				++i;
			}
		}
		return params;
	}

	static ParametersMap readMapParameters(const std::string & db)
	{
		// parameters of the last session saved in the map
		ParametersMap out;
		std::string tmp = LiveEngine::tmpDir() + "/params.db";
		UFile::erase(tmp);
		UFile::copy(db, tmp);
		DBDriver * driver = DBDriver::create();
		if(driver->openConnection(tmp, false))
		{
			out = driver->getLastParameters();
			driver->closeConnection(false);
		}
		delete driver;
		UFile::erase(tmp);
		return out;
	}

	CloudParams cloudParams() const
	{
		CloudParams c;
		c.voxel = (float)voxel_->value();
		c.maxDepth = (float)maxDepth_->value();
		return c;
	}

	void start(EngineSettings::Mode mode)
	{
		if(engine_.running() || busy_)
		{
			return;
		}
		saveSettings();
		EngineSettings s;
		s.mode = mode;
		if(mode == EngineSettings::kMapping)
		{
			QString out = outDir_->text().trimmed();
			if(out.isEmpty())
			{
				log(tr8("맵 저장 폴더를 지정하세요."), 2);
				return;
			}
			QDir().mkpath(out);
			s.db = QDir(out).absoluteFilePath(QString("map_%1.db").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"))).toStdString();
		}
		else
		{
			s.db = locMap_->text().trimmed().toStdString();
			if(!UFile::exists(s.db))
			{
				QMessageBox::warning(this, tr8("맵 없음"), tr8("로컬라이제이션에 쓸 맵(.db)을 선택하세요."));
				return;
			}
		}
		s.realsense = srcCamera_->isChecked();
		if(!s.realsense)
		{
			QString rec = recPath_->text().trimmed();
			QString euroc = rec.isEmpty()?QString():findEuroc(rec);
			bool isRecording = QFileInfo(rec).isFile() && (rec.endsWith(".bag", Qt::CaseInsensitive) || rec.endsWith(".db3", Qt::CaseInsensitive));
			if(!euroc.isEmpty() && isRecording && !euroc.endsWith("euroc_rgb") && !QDir(euroc + "/mav0").exists("rgb"))
			{
				// older conversion without color: convert again once with --rgb
				log(tr8("기존 변환본에 컬러(RGB)가 없어 컬러 포함으로 다시 변환합니다: %1").arg(euroc));
				euroc.clear();
			}
			if(euroc.isEmpty())
			{
				if(isRecording)
				{
					convertRecording(rec, mode);
					return;
				}
				QMessageBox::warning(this, tr8("녹화 파일"), tr8("녹화 파일(.bag/.db3) 또는 EuRoC 폴더를 선택하세요."));
				return;
			}
			s.euroc = euroc.toStdString();
			s.rate = rate_->value();
		}
		s.parameters = buildParameters(mode, s.db);
		s.cloud = cloudParams();
		s.cloud.stereo = s.parameters;

		clearAll();
		mode_ = mode;
		stats_["mode"]->setText(mode == EngineSettings::kMapping?tr8("실시간 매핑"):tr8("실시간 로컬라이제이션"));
		setState(mode == EngineSettings::kMapping?tr8("매핑 준비 중..."):tr8("맵 불러오는 중..."), "#555");
		setRunning(true);
		btnStop_->setEnabled(false);
		busy_ = true;
		QStringList p;
		for(ParametersMap::iterator iter=s.parameters.begin(); iter!=s.parameters.end(); ++iter)
		{
			p << QString::fromStdString(iter->first + "=" + iter->second);
		}
		log(tr8("시작 (%1): %2").arg(QString::fromStdString(s.db)).arg(p.join(" ")));
		if(worker_.joinable()) worker_.join();
		worker_ = std::thread([this, s]{
			std::string error;
			bool ok = false;
			try {ok = engine_.start(s, error);}
			catch(const std::exception & e) {error = e.what();}
			gui([this, ok, error]{
				busy_ = false;
				if(!ok)
				{
					log(tr8("시작 실패: %1").arg(QString::fromStdString(error)), 2);
					standby(tr8("시작 실패"), "#a33");
					setRunning(false);
					mode_ = -1;
					finishAuto();
				}
				else
				{
					btnStop_->setEnabled(true);
					setState(mode_ == EngineSettings::kMapping?tr8("매핑 중"):tr8("맵 매칭 대기 - 맵을 만든 곳을 비추세요"),
							mode_ == EngineSettings::kMapping?"#2a7d3a":"#8a6d1f");
				}
			});
		});
	}

	void stop()
	{
		if(convert_)
		{
			convert_->kill();
			return;
		}
		if(!engine_.running() || busy_)
		{
			return;
		}
		busy_ = true;
		btnStop_->setEnabled(false);
		bool mapping = mode_ == EngineSettings::kMapping;
		setState(mapping?tr8("맵 저장 중..."):tr8("정지 중..."), "#555");
		log(mapping?tr8("매핑 종료, 맵 저장 중..."):tr8("로컬라이제이션 종료 중..."));
		std::string db = engine_.settings().db;
		if(worker_.joinable()) worker_.join();
		worker_ = std::thread([this, mapping, db]{
			int nodes = 0;
			try {nodes = engine_.stop();}
			catch(const std::exception & e) {std::string what = e.what(); gui([this, what]{log(tr8("정지 중 오류: %1").arg(QString::fromStdString(what)), 2);});}
			gui([this, mapping, db, nodes]{
				busy_ = false;
				setRunning(false);
				mode_ = -1;
				if(mapping)
				{
					QString path = QString::fromStdString(db);
					log(tr8("맵 저장 완료: %1 (키프레임 %2개). 로컬라이제이션 맵으로 설정했습니다.").arg(path).arg(nodes));
					locMap_->setText(path);
					addRecent(path);
					standby(tr8("맵 저장 완료"));
				}
				else
				{
					standby(tr8("로컬라이제이션 종료"));
				}
				finishAuto();
			});
		});
	}

	void openMap(const QString & path)
	{
		if(engine_.running() || busy_)
		{
			QMessageBox::information(this, tr8("실행 중"), tr8("먼저 정지하세요."));
			return;
		}
		if(!QFileInfo(path).isFile())
		{
			log(tr8("맵 파일이 없습니다: %1").arg(path), 2);
			return;
		}
		addRecent(path);
		clearAll();
		locMap_->setText(path);
		mode_ = EngineSettings::kView;
		stats_["mode"]->setText(tr8("맵 보기"));
		setState(tr8("맵 불러오는 중..."), "#555");
		log(tr8("맵 불러오기: %1").arg(path));
		busy_ = true;
		setRunning(true);
		btnStop_->setEnabled(false);
		EngineSettings s;
		s.mode = EngineSettings::kView;
		s.db = path.toStdString();
		s.cloud = cloudParams();
		if(worker_.joinable()) worker_.join();
		worker_ = std::thread([this, s, path]{
			std::string error;
			bool ok = false;
			try {ok = engine_.view(s, error);}
			catch(const std::exception & e) {error = e.what();}
			gui([this, ok, error, path]{
				busy_ = false;
				setRunning(false);
				mode_ = -1;
				if(ok)
				{
					standby(tr8("맵 보기: 키프레임 %1개").arg(poses_.size()));
					QString mesh = QDir(gsDir(path)).absoluteFilePath("colmap/2dgs/mesh_hybrid.ply");
					if(QFileInfo(mesh).isFile())
					{
						log(tr8("이 맵의 메쉬와 3DGS가 있습니다: 3DGS·메쉬 > 메쉬 열기 / 3DGS 보기 (%1)").arg(gsDir(path)));
					}
					fitView();
				}
				else
				{
					log(tr8("맵을 열 수 없습니다: %1").arg(QString::fromStdString(error)), 2);
					standby(tr8("맵 열기 실패"), "#a33");
				}
				finishAuto();
			});
		});
	}

	void convertRecording(const QString & rec, EngineSettings::Mode mode)
	{
		QFileInfo fi(rec);
		QString out = QDir::toNativeSeparators(fi.dir().absoluteFilePath(fi.completeBaseName() + "/euroc_rgb"));
		QString tools = qEnvironmentVariable("RTABMAP_TOOLS");
		QString script = QDir(tools).absoluteFilePath("RealSense/realsense_bag_to_euroc.py");
		if(tools.isEmpty() || !QFileInfo(script).isFile())
		{
			QMessageBox::warning(this, tr8("녹화 변환"), tr8("realsense_bag_to_euroc.py를 찾을 수 없습니다: live_viewer.bat (rtabmap_env.bat)로 실행하세요."));
			return;
		}
		log(tr8("녹화 파일을 EuRoC 형식으로 변환 중 (한 번만): %1 -> %2").arg(rec, out));
		setState(tr8("녹화 파일 변환 중..."), "#8a6d1f");
		setRunning(true);
		convert_ = new QProcess(this);
		convert_->setProcessChannelMode(QProcess::MergedChannels);
		connect(convert_, &QProcess::readyReadStandardOutput, this, [this]{
			for(const QString & l : QString::fromLocal8Bit(convert_->readAllStandardOutput()).split('\n', Qt::SkipEmptyParts))
			{
				log(l.trimmed());
			}
		});
		connect(convert_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, rec, mode](int code, QProcess::ExitStatus){
			convert_->deleteLater();
			convert_ = 0;
			setRunning(false);
			if(code == 0 && findEuroc(rec).endsWith("euroc_rgb"))
			{
				log(tr8("변환 완료"));
				start(mode);
			}
			else
			{
				log(tr8("변환 실패 (코드 %1)").arg(code), 2);
				standby(tr8("변환 실패"), "#a33");
			}
		});
		convert_->start("python", QStringList() << script << rec << out << "--rgb");
	}

	void poll()
	{
		if(!busy_ && engine_.sourceEnded())
		{
			log(engine_.settings().realsense?tr8("카메라 입력이 멈췄습니다."):tr8("녹화 재생이 끝났습니다."), engine_.settings().realsense?1:0);
			stop();
		}
		if(mode_ == EngineSettings::kLocalization && engine_.running() && lastMatch_.isValid())
		{
			// remove the match line after a moment
			if(lastMatch_.elapsed() > 1500)
			{
				viewer_->removeLine("match");
			}
		}
	}

	void finishAuto()
	{
		if(!autoMode_)
		{
			return;
		}
		// back to the normal interactive mode: the window stays open (standby)
		// and is closed only by the user, or by --exit for scripts
		autoMode_ = false;
		bool doExit = exitWhenDone_;
		QString shot = screenshot_;
		screenshot_.clear();
		QTimer::singleShot(1500, this, [this, doExit, shot]{
			if(!shot.isEmpty())
			{
				fitView();
				QTimer::singleShot(800, this, [this, doExit, shot]{saveScreenshot(shot); if(doExit) close();});
			}
			else if(doExit)
			{
				close();
			}
		});
	}

	// overview of the keyframes (and current pose): from behind/above, or from the top
	void fitView(bool top = false)
	{
		std::vector<Transform> pts;
		for(std::map<int, Transform>::const_iterator iter=poses_.begin(); iter!=poses_.end(); ++iter) pts.push_back(iter->second);
		if(!curPose_.isNull()) pts.push_back(curPose_);
		if(pts.empty())
		{
			viewer_->resetCamera();
			viewer_->refreshView();
			return;
		}
		Eigen::Vector3f lo(1e9, 1e9, 1e9), hi(-1e9, -1e9, -1e9);
		for(const Transform & t : pts)
		{
			Eigen::Vector3f p(t.x(), t.y(), t.z());
			lo = lo.cwiseMin(p);
			hi = hi.cwiseMax(p);
		}
		Eigen::Vector3f c = (lo + hi) / 2.0f;
		float d = std::max(3.0f, (hi - lo).norm() * 1.4f);
		if(follow_->isChecked() && (mode_ == EngineSettings::kMapping || mode_ == EngineSettings::kLocalization))
		{
			// the follow mode would move it back to the camera
			follow_->setChecked(false);
		}
		if(top)
		{
			viewer_->setCameraPosition(c[0], c[1], c[2] + d, c[0], c[1], c[2], 1, 0, 0);
		}
		else
		{
			viewer_->setCameraPosition(c[0] - d*0.6f, c[1] - d*0.6f, c[2] + d*0.7f, c[0], c[1], c[2], 0, 0, 1);
		}
		viewer_->refreshView();
	}

	void saveScreenshot(const QString & path)
	{
		viewer_->refreshView();
		QPixmap img = grab();
		QImage gl = viewer_->grabFramebuffer();
		if(!gl.isNull())
		{
			QPainter painter(&img);
			painter.drawImage(viewer_->mapTo(this, QPoint(0, 0)), gl);
		}
		img.save(path);
		log(tr8("화면 저장: %1").arg(path));
	}

	// ------------------------------------------------ display (GUI thread)

	void render()
	{
		OdomSample o;
		{
			std::lock_guard<std::mutex> lock(odomMutex_);
			if(odom_.updated)
			{
				o = odom_;
				odom_.updated = false;
				odom_.image = QImage();
			}
		}
		if(o.updated)
		{
			onOdometry(o);
		}
		if(dirty_)
		{
			dirty_ = false;
			viewer_->refreshView();
		}
	}

	void onOdometry(const OdomSample & o)
	{
		if(mode_ != EngineSettings::kMapping && mode_ != EngineSettings::kLocalization)
		{
			return;
		}
		const Transform & mapPose = o.mapPose;
		bool lost = o.lost;
		fpsFrames_ += o.frames - frames_;
		frames_ = o.frames;
		if(fpsTimer_.elapsed() > 1000)
		{
			stats_["fps"]->setText(QString::number(fpsFrames_ * 1000.0 / fpsTimer_.elapsed(), 'f', 1));
			fpsTimer_.restart();
			fpsFrames_ = 0;
		}
		if(!o.image.isNull())
		{
			image_->setPixmap(QPixmap::fromImage(o.image).scaled(image_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		}
		stats_["odom"]->setText(QString("%1 ms").arg(o.ms, 0, 'f', 1));
		stats_["inliers"]->setText(QString("%1 / %2").arg(o.inliers).arg(o.features));
		stats_["frames"]->setText(QString::number(frames_));
		if(!lost && !mapPose.isNull())
		{
			curPose_ = mapPose;
			if(!lastPos_.isNull())
			{
				distance_ += lastPos_.getDistance(mapPose);
			}
			lastPos_ = mapPose;
			viewer_->updateCameraTargetPosition(mapPose);
			viewer_->addOrUpdateFrustum("camera", mapPose, CameraModel::opticalRotation(), 0.3, QColor(255, 70, 70));
			stats_["pos"]->setText(QString("%1, %2, %3").arg(mapPose.x(), 0, 'f', 2).arg(mapPose.y(), 0, 'f', 2).arg(mapPose.z(), 0, 'f', 2));
			stats_["dist"]->setText(QString("%1 m").arg(distance_, 0, 'f', 2));
		}
		if(mode_ == EngineSettings::kMapping)
		{
			if(lost) setState(tr8("추적 실패 (LOST) - 카메라를 천천히 움직이세요"), "#a33");
			else setState(tr8("매핑 중"), "#2a7d3a");
		}
		else
		{
			if(lost) setState(tr8("추적 실패 (LOST)"), "#a33");
			else if(lastMatch_.isValid() && lastMatch_.elapsed() < 2000) setState(tr8("맵 매칭됨 (로컬라이즈)"), "#2a7d3a");
			else if(lastMatch_.isValid()) setState(tr8("VIO로 추적 중 (마지막 매칭 %1초 전)").arg(lastMatch_.elapsed()/1000), "#8a6d1f");
			else setState(tr8("맵 매칭 대기 - 맵을 만든 곳을 비추세요"), "#8a6d1f");
		}
		dirty_ = true;
	}

	void onGraph(const std::map<int, Transform> & poses, const std::vector<std::pair<std::pair<int, int>, int> > & links,
			bool fullMap, int loopId, int proxId, float slamMs, const Transform & localizedPose)
	{
		if(mode_ == EngineSettings::kLocalization && !fullMap)
		{
			// the map stays, show the match
			++locUpdates_;
			int target = loopId>0?loopId:proxId;
			if(target > 0)
			{
				++locHits_;
				lastMatch_.restart();
				if(!localizedPose.isNull())
				{
					viewer_->addOrUpdateSphere(uFormat("match%d", matches_++), localizedPose, 0.04f, QColor(50, 255, 90));
					std::map<int, Transform>::const_iterator iter = poses_.find(target);
					if(iter != poses_.end())
					{
						viewer_->addOrUpdateLine("match", localizedPose, iter->second, QColor(50, 255, 90), false, true, 2.0);
					}
				}
			}
			stats_["loops"]->setText(tr8("%1 / %2 키프레임 (%3%)").arg(locHits_).arg(locUpdates_).arg(100*locHits_/std::max(locUpdates_, 1)));
			stats_["slam"]->setText(QString("%1 ms").arg(slamMs, 0, 'f', 0));
			dirty_ = true;
			return;
		}
		poses_ = poses;
		// keyframe graph
		pcl::PointCloud<pcl::PointXYZ>::Ptr graph(new pcl::PointCloud<pcl::PointXYZ>);
		for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
		{
			graph->push_back(pcl::PointXYZ(iter->second.x(), iter->second.y(), iter->second.z()));
		}
		viewer_->addOrUpdateGraph("graph", graph, QColor(70, 150, 255));
		// loop closures
		for(const std::string & id : linkIds_)
		{
			viewer_->removeLine(id);
		}
		linkIds_.clear();
		for(const auto & l : links)
		{
			std::map<int, Transform>::const_iterator a = poses.find(l.first.first);
			std::map<int, Transform>::const_iterator b = poses.find(l.first.second);
			if(a != poses.end() && b != poses.end())
			{
				std::string id = uFormat("link%d_%d", l.first.first, l.first.second);
				QColor c = l.second == Link::kGlobalClosure?QColor(255, 60, 60):l.second == Link::kLocalSpaceClosure?QColor(255, 200, 30):QColor(230, 100, 255);
				viewer_->addOrUpdateLine(id, a->second, b->second, c, false, false, 2.0);
				linkIds_.push_back(id);
			}
		}
		// clouds follow the optimized poses
		for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
		{
			showCloud(iter->first);
		}
		if(mode_ == EngineSettings::kMapping)
		{
			if(loopId > 0 || proxId > 0)
			{
				++loops_;
				log(tr8("루프 클로저: 키프레임 %1").arg(loopId>0?loopId:proxId));
			}
			stats_["loops"]->setText(QString::number(loops_));
			stats_["slam"]->setText(QString("%1 ms").arg(slamMs, 0, 'f', 0));
		}
		else
		{
			stats_["loops"]->setText(QString::number(links.size()));
		}
		stats_["keyframes"]->setText(QString::number(poses.size()));
		if(fullMap)
		{
			fitView();
		}
		dirty_ = true;
	}

	void onCloud(int id, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud)
	{
		clouds_[id] = cloud;
		points_ += (int)cloud->size();
		stats_["points"]->setText(QLocale().toString(points_));
		showCloud(id);
		dirty_ = true;
	}

	void showCloud(int id)
	{
		std::map<int, Transform>::const_iterator pose = poses_.find(id);
		std::map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>::iterator cloud = clouds_.find(id);
		if(pose == poses_.end() || cloud == clouds_.end())
		{
			return;
		}
		std::string name = uFormat("cloud%d", id);
		if(addedClouds_.find(id) != addedClouds_.end())
		{
			viewer_->updateCloudPose(name, pose->second);
		}
		else if(cloud->second->size())
		{
			viewer_->addCloud(name, cloud->second, pose->second);
			addedClouds_.insert(id);
			styleCloud(name);
		}
	}

	void styleCloud(const std::string & name)
	{
		viewer_->setCloudPointSize(name, pointSize_->value());
		viewer_->setCloudVisibility(name, showClouds_->isChecked());
		// CloudViewer color handlers (index-1): random, custom, x, y, z, rgb
		viewer_->setCloudColorIndex(name, colorMode_->currentIndex()==1?1:6);
	}

	void applyCloudStyle()
	{
		for(int id : addedClouds_)
		{
			styleCloud(uFormat("cloud%d", id));
		}
		viewer_->refreshView();
	}

	LiveEngine engine_;
	QSettings settings_;
	std::thread worker_;
	std::atomic<bool> busy_;
	int mode_;
	bool autoMode_ = false;
	QString screenshot_;
	bool exitWhenDone_ = false;
	QStringList autoArgs_;
	QProcess * convert_ = 0;
	QProcess * gsProc_ = 0;    // mesh_3dgs.ps1
	QProcess * splatProc_ = 0; // view_3dgs.py
	QString gsStep_;
	QString mesh_, splats_; // --mesh, --splats

	CloudViewer * viewer_;
	QRadioButton * srcCamera_;
	QRadioButton * srcFile_;
	QLineEdit * recPath_;
	QDoubleSpinBox * rate_;
	QComboBox * features_;
	QDoubleSpinBox * timeOffset_;
	QDoubleSpinBox * voxel_;
	QDoubleSpinBox * maxDepth_;
	QLineEdit * extraArgs_;
	QLineEdit * outDir_;
	QLineEdit * locMap_;
	QPushButton * btnMap_;
	QPushButton * btnLoc_;
	QPushButton * btnStop_;
	QComboBox * colorMode_;
	cv::Mat lastColor_; // engine thread only
	double lastColorStamp_ = 0.0;
	QSpinBox * pointSize_;
	QCheckBox * follow_;
	QCheckBox * showClouds_;
	QCheckBox * showMesh_;
	QLabel * image_;
	QLabel * state_;
	std::map<std::string, QLabel*> stats_;
	QPlainTextEdit * logView_;
	QMenu * recentMenu_;

	std::map<int, Transform> poses_;
	std::map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_;
	std::set<int> addedClouds_;
	std::vector<std::string> linkIds_;
	int frames_ = 0, keyframes_ = 0, loops_ = 0, locHits_ = 0, locUpdates_ = 0, matches_ = 0, points_ = 0;
	double distance_ = 0.0;
	Transform lastPos_;
	Transform curPose_;
	QElapsedTimer lastMatch_;
	QElapsedTimer fpsTimer_;
	QElapsedTimer imageTimer_;
	int fpsFrames_ = 0;
	std::mutex odomMutex_;
	OdomSample odom_;
	bool dirty_ = false;
};

int main(int argc, char * argv[])
{
	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kWarning);
	for(int i=1; i<argc; ++i)
	{
		if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			printf("rtabmap-live_viewer [--open MAP.db] [--mesh MESH.ply] [--splats 3DGS.ply]\n"
				   "  [--auto map|loc|view --source realsense|RECORDING --db OUT_DIR_OR_MAP [--rate F] [--features orb|aliked] [--screenshot OUT.png] [--exit]]\n"
				   "Real-time mapping and localization with a D435i (IR stereo + IMU) or a recording,\n"
				   "and offline map viewing. --auto starts one module, the viewer then waits\n"
				   "in standby for the next task (camera, recording or map); --exit closes it at the end (scripts).\n");
			return 0;
		}
	}
#ifdef _WIN32
	CoInitialize(nullptr);
#endif
#ifdef RTABMAP_PYTHON
	// the embedded Python (ALIKED / LightGlue plugins) must start on the main thread,
	// the map is created by a worker thread
	PythonInterface::instance("rtabmap-live_viewer");
#endif
	setvbuf(stdout, NULL, _IONBF, 0);
#if VTK_MAJOR_VERSION > 9 || (VTK_MAJOR_VERSION==9 && VTK_MINOR_VERSION >= 1)
	// OpenGL context VTK needs for its QOpenGLWidget (as in rtabmap and rtabmap-databaseViewer)
	QSurfaceFormat::setDefaultFormat(QVTKRenderWidget::defaultFormat());
#endif
	QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);
	LiveWindow * window = new LiveWindow(app.arguments());
	window->show();
	QTimer::singleShot(300, window, [window]{window->runAutoArgs();});
	int code = app.exec();
	delete window;
	return code;
}
