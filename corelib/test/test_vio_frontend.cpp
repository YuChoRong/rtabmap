#include <gtest/gtest.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/odometry/VIOFrontend.h>
#include <rtabmap/core/odometry/OdometryVIO.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/IMU.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/core/Version.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UTimer.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <functional>
#include <memory>

using namespace rtabmap;

namespace {

// Synthetic stereo camera inside a textured box room (world frame: z up).
const int kWidth = 640;
const int kHeight = 480;
const double kFocal = 450.0;
const double kCx = 319.5;
const double kCy = 239.5;
const double kBaseline = 0.12;
const double kRoomMin[3] = {-5.0, -4.0, 0.0};
const double kRoomMax[3] = { 5.0,  4.0, 3.0};

StereoCameraModel stereoModel()
{
	return StereoCameraModel(kFocal, kFocal, kCx, kCy, kBaseline,
			CameraModel::opticalRotation(), cv::Size(kWidth, kHeight));
}

float hash3(int a, int b, int c)
{
	uint32_t h = (uint32_t)a * 73856093u ^ (uint32_t)b * 19349663u ^ (uint32_t)c * 83492791u;
	h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
	return (h & 0xFFFF) / 65535.0f;
}

// Blocky multi-scale texture: lots of corners at cell boundaries
float texture(int face, double u, double v)
{
	const double scales[3] = {1.5, 4.0, 11.0};
	const float weights[3] = {0.5f, 0.3f, 0.2f};
	float value = 0.0f;
	for(int i=0; i<3; ++i)
	{
		value += weights[i] * hash3((int)std::floor(u*scales[i]), (int)std::floor(v*scales[i]), face*7+i);
	}
	return value;
}

// Casts a ray from origin (inside the room) along dir, returns the hit distance and the texture value.
double castRay(const Eigen::Vector3d & origin, const Eigen::Vector3d & dir, float * value = 0)
{
	double best = 1e9;
	int bestAxis = 0;
	int bestSide = 0;
	for(int axis=0; axis<3; ++axis)
	{
		if(std::fabs(dir[axis]) < 1e-12)
		{
			continue;
		}
		int side = dir[axis] > 0 ? 1 : 0;
		double bound = side ? kRoomMax[axis] : kRoomMin[axis];
		double t = (bound - origin[axis]) / dir[axis];
		if(t > 0 && t < best)
		{
			best = t;
			bestAxis = axis;
			bestSide = side;
		}
	}
	if(value)
	{
		Eigen::Vector3d p = origin + best * dir;
		int a = (bestAxis+1)%3, b = (bestAxis+2)%3;
		*value = texture(bestAxis*2+bestSide, p[a], p[b]);
	}
	return best;
}

cv::Mat renderView(const Transform & cameraPose /* optical frame in world */)
{
	Eigen::Matrix4d T = cameraPose.toEigen4d();
	Eigen::Matrix3d R = T.block<3,3>(0,0);
	Eigen::Vector3d C = T.block<3,1>(0,3);
	cv::Mat image(kHeight, kWidth, CV_8UC1);
	for(int y=0; y<kHeight; ++y)
	{
		for(int x=0; x<kWidth; ++x)
		{
			float sum = 0.0f;
			for(int s=0; s<4; ++s) // 2x2 supersampling
			{
				double px = x - 0.25 + 0.5*(s%2);
				double py = y - 0.25 + 0.5*(s/2);
				Eigen::Vector3d ray((px-kCx)/kFocal, (py-kCy)/kFocal, 1.0);
				float v;
				castRay(C, R*ray, &v);
				sum += v;
			}
			image.at<unsigned char>(y,x) = (unsigned char)std::min(255.0f, 20.0f + 215.0f*sum/4.0f);
		}
	}
	return image;
}

// Renders the stereo pair seen from the base pose in world.
void renderStereo(const Transform & basePose, cv::Mat & left, cv::Mat & right)
{
	Transform leftPose = basePose * CameraModel::opticalRotation();
	Transform rightPose = leftPose * Transform(kBaseline, 0, 0, 0, 0, 0);
	left = renderView(leftPose);
	right = renderView(rightPose);
}

// Smooth trajectory in the room, 1 m/s forward with lateral, vertical and angular oscillations
Transform groundTruth(double t)
{
	return Transform(
			-2.0 + 1.0*t, 0.3*std::sin(1.5*t), 1.5 + 0.1*std::sin(2.0*t),
			0.0, 0.05*std::sin(3.0*t), 0.3*std::sin(t));
}

// Ground truth 3D point (base frame) seen at a left image pixel
cv::Point3f groundTruthPoint(const Transform & basePose, const cv::Point2f & pixel)
{
	Transform leftPose = basePose * CameraModel::opticalRotation();
	Eigen::Matrix4d T = leftPose.toEigen4d();
	Eigen::Vector3d ray((pixel.x-kCx)/kFocal, (pixel.y-kCy)/kFocal, 1.0);
	double t = castRay(T.block<3,1>(0,3), T.block<3,3>(0,0)*ray);
	Eigen::Vector3d pCam = t * ray;
	return util3d::transformPoint(cv::Point3f(pCam[0], pCam[1], pCam[2]), CameraModel::opticalRotation());
}

double rotationAngle(const Transform & t)
{
	return Eigen::AngleAxisd(t.toEigen3d().linear()).angle();
}

ParametersMap visualOnlyParameters()
{
	ParametersMap parameters;
	parameters.insert(ParametersPair(Parameters::kOdomStrategy(), uNumber2Str((int)Odometry::kTypeVIO)));
	parameters.insert(ParametersPair(Parameters::kOdomVIOVisualOnly(), "true"));
	parameters.insert(ParametersPair(Parameters::kOdomFilteringStrategy(), "0"));
	parameters.insert(ParametersPair(Parameters::kRtabmapImagesAlreadyRectified(), "true"));
	return parameters;
}

// Runs an odometry on the synthetic trajectory, returns the final translation error (m)
double runTrajectory(Odometry & odom, int frames, double rate, int * lostFrames = 0, double * pathLength = 0)
{
	const Transform origin = groundTruth(0.0);
	int lost = 0;
	double length = 0.0;
	Transform previous;
	for(int i=0; i<frames; ++i)
	{
		double t = i / rate;
		Transform gt = groundTruth(t);
		cv::Mat left, right;
		renderStereo(gt, left, right);
		SensorData data(left, right, stereoModel(), i+1, 1000.0 + t);
		OdometryInfo info;
		if(odom.process(data, &info).isNull())
		{
			++lost;
		}
		if(!previous.isNull())
		{
			length += previous.getDistance(gt);
		}
		previous = gt;
	}
	if(lostFrames) *lostFrames = lost;
	if(pathLength) *pathLength = length;
	Transform expected = origin.inverse() * groundTruth((frames-1) / rate);
	return expected.getDistance(odom.getPose());
}

} // namespace

TEST(VIOFrontendTest, StaticPairKeepsTrackIds)
{
	VIOFrontend frontend;
	cv::Mat left, right;
	renderStereo(groundTruth(0.0), left, right);

	ASSERT_TRUE(frontend.process(left, right, stereoModel()));
	ASSERT_GT(frontend.tracks().size(), 100u);
	std::vector<int> firstIds;
	for(size_t i=0; i<frontend.tracks().size(); ++i)
	{
		firstIds.push_back(frontend.tracks()[i].id);
		EXPECT_EQ(frontend.tracks()[i].age, 1);
	}

	Transform motion;
	ASSERT_TRUE(frontend.process(left, right, stereoModel(), Transform(), &motion));
	ASSERT_FALSE(motion.isNull());
	EXPECT_LT(motion.getNorm(), 1e-3);
	EXPECT_LT(rotationAngle(motion), 1e-3);
	EXPECT_EQ(frontend.stats().added, 0);
	ASSERT_EQ(frontend.tracks().size(), firstIds.size());
	for(size_t i=0; i<frontend.tracks().size(); ++i)
	{
		EXPECT_EQ(frontend.tracks()[i].id, firstIds[i]);
		EXPECT_EQ(frontend.tracks()[i].age, 2);
	}
}

TEST(VIOFrontendTest, StereoDepthMatchesGeometry)
{
	VIOFrontend frontend;
	const Transform pose = groundTruth(0.0);
	cv::Mat left, right;
	renderStereo(pose, left, right);
	ASSERT_TRUE(frontend.process(left, right, stereoModel()));

	std::vector<double> errors;
	for(size_t i=0; i<frontend.tracks().size(); ++i)
	{
		const VIOFrontend::Track & track = frontend.tracks()[i];
		if(track.hasDepth())
		{
			cv::Point3f gt = groundTruthPoint(pose, track.left);
			double range = cv::norm(gt);
			errors.push_back(cv::norm(track.point - gt) / range);
		}
	}
	ASSERT_GT(errors.size(), frontend.tracks().size() * 8 / 10);
	std::sort(errors.begin(), errors.end());
	EXPECT_LT(errors[errors.size()/2], 0.02);      // median relative error
	EXPECT_LT(errors[errors.size()*9/10], 0.05);   // 90th percentile
}

TEST(VIOFrontendTest, TracksFollowMotionAndRejectMovingPatch)
{
	VIOFrontend frontend;
	cv::Mat left0, right0, left1, right1;
	renderStereo(groundTruth(0.0), left0, right0);
	renderStereo(groundTruth(0.05), left1, right1);

	// An independently moving object: a patch of the first frame pasted 15 px aside in the second frame
	const cv::Rect source(380, 160, 160, 160);
	const cv::Rect moved = source + cv::Point(15, 8);
	left0(source).copyTo(left1(moved));
	right0(source).copyTo(right1(moved));

	ASSERT_TRUE(frontend.process(left0, right0, stereoModel()));
	Transform motion;
	ASSERT_TRUE(frontend.process(left1, right1, stereoModel(), Transform(), &motion));
	ASSERT_FALSE(motion.isNull());

	Transform expected = groundTruth(0.0).inverse() * groundTruth(0.05);
	EXPECT_LT(motion.getDistance(expected), 0.005);
	EXPECT_LT(rotationAngle(motion.inverse()*expected), 0.002);

	cv::Rect inner(moved.x+10, moved.y+10, moved.width-20, moved.height-20);
	int keptInPatch = 0;
	int kept = 0;
	for(size_t i=0; i<frontend.tracks().size(); ++i)
	{
		const VIOFrontend::Track & track = frontend.tracks()[i];
		if(track.age == 2)
		{
			++kept;
			if(inner.contains(track.left))
			{
				++keptInPatch;
			}
		}
	}
	EXPECT_GT(kept, 80);
	EXPECT_EQ(keptInPatch, 0);
	EXPECT_GT(frontend.stats().tracked - frontend.stats().inliers, 3);
}

TEST(OdometryVIOVisualTest, VisualOnlyFollowsTrajectoryLikeF2M)
{
	const int frames = 40;
	const double rate = 20.0;

	ParametersMap parameters = visualOnlyParameters();
	std::unique_ptr<Odometry> vio(Odometry::create(parameters));
	ASSERT_EQ(vio->getType(), Odometry::kTypeVIO);
	int lost = 0;
	double length = 0.0;
	double vioError = runTrajectory(*vio, frames, rate, &lost, &length);

	ParametersMap f2mParameters = parameters;
	f2mParameters[Parameters::kOdomStrategy()] = uNumber2Str((int)Odometry::kTypeF2M);
	std::unique_ptr<Odometry> f2m(Odometry::create(f2mParameters));
	int f2mLost = 0;
	double f2mError = runTrajectory(*f2m, frames, rate, &f2mLost);

	std::cout << "Path length " << length << " m, final error: VIO visual-only " << vioError
			<< " m (lost " << lost << "), F2M " << f2mError << " m (lost " << f2mLost << ")" << std::endl;
	RecordProperty("vio_error_m", uNumber2Str(vioError));
	RecordProperty("f2m_error_m", uNumber2Str(f2mError));

	EXPECT_EQ(lost, 0);
	EXPECT_LT(vioError, 0.01 * length);
	// "Similar to F2M": no worse than twice F2M's error (with a 1 cm floor)
	EXPECT_LT(vioError, std::max(2.0 * f2mError, 0.01));
}

TEST(OdometryVIOVisualTest, ImuModeReportsTracks)
{
	ParametersMap parameters;
	parameters.insert(ParametersPair(Parameters::kOdomFilteringStrategy(), "0"));
	parameters.insert(ParametersPair(Parameters::kOdomGuessMotion(), "false"));
	parameters.insert(ParametersPair(Parameters::kRtabmapImagesAlreadyRectified(), "true"));
	OdometryVIO odom(parameters);

	cv::Mat left, right;
	renderStereo(groundTruth(0.0), left, right);
	int imagesWithPose = 0;
	OdometryInfo info;
	for(int i=0; i<=100; ++i) // 0.5 s static, IMU at 200 Hz, images at 20 Hz
	{
		double stamp = 1000.0 + i / 200.0;
		SensorData imu;
		imu.setStamp(stamp);
		imu.setIMU(IMU(cv::Vec3d(0,0,0), cv::Mat::eye(3,3,CV_64FC1), cv::Vec3d(0,0,9.81), cv::Mat::eye(3,3,CV_64FC1), Transform::getIdentity()));
		odom.process(imu);
		if(i % 10 == 0)
		{
			SensorData image(left, right, stereoModel(), i/10+1, stamp);
			info = OdometryInfo();
			if(!odom.process(image, &info).isNull())
			{
				++imagesWithPose;
			}
		}
	}
	if(imagesWithPose == 0)
	{
		GTEST_SKIP() << "RTAB-Map built without GTSAM";
	}
	EXPECT_GT(info.features, 100);
	EXPECT_GT(info.reg.inliers, 100);
	EXPECT_EQ(info.words.size(), (size_t)info.features);
	EXPECT_LT(odom.getPose().getNorm(), 1e-3);
}

namespace {

// Visual-inertial sequence: static for 0.25 s (gravity initialization), then
// the room trajectory with a smooth start (zero velocity at t=0).
const double kImuRateVio = 200.0;
const double kCameraRateVio = 20.0;
const double kStaticTime = 0.25;
const double kMotionTime = 3.0;
// IMU mounted rotated and offset in the base frame
const Transform kImuLocal(0.05, -0.02, 0.03, 0.1, -0.05, M_PI/2.0);

double warp(double t)
{
	const double k = 3.0;
	return t <= 0.0 ? 0.0 : t - (1.0 - std::exp(-k*t)) / k;
}

Transform groundTruthVio(double t)
{
	return groundTruth(warp(t));
}

// Exact IMU measurements (in the IMU frame) of the ground truth trajectory, by finite differences
void imuAt(double t, cv::Vec3d & gyro, cv::Vec3d & acc)
{
	const double h = 1e-3;
	Eigen::Matrix4d T0 = (groundTruthVio(t-h) * kImuLocal).toEigen4d();
	Eigen::Matrix4d T1 = (groundTruthVio(t) * kImuLocal).toEigen4d();
	Eigen::Matrix4d T2 = (groundTruthVio(t+h) * kImuLocal).toEigen4d();
	Eigen::Matrix3d R1 = T1.block<3,3>(0,0);
	Eigen::AngleAxisd aa(T0.block<3,3>(0,0).transpose() * T2.block<3,3>(0,0));
	Eigen::Vector3d w = aa.angle() * aa.axis() / (2.0*h);
	Eigen::Vector3d a = (T2.block<3,1>(0,3) - 2.0*T1.block<3,1>(0,3) + T0.block<3,1>(0,3)) / (h*h);
	Eigen::Vector3d f = R1.transpose() * (a + Eigen::Vector3d(0, 0, 9.81));
	gyro = cv::Vec3d(w[0], w[1], w[2]);
	acc = cv::Vec3d(f[0], f[1], f[2]);
}

// Rendered stereo frames of the visual-inertial sequence, shared by the tests
const std::pair<cv::Mat, cv::Mat> & vioFrame(int index)
{
	static std::map<int, std::pair<cv::Mat, cv::Mat> > cache;
	std::map<int, std::pair<cv::Mat, cv::Mat> >::iterator iter = cache.find(index);
	if(iter == cache.end())
	{
		std::pair<cv::Mat, cv::Mat> frame;
		renderStereo(groundTruthVio(-kStaticTime + index / kCameraRateVio), frame.first, frame.second);
		iter = cache.insert(std::make_pair(index, frame)).first;
	}
	return iter->second;
}

enum CameraInput {kStereo, kMono, kNoImu};

struct VioRunOptions
{
	CameraInput input = kStereo;
	cv::Vec3d gyroBias = cv::Vec3d(0,0,0);
	cv::Vec3d accBias = cv::Vec3d(0,0,0);
	int blackoutBegin = -1; // frames [begin, end) are uniform gray
	int blackoutEnd = -1;
};

struct VioRunResult
{
	double finalError = 0.0;
	double maxError = 0.0;
	double pathLength = 0.0;
	int nullPoses = 0;   // image frames without pose after initialization
	double msPerFrame = 0.0;
};

VioRunResult runVio(Odometry & odom, const VioRunOptions & options)
{
	VioRunResult result;
	const int imuPerImage = (int)std::lround(kImuRateVio / kCameraRateVio);
	const int steps = (int)std::lround((kStaticTime + kMotionTime) * kImuRateVio);
	const Transform origin = groundTruthVio(0.0);
	const CameraModel mono(kFocal, kFocal, kCx, kCy, CameraModel::opticalRotation(), 0, cv::Size(kWidth, kHeight));
	bool started = false;
	Transform previous;
	double processingTime = 0.0;
	int frames = 0;
	for(int i=0; i<=steps; ++i)
	{
		double t = -kStaticTime + i / kImuRateVio;
		double stamp = 1000.0 + i / kImuRateVio;
		if(options.input != kNoImu)
		{
			cv::Vec3d gyro, acc;
			imuAt(t, gyro, acc);
			SensorData imu;
			imu.setStamp(stamp);
			imu.setIMU(IMU(gyro + options.gyroBias, cv::Mat::eye(3,3,CV_64FC1),
					acc + options.accBias, cv::Mat::eye(3,3,CV_64FC1), kImuLocal));
			odom.process(imu);
		}
		if(i % imuPerImage != 0)
		{
			continue;
		}
		int index = i / imuPerImage;
		std::pair<cv::Mat, cv::Mat> images = vioFrame(index);
		if(index >= options.blackoutBegin && index < options.blackoutEnd)
		{
			images.first = cv::Mat(kHeight, kWidth, CV_8UC1, cv::Scalar(128));
			images.second = images.first;
		}
		SensorData data = options.input == kMono ?
				SensorData(images.first, mono, index+1, stamp) :
				SensorData(images.first, images.second, stereoModel(), index+1, stamp);
		OdometryInfo info;
		UTimer timer;
		Transform pose = odom.process(data, &info);
		processingTime += timer.ticks();
		++frames;
		if(pose.isNull())
		{
			if(started) ++result.nullPoses;
			continue;
		}
		started = true;
		Transform gt = groundTruthVio(t);
		if(t >= 0.0)
		{
			double error = (origin.inverse() * gt).getDistance(odom.getPose());
			result.maxError = std::max(result.maxError, error);
			result.finalError = error;
			if(!previous.isNull())
			{
				result.pathLength += previous.getDistance(gt);
			}
			previous = gt;
		}
	}
	result.msPerFrame = 1000.0 * processingTime / frames;
	return result;
}

ParametersMap vioParameters()
{
	ParametersMap parameters;
	parameters.insert(ParametersPair(Parameters::kOdomStrategy(), uNumber2Str((int)Odometry::kTypeVIO)));
	parameters.insert(ParametersPair(Parameters::kOdomFilteringStrategy(), "0"));
	parameters.insert(ParametersPair(Parameters::kOdomGuessMotion(), "false"));
	parameters.insert(ParametersPair(Parameters::kRtabmapImagesAlreadyRectified(), "true"));
	return parameters;
}

bool backendAvailable()
{
#ifdef RTABMAP_GTSAM_UNSTABLE
	return true;
#else
	return false;
#endif
}

void printResult(const std::string & name, const VioRunResult & r)
{
	std::cout << name << ": final error " << r.finalError << " m, max error " << r.maxError
			<< " m over " << r.pathLength << " m, " << r.nullPoses << " frames without pose, "
			<< r.msPerFrame << " ms/frame" << std::endl;
}

} // namespace

TEST(OdometryVIOFusionTest, FusionFollowsTrajectory)
{
	if(!backendAvailable()) GTEST_SKIP() << "RTAB-Map built without gtsam_unstable";

	std::unique_ptr<Odometry> fusion(Odometry::create(vioParameters()));
	VioRunResult fused = runVio(*fusion, VioRunOptions());

	ParametersMap visualParameters = vioParameters();
	visualParameters[Parameters::kOdomVIOVisualOnly()] = "true";
	std::unique_ptr<Odometry> visual(Odometry::create(visualParameters));
	VioRunOptions visualOptions;
	visualOptions.input = kNoImu;
	VioRunResult visualOnly = runVio(*visual, visualOptions);

	printResult("VIO (IMU + stereo)", fused);
	printResult("Visual only", visualOnly);

	EXPECT_EQ(fused.nullPoses, 0);
	EXPECT_GT(fused.pathLength, 2.0);
	EXPECT_LT(fused.finalError, 0.0075 * fused.pathLength);
	EXPECT_LT(fused.maxError, 0.01 * fused.pathLength);
}

TEST(OdometryVIOFusionTest, FusionCorrectsImuBias)
{
	if(!backendAvailable()) GTEST_SKIP() << "RTAB-Map built without gtsam_unstable";

	VioRunOptions options;
	options.gyroBias = cv::Vec3d(0.005, -0.004, 0.01);  // rad/s
	options.accBias = cv::Vec3d(0.05, -0.04, 0.03);     // m/s^2

	std::unique_ptr<Odometry> fusion(Odometry::create(vioParameters()));
	VioRunResult fused = runVio(*fusion, options);

	std::unique_ptr<Odometry> imuOnly(Odometry::create(vioParameters()));
	VioRunOptions imuOptions = options;
	imuOptions.input = kMono; // no stereo: IMU propagation only
	VioRunResult propagated = runVio(*imuOnly, imuOptions);

	printResult("VIO with biased IMU", fused);
	printResult("IMU propagation with biased IMU", propagated);

	EXPECT_EQ(fused.nullPoses, 0);
	EXPECT_LT(fused.finalError, 0.01 * fused.pathLength);
	EXPECT_LT(fused.finalError * 10.0, propagated.finalError);
}

TEST(OdometryVIOFusionTest, BridgesVisualBlackout)
{
	if(!backendAvailable()) GTEST_SKIP() << "RTAB-Map built without gtsam_unstable";

	VioRunOptions options;
	options.blackoutBegin = 30; // 0.3 s without any feature in the middle of the motion
	options.blackoutEnd = 36;
	std::unique_ptr<Odometry> fusion(Odometry::create(vioParameters()));
	VioRunResult fused = runVio(*fusion, options);

	ParametersMap visualParameters = vioParameters();
	visualParameters[Parameters::kOdomVIOVisualOnly()] = "true";
	std::unique_ptr<Odometry> visual(Odometry::create(visualParameters));
	VioRunOptions visualOptions = options;
	visualOptions.input = kNoImu;
	VioRunResult visualOnly = runVio(*visual, visualOptions);

	printResult("VIO with 0.3 s visual blackout", fused);
	printResult("Visual only with 0.3 s visual blackout", visualOnly);

	EXPECT_EQ(fused.nullPoses, 0);
	EXPECT_LT(fused.finalError, 0.01 * fused.pathLength);
	EXPECT_LT(fused.finalError * 10.0, visualOnly.finalError);
}


#ifdef RTABMAP_GTSAM_UNSTABLE
#include "odometry/VIOBackend.h" // private header (corelib/src)

// The back-end alone, fed with exact IMU and exact stereo projections of
// landmarks on the room walls (no image processing): it must follow the
// ground truth closely, which validates the factors, frames and bookkeeping.
TEST(VIOBackendTest, FollowsGroundTruthWithIdealMeasurements)
{
	std::vector<Eigen::Vector3d> landmarks;
	cv::RNG rng(1);
	while(landmarks.size() < 2000)
	{
		Eigen::Vector3d o(0,0,1.5), d(rng.uniform(-1.0,1.0), rng.uniform(-1.0,1.0), rng.uniform(-1.0,1.0));
		d.normalize();
		landmarks.push_back(o + castRay(o, d) * d * 0.999);
	}
	const double imuRate = 1000.0;
	const double keyframeInterval = 0.1;
	auto params = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);
	params->accelerometerCovariance = gtsam::I_3x3 * 4e-6;
	params->gyroscopeCovariance = gtsam::I_3x3 * 3e-8;
	params->biasAccCovariance = gtsam::I_3x3 * 9e-6;
	params->biasOmegaCovariance = gtsam::I_3x3 * 4e-10;
	params->integrationCovariance = gtsam::I_3x3 * 1e-8;
	params->biasAccOmegaInt = gtsam::I_6x6 * 1e-5;
	gtsam::PreintegratedCombinedMeasurements preintegrated(params, gtsam::imuBias::ConstantBias());

	// Up to 150 visible landmarks, with their exact rectified stereo projections
	std::function<std::vector<VIOFrontend::Track>(double)> tracksAt = [&](double t) {
		std::vector<VIOFrontend::Track> tracks;
		Eigen::Matrix4d T = (groundTruthVio(t) * CameraModel::opticalRotation()).toEigen4d();
		for(size_t i=0; i<landmarks.size() && tracks.size() < 150; ++i)
		{
			Eigen::Vector3d pc = T.block<3,3>(0,0).transpose() * (landmarks[i] - T.block<3,1>(0,3));
			if(pc[2] < 0.3) continue;
			double u = kFocal*pc[0]/pc[2]+kCx, v = kFocal*pc[1]/pc[2]+kCy, ur = kFocal*(pc[0]-kBaseline)/pc[2]+kCx;
			if(u < 0 || v < 0 || u >= kWidth || v >= kHeight || ur < 0) continue;
			VIOFrontend::Track track;
			track.id = (int)i+1;
			track.age = 2;
			track.left = cv::Point2f(u, v);
			track.right = cv::Point2f(ur, v);
			track.point = cv::Point3f(pc[2], -pc[0], -pc[1]);
			tracks.push_back(track);
		}
		return tracks;
	};
	std::function<gtsam::NavState(double)> stateAt = [&](double t) {
		const double h = 1e-4;
		Eigen::Matrix4d T0 = (groundTruthVio(t-h) * kImuLocal).toEigen4d();
		Eigen::Matrix4d T1 = (groundTruthVio(t) * kImuLocal).toEigen4d();
		Eigen::Matrix4d T2 = (groundTruthVio(t+h) * kImuLocal).toEigen4d();
		return gtsam::NavState(gtsam::Pose3(T1), gtsam::Vector3((T2.block<3,1>(0,3)-T0.block<3,1>(0,3))/(2*h)));
	};

	VIOBackend backend(ParametersMap{});
	const gtsam::Pose3 imuToCamera((kImuLocal.inverse() * CameraModel::opticalRotation()).toEigen4d());
	ASSERT_TRUE(backend.initialize(0.0, stateAt(0.0), gtsam::imuBias::ConstantBias(), imuToCamera, stereoModel(), tracksAt(0.0), 0.01));

	cv::Vec3d gyro0, acc0;
	imuAt(0.0, gyro0, acc0);
	const int steps = (int)std::lround(kMotionTime * imuRate);
	const int imuPerKeyframe = (int)std::lround(keyframeInterval * imuRate);
	double maxError = 0.0;
	for(int i=1; i<=steps; ++i)
	{
		double t = i / imuRate;
		cv::Vec3d gyro1, acc1;
		imuAt(t, gyro1, acc1);
		cv::Vec3d acc = 0.5*(acc0+acc1), gyro = 0.5*(gyro0+gyro1);
		preintegrated.integrateMeasurement(gtsam::Vector3(acc[0], acc[1], acc[2]), gtsam::Vector3(gyro[0], gyro[1], gyro[2]), 1.0/imuRate);
		gyro0 = gyro1;
		acc0 = acc1;
		if(i % imuPerKeyframe == 0)
		{
			gtsam::NavState predicted = preintegrated.predict(backend.state(), backend.bias());
			ASSERT_TRUE(backend.addKeyframe(t, preintegrated, predicted, tracksAt(t)));
			preintegrated.resetIntegrationAndSetBias(backend.bias());
			EXPECT_GT(backend.stats().smartFactors, 50);
			maxError = std::max(maxError, (backend.state().pose().translation() - stateAt(t).pose().translation()).norm());
		}
	}
	gtsam::NavState truth = stateAt(kMotionTime);
	std::cout << "Back-end with ideal measurements: max error " << maxError << " m, final velocity error "
			<< (backend.state().v() - truth.v()).norm() << " m/s, "
			<< backend.stats().keyframes << " keyframes in window" << std::endl;
	EXPECT_LT(maxError, 0.002);
	EXPECT_LT((backend.state().v() - truth.v()).norm(), 0.005);
	EXPECT_LT(gtsam::Rot3::Logmap(backend.state().pose().rotation().between(truth.pose().rotation())).norm(), 0.001);
	EXPECT_LT(backend.bias().vector().norm(), 0.01);
	EXPECT_EQ(backend.stats().keyframes, 15); // keyframes newer than 1.5 s at 10 Hz
}
#endif
