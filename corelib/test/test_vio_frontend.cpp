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
#include <rtabmap/utilite/UConversion.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
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
