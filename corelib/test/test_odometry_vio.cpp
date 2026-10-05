#include <gtest/gtest.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/odometry/OdometryVIO.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/IMU.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/utilite/UConversion.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <functional>
#include <memory>

using namespace rtabmap;

namespace {

const double kGravity = 9.81;
const double kImuRate = 200.0;
const double kCameraRate = 20.0;

ParametersMap vioTestParameters()
{
	ParametersMap parameters;
	parameters.insert(ParametersPair(Parameters::kOdomStrategy(), uNumber2Str((int)Odometry::kTypeVIO)));
	parameters.insert(ParametersPair(Parameters::kOdomGuessMotion(), "false"));
	parameters.insert(ParametersPair(Parameters::kOdomFilteringStrategy(), "0"));
	parameters.insert(ParametersPair(Parameters::kRtabmapImagesAlreadyRectified(), "true"));
	parameters.insert(ParametersPair(Parameters::kOdomVIOGravity(), uNumber2Str(kGravity)));
	parameters.insert(ParametersPair(Parameters::kOdomVIOInitImuSamples(), "20"));
	return parameters;
}

SensorData makeImuData(double stamp, const cv::Vec3d & gyro, const cv::Vec3d & acc, const Transform & localTransform)
{
	SensorData data;
	data.setStamp(stamp);
	data.setIMU(IMU(gyro, cv::Mat::eye(3,3,CV_64FC1), acc, cv::Mat::eye(3,3,CV_64FC1), localTransform));
	return data;
}

SensorData makeImageData(int id, double stamp)
{
	const cv::Mat image = cv::Mat::zeros(32, 32, CV_8UC1);
	const CameraModel model(100.0, 100.0, 16.0, 16.0, CameraModel::opticalRotation());
	SensorData data(image, model, id);
	data.setStamp(stamp);
	return data;
}

// Feeds IMU at 200 Hz and an image at 20 Hz from t=0 to duration.
// imuAt(t) returns the measured {gyro, acc} in the IMU frame.
// Returns the odometry pose after the last image.
Transform runSequence(
		Odometry & odom,
		double duration,
		const std::function<std::pair<cv::Vec3d, cv::Vec3d>(double)> & imuAt,
		const Transform & imuLocalTransform = Transform::getIdentity(),
		int * lostFrames = 0)
{
	const int imuPerImage = (int)std::lround(kImuRate / kCameraRate);
	const int steps = (int)std::lround(duration * kImuRate);
	int imageId = 1;
	int lost = 0;
	for(int i=0; i<=steps; ++i)
	{
		double stamp = 1000.0 + i / kImuRate; // arbitrary start time
		std::pair<cv::Vec3d, cv::Vec3d> m = imuAt(i / kImuRate);
		SensorData imu = makeImuData(stamp, m.first, m.second, imuLocalTransform);
		odom.process(imu);
		if(i % imuPerImage == 0)
		{
			SensorData image = makeImageData(imageId++, stamp);
			OdometryInfo info;
			Transform t = odom.process(image, &info);
			if(t.isNull())
			{
				++lost;
			}
		}
	}
	if(lostFrames)
	{
		*lostFrames = lost;
	}
	return odom.getPose();
}

bool vioAvailable()
{
	// Without GTSAM the factory still creates OdometryVIO, but it never outputs a pose.
	ParametersMap parameters = vioTestParameters();
	std::unique_ptr<Odometry> odom(Odometry::create(parameters));
	int lost = 0;
	runSequence(*odom, 0.5, [](double) {
		return std::make_pair(cv::Vec3d(0,0,0), cv::Vec3d(0,0,kGravity));
	}, Transform::getIdentity(), &lost);
	return lost < 10;
}

} // namespace

TEST(OdometryVIOTest, FactoryCreatesVIO)
{
	ParametersMap parameters = vioTestParameters();
	std::unique_ptr<Odometry> odom(Odometry::create(parameters));
	ASSERT_NE(odom.get(), nullptr);
	EXPECT_EQ(odom->getType(), Odometry::kTypeVIO);
	EXPECT_TRUE(odom->canProcessAsyncIMU());
}

TEST(OdometryVIOTest, WaitsForGravityInitialization)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// An image before any IMU sample cannot be processed
	SensorData image = makeImageData(1, 1000.0);
	EXPECT_TRUE(odom.process(image).isNull());
}

TEST(OdometryVIOTest, StaticSensorStaysAtOrigin)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	Transform pose = runSequence(odom, 2.0, [](double) {
		return std::make_pair(cv::Vec3d(0,0,0), cv::Vec3d(0,0,kGravity));
	});
	EXPECT_NEAR(pose.x(), 0.0, 1e-3);
	EXPECT_NEAR(pose.y(), 0.0, 1e-3);
	EXPECT_NEAR(pose.z(), 0.0, 1e-3);
	EXPECT_NEAR(pose.theta(), 0.0, 1e-4);
}

TEST(OdometryVIOTest, ConstantAccelerationFollowsKinematics)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// Static for 0.5 s (gravity initialization), then 0.5 m/s^2 along x for 2 s
	const double t0 = 0.5, a = 0.5, duration = 2.5;
	Transform pose = runSequence(odom, duration, [=](double t) {
		return std::make_pair(cv::Vec3d(0,0,0), cv::Vec3d(t > t0 ? a : 0.0, 0, kGravity));
	});
	const double T = duration - t0;
	EXPECT_NEAR(pose.x(), 0.5 * a * T * T, 0.02);
	EXPECT_NEAR(pose.y(), 0.0, 1e-3);
	EXPECT_NEAR(pose.z(), 0.0, 1e-3);
}

TEST(OdometryVIOTest, ConstantYawRateIntegratesHeading)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	const double t0 = 0.5, w = 0.5, duration = 2.5;
	Transform pose = runSequence(odom, duration, [=](double t) {
		return std::make_pair(cv::Vec3d(0, 0, t > t0 ? w : 0.0), cv::Vec3d(0,0,kGravity));
	});
	EXPECT_NEAR(pose.theta(), w * (duration - t0), 0.02);
	EXPECT_NEAR(pose.x(), 0.0, 1e-3);
	EXPECT_NEAR(pose.y(), 0.0, 1e-3);
}

TEST(OdometryVIOTest, RotatedImuIsExpressedInBaseFrame)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// IMU x axis points along base y (yaw +90 deg). The base accelerates along its x axis,
	// which the IMU measures along its -y axis.
	const Transform imuLocal(0,0,0, 0,0,M_PI/2.0);
	const double t0 = 0.5, a = 0.5, duration = 2.5;
	Transform pose = runSequence(odom, duration, [=](double t) {
		return std::make_pair(cv::Vec3d(0,0,0), cv::Vec3d(0, t > t0 ? -a : 0.0, kGravity));
	}, imuLocal);
	const double T = duration - t0;
	EXPECT_NEAR(pose.x(), 0.5 * a * T * T, 0.02);
	EXPECT_NEAR(pose.y(), 0.0, 1e-3);
	EXPECT_NEAR(pose.theta(), 0.0, 1e-3);
}

TEST(OdometryVIOTest, TiltedStartIsAlignedWithGravity)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// Sensor pitched by 0.3 rad: gravity appears partly along x
	const double pitch = 0.3;
	Transform pose = runSequence(odom, 1.0, [=](double) {
		return std::make_pair(cv::Vec3d(0,0,0), cv::Vec3d(-kGravity*std::sin(pitch), 0, kGravity*std::cos(pitch)));
	});
	float roll, p, yaw;
	pose.getEulerAngles(roll, p, yaw);
	EXPECT_NEAR(p, pitch, 1e-3);
	EXPECT_NEAR(roll, 0.0, 1e-3);
	EXPECT_NEAR(pose.x(), 0.0, 1e-3);
	EXPECT_NEAR(pose.z(), 0.0, 1e-3);
}

TEST(OdometryVIOTest, EstimatesGyroBiasWhileStatic)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// Gyro bias of 0.03 rad/s: without its estimation, the heading would drift by 0.06 rad
	const cv::Vec3d bias(0.01, -0.02, 0.015);
	const double t0 = 0.5, w = 0.5, duration = 2.5;
	Transform pose = runSequence(odom, duration, [=](double t) {
		return std::make_pair(cv::Vec3d(0, 0, t > t0 ? w : 0.0) + bias, cv::Vec3d(0,0,kGravity));
	});
	float roll, pitch, yaw;
	pose.getEulerAngles(roll, pitch, yaw);
	EXPECT_NEAR(roll, 0.0, 2e-3);
	EXPECT_NEAR(pitch, 0.0, 2e-3);
	// The bias is exactly estimated: what remains is the midpoint integration of
	// the angular velocity step (half a sample at 0.5 rad/s = 1.25e-3 rad)
	EXPECT_NEAR(pose.theta(), w * (duration - t0), 2e-3);
}

TEST(OdometryVIOTest, WaitsForStaticWindowAfterMovingStart)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// Shaken for 1 s, static for 0.5 s, then 0.5 m/s^2 along x for 1 s
	const double shake = 1.0, t0 = 1.5, a = 0.5, duration = 2.5;
	int lost = 0;
	Transform pose = runSequence(odom, duration, [=](double t) {
		if(t < shake)
		{
			return std::make_pair(cv::Vec3d(0.3*std::sin(15*t), 0.2*std::cos(11*t), 0),
					cv::Vec3d(2.0*std::sin(10*t), 1.0*std::cos(7*t), kGravity));
		}
		return std::make_pair(cv::Vec3d(0,0,0), cv::Vec3d(t > t0 ? a : 0.0, 0, kGravity));
	}, Transform::getIdentity(), &lost);
	// No pose before the end of the first static window (shake + 20 samples)
	EXPECT_GE(lost, (int)(shake * kCameraRate));
	EXPECT_LE(lost, (int)((shake + 0.15) * kCameraRate));
	const double T = duration - t0;
	EXPECT_NEAR(pose.x(), 0.5 * a * T * T, 0.01);
	EXPECT_NEAR(pose.y(), 0.0, 1e-3);
	EXPECT_NEAR(pose.z(), 0.0, 1e-3);
}

TEST(OdometryVIOTest, ConstantRotationIsNotStatic)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// Spinning at a constant 0.5 rad/s: low variance, but too fast to be a gyro bias
	int lost = 0;
	runSequence(odom, 1.0, [](double) {
		return std::make_pair(cv::Vec3d(0,0,0.5), cv::Vec3d(0,0,kGravity));
	}, Transform::getIdentity(), &lost);
	EXPECT_EQ(lost, (int)(1.0 * kCameraRate) + 1);
}

TEST(OdometryVIOTest, NoisyStaticStartInitializes)
{
	if(!vioAvailable()) GTEST_SKIP() << "RTAB-Map built without GTSAM";
	OdometryVIO odom(vioTestParameters());
	// White noise like a real IMU at rest (acc 0.03 m/s^2, gyro 0.003 rad/s per sample)
	cv::RNG rng(7);
	int lost = 0;
	Transform pose = runSequence(odom, 0.5, [&](double) {
		return std::make_pair(
				cv::Vec3d(rng.gaussian(0.003), rng.gaussian(0.003), rng.gaussian(0.003)),
				cv::Vec3d(rng.gaussian(0.03), rng.gaussian(0.03), kGravity + rng.gaussian(0.03)));
	}, Transform::getIdentity(), &lost);
	EXPECT_LE(lost, 3);
	EXPECT_LT(pose.getNorm(), 0.01);
}
