/*
Copyright (c) 2010-2026, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
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

#include "rtabmap/core/odometry/OdometryVIO.h"
#include "rtabmap/core/odometry/VIOFrontend.h"
#include "rtabmap/core/OdometryInfo.h"
#include "rtabmap/core/util3d_motion_estimation.h"
#include "rtabmap/utilite/ULogger.h"
#include <algorithm>
#include <deque>

#ifdef RTABMAP_GTSAM
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#endif
#ifdef RTABMAP_GTSAM_UNSTABLE
#include "VIOBackend.h"
#endif

namespace rtabmap {

struct OdometryVIO::Impl
{
#ifdef RTABMAP_GTSAM
	// boost::shared_ptr before GTSAM 4.3, std::shared_ptr after
	decltype(gtsam::PreintegrationCombinedParams::MakeSharedU()) params;
	std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> preintegrated;
	gtsam::NavState state; // IMU frame in world (z up)
	gtsam::imuBias::ConstantBias bias;
#endif
	bool initialized = false;
	double lastImuStamp = 0.0;
	Eigen::Vector3d lastAcc = Eigen::Vector3d::Zero();
	Eigen::Vector3d lastOmega = Eigen::Vector3d::Zero();
	// IMU samples received after initialization, not integrated yet (stamp, acc, gyro)
	std::deque<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d> > > imuQueue;
	// Initialization: last IMU samples (stamp, acc, gyro)
	std::deque<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d> > > initSamples;
	// Stereo images before initialization: stamp, track positions and 3D points (base frame)
	std::deque<std::pair<double, std::vector<VIOFrontend::Track> > > initImages;
	StereoCameraModel initModel;
	std::pair<double, double> initMotionStamps; // images of the last camera motion computed
	double initMotion = -1.0;
	int initFilterSamples = 1;
	double initMaxAccStd = 0.0;
	double initMaxGyroStd = 0.0;
	double initMaxGyroBias = 0.0;
	double initMaxMotion = 0.0;
	Transform imuLocalTransform;     // base -> imu
	Transform imuLocalTransformInv;  // imu -> base
	Transform previousPoseInv;       // inverse of last base pose returned
	std::unique_ptr<VIOFrontend> frontend;
#ifdef RTABMAP_GTSAM_UNSTABLE
	std::unique_ptr<VIOBackend> backend;
	std::unique_ptr<VIOLocalMap> localMap; // null if disabled
	bool backendStarted = false; // the back-end has been initialized at least once since clear()
#endif
	double keyframeInterval = 0.0;

	// Integrates the queued IMU samples up to stamp, interpolating the
	// sample at stamp when it falls between two samples.
	void integrateImu(double stamp)
	{
		while(!imuQueue.empty() && lastImuStamp < stamp)
		{
			double t = imuQueue.front().first;
			Eigen::Vector3d acc = imuQueue.front().second.first;
			Eigen::Vector3d omega = imuQueue.front().second.second;
			if(t > stamp)
			{
				double ratio = (stamp - lastImuStamp) / (t - lastImuStamp);
				acc = lastAcc + ratio * (acc - lastAcc);
				omega = lastOmega + ratio * (omega - lastOmega);
				t = stamp;
			}
			else
			{
				imuQueue.pop_front();
			}
#ifdef RTABMAP_GTSAM
			// Midpoint between the two samples bounding the interval
			preintegrated->integrateMeasurement(0.5*(lastAcc+acc), 0.5*(lastOmega+omega), t - lastImuStamp);
#endif
			lastImuStamp = t;
			lastAcc = acc;
			lastOmega = omega;
		}
	}

	void clear()
	{
#ifdef RTABMAP_GTSAM
		preintegrated.reset();
		state = gtsam::NavState();
		bias = gtsam::imuBias::ConstantBias();
#endif
		initialized = false;
		lastImuStamp = 0.0;
		imuQueue.clear();
		initSamples.clear();
		initImages.clear();
		initMotionStamps = std::make_pair(0.0, 0.0);
		initMotion = -1.0;
		imuLocalTransform.setNull();
		imuLocalTransformInv.setNull();
		previousPoseInv.setNull();
		if(frontend)
		{
			frontend->reset();
		}
#ifdef RTABMAP_GTSAM_UNSTABLE
		if(backend)
		{
			backend->reset();
		}
		if(localMap)
		{
			localMap->reset();
		}
		backendStarted = false;
#endif
	}
};

OdometryVIO::OdometryVIO(const ParametersMap & parameters) :
	Odometry(parameters),
	impl_(new Impl),
	initGravity_(false),
	initImuSamples_(Parameters::defaultOdomVIOInitImuSamples()),
	visualOnly_(Parameters::defaultOdomVIOVisualOnly())
{
	impl_->frontend.reset(new VIOFrontend(parameters));
#ifdef RTABMAP_GTSAM_UNSTABLE
	impl_->backend.reset(new VIOBackend(parameters));
	bool localMap = Parameters::defaultOdomVIOLocalMap();
	Parameters::parse(parameters, Parameters::kOdomVIOLocalMap(), localMap);
	if(localMap)
	{
		impl_->localMap.reset(new VIOLocalMap(parameters));
	}
#endif
	impl_->keyframeInterval = Parameters::defaultOdomVIOKeyframeInterval();
	Parameters::parse(parameters, Parameters::kOdomVIOKeyframeInterval(), impl_->keyframeInterval);
	UASSERT(impl_->keyframeInterval >= 0.0);
	impl_->initFilterSamples = Parameters::defaultOdomVIOInitFilterSamples();
	impl_->initMaxAccStd = Parameters::defaultOdomVIOInitMaxAccStd();
	impl_->initMaxGyroStd = Parameters::defaultOdomVIOInitMaxGyroStd();
	impl_->initMaxGyroBias = Parameters::defaultOdomVIOInitMaxGyroBias();
	impl_->initMaxMotion = Parameters::defaultOdomVIOInitMaxMotion();
	Parameters::parse(parameters, Parameters::kOdomVIOInitFilterSamples(), impl_->initFilterSamples);
	UASSERT(impl_->initFilterSamples >= 1);
	Parameters::parse(parameters, Parameters::kOdomVIOInitMaxGyroBias(), impl_->initMaxGyroBias);
	Parameters::parse(parameters, Parameters::kOdomVIOInitMaxAccStd(), impl_->initMaxAccStd);
	Parameters::parse(parameters, Parameters::kOdomVIOInitMaxGyroStd(), impl_->initMaxGyroStd);
	Parameters::parse(parameters, Parameters::kOdomVIOInitMaxMotion(), impl_->initMaxMotion);
	Parameters::parse(parameters, Parameters::kOdomVIOVisualOnly(), visualOnly_);
#ifdef RTABMAP_GTSAM
	double accNoise = Parameters::defaultOdomVIOAccNoise();
	double gyroNoise = Parameters::defaultOdomVIOGyroNoise();
	double accBiasNoise = Parameters::defaultOdomVIOAccBiasNoise();
	double gyroBiasNoise = Parameters::defaultOdomVIOGyroBiasNoise();
	double gravity = Parameters::defaultOdomVIOGravity();
	Parameters::parse(parameters, Parameters::kOdomVIOAccNoise(), accNoise);
	Parameters::parse(parameters, Parameters::kOdomVIOGyroNoise(), gyroNoise);
	Parameters::parse(parameters, Parameters::kOdomVIOAccBiasNoise(), accBiasNoise);
	Parameters::parse(parameters, Parameters::kOdomVIOGyroBiasNoise(), gyroBiasNoise);
	Parameters::parse(parameters, Parameters::kOdomVIOGravity(), gravity);
	UASSERT(accNoise > 0.0 && gyroNoise > 0.0 && accBiasNoise > 0.0 && gyroBiasNoise > 0.0);
	UASSERT(gravity > 0.0);

	// "U" = navigation frame with z up, gravity along -z
	impl_->params = gtsam::PreintegrationCombinedParams::MakeSharedU(gravity);
	impl_->params->accelerometerCovariance = gtsam::I_3x3 * accNoise * accNoise;
	impl_->params->gyroscopeCovariance = gtsam::I_3x3 * gyroNoise * gyroNoise;
	impl_->params->biasAccCovariance = gtsam::I_3x3 * accBiasNoise * accBiasNoise;
	impl_->params->biasOmegaCovariance = gtsam::I_3x3 * gyroBiasNoise * gyroBiasNoise;
	impl_->params->integrationCovariance = gtsam::I_3x3 * 1e-8;
	impl_->params->biasAccOmegaInt = gtsam::I_6x6 * 1e-5;
#endif
	Parameters::parse(parameters, Parameters::kOdomVIOInitImuSamples(), initImuSamples_);
	UASSERT(initImuSamples_ >= 1);
}

OdometryVIO::~OdometryVIO()
{
}

void OdometryVIO::reset(const Transform & initialPose)
{
	Odometry::reset(initialPose);
	if(!initGravity_)
	{
		impl_->clear();
	}
	initGravity_ = false;
}

namespace {

// Standard deviation (RMS distance to the mean) of 3D samples
double stdDev(const std::vector<Eigen::Vector3d> & v, Eigen::Vector3d & mean)
{
	mean.setZero();
	for(size_t i=0; i<v.size(); ++i) mean += v[i];
	mean /= (double)v.size();
	double sum = 0.0;
	for(size_t i=0; i<v.size(); ++i) sum += (v[i]-mean).squaredNorm();
	return std::sqrt(sum / (double)v.size());
}

// Means of consecutive groups of n samples (low-pass filter against vibrations)
std::vector<Eigen::Vector3d> groupMeans(const std::vector<Eigen::Vector3d> & v, int n)
{
	std::vector<Eigen::Vector3d> means;
	for(size_t i=0; i+n<=v.size(); i+=n)
	{
		Eigen::Vector3d mean = Eigen::Vector3d::Zero();
		for(int j=0; j<n; ++j) mean += v[i+j];
		means.push_back(mean / (double)n);
	}
	return means;
}

// Camera translation (m) between two images of the same tracks (PnP), -1 if it cannot be estimated
double cameraMotion(
		const std::vector<VIOFrontend::Track> & from,
		const std::vector<VIOFrontend::Track> & to,
		const CameraModel & model)
{
	std::map<int, cv::Point3f> words3;
	std::map<int, cv::KeyPoint> words2;
	for(size_t i=0; i<from.size(); ++i)
	{
		if(from[i].hasDepth())
		{
			words3.insert(std::make_pair(from[i].id, from[i].point));
		}
	}
	for(size_t i=0; i<to.size(); ++i)
	{
		words2.insert(std::make_pair(to[i].id, cv::KeyPoint(to[i].left, 1.0f)));
	}
	Transform motion = util3d::estimateMotion3DTo2D(words3, words2, model, 10);
	return motion.isNull() ? -1.0 : (double)motion.getNorm();
}

bool hasStereo(const SensorData & data)
{
	return data.stereoCameraModels().size() == 1 &&
			!data.imageRaw().empty() && !data.rightRaw().empty();
}

void fillFrontendInfo(const VIOFrontend & frontend, OdometryInfo * info)
{
	if(!info)
	{
		return;
	}
	const VIOFrontend::Stats & stats = frontend.stats();
	info->features = (int)frontend.tracks().size();
	info->reg.matches = stats.tracked;
	info->reg.inliers = stats.inliers;
	info->refCorners = frontend.previousCorners();
	info->newCorners = frontend.currentCorners();
	info->cornerInliers = frontend.cornerInliers();
	info->words.clear();
	info->reg.inliersIDs.clear();
	for(size_t i=0; i<frontend.tracks().size(); ++i)
	{
		const VIOFrontend::Track & track = frontend.tracks()[i];
		info->words.insert(std::make_pair(track.id, cv::KeyPoint(track.left, 1.0f)));
		if(track.age > 1)
		{
			info->reg.inliersIDs.push_back(track.id);
		}
	}
}

} // namespace

// return not null transform if odometry is correctly computed
Transform OdometryVIO::computeTransform(
		SensorData & data,
		const Transform & guess,
		OdometryInfo * info)
{
	Transform t;
	Impl & d = *impl_;

	if(visualOnly_)
	{
		// Stereo visual odometry of the front-end only, IMU is ignored
		if(hasStereo(data))
		{
			bool firstFrame = !d.frontend->hasPrevious();
			cv::Mat covariance;
			if(d.frontend->process(data.imageRaw(), data.rightRaw(), data.stereoCameraModels()[0], guess, &t, &covariance))
			{
				if(firstFrame)
				{
					t = Transform::getIdentity();
					covariance = cv::Mat::eye(6, 6, CV_64FC1) * 1e-9;
				}
				if(info)
				{
					info->type = this->getType();
					fillFrontendInfo(*d.frontend, info);
					if(!t.isNull())
					{
						info->reg.covariance = covariance;
					}
				}
			}
			if(t.isNull() && info)
			{
				// Lost: callers read the covariance of every frame
				info->reg.covariance = cv::Mat::eye(6, 6, CV_64FC1) * 9999.0;
			}
		}
		else if(!data.imageRaw().empty())
		{
			UERROR("%s=true requires rectified stereo images", Parameters::kOdomVIOVisualOnly().c_str());
		}
		return t;
	}

#ifdef RTABMAP_GTSAM

	if(!data.imu().empty())
	{
		if(d.imuLocalTransform.isNull())
		{
			d.imuLocalTransform = data.imu().localTransform();
			UASSERT(!d.imuLocalTransform.isNull());
			d.imuLocalTransformInv = d.imuLocalTransform.inverse();
		}

		const cv::Vec3d & w = data.imu().angularVelocity();
		const cv::Vec3d & a = data.imu().linearAcceleration();
		Eigen::Vector3d omega(w.val[0], w.val[1], w.val[2]);
		Eigen::Vector3d acc(a.val[0], a.val[1], a.val[2]);

		if(!d.initialized)
		{
			// Static initialization: wait for a window of OdomVIO/InitImuSamples
			// samples where the IMU is still (low variance) and, with stereo
			// images, the image does not move either (an IMU cannot tell
			// constant velocity from no motion). The mean specific force then
			// points up and the mean angular velocity is the gyroscope bias.
			d.initSamples.push_back(std::make_pair(data.stamp(), std::make_pair(acc, omega)));
			while((int)d.initSamples.size() > initImuSamples_)
			{
				d.initSamples.pop_front();
			}
			if((int)d.initSamples.size() == initImuSamples_)
			{
				std::vector<Eigen::Vector3d> accs(d.initSamples.size()), gyros(d.initSamples.size());
				for(size_t i=0; i<d.initSamples.size(); ++i)
				{
					accs[i] = d.initSamples[i].second.first;
					gyros[i] = d.initSamples[i].second.second;
				}
				// Vibrations average out in groups of samples, motion does not
				int groupSize = std::min(d.initFilterSamples, initImuSamples_);
				Eigen::Vector3d accMean, gyroMean;
				double accStd = stdDev(groupMeans(accs, groupSize), accMean);
				double gyroStd = stdDev(groupMeans(gyros, groupSize), gyroMean);
				// Gravity and gyroscope bias from all samples (the groups may leave some out)
				stdDev(accs, accMean);
				stdDev(gyros, gyroMean);
				// A constant rotation also has a low variance: the mean angular velocity must look like a bias
				bool imuStill = accStd <= d.initMaxAccStd && gyroStd <= d.initMaxGyroStd && gyroMean.norm() <= d.initMaxGyroBias;
				// With stereo images, the camera must not have moved since the start of the window
				bool imageStill = true;
				if(imuStill && !d.initImages.empty())
				{
					std::deque<std::pair<double, std::vector<VIOFrontend::Track> > >::const_iterator first = d.initImages.begin();
					while(first != d.initImages.end() && first->first < d.initSamples.front().first)
					{
						++first;
					}
					double motion = -1.0;
					if(first != d.initImages.end())
					{
						std::pair<double, double> stamps(first->first, d.initImages.back().first);
						if(stamps != d.initMotionStamps)
						{
							d.initMotion = cameraMotion(first->second, d.initImages.back().second, d.initModel.left());
							d.initMotionStamps = stamps;
						}
						motion = d.initMotion;
					}
					imageStill = motion >= 0.0 && motion <= d.initMaxMotion &&
							d.initImages.front().first < d.initSamples.front().first; // images cover the window
				}
				if(imuStill && imageStill)
				{
					Eigen::Vector3d up = accMean.normalized();
					gtsam::Rot3 R_wi(Eigen::Quaterniond::FromTwoVectors(up, Eigen::Vector3d::UnitZ()));
					// Gravity gives roll and pitch only: start with the base frame (not the IMU) at yaw 0
					Transform baseRotation = Transform::fromEigen4d(gtsam::Pose3(R_wi, gtsam::Point3(0,0,0)).matrix()) * d.imuLocalTransformInv.rotation();
					R_wi = gtsam::Rot3::Rz(-baseRotation.theta()) * R_wi;
					d.state = gtsam::NavState(R_wi, gtsam::Point3(0,0,0), gtsam::Velocity3(0,0,0));
					d.bias = gtsam::imuBias::ConstantBias(gtsam::Vector3(0,0,0), gyroMean);
					d.preintegrated.reset(new gtsam::PreintegratedCombinedMeasurements(d.params, d.bias));
					d.initialized = true;
					d.initSamples.clear();
					d.initImages.clear();
					UINFO("VIO initialized from %d static IMU samples (acc std=%f, gyro std=%f): "
							"roll=%f pitch=%f, gyro bias=(%f,%f,%f)",
							initImuSamples_, accStd, gyroStd, R_wi.roll(), R_wi.pitch(),
							gyroMean[0], gyroMean[1], gyroMean[2]);
				}
				else
				{
					UDEBUG("Waiting for a static window to initialize VIO (acc std=%f/%f, gyro std=%f/%f, gyro mean=%f/%f, image still=%d (-1: not checked))",
							accStd, d.initMaxAccStd, gyroStd, d.initMaxGyroStd, gyroMean.norm(), d.initMaxGyroBias, imuStill?(imageStill?1:0):-1);
				}
			}
		}
		else
		{
			// Integrated up to the stamp of the next image
			if(data.stamp() > (d.imuQueue.empty()?d.lastImuStamp:d.imuQueue.back().first))
			{
				d.imuQueue.push_back(std::make_pair(data.stamp(), std::make_pair(acc, omega)));
				// Without images, do not keep the samples forever
				while(d.imuQueue.back().first - d.imuQueue.front().first > 1.0)
				{
					d.integrateImu(d.imuQueue.front().first);
				}
			}
			else
			{
				UWARN("Ignoring IMU measurement with non-increasing stamp (%f)", data.stamp());
			}
		}
		if(!d.initialized || d.lastImuStamp == 0.0)
		{
			d.lastImuStamp = data.stamp();
			d.lastAcc = acc;
			d.lastOmega = omega;
		}
	}

	bool isImageFrame = !data.imageRaw().empty() ||
			!data.cameraModels().empty() ||
			!data.stereoCameraModels().empty();
	if(isImageFrame)
	{
		if(!d.initialized)
		{
			// Track the stereo images to know whether the camera is still
			if(hasStereo(data))
			{
				d.frontend->process(data.imageRaw(), data.rightRaw(), data.stereoCameraModels()[0]);
				fillFrontendInfo(*d.frontend, info);
				d.initModel = data.stereoCameraModels()[0];
				d.initImages.push_back(std::make_pair(data.stamp(), d.frontend->tracks()));
				// Keep one image older than the IMU window (and a bounded history without IMU)
				while(d.initImages.size() > 2 &&
					  ((!d.initSamples.empty() && d.initImages[1].first <= d.initSamples.front().first) || d.initImages.size() > 100))
				{
					d.initImages.pop_front();
				}
			}
			UWARN("VIO not initialized yet, waiting for the sensor to be static (%d/%d IMU samples)...",
					(int)d.initSamples.size(), initImuSamples_);
			if(info)
			{
				info->reg.covariance = cv::Mat::eye(6, 6, CV_64FC1) * 9999.0;
			}
			return t;
		}

		// Predict at the stamp of the image (IMU is expected to be fed before
		// the image of the same time, later samples stay queued). d.state is
		// the state where the pre-integration started: the last keyframe with
		// the back-end, the last image frame otherwise.
		d.integrateImu(data.stamp());
		if(d.lastImuStamp < data.stamp() - 0.01)
		{
			UWARN("Image at %f is %f s after the last IMU sample, the IMU should be fed before the image",
					data.stamp(), data.stamp() - d.lastImuStamp);
		}
		gtsam::NavState predicted = d.preintegrated->predict(d.state, d.bias);
		Eigen::Matrix<double, 15, 15> cov = d.preintegrated->preintMeasCov();
		gtsam::NavState output = predicted;
		bool restartIntegration = true;

		if(hasStereo(data))
		{
			// The IMU motion initializes KLT
			Transform motionGuess;
			if(!d.previousPoseInv.isNull())
			{
				motionGuess = d.previousPoseInv * Transform::fromEigen4d(predicted.pose().matrix()) * d.imuLocalTransformInv;
			}
			const StereoCameraModel & model = data.stereoCameraModels()[0];
			d.frontend->process(data.imageRaw(), data.rightRaw(), model, motionGuess);
			fillFrontendInfo(*d.frontend, info);

#ifdef RTABMAP_GTSAM_UNSTABLE
			// (1e-6: stamps are large numbers, 1000.1-1000.0 < 0.1)
			bool keyframe = !d.backend->initialized() ||
					(data.stamp() > d.backend->lastKeyframeStamp() &&
					 data.stamp() - d.backend->lastKeyframeStamp() >= d.keyframeInterval - 1e-6);
			if(keyframe)
			{
				bool success;
				const gtsam::Pose3 imuToCamera((d.imuLocalTransformInv * model.left().localTransform()).toEigen4d());
				const gtsam::Pose3 imuToBase(d.imuLocalTransformInv.toEigen4d());
				std::map<int, VIOBackend::Landmark> landmarks;
				if(d.localMap)
				{
					landmarks = d.localMap->match(data.imageRaw(), d.frontend->tracks(), model, predicted.pose(), imuToCamera, imuToBase);
				}
				if(!d.backend->initialized())
				{
					if(d.localMap && d.backendStarted)
					{
						// Restarting after a failure: the map is in the old frame
						d.localMap->reset();
						landmarks = d.localMap->match(data.imageRaw(), d.frontend->tracks(), model, predicted.pose(), imuToCamera, imuToBase);
					}
					// Static start: zero velocity. After a failure, the velocity is the IMU prediction.
					double velocitySigma = d.backendStarted ? 1.0 : 0.01;
					success = d.backend->initialize(data.stamp(), predicted, d.bias, imuToCamera, model,
							d.frontend->tracks(), velocitySigma);
					d.backendStarted = true;
				}
				else
				{
					success = d.backend->addKeyframe(data.stamp(), *d.preintegrated, predicted, d.frontend->tracks(), landmarks);
				}
				if(success)
				{
					output = d.backend->state();
					d.bias = d.backend->bias();
					if(d.localMap)
					{
						d.localMap->update(data.stamp(), d.frontend->tracks(), output.pose(), imuToBase, d.backend->landmarkEstimates());
					}
				}
				// else: keep the IMU prediction, the back-end restarts at the next keyframe
			}
			else
			{
				// Keep integrating from the last keyframe
				restartIntegration = false;
			}
#endif
		}

		Transform imuPose = Transform::fromEigen4d(output.pose().matrix());
		Transform p = imuPose * d.imuLocalTransformInv; // base pose in world

		if(this->getPose().rotation().isIdentity() && d.previousPoseInv.isNull())
		{
			// Align the odometry frame with gravity, keep our state on reset
			initGravity_ = true;
			this->reset(this->getPose() * p.rotation());
		}

		if(d.previousPoseInv.isNull())
		{
			d.previousPoseInv = p.inverse();
		}

		t = d.previousPoseInv * p;
		d.previousPoseInv = p.inverse();

		if(restartIntegration)
		{
			d.state = output;
			d.preintegrated->resetIntegrationAndSetBias(d.bias);
		}

		if(info)
		{
			info->type = this->getType();
			// GTSAM order: [rot, pos, vel, ba, bg] -> rtabmap [x y z roll pitch yaw]
			info->reg.covariance = cv::Mat::eye(6, 6, CV_64FC1) * 1e-9;
			for(int i=0; i<3; ++i)
			{
				for(int j=0; j<3; ++j)
				{
					info->reg.covariance.at<double>(i,j) += cov(3+i,3+j);
					info->reg.covariance.at<double>(3+i,3+j) += cov(i,j);
				}
			}
		}
	}

#else
	UERROR("RTAB-Map is not built with GTSAM support! Select another odometry approach.");
#endif
	return t;
}

} // namespace rtabmap
