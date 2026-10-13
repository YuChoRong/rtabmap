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
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#endif
#ifdef RTABMAP_GTSAM_UNSTABLE
#include "VIOBackend.h"
#endif

namespace rtabmap {

namespace {

// Online estimation of the time offset between the camera and the IMU
// clocks (t_imu = t_camera + offset), by aligning the rotations of the
// stereo visual odometry with the gyroscope integrated over the same
// intervals shifted by the offset. The rotation integrated over [a+td, b+td]
// changes with td by the angular velocity at its ends, so the angular
// accelerations of the motion make the offset observable.
class TimeOffsetEstimator
{
public:
	TimeOffsetEstimator(double window, double maxOffset) :
		window_(window), maxOffset_(maxOffset) {}

	void clear()
	{
		gyros_.clear();
		rotations_.clear();
	}

	void addGyro(double stamp, const Eigen::Vector3d & omega)
	{
		if(gyros_.empty() || stamp > gyros_.back().first)
		{
			gyros_.push_back(std::make_pair(stamp, omega));
		}
		while(gyros_.size() > 2 && gyros_.front().first < stamp - window_ - 2.0*maxOffset_ - 1.0)
		{
			gyros_.pop_front();
		}
	}

	// Camera stamp and IMU frame rotation (in an arbitrary frame) of a
	// visual odometry chain, null rotation when the chain breaks
	void addRotation(double stamp, const Eigen::Matrix3d & rotation, bool chained)
	{
		if(!chained)
		{
			rotations_.clear();
		}
		rotations_.push_back(Rotation(stamp, rotation));
		while(rotations_.size() > 2 && rotations_.front().stamp < stamp - window_)
		{
			rotations_.pop_front();
		}
	}

	// Estimates the offset from the window starting at offset (the current
	// one), returns false if the window does not constrain it enough
	bool estimate(const Eigen::Vector3d & gyroBias, double & offset, double & sigma, int & pairs) const
	{
		const double interval = 0.2; // s between the images of a pair
		std::vector<std::pair<size_t, size_t> > p;
		for(size_t i=0, j=0; i<rotations_.size(); ++i)
		{
			while(j < rotations_.size() && rotations_[j].stamp < rotations_[i].stamp + interval - 1e-6)
			{
				++j;
			}
			if(j >= rotations_.size())
			{
				break;
			}
			p.push_back(std::make_pair(i, j));
		}
		pairs = (int)p.size();
		if(pairs < 10 || gyros_.size() < 2)
		{
			return false;
		}
		double td = offset;
		const double h = 1e-3;
		double information = 0.0;
		double s2 = 0.0;
		for(int iteration=0; iteration<10; ++iteration)
		{
			double JtJ = 0.0, Jtr = 0.0, rtr = 0.0;
			int n = 0;
			for(size_t k=0; k<p.size(); ++k)
			{
				const Rotation & a = rotations_[p[k].first];
				const Rotation & b = rotations_[p[k].second];
				if(a.stamp + td - h < gyros_.front().first || b.stamp + td + h > gyros_.back().first)
				{
					continue;
				}
				Eigen::Matrix3d visual = a.rotation.transpose() * b.rotation;
				Eigen::Vector3d r = residual(a.stamp, b.stamp, td, gyroBias, visual);
				Eigen::Vector3d J = (residual(a.stamp, b.stamp, td+h, gyroBias, visual) - residual(a.stamp, b.stamp, td-h, gyroBias, visual)) / (2.0*h);
				// Huber weight against wrong visual rotations
				const double delta = 0.005; // rad
				double norm = r.norm();
				double w = norm <= delta ? 1.0 : delta / norm;
				JtJ += w * J.dot(J);
				Jtr += w * J.dot(r);
				rtr += w * r.dot(r);
				++n;
			}
			if(n < 10 || JtJ <= 0.0)
			{
				return false;
			}
			double step = -Jtr / JtJ;
			step = std::max(-0.02, std::min(0.02, step));
			td += step;
			information = JtJ;
			s2 = rtr / std::max(1, 3*n - 1);
			if(std::fabs(step) < 1e-5)
			{
				break;
			}
		}
		sigma = std::sqrt(s2 / information);
		if(std::fabs(td) > maxOffset_)
		{
			return false;
		}
		offset = td;
		return true;
	}

private:
	struct Rotation
	{
		Rotation(double s, const Eigen::Matrix3d & r) : stamp(s), rotation(r) {}
		double stamp;
		Eigen::Matrix3d rotation;
	};

	Eigen::Vector3d gyroAt(double stamp) const
	{
		std::deque<std::pair<double, Eigen::Vector3d> >::const_iterator iter = std::lower_bound(
				gyros_.begin(), gyros_.end(), stamp,
				[](const std::pair<double, Eigen::Vector3d> & s, double t) {return s.first < t;});
		if(iter == gyros_.begin()) return iter->second;
		if(iter == gyros_.end()) return gyros_.back().second;
		std::deque<std::pair<double, Eigen::Vector3d> >::const_iterator prev = iter - 1;
		double ratio = (stamp - prev->first) / (iter->first - prev->first);
		return prev->second + ratio * (iter->second - prev->second);
	}

	// Gyroscope rotation between two IMU stamps (midpoint integration)
	Eigen::Matrix3d integrate(double from, double to, const Eigen::Vector3d & bias) const
	{
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		std::deque<std::pair<double, Eigen::Vector3d> >::const_iterator iter = std::upper_bound(
				gyros_.begin(), gyros_.end(), from,
				[](double t, const std::pair<double, Eigen::Vector3d> & s) {return t < s.first;});
		double t = from;
		Eigen::Vector3d last = gyroAt(from);
		while(t < to)
		{
			double next = iter != gyros_.end() ? std::min(iter->first, to) : to;
			Eigen::Vector3d omega = iter != gyros_.end() && next < to ? iter->second : gyroAt(to);
			Eigen::Vector3d phi = (0.5*(last+omega) - bias) * (next - t);
			double angle = phi.norm();
			if(angle > 1e-12)
			{
				R = R * Eigen::AngleAxisd(angle, phi/angle).toRotationMatrix();
			}
			t = next;
			last = omega;
			if(iter != gyros_.end()) ++iter;
		}
		return R;
	}

	Eigen::Vector3d residual(double a, double b, double td, const Eigen::Vector3d & bias, const Eigen::Matrix3d & visual) const
	{
		Eigen::AngleAxisd aa(integrate(a+td, b+td, bias).transpose() * visual);
		return aa.angle() * aa.axis();
	}

	double window_;
	double maxOffset_;
	std::deque<std::pair<double, Eigen::Vector3d> > gyros_;
	std::deque<Rotation> rotations_;
};

} // namespace

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
	// IMU samples received after initialization (last second), to restart the integration earlier
	std::deque<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d> > > imuHistory;
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
	// Dynamic initialization: IMU samples and stereo visual odometry poses (base frame) of the window
	double initDynamicWindow = 0.0;
	std::deque<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d> > > dynamicSamples;
	std::deque<std::pair<double, Transform> > dynamicPoses;
	double initVelocitySigma = 0.01; // velocity prior of the back-end start
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
	// Time offset between the camera and IMU clocks: t_imu = t_camera + timeOffset
	double timeOffset = 0.0;
	Eigen::Matrix3d offsetRotation = Eigen::Matrix3d::Identity();
	double initialTimeOffset = 0.0;
	std::unique_ptr<TimeOffsetEstimator> offsetEstimator; // null if not estimated

#ifdef RTABMAP_GTSAM
	// State extrapolated by dt with the last IMU sample
	gtsam::NavState extrapolate(const gtsam::NavState & from, double dt) const
	{
		gtsam::PreintegratedCombinedMeasurements extra(params, bias);
		extra.integrateMeasurement(lastAcc, lastOmega, dt);
		return extra.predict(from, bias);
	}
#endif

#ifdef RTABMAP_GTSAM_UNSTABLE
	// Back-end keyframe, waiting for its IMU samples when the camera is late
	struct Keyframe
	{
		double stamp; // IMU clock
		cv::Mat image;
		std::vector<VIOFrontend::Track> tracks;
		StereoCameraModel model;
	};
	std::unique_ptr<Keyframe> pending;

	// Adds a keyframe to the back-end, IMU integrated up to its stamp.
	// Returns the optimized state, or predicted if the back-end failed (it
	// restarts at the next keyframe).
	gtsam::NavState processKeyframe(const Keyframe & frame, const gtsam::NavState & predicted)
	{
		gtsam::NavState output = predicted;
		bool success;
		const gtsam::Pose3 imuToCamera((imuLocalTransformInv * frame.model.left().localTransform()).toEigen4d());
		const gtsam::Pose3 imuToBase(imuLocalTransformInv.toEigen4d());
		std::map<int, VIOBackend::Landmark> landmarks;
		if(localMap)
		{
			landmarks = localMap->match(frame.image, frame.tracks, frame.model, predicted.pose(), imuToCamera, imuToBase);
		}
		if(!backend->initialized())
		{
			if(localMap && backendStarted)
			{
				// Restarting after a failure: the map is in the old frame
				localMap->reset();
				landmarks = localMap->match(frame.image, frame.tracks, frame.model, predicted.pose(), imuToCamera, imuToBase);
			}
			// Static start: zero velocity, dynamic start: aligned velocity. After a failure, the velocity is the IMU prediction.
			double velocitySigma = backendStarted ? 1.0 : initVelocitySigma;
			success = backend->initialize(frame.stamp, predicted, bias, imuToCamera, frame.model,
					frame.tracks, velocitySigma);
			backendStarted = true;
		}
		else
		{
			success = backend->addKeyframe(frame.stamp, *preintegrated, predicted, frame.tracks, landmarks);
		}
		if(success)
		{
			output = backend->state();
			bias = backend->bias();
			if(offsetEstimator)
			{
				double offset = timeOffset, sigma = 0.0;
				int pairs = 0;
				// Changed only when significantly different (each change restarts the IMU integration)
				if(offsetEstimator->estimate(bias.gyroscope(), offset, sigma, pairs) && sigma < 0.0005 &&
				   std::fabs(offset - timeOffset) > std::max(3.0*sigma, 0.002))
				{
					// The next IMU factor starts at the keyframe in the new
					// clock, so that it spans the same time as the images
					restartImuAt(frame.stamp + offset - timeOffset, output);
					timeOffset = offset;
				}
				UINFO("Time offset: %f s (window estimate %f s, sigma=%f s, %d pairs)", timeOffset, offset, sigma, pairs);
			}
			if(localMap)
			{
				localMap->update(frame.stamp, frame.tracks, output.pose(), imuToBase, backend->landmarkEstimates());
			}
		}
		return output;
	}
#endif

#ifdef RTABMAP_GTSAM
	// Restarts the IMU integration at stamp (the state stays the same)
	void restartImuAt(double stamp, const gtsam::NavState & state)
	{
		imuQueue.clear();
		size_t i = 0;
		while(i < imuHistory.size() && imuHistory[i].first <= stamp)
		{
			++i;
		}
		if(i > 0 && i < imuHistory.size())
		{
			double ratio = (stamp - imuHistory[i-1].first) / (imuHistory[i].first - imuHistory[i-1].first);
			lastAcc = imuHistory[i-1].second.first + ratio * (imuHistory[i].second.first - imuHistory[i-1].second.first);
			lastOmega = imuHistory[i-1].second.second + ratio * (imuHistory[i].second.second - imuHistory[i-1].second.second);
		}
		else if(!imuHistory.empty())
		{
			lastAcc = imuHistory[i>0?i-1:0].second.first;
			lastOmega = imuHistory[i>0?i-1:0].second.second;
		}
		imuQueue.insert(imuQueue.end(), imuHistory.begin()+i, imuHistory.end());
		lastImuStamp = stamp;
		this->state = state;
		preintegrated->resetIntegrationAndSetBias(bias);
	}
#endif

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
		imuHistory.clear();
		initSamples.clear();
		initImages.clear();
		initMotionStamps = std::make_pair(0.0, 0.0);
		initMotion = -1.0;
		dynamicSamples.clear();
		dynamicPoses.clear();
		initVelocitySigma = 0.01;
		timeOffset = initialTimeOffset;
		offsetRotation.setIdentity();
#ifdef RTABMAP_GTSAM_UNSTABLE
		pending.reset();
#endif
		if(offsetEstimator)
		{
			offsetEstimator->clear();
		}
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
	impl_->initDynamicWindow = Parameters::defaultOdomVIOInitDynamicWindow();
	Parameters::parse(parameters, Parameters::kOdomVIOInitDynamicWindow(), impl_->initDynamicWindow);
	UASSERT(impl_->initDynamicWindow >= 0.0);
	Parameters::parse(parameters, Parameters::kOdomVIOVisualOnly(), visualOnly_);
	impl_->initialTimeOffset = Parameters::defaultOdomVIOTimeOffset();
	Parameters::parse(parameters, Parameters::kOdomVIOTimeOffset(), impl_->initialTimeOffset);
	impl_->timeOffset = impl_->initialTimeOffset;
	bool estimateTimeOffset = Parameters::defaultOdomVIOEstimateTimeOffset();
	Parameters::parse(parameters, Parameters::kOdomVIOEstimateTimeOffset(), estimateTimeOffset);
	if(estimateTimeOffset)
	{
		impl_->offsetEstimator.reset(new TimeOffsetEstimator(20.0, 0.1));
	}
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
#if GTSAM_VERSION_NUMERIC >= 40300
	// GTSAM 4.3 no longer adds biasAccOmegaInt to the preintegration covariance.
	// In 4.2 it entered exactly like extra accelerometer/gyroscope white noise
	// (same Jacobians), so add it there to keep the tuned behavior.
	impl_->params->accelerometerCovariance += gtsam::I_3x3 * 1e-5;
	impl_->params->gyroscopeCovariance += gtsam::I_3x3 * 1e-5;
#else
	impl_->params->biasAccOmegaInt = gtsam::I_6x6 * 1e-5;
#endif
#endif
	Parameters::parse(parameters, Parameters::kOdomVIOInitImuSamples(), initImuSamples_);
	UASSERT(initImuSamples_ >= 1);
}

OdometryVIO::~OdometryVIO()
{
}

double OdometryVIO::timeOffset() const
{
	return impl_->timeOffset;
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


#ifdef RTABMAP_GTSAM
typedef std::deque<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d> > > ImuSamples;

// IMU sample (acc, gyro) at a stamp, interpolated between the samples around it
std::pair<Eigen::Vector3d, Eigen::Vector3d> imuSampleAt(const ImuSamples & samples, double stamp)
{
	UASSERT(!samples.empty());
	if(stamp <= samples.front().first)
	{
		return samples.front().second;
	}
	for(size_t i=1; i<samples.size(); ++i)
	{
		if(samples[i].first >= stamp)
		{
			double ratio = (stamp - samples[i-1].first) / (samples[i].first - samples[i-1].first);
			return std::make_pair(
					samples[i-1].second.first + ratio * (samples[i].second.first - samples[i-1].second.first),
					samples[i-1].second.second + ratio * (samples[i].second.second - samples[i-1].second.second));
		}
	}
	return samples.back().second;
}

// Pre-integrates the samples between two stamps (midpoint integration, as OdometryVIO::Impl::integrateImu())
void preintegrate(const ImuSamples & samples, double from, double to, gtsam::PreintegratedCombinedMeasurements & preintegrated)
{
	double t = from;
	std::pair<Eigen::Vector3d, Eigen::Vector3d> last = imuSampleAt(samples, from);
	for(size_t i=0; i<=samples.size() && t < to; ++i)
	{
		double next = i < samples.size() ? std::min(samples[i].first, to) : to;
		if(next <= t)
		{
			continue;
		}
		std::pair<Eigen::Vector3d, Eigen::Vector3d> sample = next < to && i < samples.size() ? samples[i].second : imuSampleAt(samples, to);
		preintegrated.integrateMeasurement(0.5*(last.first+sample.first), 0.5*(last.second+sample.second), next - t);
		t = next;
		last = sample;
	}
}

// Dynamic initialization: aligns a window of stereo visual odometry poses
// (base frame, metric but in an arbitrary frame) with the IMU pre-integrated
// between them. The unknowns are the IMU states (gravity direction, velocity)
// and the biases. Returns the state and bias at the last pose.
bool alignVisualInertial(
		const std::deque<std::pair<double, Transform> > & poses,
		const ImuSamples & samples,
		const decltype(gtsam::PreintegrationCombinedParams::MakeSharedU()) & params,
		const Transform & imuLocalTransform, // base -> imu
		double maxGyroBias,
		gtsam::NavState & state,
		gtsam::imuBias::ConstantBias & bias)
{
	using gtsam::symbol_shorthand::X;
	using gtsam::symbol_shorthand::V;
	using gtsam::symbol_shorthand::B;

	// Keyframes at least 0.1 s apart, the last pose included
	std::vector<size_t> keyframes(1, poses.size()-1);
	for(int i=(int)poses.size()-2; i>=0; --i)
	{
		if(poses[keyframes.back()].first - poses[i].first >= 0.1 - 1e-6)
		{
			keyframes.push_back(i);
		}
	}
	std::reverse(keyframes.begin(), keyframes.end());
	if(keyframes.size() < 5 || samples.empty() ||
	   samples.front().first > poses[keyframes.front()].first ||
	   samples.back().first < poses[keyframes.back()].first - 0.01)
	{
		return false;
	}

	// IMU poses in the visual odometry frame
	std::vector<Eigen::Isometry3d> imuPoses(keyframes.size());
	for(size_t k=0; k<keyframes.size(); ++k)
	{
		imuPoses[k] = Eigen::Isometry3d((poses[keyframes[k]].second * imuLocalTransform).toEigen4d());
	}
	const double duration = poses[keyframes.back()].first - poses[keyframes.front()].first;

	// Gravity direction (up) in the first IMU frame: mean specific force minus
	// the mean acceleration seen by the visual odometry
	Eigen::Vector3d meanForce = Eigen::Vector3d::Zero();
	int count = 0;
	for(size_t i=0; i<samples.size(); ++i)
	{
		double t = samples[i].first;
		if(t < poses[keyframes.front()].first || t > poses[keyframes.back()].first)
		{
			continue;
		}
		// rotation of the closest keyframe (the rotation is small between keyframes)
		size_t k = 0;
		while(k+1 < keyframes.size() && poses[keyframes[k+1]].first <= t) ++k;
		meanForce += imuPoses[0].linear().transpose() * imuPoses[k].linear() * samples[i].second.first;
		++count;
	}
	if(count == 0)
	{
		return false;
	}
	meanForce /= (double)count;
	std::vector<Eigen::Vector3d> velocities(keyframes.size()); // first IMU frame
	for(size_t k=0; k<keyframes.size(); ++k)
	{
		size_t a = k>0?k-1:k, b = k+1<keyframes.size()?k+1:k;
		double dt = poses[keyframes[b]].first - poses[keyframes[a]].first;
		velocities[k] = imuPoses[0].linear().transpose() * (imuPoses[b].translation() - imuPoses[a].translation()) / dt;
	}
	Eigen::Vector3d up = (meanForce - (velocities.back() - velocities.front()) / duration).normalized();

	// World frame: z up, origin and yaw of the base at the first keyframe (as the static initialization)
	gtsam::Rot3 R_wi(Eigen::Quaterniond::FromTwoVectors(up, Eigen::Vector3d::UnitZ()));
	Transform baseRotation = Transform::fromEigen4d(gtsam::Pose3(R_wi, gtsam::Point3(0,0,0)).matrix()) * imuLocalTransform.inverse().rotation();
	R_wi = gtsam::Rot3::Rz(-baseRotation.theta()) * R_wi;
	const Eigen::Isometry3d worldFromVo = Eigen::Isometry3d(gtsam::Pose3(R_wi, gtsam::Point3(0,0,0)).matrix()) * imuPoses[0].inverse();

	gtsam::NonlinearFactorGraph graph;
	gtsam::Values values;
	// Gauge: position and yaw of the first keyframe. Roll and pitch are free,
	// the IMU observes them (rotation tangent in the body frame).
	Eigen::Vector3d g = R_wi.transpose() * Eigen::Vector3d::UnitZ();
	gtsam::Matrix6 information = gtsam::Matrix6::Zero();
	information.block<3,3>(0,0) = g*g.transpose() / (0.001*0.001) + (Eigen::Matrix3d::Identity() - g*g.transpose()) / (1.0*1.0);
	information.block<3,3>(3,3) = Eigen::Matrix3d::Identity() / (0.001*0.001);
	gtsam::Vector6 biasSigmas;
	biasSigmas << 0.2, 0.2, 0.2, 0.05, 0.05, 0.05; // acc (m/s^2), gyro (rad/s)
	// Stereo visual odometry between keyframes 0.1 s apart
	gtsam::Vector6 odomSigmas;
	odomSigmas << 0.01, 0.01, 0.01, 0.02, 0.02, 0.02; // rot (rad), pos (m)

	for(size_t k=0; k<keyframes.size(); ++k)
	{
		gtsam::Pose3 pose((worldFromVo * imuPoses[k]).matrix());
		values.insert(X(k), pose);
		values.insert(V(k), gtsam::Vector3(R_wi.matrix() * velocities[k]));
		values.insert(B(k), gtsam::imuBias::ConstantBias());
		if(k == 0)
		{
			graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3> >(X(0), pose, gtsam::noiseModel::Gaussian::Information(information));
			graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias> >(B(0), gtsam::imuBias::ConstantBias(), gtsam::noiseModel::Diagonal::Sigmas(biasSigmas));
		}
		else
		{
			gtsam::Pose3 relative((imuPoses[k-1].inverse() * imuPoses[k]).matrix());
			graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3> >(X(k-1), X(k), relative, gtsam::noiseModel::Diagonal::Sigmas(odomSigmas));
			gtsam::PreintegratedCombinedMeasurements preintegrated(params, gtsam::imuBias::ConstantBias());
			preintegrate(samples, poses[keyframes[k-1]].first, poses[keyframes[k]].first, preintegrated);
			graph.emplace_shared<gtsam::CombinedImuFactor>(X(k-1), V(k-1), X(k), V(k), B(k-1), B(k), preintegrated);
		}
	}

	gtsam::Values result;
	try
	{
		gtsam::LevenbergMarquardtParams lmParams;
		lmParams.maxIterations = 30;
		result = gtsam::LevenbergMarquardtOptimizer(graph, values, lmParams).optimize();
	}
	catch(const std::exception & e)
	{
		UWARN("Dynamic initialization failed: %s", e.what());
		return false;
	}

	// Consistency: chi2 of the IMU and visual odometry factors, plausible biases
	double chi2 = 2.0 * graph.error(result) / (double)(keyframes.size()-1);
	size_t last = keyframes.size()-1;
	bias = result.at<gtsam::imuBias::ConstantBias>(B(last));
	state = gtsam::NavState(result.at<gtsam::Pose3>(X(last)), result.at<gtsam::Vector3>(V(last)));
	double tilt = gtsam::Rot3::Logmap(R_wi.between(result.at<gtsam::Pose3>(X(0)).rotation())).norm();
	bool accepted = chi2 < 50.0 && bias.accelerometer().norm() < 0.5 && bias.gyroscope().norm() < maxGyroBias;
	UINFO("Dynamic initialization over %.2f s (%d keyframes): %s (chi2/keyframe=%f, gravity correction=%f deg, "
			"acc bias=(%f,%f,%f), gyro bias=(%f,%f,%f), speed=%f m/s)",
			duration, (int)keyframes.size(), accepted?"accepted":"rejected", chi2, tilt*180.0/M_PI,
			bias.accelerometer()[0], bias.accelerometer()[1], bias.accelerometer()[2],
			bias.gyroscope()[0], bias.gyroscope()[1], bias.gyroscope()[2], state.v().norm());
	return accepted;
}
#endif

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
			if(d.initDynamicWindow > 0.0)
			{
				d.dynamicSamples.push_back(d.initSamples.back());
				while(d.dynamicSamples.front().first < data.stamp() - d.initDynamicWindow - 0.5)
				{
					d.dynamicSamples.pop_front();
				}
			}
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
			if(d.imuHistory.empty() || data.stamp() > d.imuHistory.back().first)
			{
				d.imuHistory.push_back(std::make_pair(data.stamp(), std::make_pair(acc, omega)));
				while(d.imuHistory.front().first < data.stamp() - 1.0)
				{
					d.imuHistory.pop_front();
				}
				if(data.stamp() > (d.imuQueue.empty()?d.lastImuStamp:d.imuQueue.back().first))
				{
					d.imuQueue.push_back(d.imuHistory.back());
				}
				if(d.offsetEstimator)
				{
					d.offsetEstimator->addGyro(data.stamp(), omega);
				}
#ifdef RTABMAP_GTSAM_UNSTABLE
				if(d.pending && data.stamp() >= d.pending->stamp)
				{
					// The IMU samples of the waiting keyframe are received
					d.integrateImu(d.pending->stamp);
					d.state = d.processKeyframe(*d.pending, d.preintegrated->predict(d.state, d.bias));
					d.preintegrated->resetIntegrationAndSetBias(d.bias);
					d.pending.reset();
				}
#endif
				// Without images, do not keep the samples forever
				while(!d.imuQueue.empty() && d.imuQueue.back().first - d.imuQueue.front().first > 1.0)
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
		// Image stamp in the IMU clock
		const double imageStamp = data.stamp() + d.timeOffset;
		bool tracked = false; // the frontend already processed this image
		if(!d.initialized)
		{
			// Track the stereo images to know whether the camera is still
			if(hasStereo(data))
			{
				bool hadPrevious = d.frontend->hasPrevious();
				Transform motion;
				bool ok = d.frontend->process(data.imageRaw(), data.rightRaw(), data.stereoCameraModels()[0], Transform(), &motion);
				fillFrontendInfo(*d.frontend, info);
				tracked = true;
				if(d.initDynamicWindow > 0.0)
				{
					// Stereo visual odometry poses of the window, restarted when tracking fails
					if(!ok || (hadPrevious && motion.isNull()))
					{
						d.dynamicPoses.clear();
					}
					else
					{
						d.dynamicPoses.push_back(std::make_pair(imageStamp,
								d.dynamicPoses.empty() || !hadPrevious ? Transform::getIdentity() : d.dynamicPoses.back().second * motion));
						while(d.dynamicPoses.size() > 1 && d.dynamicPoses[1].first <= imageStamp - d.initDynamicWindow + 1e-6)
						{
							d.dynamicPoses.pop_front();
						}
					}
				}
				d.initModel = data.stereoCameraModels()[0];
				d.initImages.push_back(std::make_pair(imageStamp, d.frontend->tracks()));
				// Keep one image older than the IMU window (and a bounded history without IMU)
				while(d.initImages.size() > 2 &&
					  ((!d.initSamples.empty() && d.initImages[1].first <= d.initSamples.front().first) || d.initImages.size() > 100))
				{
					d.initImages.pop_front();
				}
			}
			// Dynamic initialization when the images move over a full window
			// (a still camera waits for the static initialization)
			if(d.initDynamicWindow > 0.0 && !d.dynamicPoses.empty() &&
			   d.dynamicPoses.back().first - d.dynamicPoses.front().first >= d.initDynamicWindow - 1e-6)
			{
				double maxMotion = 0.0, maxRotation = 0.0;
				for(size_t i=1; i<d.dynamicPoses.size(); ++i)
				{
					Transform motion = d.dynamicPoses.front().second.inverse() * d.dynamicPoses[i].second;
					maxMotion = std::max(maxMotion, (double)motion.getNorm());
					maxRotation = std::max(maxRotation, (double)Eigen::AngleAxisf(motion.toEigen3f().linear()).angle());
				}
				if((maxMotion > 2.0*d.initMaxMotion || maxRotation > 2.0*M_PI/180.0) &&
				   alignVisualInertial(d.dynamicPoses, d.dynamicSamples, d.params, d.imuLocalTransform,
						   d.initMaxGyroBias, d.state, d.bias))
				{
					d.preintegrated.reset(new gtsam::PreintegratedCombinedMeasurements(d.params, d.bias));
					d.initialized = true;
					d.initVelocitySigma = 0.1;
					std::pair<Eigen::Vector3d, Eigen::Vector3d> sample = imuSampleAt(d.dynamicSamples, imageStamp);
					d.lastImuStamp = imageStamp;
					d.lastAcc = sample.first;
					d.lastOmega = sample.second;
					d.imuQueue.clear();
					for(size_t i=0; i<d.dynamicSamples.size(); ++i)
					{
						if(d.dynamicSamples[i].first > imageStamp)
						{
							d.imuQueue.push_back(d.dynamicSamples[i]);
						}
					}
					d.initSamples.clear();
					d.initImages.clear();
					d.dynamicSamples.clear();
					d.dynamicPoses.clear();
				}
			}
			if(!d.initialized)
			{
				UWARN("VIO not initialized yet, waiting for the sensor to be static (%d/%d IMU samples)...",
						(int)d.initSamples.size(), initImuSamples_);
				if(info)
				{
					info->reg.covariance = cv::Mat::eye(6, 6, CV_64FC1) * 9999.0;
				}
				return t;
			}
		}

		// Predict at the stamp of the image (IMU is expected to be fed before
		// the image of the same time, later samples stay queued). d.state is
		// the state where the pre-integration started: the last keyframe with
		// the back-end, the last image frame otherwise.
		// With a positive time offset, the IMU samples up to the image may not
		// be received yet: the prediction is extrapolated with the last
		// sample and the keyframe update waits for the next IMU samples.
		d.integrateImu(imageStamp);
		const double imuGap = imageStamp - d.lastImuStamp; // > 0: missing IMU samples
		if(imuGap > std::max(0.0, d.timeOffset) + 0.01)
		{
			UWARN("Image at %f is %f s after the last IMU sample, the IMU should be fed before the image",
					imageStamp, imuGap);
		}
		gtsam::NavState predicted = d.preintegrated->predict(d.state, d.bias); // at the last IMU sample
		const gtsam::NavState predictedImage = imuGap > 1e-6 ? d.extrapolate(predicted, imuGap) : predicted;
		Eigen::Matrix<double, 15, 15> cov = d.preintegrated->preintMeasCov();
		gtsam::NavState output = predicted;
		bool restartIntegration = true;

		if(hasStereo(data))
		{
			// The IMU motion initializes KLT
			Transform motionGuess;
			if(!d.previousPoseInv.isNull())
			{
				motionGuess = d.previousPoseInv * Transform::fromEigen4d(predictedImage.pose().matrix()) * d.imuLocalTransformInv;
			}
			const StereoCameraModel & model = data.stereoCameraModels()[0];
			Transform motion;
			if(!tracked)
			{
				d.frontend->process(data.imageRaw(), data.rightRaw(), model, motionGuess, &motion);
				fillFrontendInfo(*d.frontend, info);
			}
			if(d.offsetEstimator)
			{
				// Visual rotations of the IMU, in the camera clock
				if(motion.isNull())
				{
					d.offsetRotation.setIdentity();
				}
				else
				{
					d.offsetRotation = d.offsetRotation * (d.imuLocalTransformInv * motion * d.imuLocalTransform).toEigen3d().linear();
				}
				d.offsetEstimator->addRotation(data.stamp(), d.offsetRotation, !motion.isNull());
			}

#ifdef RTABMAP_GTSAM_UNSTABLE
			// (1e-6: stamps are large numbers, 1000.1-1000.0 < 0.1)
			bool keyframe = !d.pending && (!d.backend->initialized() ||
					(imageStamp > d.backend->lastKeyframeStamp() &&
					 imageStamp - d.backend->lastKeyframeStamp() >= d.keyframeInterval - 1e-6));
			if(keyframe)
			{
				std::unique_ptr<Impl::Keyframe> frame(new Impl::Keyframe);
				frame->stamp = imageStamp;
				frame->image = data.imageRaw();
				frame->tracks = d.frontend->tracks();
				frame->model = model;
				if(imuGap > 1e-6)
				{
					// Wait for the IMU samples of the image
					d.pending = std::move(frame);
					restartIntegration = false;
				}
				else
				{
					output = d.processKeyframe(*frame, predicted);
				}
			}
			else
			{
				// Keep integrating from the last keyframe
				restartIntegration = false;
			}
#endif
		}

		// Pose at the image (output is at the last IMU sample when samples are missing)
		Transform imuPose = Transform::fromEigen4d((imuGap > 1e-6 ? predictedImage : output).pose().matrix());
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
