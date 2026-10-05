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
#include "rtabmap/core/OdometryInfo.h"
#include "rtabmap/utilite/ULogger.h"

#ifdef RTABMAP_GTSAM
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
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
	Eigen::Vector3d accSum = Eigen::Vector3d::Zero();
	int accSamples = 0;
	Transform imuLocalTransform;     // base -> imu
	Transform imuLocalTransformInv;  // imu -> base
	Transform previousPoseInv;       // inverse of last base pose returned

	void clear()
	{
#ifdef RTABMAP_GTSAM
		preintegrated.reset();
		state = gtsam::NavState();
		bias = gtsam::imuBias::ConstantBias();
#endif
		initialized = false;
		lastImuStamp = 0.0;
		accSum.setZero();
		accSamples = 0;
		imuLocalTransform.setNull();
		imuLocalTransformInv.setNull();
		previousPoseInv.setNull();
	}
};

OdometryVIO::OdometryVIO(const ParametersMap & parameters) :
	Odometry(parameters),
	impl_(new Impl),
	initGravity_(false),
	initImuSamples_(Parameters::defaultOdomVIOInitImuSamples())
{
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

// return not null transform if odometry is correctly computed
Transform OdometryVIO::computeTransform(
		SensorData & data,
		const Transform & guess,
		OdometryInfo * info)
{
	Transform t;
#ifdef RTABMAP_GTSAM
	Impl & d = *impl_;

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
			// Static initialization: the mean specific force points up (opposite to gravity).
			d.accSum += acc;
			++d.accSamples;
			if(d.accSamples >= initImuSamples_)
			{
				Eigen::Vector3d up = (d.accSum / d.accSamples).normalized();
				gtsam::Rot3 R_wi(Eigen::Quaterniond::FromTwoVectors(up, Eigen::Vector3d::UnitZ()));
				// Gravity gives roll and pitch only: start with the base frame (not the IMU) at yaw 0
				Transform baseRotation = Transform::fromEigen4d(gtsam::Pose3(R_wi, gtsam::Point3()).matrix()) * d.imuLocalTransformInv.rotation();
				R_wi = gtsam::Rot3::Rz(-baseRotation.theta()) * R_wi;
				d.state = gtsam::NavState(R_wi, gtsam::Point3(0,0,0), gtsam::Velocity3(0,0,0));
				d.bias = gtsam::imuBias::ConstantBias();
				d.preintegrated.reset(new gtsam::PreintegratedCombinedMeasurements(d.params, d.bias));
				d.initialized = true;
				UINFO("VIO initialized with gravity from %d IMU samples (roll=%f pitch=%f)",
						d.accSamples, R_wi.roll(), R_wi.pitch());
			}
		}
		else
		{
			double dt = data.stamp() - d.lastImuStamp;
			if(dt > 0.0)
			{
				d.preintegrated->integrateMeasurement(acc, omega, dt);
			}
			else
			{
				UWARN("Ignoring IMU measurement with non-increasing stamp (dt=%f)", dt);
			}
		}
		d.lastImuStamp = data.stamp();
	}

	bool isImageFrame = !data.imageRaw().empty() ||
			!data.cameraModels().empty() ||
			!data.stereoCameraModels().empty();
	if(isImageFrame)
	{
		if(!d.initialized)
		{
			UWARN("VIO not initialized yet (%d/%d IMU samples received), waiting for IMU data...",
					d.accSamples, initImuSamples_);
			return t;
		}

		// Predict at the stamp of the last IMU sample received (IMU is expected
		// to be fed before the image of the same time).
		gtsam::NavState predicted = d.preintegrated->predict(d.state, d.bias);
		Eigen::Matrix<double, 15, 15> cov = d.preintegrated->preintMeasCov();

		Transform imuPose = Transform::fromEigen4d(predicted.pose().matrix());
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

		d.state = predicted;
		d.preintegrated->resetIntegrationAndSetBias(d.bias);

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
