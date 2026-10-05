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

#ifndef VIOBACKEND_H_
#define VIOBACKEND_H_

// Private header of OdometryVIO, requires GTSAM and gtsam_unstable

#include "rtabmap/core/Parameters.h"
#include "rtabmap/core/odometry/VIOFrontend.h"
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <map>
#include <memory>
#include <vector>

namespace gtsam {
class IncrementalFixedLagSmoother;
}

namespace rtabmap {

/**
 * Sliding window back-end of OdometryVIO.
 *
 * Each keyframe n adds a pose X(n) (IMU frame in world), a velocity V(n) and
 * an IMU bias B(n), linked to the previous keyframe by a CombinedImuFactor.
 * Each stereo track becomes one SmartStereoProjectionPoseFactor holding its
 * observations in the window (the 3D point is not a variable). When a track
 * is observed again, its factor is replaced by one with the new observation.
 * Everything is optimized by an IncrementalFixedLagSmoother that
 * marginalizes the keyframes older than OdomVIO/WindowSize.
 */
class VIOBackend
{
public:
	struct Stats
	{
		int keyframes = 0;      ///< Keyframes in the window.
		int smartFactors = 0;   ///< Smart factors added or updated at the last keyframe.
		int observations = 0;   ///< Stereo observations added at the last keyframe.
	};

public:
	VIOBackend(const ParametersMap & parameters);
	~VIOBackend();

	bool initialized() const {return smoother_.get() != 0;}
	void reset();

	/**
	 * Starts the window at a first keyframe.
	 * @param imuToCamera Pose of the left camera (optical frame) in the IMU frame.
	 * @param velocitySigma Standard deviation of the velocity prior (m/s).
	 * @return false if the optimization failed (the back-end is then reset).
	 */
	bool initialize(
			double stamp,
			const gtsam::NavState & state,
			const gtsam::imuBias::ConstantBias & bias,
			const gtsam::Pose3 & imuToCamera,
			const StereoCameraModel & model,
			const std::vector<VIOFrontend::Track> & tracks,
			double velocitySigma);

	/**
	 * Adds a keyframe.
	 * @param preintegrated IMU measurements since the previous keyframe.
	 * @param predicted State predicted from the previous keyframe with preintegrated (initial estimate).
	 * @return false if the optimization failed (the back-end is then reset).
	 */
	bool addKeyframe(
			double stamp,
			const gtsam::PreintegratedCombinedMeasurements & preintegrated,
			const gtsam::NavState & predicted,
			const std::vector<VIOFrontend::Track> & tracks);

	double lastKeyframeStamp() const {return lastStamp_;}
	/** Optimized state and bias of the last keyframe. */
	const gtsam::NavState & state() const {return state_;}
	const gtsam::imuBias::ConstantBias & bias() const {return bias_;}
	const Stats & stats() const {return stats_;}

private:
	struct TrackFactor
	{
		std::vector<std::pair<int, gtsam::StereoPoint2> > observations; // keyframe index, measurement
		long slot = -1; // index of its factor in the smoother, -1 if none
	};

	void addObservations(
			const std::vector<VIOFrontend::Track> & tracks,
			gtsam::NonlinearFactorGraph & graph,
			std::vector<int> & updatedTracks,
			gtsam::FactorIndices & factorsToRemove);
	bool optimize(
			const gtsam::NonlinearFactorGraph & graph,
			const gtsam::Values & values,
			const std::map<gtsam::Key, double> & timestamps,
			const gtsam::FactorIndices & factorsToRemove,
			const std::vector<int> & updatedTracks,
			size_t firstSmartFactor);

private:
	// parameters
	double windowSize_;
	double pixelNoise_;
	int extraIterations_;
	bool monoObservations_;

	std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
	gtsam::Pose3 imuToCamera_;
	gtsam::Cal3_S2Stereo::shared_ptr K_;
	int index_;
	double lastStamp_;
	std::map<int, double> keyframeStamps_; // keyframe index -> stamp, in the window
	std::map<int, TrackFactor> trackFactors_; // track id -> factor
	gtsam::NavState state_;
	gtsam::imuBias::ConstantBias bias_;
	Stats stats_;
};

}

#endif /* VIOBACKEND_H_ */
