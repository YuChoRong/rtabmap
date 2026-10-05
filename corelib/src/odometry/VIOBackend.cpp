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

#include "VIOBackend.h"
#include "rtabmap/utilite/ULogger.h"
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>
#include <cmath>
#include <limits>

namespace rtabmap {

using gtsam::symbol_shorthand::X; // IMU pose in world
using gtsam::symbol_shorthand::V; // velocity in world
using gtsam::symbol_shorthand::B; // IMU bias

VIOBackend::VIOBackend(const ParametersMap & parameters) :
	windowSize_(Parameters::defaultOdomVIOWindowSize()),
	pixelNoise_(Parameters::defaultOdomVIOPixelNoise()),
	extraIterations_(Parameters::defaultOdomVIOBackendIterations()),
	monoObservations_(Parameters::defaultOdomVIOMonoObservations()),
	index_(0),
	lastStamp_(0.0)
{
	Parameters::parse(parameters, Parameters::kOdomVIOWindowSize(), windowSize_);
	Parameters::parse(parameters, Parameters::kOdomVIOPixelNoise(), pixelNoise_);
	Parameters::parse(parameters, Parameters::kOdomVIOBackendIterations(), extraIterations_);
	Parameters::parse(parameters, Parameters::kOdomVIOMonoObservations(), monoObservations_);
	UASSERT(windowSize_ > 0.0);
	UASSERT(pixelNoise_ > 0.0);
	UASSERT(extraIterations_ >= 0);
}

VIOBackend::~VIOBackend()
{
}

void VIOBackend::reset()
{
	smoother_.reset();
	K_.reset();
	index_ = 0;
	lastStamp_ = 0.0;
	keyframeStamps_.clear();
	trackFactors_.clear();
	state_ = gtsam::NavState();
	bias_ = gtsam::imuBias::ConstantBias();
	stats_ = Stats();
}

bool VIOBackend::initialize(
		double stamp,
		const gtsam::NavState & state,
		const gtsam::imuBias::ConstantBias & bias,
		const gtsam::Pose3 & imuToCamera,
		const StereoCameraModel & model,
		const std::vector<VIOFrontend::Track> & tracks,
		double velocitySigma)
{
	reset();

	gtsam::ISAM2Params isamParams;
	isamParams.relinearizeThreshold = 0.01;
	isamParams.relinearizeSkip = 1;
	// Slots are not reused: the factor indices kept for the smart factors stay valid
	isamParams.findUnusedFactorSlots = false;
	smoother_.reset(new gtsam::IncrementalFixedLagSmoother(windowSize_, isamParams));

	imuToCamera_ = imuToCamera;
	K_.reset(new gtsam::Cal3_S2Stereo(
			model.left().fx(), model.left().fy(), 0.0,
			model.left().cx(), model.left().cy(), model.baseline()));

	gtsam::NonlinearFactorGraph graph;
	gtsam::Values values;
	std::map<gtsam::Key, double> timestamps;

	// The start fixes the gauge: position and yaw are unobservable, roll and
	// pitch come from the gravity initialization.
	gtsam::Vector6 poseSigmas;
	poseSigmas << 0.01, 0.01, 0.01, 0.001, 0.001, 0.001; // rot (rad), pos (m)
	gtsam::Vector6 biasSigmas;
	biasSigmas << 0.2, 0.2, 0.2, 0.05, 0.05, 0.05; // acc (m/s^2), gyro (rad/s)
	graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3> >(X(0), state.pose(), gtsam::noiseModel::Diagonal::Sigmas(poseSigmas));
	graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3> >(V(0), state.v(), gtsam::noiseModel::Isotropic::Sigma(3, velocitySigma));
	graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias> >(B(0), bias, gtsam::noiseModel::Diagonal::Sigmas(biasSigmas));
	values.insert(X(0), state.pose());
	values.insert(V(0), state.v());
	values.insert(B(0), bias);
	timestamps[X(0)] = timestamps[V(0)] = timestamps[B(0)] = stamp;

	index_ = 0;
	lastStamp_ = stamp;
	keyframeStamps_[0] = stamp;
	state_ = state;
	bias_ = bias;

	std::vector<int> updatedTracks;
	gtsam::FactorIndices factorsToRemove;
	size_t firstSmartFactor = graph.size();
	addObservations(tracks, graph, updatedTracks, factorsToRemove);
	return optimize(graph, values, timestamps, factorsToRemove, updatedTracks, firstSmartFactor);
}

bool VIOBackend::addKeyframe(
		double stamp,
		const gtsam::PreintegratedCombinedMeasurements & preintegrated,
		const gtsam::NavState & predicted,
		const std::vector<VIOFrontend::Track> & tracks)
{
	UASSERT(initialized());
	UASSERT(stamp > lastStamp_);

	const int n = ++index_;
	gtsam::NonlinearFactorGraph graph;
	gtsam::Values values;
	std::map<gtsam::Key, double> timestamps;

	graph.emplace_shared<gtsam::CombinedImuFactor>(X(n-1), V(n-1), X(n), V(n), B(n-1), B(n), preintegrated);
	values.insert(X(n), predicted.pose());
	values.insert(V(n), predicted.v());
	values.insert(B(n), bias_);
	timestamps[X(n)] = timestamps[V(n)] = timestamps[B(n)] = stamp;

	lastStamp_ = stamp;
	keyframeStamps_[n] = stamp;
	// Keyframes leaving the window in this update (same rule as the smoother)
	const double cutoff = stamp - windowSize_;
	while(!keyframeStamps_.empty() && keyframeStamps_.begin()->second < cutoff + 1e-6)
	{
		keyframeStamps_.erase(keyframeStamps_.begin());
	}

	std::vector<int> updatedTracks;
	gtsam::FactorIndices factorsToRemove;
	size_t firstSmartFactor = graph.size();
	addObservations(tracks, graph, updatedTracks, factorsToRemove);
	return optimize(graph, values, timestamps, factorsToRemove, updatedTracks, firstSmartFactor);
}

void VIOBackend::addObservations(
		const std::vector<VIOFrontend::Track> & tracks,
		gtsam::NonlinearFactorGraph & graph,
		std::vector<int> & updatedTracks,
		gtsam::FactorIndices & factorsToRemove)
{
	gtsam::SmartStereoProjectionParams params(gtsam::HESSIAN, gtsam::ZERO_ON_DEGENERACY);
	params.setRankTolerance(1.0);
	params.setLandmarkDistanceThreshold(50.0);
	params.setRetriangulationThreshold(1e-3);
	params.setDynamicOutlierRejectionThreshold(8.0 * pixelNoise_);
	params.setEnableEPI(false);
	gtsam::SharedNoiseModel noise = gtsam::noiseModel::Isotropic::Sigma(3, pixelNoise_);
	const gtsam::NonlinearFactorGraph & factors = smoother_->getFactors();

	std::map<int, TrackFactor> current;
	stats_.observations = 0;
	for(size_t i=0; i<tracks.size(); ++i)
	{
		const VIOFrontend::Track & track = tracks[i];
		if(!track.hasDepth() && !monoObservations_)
		{
			continue;
		}
		TrackFactor entry;
		std::map<int, TrackFactor>::iterator iter = trackFactors_.find(track.id);
		if(iter != trackFactors_.end())
		{
			entry = iter->second;
			bool factorAlive = entry.slot >= 0 && (size_t)entry.slot < factors.size() && factors[entry.slot];
			if(entry.slot >= 0 && !factorAlive)
			{
				// Marginalized with its oldest keyframe: its information is in the
				// marginal now, start over to avoid counting it twice.
				entry.observations.clear();
				entry.slot = -1;
			}
			// Drop observations of keyframes leaving the window
			std::vector<std::pair<int, gtsam::StereoPoint2> > kept;
			for(size_t j=0; j<entry.observations.size(); ++j)
			{
				if(keyframeStamps_.find(entry.observations[j].first) != keyframeStamps_.end())
				{
					kept.push_back(entry.observations[j]);
				}
			}
			entry.observations.swap(kept);
		}
		bool hasStereo = false;
		for(size_t j=0; j<entry.observations.size() && !hasStereo; ++j)
		{
			hasStereo = !std::isnan(entry.observations[j].second.uR());
		}
		if(!track.hasDepth() && !hasStereo)
		{
			// Left only: triangulated from at least one stereo observation
			if(iter != trackFactors_.end())
			{
				current.insert(std::make_pair(track.id, entry));
			}
			continue;
		}
		// The right coordinate is NaN for a left only observation
		entry.observations.push_back(std::make_pair(index_,
				gtsam::StereoPoint2(track.left.x,
						track.hasDepth()?track.right.x:std::numeric_limits<double>::quiet_NaN(),
						track.left.y)));
		++stats_.observations;

		if(entry.observations.size() >= 2)
		{
			if(entry.slot >= 0)
			{
				factorsToRemove.push_back(entry.slot);
				entry.slot = -1;
			}
			gtsam::SmartStereoProjectionPoseFactor::shared_ptr factor(
					new gtsam::SmartStereoProjectionPoseFactor(noise, params, imuToCamera_));
			for(size_t j=0; j<entry.observations.size(); ++j)
			{
				factor->add(entry.observations[j].second, X(entry.observations[j].first), K_);
			}
			graph.push_back(factor);
			updatedTracks.push_back(track.id);
		}
		current.insert(std::make_pair(track.id, entry));
	}
	// Lost tracks keep their factor in the smoother until it is marginalized
	trackFactors_.swap(current);
	stats_.smartFactors = (int)updatedTracks.size();
}

bool VIOBackend::optimize(
		const gtsam::NonlinearFactorGraph & graph,
		const gtsam::Values & values,
		const std::map<gtsam::Key, double> & timestamps,
		const gtsam::FactorIndices & factorsToRemove,
		const std::vector<int> & updatedTracks,
		size_t firstSmartFactor)
{
	try
	{
		gtsam::FixedLagSmoother::KeyTimestampMap stamps(timestamps.begin(), timestamps.end());
		smoother_->update(graph, values, stamps, factorsToRemove);

		const gtsam::FactorIndices & newIndices = smoother_->getISAM2Result().newFactorsIndices;
		UASSERT(newIndices.size() == graph.size());
		for(size_t i=0; i<updatedTracks.size(); ++i)
		{
			trackFactors_.at(updatedTracks[i]).slot = (long)newIndices[firstSmartFactor + i];
		}

		for(int i=0; i<extraIterations_; ++i)
		{
			smoother_->update();
		}

		gtsam::Values estimate = smoother_->calculateEstimate();
		state_ = gtsam::NavState(estimate.at<gtsam::Pose3>(X(index_)), estimate.at<gtsam::Vector3>(V(index_)));
		bias_ = estimate.at<gtsam::imuBias::ConstantBias>(B(index_));
		stats_.keyframes = (int)keyframeStamps_.size();
	}
	catch(const std::exception & e)
	{
		UERROR("VIO back-end optimization failed, resetting it: %s", e.what());
		reset();
		return false;
	}
	UDEBUG("Keyframe %d: %d smart factors updated (%d observations), %d keyframes in window, bias acc=(%f,%f,%f) gyro=(%f,%f,%f)",
			index_, stats_.smartFactors, stats_.observations, stats_.keyframes,
			bias_.accelerometer()[0], bias_.accelerometer()[1], bias_.accelerometer()[2],
			bias_.gyroscope()[0], bias_.gyroscope()[1], bias_.gyroscope()[2]);
	return true;
}

}
