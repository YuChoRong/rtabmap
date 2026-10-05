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

#include "rtabmap/core/odometry/VIOFrontend.h"
#include "rtabmap/core/Stereo.h"
#include "rtabmap/core/util3d.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/util3d_motion_estimation.h"
#include "rtabmap/utilite/ULogger.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace rtabmap {

VIOFrontend::VIOFrontend(const ParametersMap & parameters) :
	maxFeatures_(Parameters::defaultOdomVIOMaxFeatures()),
	minFeatureDistance_(Parameters::defaultOdomVIOMinFeatureDistance()),
	featureQuality_(Parameters::defaultOdomVIOFeatureQuality()),
	flowWinSize_(Parameters::defaultOdomVIOFlowWinSize()),
	flowMaxLevel_(Parameters::defaultOdomVIOFlowMaxLevel()),
	flowBackCheck_(Parameters::defaultOdomVIOFlowBackCheck()),
	fundamentalThreshold_(Parameters::defaultOdomVIOFundamentalThreshold()),
	pnpReprojError_(Parameters::defaultOdomVIOPnPReprojError()),
	pnpIterations_(Parameters::defaultOdomVIOPnPIterations()),
	minInliers_(Parameters::defaultOdomVIOMinInliers()),
	stereo_(Stereo::create(parameters)),
	nextId_(1)
{
	Parameters::parse(parameters, Parameters::kOdomVIOMaxFeatures(), maxFeatures_);
	Parameters::parse(parameters, Parameters::kOdomVIOMinFeatureDistance(), minFeatureDistance_);
	Parameters::parse(parameters, Parameters::kOdomVIOFeatureQuality(), featureQuality_);
	Parameters::parse(parameters, Parameters::kOdomVIOFlowWinSize(), flowWinSize_);
	Parameters::parse(parameters, Parameters::kOdomVIOFlowMaxLevel(), flowMaxLevel_);
	Parameters::parse(parameters, Parameters::kOdomVIOFlowBackCheck(), flowBackCheck_);
	Parameters::parse(parameters, Parameters::kOdomVIOFundamentalThreshold(), fundamentalThreshold_);
	Parameters::parse(parameters, Parameters::kOdomVIOPnPReprojError(), pnpReprojError_);
	Parameters::parse(parameters, Parameters::kOdomVIOPnPIterations(), pnpIterations_);
	Parameters::parse(parameters, Parameters::kOdomVIOMinInliers(), minInliers_);
	UASSERT(maxFeatures_ > 0);
	UASSERT(minFeatureDistance_ >= 0.0);
	UASSERT(featureQuality_ > 0.0);
	UASSERT(flowWinSize_ >= 3);
	UASSERT(flowMaxLevel_ >= 0);
	UASSERT(pnpReprojError_ > 0.0);
	UASSERT(pnpIterations_ > 0);
	UASSERT(minInliers_ >= 4);
}

VIOFrontend::~VIOFrontend()
{
}

void VIOFrontend::reset()
{
	previousLeft_ = cv::Mat();
	tracks_.clear();
	nextId_ = 1;
	previousCorners_.clear();
	currentCorners_.clear();
	cornerInliers_.clear();
	stats_ = Stats();
}

bool VIOFrontend::process(
		const cv::Mat & leftIn,
		const cv::Mat & rightIn,
		const StereoCameraModel & model,
		const Transform & motionGuess,
		Transform * motion,
		cv::Mat * covariance)
{
	if(motion)
	{
		motion->setNull();
	}
	if(leftIn.empty() || rightIn.empty() || leftIn.size() != rightIn.size() ||
	   !model.isValidForProjection() || model.left().localTransform().isNull())
	{
		UERROR("VIOFrontend requires a rectified stereo pair with a valid stereo model "
				"(left=%dx%d right=%dx%d valid=%d)",
				leftIn.cols, leftIn.rows, rightIn.cols, rightIn.rows, model.isValidForProjection()?1:0);
		return false;
	}

	cv::Mat left = leftIn;
	cv::Mat right = rightIn;
	if(left.channels() > 1)
	{
		cv::cvtColor(leftIn, left, cv::COLOR_BGR2GRAY);
	}
	if(right.channels() > 1)
	{
		cv::cvtColor(rightIn, right, cv::COLOR_BGR2GRAY);
	}

	stats_ = Stats();
	previousCorners_.clear();
	currentCorners_.clear();
	cornerInliers_.clear();

	const CameraModel & leftModel = model.left();
	Transform motionEstimate;

	if(!previousLeft_.empty() && !tracks_.empty() && previousLeft_.size() == left.size())
	{
		// 1. KLT tracking, initialized from the guess for tracks with depth
		std::vector<cv::Point2f> prevPts(tracks_.size());
		std::vector<cv::Point2f> curPts(tracks_.size());
		Transform guessInv = motionGuess.isNull()?Transform():motionGuess.inverse();
		Transform opticalInv = leftModel.localTransform().inverse();
		for(size_t i=0; i<tracks_.size(); ++i)
		{
			prevPts[i] = tracks_[i].left;
			curPts[i] = tracks_[i].left;
			if(!guessInv.isNull() && tracks_[i].hasDepth())
			{
				cv::Point3f pt = util3d::transformPoint(tracks_[i].point, opticalInv * guessInv);
				if(pt.z > 0.0f)
				{
					float u, v;
					leftModel.reproject(pt.x, pt.y, pt.z, u, v);
					if(u >= 0.0f && v >= 0.0f && u < (float)left.cols && v < (float)left.rows)
					{
						curPts[i] = cv::Point2f(u, v);
					}
				}
			}
		}

		const cv::Size winSize(flowWinSize_, flowWinSize_);
		const cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
		std::vector<unsigned char> status;
		std::vector<float> err;
		cv::calcOpticalFlowPyrLK(previousLeft_, left, prevPts, curPts, status, err,
				winSize, flowMaxLevel_, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

		std::vector<unsigned char> backStatus(status.size(), 1);
		std::vector<cv::Point2f> backPts = prevPts;
		if(flowBackCheck_ > 0.0)
		{
			cv::calcOpticalFlowPyrLK(left, previousLeft_, curPts, backPts, backStatus, err,
					winSize, flowMaxLevel_, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
		}

		std::vector<int> tracked; // indices in tracks_
		for(size_t i=0; i<tracks_.size(); ++i)
		{
			if(status[i] && backStatus[i] &&
			   curPts[i].x >= 0.0f && curPts[i].y >= 0.0f &&
			   curPts[i].x < (float)left.cols && curPts[i].y < (float)left.rows &&
			   (flowBackCheck_ <= 0.0 || cv::norm(backPts[i] - prevPts[i]) <= flowBackCheck_))
			{
				tracked.push_back((int)i);
			}
		}
		stats_.tracked = (int)tracked.size();

		std::vector<bool> keep(tracked.size(), true);

		// 2a. Fundamental matrix RANSAC on all tracks (also those without depth)
		if(fundamentalThreshold_ > 0.0 && tracked.size() >= 8)
		{
			std::vector<cv::Point2f> a(tracked.size()), b(tracked.size());
			for(size_t k=0; k<tracked.size(); ++k)
			{
				a[k] = prevPts[tracked[k]];
				b[k] = curPts[tracked[k]];
			}
			std::vector<unsigned char> mask;
			cv::Mat F = cv::findFundamentalMat(a, b, cv::FM_RANSAC, fundamentalThreshold_, 0.99, mask);
			if(!F.empty() && mask.size() == tracked.size())
			{
				for(size_t k=0; k<tracked.size(); ++k)
				{
					keep[k] = mask[k] != 0;
				}
			}
		}

		// 2b. PnP RANSAC with the previous 3D points: motion estimate
		std::map<int, cv::Point3f> words3A;
		std::map<int, cv::KeyPoint> words2B;
		for(size_t k=0; k<tracked.size(); ++k)
		{
			const Track & track = tracks_[tracked[k]];
			if(keep[k] && track.hasDepth())
			{
				words3A.insert(std::make_pair(track.id, track.point));
				words2B.insert(std::make_pair(track.id, cv::KeyPoint(curPts[tracked[k]], 1.0f)));
			}
		}
		stats_.pnpMatches = (int)words3A.size();
		if((int)words3A.size() >= minInliers_)
		{
			std::vector<int> matches, inliers;
			cv::Mat cov;
			motionEstimate = util3d::estimateMotion3DTo2D(
					words3A, words2B, leftModel,
					minInliers_, pnpIterations_, pnpReprojError_,
					0, 1, 4, 0,
					motionGuess.isNull()?Transform::getIdentity():motionGuess,
					std::map<int, cv::Point3f>(),
					&cov, &matches, &inliers);
			stats_.pnpInliers = (int)inliers.size();
			if(!motionEstimate.isNull())
			{
				std::set<int> inlierIds(inliers.begin(), inliers.end());
				for(size_t k=0; k<tracked.size(); ++k)
				{
					const Track & track = tracks_[tracked[k]];
					if(keep[k] && words3A.find(track.id) != words3A.end() && inlierIds.find(track.id) == inlierIds.end())
					{
						keep[k] = false;
					}
				}
				if(covariance)
				{
					*covariance = cov;
				}
			}
			else
			{
				UWARN("PnP failed (%d/%d inliers, min=%d)", (int)inliers.size(), (int)words3A.size(), minInliers_);
			}
		}
		else
		{
			UWARN("Not enough tracks with depth for PnP (%d, min=%d)", (int)words3A.size(), minInliers_);
		}

		// Keep the inliers
		std::vector<Track> survivors;
		survivors.reserve(tracked.size());
		for(size_t k=0; k<tracked.size(); ++k)
		{
			previousCorners_.push_back(prevPts[tracked[k]]);
			currentCorners_.push_back(curPts[tracked[k]]);
			if(keep[k])
			{
				cornerInliers_.push_back((int)k);
				Track track = tracks_[tracked[k]];
				track.left = curPts[tracked[k]];
				++track.age;
				survivors.push_back(track);
			}
		}
		tracks_.swap(survivors);
		stats_.inliers = (int)tracks_.size();
	}
	else
	{
		tracks_.clear();
	}

	// 3. New features
	detectNewFeatures(left);

	// 4. Stereo depth of every track
	computeDepth(left, right, model);

	previousLeft_ = left;
	if(motion)
	{
		*motion = motionEstimate;
	}

	UDEBUG("tracked=%d inliers=%d pnp=%d/%d added=%d stereo=%d motion=%s",
			stats_.tracked, stats_.inliers, stats_.pnpInliers, stats_.pnpMatches,
			stats_.added, stats_.stereo, motionEstimate.prettyPrint().c_str());
	return true;
}

void VIOFrontend::detectNewFeatures(const cv::Mat & image)
{
	int toAdd = maxFeatures_ - (int)tracks_.size();
	if(toAdd <= 0)
	{
		return;
	}
	cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
	int radius = (int)std::ceil(minFeatureDistance_);
	for(size_t i=0; i<tracks_.size(); ++i)
	{
		cv::circle(mask, tracks_[i].left, radius, cv::Scalar(0), -1);
	}
	std::vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(image, corners, toAdd, featureQuality_, minFeatureDistance_, mask);
	if(corners.empty())
	{
		return;
	}
	cv::cornerSubPix(image, corners, cv::Size(3,3), cv::Size(-1,-1),
			cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.03));
	const float nan = std::numeric_limits<float>::quiet_NaN();
	for(size_t i=0; i<corners.size(); ++i)
	{
		Track track;
		track.id = nextId_++;
		track.age = 1;
		track.left = corners[i];
		track.right = cv::Point2f(-1.0f, -1.0f);
		track.point = cv::Point3f(nan, nan, nan);
		tracks_.push_back(track);
	}
	stats_.added = (int)corners.size();
}

void VIOFrontend::computeDepth(const cv::Mat & left, const cv::Mat & right, const StereoCameraModel & model)
{
	const float nan = std::numeric_limits<float>::quiet_NaN();
	std::vector<cv::Point2f> leftPts(tracks_.size());
	for(size_t i=0; i<tracks_.size(); ++i)
	{
		leftPts[i] = tracks_[i].left;
		tracks_[i].right = cv::Point2f(-1.0f, -1.0f);
		tracks_[i].point = cv::Point3f(nan, nan, nan);
	}
	if(leftPts.empty())
	{
		return;
	}
	std::vector<unsigned char> status;
	std::vector<cv::Point2f> rightPts = stereo_->computeCorrespondences(left, right, leftPts, status);
	int withDepth = 0;
	for(size_t i=0; i<tracks_.size() && i<rightPts.size(); ++i)
	{
		if(!status[i])
		{
			continue;
		}
		float disparity = leftPts[i].x - rightPts[i].x;
		cv::Point3f pt = util3d::projectDisparityTo3D(leftPts[i], disparity, model);
		if(util3d::isFinite(pt))
		{
			tracks_[i].right = rightPts[i];
			tracks_[i].point = util3d::transformPoint(pt, model.left().localTransform());
			++withDepth;
		}
	}
	stats_.stereo = withDepth;
}

}
