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

#ifndef VIOFRONTEND_H_
#define VIOFRONTEND_H_

#include <rtabmap/core/rtabmap_core_export.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/Transform.h>
#include <opencv2/core/core.hpp>
#include <memory>
#include <vector>

namespace rtabmap {

class Stereo;

/**
 * Stereo feature tracking front-end of OdometryVIO.
 *
 * For each rectified stereo pair:
 * 1. Tracks the features of the previous left image with pyramidal KLT
 *    (initialized from the motion guess when given), with a backward check.
 * 2. Rejects outliers with a fundamental matrix RANSAC on all tracks, then a
 *    3D-to-2D PnP RANSAC on the tracks that had a stereo depth. The PnP
 *    result is the visual motion estimate.
 * 3. Detects new GFTT corners away from the surviving tracks, up to
 *    OdomVIO/MaxFeatures.
 * 4. Matches every track in the right image (Stereo parameters) to get
 *    its depth.
 *
 * Each track keeps the same id for as long as it is followed, which is what
 * the smart stereo factors of the back-end need.
 */
class RTABMAP_CORE_EXPORT VIOFrontend
{
public:
	struct Track
	{
		int id;
		int age;              ///< Number of frames this track has been seen in (1 = new).
		cv::Point2f left;     ///< Position in the left image.
		cv::Point2f right;    ///< Position in the right image, x<0 if no stereo match.
		cv::Point3f point;    ///< 3D point in the base frame of the current frame, NaN if no stereo match.
		bool hasDepth() const {return right.x >= 0.0f;}
	};

	struct Stats
	{
		int tracked = 0;      ///< Tracks followed from the previous frame (after KLT and backward check).
		int inliers = 0;      ///< Tracks kept after RANSAC.
		int pnpMatches = 0;   ///< Tracks with a previous 3D point given to PnP.
		int pnpInliers = 0;   ///< PnP inliers.
		int added = 0;        ///< New tracks detected in this frame.
		int stereo = 0;       ///< Tracks with a stereo depth.
	};

public:
	VIOFrontend(const ParametersMap & parameters = ParametersMap());
	~VIOFrontend();

	void reset();

	/**
	 * @param left, right Rectified images (grayscale or BGR).
	 * @param model Rectified stereo model (with its local transform base -> left optical frame).
	 * @param motionGuess Expected motion of the base frame since the previous frame (previous -> current), can be null.
	 * @param motion Output: visual motion of the base frame since the previous frame, null if it could not be estimated
	 *        (first frame, or not enough PnP inliers).
	 * @param covariance Output: 6x6 covariance of motion (when not null).
	 * @return false if the input is invalid.
	 */
	bool process(
			const cv::Mat & left,
			const cv::Mat & right,
			const StereoCameraModel & model,
			const Transform & motionGuess = Transform(),
			Transform * motion = 0,
			cv::Mat * covariance = 0);

	/** True when a previous frame is available to track from. */
	bool hasPrevious() const {return !previousLeft_.empty();}
	const std::vector<Track> & tracks() const {return tracks_;}
	/** Left image positions in the previous frame of the tracks followed in the last call, same order as previousCorners/currentCorners. */
	const std::vector<cv::Point2f> & previousCorners() const {return previousCorners_;}
	const std::vector<cv::Point2f> & currentCorners() const {return currentCorners_;}
	/** Indices in previousCorners()/currentCorners() that survived outlier rejection. */
	const std::vector<int> & cornerInliers() const {return cornerInliers_;}
	const Stats & stats() const {return stats_;}

private:
	void detectNewFeatures(const cv::Mat & image);
	void computeDepth(const cv::Mat & left, const cv::Mat & right, const StereoCameraModel & model);

private:
	// parameters
	int maxFeatures_;
	int histogramEqualization_;
	int gridCells_;
	double minFeatureDistance_;
	double featureQuality_;
	int flowWinSize_;
	int flowMaxLevel_;
	double flowBackCheck_;
	double stereoBackCheck_;
	double fundamentalThreshold_;
	double pnpReprojError_;
	int pnpIterations_;
	int minInliers_;
	std::unique_ptr<Stereo> stereo_;

	// state
	cv::Mat previousLeft_;
	std::vector<Track> tracks_;
	int nextId_;
	std::vector<cv::Point2f> previousCorners_;
	std::vector<cv::Point2f> currentCorners_;
	std::vector<int> cornerInliers_;
	Stats stats_;
};

}

#endif /* VIOFRONTEND_H_ */
