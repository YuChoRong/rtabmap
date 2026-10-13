#ifndef COLMAPMODEL_H
#define COLMAPMODEL_H

#include <Eigen/Geometry>
#include <QString>
#include <map>
#include <string>
#include <vector>

/** Camera pose in world (camera optical frame -> world). */
struct Pose
{
	Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
	Eigen::Vector3d t = Eigen::Vector3d::Zero();
	Eigen::Vector3d transform(const Eigen::Vector3d & p) const {return R * p + t;}
};

struct Frame
{
	int imageId = 0;
	std::string name;     ///< Image file name in images/.
	int key = -1;         ///< Integer stem of the name (RTAB-Map node id), -1 if not a number.
	double stamp = -1.0;  ///< From camera_poses.txt, -1 if unknown.
	Pose estimate;        ///< From images.txt.
	bool hasGt = false;
	Pose gt;              ///< Ground truth, aligned to the estimate.
	double error = 0.0;   ///< Position error after alignment (m).
	double rotError = 0.0;///< Rotation error after alignment (deg).
};

struct Point3D
{
	Eigen::Vector3f xyz;
	unsigned char rgb[3];
};

struct Intrinsics
{
	std::string model;
	int width = 0, height = 0;
	double fx = 0, fy = 0, cx = 0, cy = 0;
};

struct Stats
{
	int matched = 0;
	double rmse = 0, mean = 0, median = 0, max = 0;
	double rotRmse = 0;
	double rotOffset = 0;    ///< Angle (deg) of the constant rotation between the estimated and ground truth camera frames.
	double rotRmseNoOffset = 0; ///< Rotation RMSE (deg) after removing that constant rotation.
	double scale = 1.0;      ///< Similarity scale estimate -> ground truth.
	double pathLength = 0;
};

class ColmapModel
{
public:
	/** Loads DIR/sparse/0 (or DIR itself) and DIR/camera_poses.txt, DIR/gt_camera_poses.txt when present. */
	bool load(const QString & dir, QString * error);
	/**
	 * Ground truth in TUM format (stamp x y z qx qy qz qw [id]), camera optical frame.
	 * Associated by id when the file has one, otherwise by nearest stamp (needs camera_poses.txt).
	 */
	bool loadGroundTruth(const QString & path, QString * error);
	/** Aligns the ground truth on the estimate (rigid, or similarity when withScale) and computes the errors. */
	void align(bool withScale);

	const QString & directory() const {return dir_;}
	QString imagePath(const Frame & frame) const;
	const Intrinsics & intrinsics() const {return intrinsics_;}
	const std::vector<Frame> & frames() const {return frames_;}
	const std::vector<Point3D> & points() const {return points_;}
	const Stats & stats() const {return stats_;}
	bool hasGroundTruth() const {return !rawGt_.empty();}
	const QString & groundTruthPath() const {return gtPath_;}

private:
	QString dir_;
	QString modelDir_;
	QString gtPath_;
	Intrinsics intrinsics_;
	std::vector<Frame> frames_;
	std::vector<Point3D> points_;
	std::map<size_t, Pose> rawGt_; // frame index -> ground truth in its own frame
	Stats stats_;
};

#endif
