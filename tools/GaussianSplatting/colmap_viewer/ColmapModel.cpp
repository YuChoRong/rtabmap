#include "ColmapModel.h"
#include <Eigen/SVD>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <algorithm>
#include <cmath>

namespace {

const double kRadToDeg = 180.0 / 3.14159265358979323846; // M_PI is not defined by MSVC

Pose fromTum(double x, double y, double z, double qx, double qy, double qz, double qw)
{
	Pose p;
	p.R = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
	p.t = Eigen::Vector3d(x, y, z);
	return p;
}

QStringList dataLines(const QString & path, QString * error)
{
	QFile file(path);
	QStringList lines;
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		if(error) *error = QString("Cannot open %1").arg(path);
		return lines;
	}
	QTextStream in(&file);
	while(!in.atEnd())
	{
		QString line = in.readLine();
		if(!line.startsWith('#'))
		{
			lines.push_back(line);
		}
	}
	return lines;
}

struct TumPose
{
	double stamp;
	Pose pose;
	int id;
};

std::vector<TumPose> readTum(const QString & path, QString * error)
{
	std::vector<TumPose> poses;
	QStringList lines = dataLines(path, error);
	for(const QString & line : lines)
	{
		QStringList v = line.simplified().split(' ', Qt::SkipEmptyParts);
		if(v.size() < 8)
		{
			continue;
		}
		TumPose p;
		p.stamp = v[0].toDouble();
		p.pose = fromTum(v[1].toDouble(), v[2].toDouble(), v[3].toDouble(),
				v[4].toDouble(), v[5].toDouble(), v[6].toDouble(), v[7].toDouble());
		p.id = v.size() >= 9 ? v[8].toInt() : -1;
		poses.push_back(p);
	}
	return poses;
}

} // namespace

bool ColmapModel::load(const QString & dir, QString * error)
{
	dir_ = dir;
	frames_.clear();
	points_.clear();
	rawGt_.clear();
	gtPath_.clear();
	stats_ = Stats();
	intrinsics_ = Intrinsics();

	modelDir_ = dir + "/sparse/0";
	if(!QFileInfo::exists(modelDir_ + "/images.txt"))
	{
		modelDir_ = dir + "/sparse";
		if(!QFileInfo::exists(modelDir_ + "/images.txt"))
		{
			modelDir_ = dir;
		}
	}
	if(!QFileInfo::exists(modelDir_ + "/images.txt"))
	{
		if(error) *error = QString("No images.txt in %1 (text model expected: sparse/0/{cameras,images,points3D}.txt)").arg(dir);
		return false;
	}

	// cameras.txt: first camera (all images of the RTAB-Map export share it)
	for(const QString & line : dataLines(modelDir_ + "/cameras.txt", error))
	{
		QStringList v = line.simplified().split(' ', Qt::SkipEmptyParts);
		if(v.size() < 5)
		{
			continue;
		}
		intrinsics_.model = v[1].toStdString();
		intrinsics_.width = v[2].toInt();
		intrinsics_.height = v[3].toInt();
		if(intrinsics_.model == "SIMPLE_PINHOLE" || intrinsics_.model == "SIMPLE_RADIAL" || intrinsics_.model == "RADIAL")
		{
			intrinsics_.fx = intrinsics_.fy = v[4].toDouble();
			intrinsics_.cx = v[5].toDouble();
			intrinsics_.cy = v[6].toDouble();
		}
		else if(v.size() >= 8) // PINHOLE, OPENCV, ... (distortion ignored)
		{
			intrinsics_.fx = v[4].toDouble();
			intrinsics_.fy = v[5].toDouble();
			intrinsics_.cx = v[6].toDouble();
			intrinsics_.cy = v[7].toDouble();
		}
		break;
	}

	// images.txt: one line of pose, one line of 2D points
	QStringList lines;
	{
		QFile file(modelDir_ + "/images.txt");
		file.open(QIODevice::ReadOnly | QIODevice::Text);
		QTextStream in(&file);
		while(!in.atEnd()) lines.push_back(in.readLine());
	}
	for(int i=0; i<lines.size(); ++i)
	{
		if(lines[i].startsWith('#') || lines[i].trimmed().isEmpty())
		{
			continue;
		}
		QStringList v = lines[i].simplified().split(' ', Qt::SkipEmptyParts);
		if(v.size() < 10)
		{
			continue;
		}
		Frame f;
		f.imageId = v[0].toInt();
		Eigen::Matrix3d Rcw = Eigen::Quaterniond(v[1].toDouble(), v[2].toDouble(), v[3].toDouble(), v[4].toDouble()).normalized().toRotationMatrix();
		Eigen::Vector3d tcw(v[5].toDouble(), v[6].toDouble(), v[7].toDouble());
		f.estimate.R = Rcw.transpose();
		f.estimate.t = -Rcw.transpose() * tcw;
		f.name = v[9].toStdString();
		bool ok = false;
		int key = QFileInfo(v[9]).completeBaseName().toInt(&ok);
		f.key = ok ? key : -1;
		frames_.push_back(f);
		++i; // skip the 2D points line
	}
	std::sort(frames_.begin(), frames_.end(), [](const Frame & a, const Frame & b) {
		return a.key >= 0 && b.key >= 0 ? a.key < b.key : a.name < b.name;
	});
	if(frames_.empty())
	{
		if(error) *error = "No image in images.txt";
		return false;
	}

	// Stamps of the RTAB-Map export
	if(QFileInfo::exists(dir + "/camera_poses.txt"))
	{
		std::map<int, double> stamps;
		for(const TumPose & p : readTum(dir + "/camera_poses.txt", error))
		{
			stamps[p.id] = p.stamp;
		}
		for(Frame & f : frames_)
		{
			if(stamps.count(f.key)) f.stamp = stamps[f.key];
		}
	}

	for(const QString & line : dataLines(modelDir_ + "/points3D.txt", 0))
	{
		QStringList v = line.simplified().split(' ', Qt::SkipEmptyParts);
		if(v.size() < 7)
		{
			continue;
		}
		Point3D p;
		p.xyz = Eigen::Vector3f(v[1].toFloat(), v[2].toFloat(), v[3].toFloat());
		p.rgb[0] = (unsigned char)v[4].toInt();
		p.rgb[1] = (unsigned char)v[5].toInt();
		p.rgb[2] = (unsigned char)v[6].toInt();
		points_.push_back(p);
	}

	stats_.pathLength = 0;
	for(size_t i=1; i<frames_.size(); ++i)
	{
		stats_.pathLength += (frames_[i].estimate.t - frames_[i-1].estimate.t).norm();
	}

	if(QFileInfo::exists(dir + "/gt_camera_poses.txt"))
	{
		loadGroundTruth(dir + "/gt_camera_poses.txt", 0);
	}
	return true;
}

bool ColmapModel::loadGroundTruth(const QString & path, QString * error)
{
	std::vector<TumPose> gt = readTum(path, error);
	if(gt.empty())
	{
		if(error && error->isEmpty()) *error = QString("No pose in %1").arg(path);
		return false;
	}
	rawGt_.clear();
	bool byId = gt.front().id >= 0;
	if(byId)
	{
		std::map<int, Pose> poses;
		for(const TumPose & p : gt) poses[p.id] = p.pose;
		for(size_t i=0; i<frames_.size(); ++i)
		{
			if(poses.count(frames_[i].key)) rawGt_[i] = poses[frames_[i].key];
		}
	}
	else
	{
		std::sort(gt.begin(), gt.end(), [](const TumPose & a, const TumPose & b) {return a.stamp < b.stamp;});
		for(size_t i=0; i<frames_.size(); ++i)
		{
			if(frames_[i].stamp < 0) continue;
			auto it = std::lower_bound(gt.begin(), gt.end(), frames_[i].stamp,
					[](const TumPose & a, double s) {return a.stamp < s;});
			const TumPose * best = 0;
			if(it != gt.end()) best = &*it;
			if(it != gt.begin() && (!best || std::fabs((it-1)->stamp - frames_[i].stamp) < std::fabs(best->stamp - frames_[i].stamp)))
				best = &*(it-1);
			if(best && std::fabs(best->stamp - frames_[i].stamp) < 0.02) rawGt_[i] = best->pose;
		}
	}
	if(rawGt_.size() < 3)
	{
		if(error) *error = QString("Only %1 ground truth poses match the images (by %2)").arg(rawGt_.size()).arg(byId ? "id" : "stamp");
		rawGt_.clear();
		return false;
	}
	gtPath_ = path;
	align(false);
	return true;
}

void ColmapModel::align(bool withScale)
{
	for(Frame & f : frames_) f.hasGt = false;
	Stats s;
	s.pathLength = stats_.pathLength;
	if(rawGt_.size() < 3)
	{
		stats_ = s;
		return;
	}
	// Umeyama: gt aligned = scale * R * gt + t, fitted on the estimated positions
	const int n = (int)rawGt_.size();
	Eigen::Matrix3Xd src(3, n), dst(3, n);
	int k = 0;
	for(const auto & it : rawGt_)
	{
		src.col(k) = it.second.t;
		dst.col(k) = frames_[it.first].estimate.t;
		++k;
	}
	Eigen::Matrix4d T = Eigen::umeyama(src, dst, true);
	double scale = std::cbrt(T.block<3,3>(0,0).determinant());
	Eigen::Matrix3d R = T.block<3,3>(0,0) / scale;
	s.scale = 1.0 / scale; // ground truth size / estimate size
	if(!withScale)
	{
		T = Eigen::umeyama(src, dst, false);
		R = T.block<3,3>(0,0);
		scale = 1.0;
	}
	Eigen::Vector3d t = T.block<3,1>(0,3);

	std::vector<double> errors;
	double rotSum = 0;
	for(const auto & it : rawGt_)
	{
		Frame & f = frames_[it.first];
		f.hasGt = true;
		f.gt.R = R * it.second.R;
		f.gt.t = scale * R * it.second.t + t;
		f.error = (f.gt.t - f.estimate.t).norm();
		Eigen::AngleAxisd aa(f.gt.R.transpose() * f.estimate.R);
		f.rotError = std::fabs(aa.angle()) * kRadToDeg;
		errors.push_back(f.error);
		s.rmse += f.error * f.error;
		s.mean += f.error;
		s.max = std::max(s.max, f.error);
		rotSum += f.rotError * f.rotError;
	}
	// Constant rotation between the two camera frames (e.g., the ground truth
	// body frame is not exactly the camera frame): chordal mean of gt^T * estimate
	Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
	for(const auto & it : rawGt_)
	{
		const Frame & f = frames_[it.first];
		sum += f.gt.R.transpose() * f.estimate.R;
	}
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d D = svd.matrixU() * svd.matrixV().transpose();
	if(D.determinant() < 0)
	{
		Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
		S(2,2) = -1;
		D = svd.matrixU() * S * svd.matrixV().transpose();
	}
	s.rotOffset = std::fabs(Eigen::AngleAxisd(D).angle()) * kRadToDeg;
	double rotSumNoOffset = 0;
	for(const auto & it : rawGt_)
	{
		const Frame & f = frames_[it.first];
		double a = std::fabs(Eigen::AngleAxisd((f.gt.R * D).transpose() * f.estimate.R).angle()) * kRadToDeg;
		rotSumNoOffset += a * a;
	}
	s.rotRmseNoOffset = std::sqrt(rotSumNoOffset / n);
	s.matched = n;
	s.rmse = std::sqrt(s.rmse / n);
	s.mean /= n;
	s.rotRmse = std::sqrt(rotSum / n);
	std::sort(errors.begin(), errors.end());
	s.median = errors[errors.size()/2];
	stats_ = s;
}

QString ColmapModel::imagePath(const Frame & frame) const
{
	QString name = QString::fromStdString(frame.name);
	for(const QString & d : {dir_ + "/images", dir_ + "/input", dir_})
	{
		if(QFileInfo::exists(d + "/" + name)) return d + "/" + name;
	}
	return QString();
}
