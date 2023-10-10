#pragma once
#include<vector>
#include<map>
#include<unordered_map>
#include <opencv2/opencv.hpp>

#include "Eigen/Core"
#include "../../../thirdparty/Sophus/sophus/se3.hpp"
#include "../../../thirdparty/Sophus/sophus/geometry.hpp"

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include "ImuTypes.h"
#include "KeyPoint.h"
#include "utils/sensor_data.h"

using namespace std;
using namespace Eigen;

class KeyPoint;
class Frame
{
private:
    /* data */
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Frame();
    Frame(const Frame &frame);
    Frame(const double TimeStamp, const size_t Id);
    Frame(const double TimeStamp, const size_t Id, unordered_map<size_t, vector<double>> &features);
    ~Frame();

    void setIntegrated();

    void SetGpsPose( const Eigen::Vector3f &gps);

    double mTimeStamp;
    size_t mnId;
    bool have_gps = false;
    unordered_map<size_t, vector<double>>  mFeatures;

    void SetPose(ov_core::Pose_and_calib pose_);
    void SetVelocity(const Eigen::Vector3f &Vw_);

    ov_core::Pose_and_calib GetPose();

    Eigen::Vector3f GetVelocity();


    void SetNewBias(const IMU::Bias &b);
    void SetNewBiasInit(const IMU::Bias &b);
    
    Eigen::Vector3f GetGyroBias();

    Eigen::Vector3f GetAccBias();

    IMU::Bias GetImuBias();

    Eigen::Vector3f GetGpsPose();

    void AddFeatures(size_t featid, vector<double> feature, int camId);

    std::unordered_map<size_t, std::vector<double>> GetFeatures(int camId);

    void eraseFeatures(size_t featid, int camId);

public:
    bool mbImuPreintegrated;


    // Calibration parameters
    float mbf, mb;
    // cv::Mat mDistCoef;

    // Preintegrated IMU measurements from previous keyframe
    Frame* mPrevKF;

    Frame* leftPrevKF = nullptr;
    Frame* rightPrevKF = nullptr;

    Frame* leftPrevKF1 = nullptr;
    Frame* rightPrevKF1 = nullptr;
    Frame* leftPrevKF2 = nullptr;
    Frame* rightPrevKF2 = nullptr;
    IMU::Preintegrated* mpImuPreintegrated;
    IMU::Calib mImuCalib, mImuCalib1, mImuCalib2;//, mImuCalib2;

    // for lidar
    bool is_lidar = false;

    ov_core::Pose_and_calib old_pose;
    bool has_optimize = false;
protected:
    ov_core::Pose_and_calib pose;
    std::unordered_map<size_t, std::vector<double>> features, features1, features2;

    // Imu bias
    IMU::Bias mImuBias;
    Eigen::Vector3f mVw;

    // GPS
    Eigen::Vector3f pGps;

    std::mutex mMutexPose, mMutexImu, mMutexFeatures;;
    
};


