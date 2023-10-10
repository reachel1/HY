#include "Frame.h"


Frame::Frame(const double TimeStamp, const size_t Id, unordered_map<size_t, vector<double>> &features)
    :mTimeStamp(TimeStamp),mnId(Id),mFeatures(features){}

Frame::Frame(const double TimeStamp, const size_t Id):mTimeStamp(TimeStamp),mnId(Id){}

Frame::Frame(const Frame &frame)
    :mTimeStamp(frame.mTimeStamp),mnId(frame.mnId),mFeatures(frame.mFeatures),
     mbImuPreintegrated(frame.mbImuPreintegrated),mPrevKF(frame.mPrevKF),mpImuPreintegrated(frame.mpImuPreintegrated),
     mImuCalib(frame.mImuCalib),mImuCalib1(frame.mImuCalib1),mImuCalib2(frame.mImuCalib2),
     is_lidar(frame.is_lidar),has_optimize(frame.has_optimize),
     mVw(frame.mVw),pose(frame.pose),have_gps(frame.have_gps),old_pose(frame.old_pose),
     mImuBias(frame.mImuBias),pGps(frame.pGps),leftPrevKF(frame.leftPrevKF),rightPrevKF(frame.rightPrevKF),
     mbf(frame.mbf),mb(frame.mb),leftPrevKF1(frame.leftPrevKF1),rightPrevKF1(frame.rightPrevKF1),
     leftPrevKF2(frame.leftPrevKF2),rightPrevKF2(frame.rightPrevKF2){}

Frame::~Frame()
{
}

void Frame::setIntegrated()
{
    unique_lock<mutex> lock1(mMutexImu);
    mbImuPreintegrated = true;
}

void Frame::SetGpsPose( const Eigen::Vector3f &gps)
{
    pGps = gps;
    have_gps = true;
}
// 设置当前关键帧的位姿
void Frame::SetPose(ov_core::Pose_and_calib pose_)
{
    unique_lock<mutex> lock(mMutexPose);
    pose = pose_;
}

void Frame::SetVelocity(const Eigen::Vector3f &Vw)
{
    unique_lock<mutex> lock(mMutexImu);
    mVw = Vw;
    pose.velocity = Vw.cast<double>();
}

Eigen::Vector3f Frame::GetGpsPose()
{
    return pGps;
}

ov_core::Pose_and_calib Frame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return pose;
}

Eigen::Vector3f Frame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexImu);
    mVw = pose.velocity.cast<float>();
    return mVw;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock1(mMutexImu);
    mImuBias = b;
    if (mpImuPreintegrated){
        mpImuPreintegrated->SetNewBias(b);
    }
        
}

void Frame::SetNewBiasInit(const IMU::Bias &b)
{
    mImuBias = b;
}

Eigen::Vector3f Frame::GetGyroBias()
{
    unique_lock<mutex> lock1(mMutexImu);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f Frame::GetAccBias()
{
    unique_lock<mutex> lock1(mMutexImu);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias Frame::GetImuBias()
{
    unique_lock<mutex> lock1(mMutexImu);
    return mImuBias;
}

void Frame::AddFeatures(size_t featid, vector<double> feature, int camId)
{
    unique_lock<mutex> lock2(mMutexFeatures);
    if(camId == 0){
        features[featid] = feature;
    }else if(camId == 1){
        features1[featid] = feature;
    }else{
        features2[featid] = feature;
    }
}

std::unordered_map<size_t, std::vector<double>> Frame::GetFeatures(int camId)
{
    unique_lock<mutex> lock2(mMutexFeatures);
    if(camId == 0){
        return features;
    }else if(camId == 1){
        return features1;
    }
    return features2;
}

void Frame::eraseFeatures(size_t featid, int camId)
{
    unique_lock<mutex> lock2(mMutexFeatures);
    if(camId == 0){
        auto it = features.find(featid);
        if(it != features.end()) it = features.erase(it);
    }else if(camId == 1){
        auto it = features1.find(featid);
        if(it != features1.end()) it = features1.erase(it);
    }else{
        auto it = features2.find(featid);
        if(it != features2.end()) it = features2.erase(it);
    }
}

