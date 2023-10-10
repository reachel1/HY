#include "KeyPoint.h"

KeyPoint::KeyPoint()
{
}

KeyPoint::KeyPoint(const KeyPoint &mp)
:mnId(mp.mnId), nObs(mp.nObs), cam_id(mp.cam_id), isBad(mp.isBad), firstObserveTime(mp.firstObserveTime),
 mObservations(mp.mObservations), mWorldPos(mp.mWorldPos),has_optimize(mp.has_optimize){}

KeyPoint::KeyPoint(const Eigen::Vector3f &Pos, const size_t id, const int camId)
{
    SetWorldPos(Pos);
    mnId = id;
    cam_id = camId;
    isBad = false;
}

void KeyPoint::SetWorldPos(const Eigen::Vector3f &Pos)
{
    //unique_lock<mutex> lock(mMutexPos);
    unique_lock<mutex> lock(mMutexFeatures);
    mWorldPos = Pos;
}

Eigen::Vector3f KeyPoint::GetWorldPos()
{
    //unique_lock<mutex> lock(mMutexPos);
    unique_lock<mutex> lock(mMutexFeatures);
    return mWorldPos;
}

KeyPoint::~KeyPoint()
{
}



/**
 * @brief 返回被观测次数，双目一帧算两次，左右目各算各的
 * @return nObs
 */
int KeyPoint::Observations()
{
    return nObs;
}

void KeyPoint::AddObservation(const double timestamp)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mObservations.insert(timestamp);
    ++nObs;
}

unordered_set<double> KeyPoint::GetObservation()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

void KeyPoint::eraseObservation(const double timestamp)
{
    unique_lock<mutex> lock(mMutexFeatures);
    auto it = mObservations.find(timestamp);
    if(it != mObservations.end()){
        --nObs;
        it = mObservations.erase(it);
    }
    
}
