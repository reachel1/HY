#pragma once

#include "Frame.h"
#include <unordered_map>
#include <unordered_set>

using namespace std;


class KeyPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyPoint();
    KeyPoint(const KeyPoint &mp);
    KeyPoint(const Eigen::Vector3f &Pos, const size_t id, const int camId);


    void SetWorldPos(const Eigen::Vector3f &Pos);
    
    int Observations();

    Eigen::Vector3f GetWorldPos();

    std::vector<Eigen::Vector3f> getFeatPos();

    void AddObservation(const double timestamp);

    unordered_set<double> GetObservation();

    void eraseObservation(const double timestamp);

    ~KeyPoint();


    // at frame, we need the id-u-v; made of map
public:
    size_t mnId;
    int nObs;
    int cam_id;
    bool isBad;
    double firstObserveTime;
    bool has_optimize = false;


protected:    

    //obs:frame: time + u v
    unordered_set<double> mObservations;
    // Position in absolute coordinates
    Eigen::Vector3f mWorldPos;
    mutex mMutexFeatures;
};


