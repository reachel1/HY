#ifndef OV_MSCKF_OPTIMIZER_H
#define OV_MSCKF_OPTIMIZER_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "state/State.h"
#include "feat/FeatureDatabase.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"
#include "core/VioManagerOptions.h"

#include <Eigen/StdVector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <condition_variable>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>

#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"
// #include "control/Control.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <unordered_set>
#include <unsupported/Eigen/MatrixFunctions>
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <utility>
#include <queue>
#include <list>

#include  <g2o/types/sim3/types_seven_dof_expmap.h>
#include  <g2o/types/sba/types_sba.h>
#include  <g2o/core/sparse_optimizer.h>
#include <g2o/core/eigen_types.h>
#include <g2o/types/slam3d/dquat2mat.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/core/factory.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

#include "Frame.h"
#include "KeyPoint.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "ImuTypes.h"
#include "tic_toc.h"

using namespace std;
using namespace Eigen;
using namespace ov_core;
using namespace ov_type;

namespace ov_msckf {

/**
 * @brief State of our filter
 *
 * This state has all the current estimates for the filter.
 * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
 * We additionally have more parameters for online estimation of calibration and SLAM features.
 * We also have the covariance of the system, which should be managed using the StateHelper class.
 */
class Optimizer {

public:
  /**
   * @brief Default Constructor (will initialize variables to defaults)
   * @param options_ Options structure containing filter options
   */
  Optimizer(int max_clone_size){
    max_clone = max_clone_size;
    feat_database = std::make_shared<ov_core::FeatureDatabase>();
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );
    // solver->setUserLambdaInit(1e-5);
    localOptimizer.setAlgorithm(solver);
    localOptimizer.setVerbose(true);

    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver1;
    linearSolver1 = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver1 = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver1))
    );
    // solver1->setUserLambdaInit(1e-5);
    globalOptimizer.setAlgorithm(solver1);
    globalOptimizer.setVerbose(true);

    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver2;
    linearSolver2 = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver2 = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver2))
    );
    // solver2->setUserLambdaInit(1e-16);
    essentialGraphOptimizer.setAlgorithm(solver2);
    essentialGraphOptimizer.setVerbose(true);

    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver3;
    linearSolver3 = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver3 = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver3))
    );
    globalOptimizer1.setAlgorithm(solver3);
    globalOptimizer1.setVerbose(true);
  }

  void initOptimizerParameters(const vector<double> &imuNoises, std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> &camaraIntris, const double &imuFreq, const double &gravity)
  {
    ACC_N = imuNoises[0];
    ACC_W = imuNoises[1];
    GYR_N = imuNoises[2];
    GYR_W = imuNoises[3];
    mImuFreq = imuFreq;
    G(2) = gravity;

    cam_intins = camaraIntris;
    // mb的取值方式要改
    mb = 1398.758588 / 1896.453734;
  }

  ~Optimizer() {}

  void get_pose(std::shared_ptr<ov_msckf::State> state){
    //可优化的time
    cur_marg_time = state->margtimestep();
    double timestamp = state->_timestamp;
    ov_core::Pose_and_calib pose;
    pose.vio = 1;
    pose.timestamp = state->_timestamp;
    pose.R = state->_imu->Rot();
    pose.t = state->_imu->pos();
    pose.R_calib0 = state->_calib_IMUtoCAM[0]->Rot();
    pose.t_calib0 = state->_calib_IMUtoCAM[0]->pos();
    pose.R_calib1 = state->_calib_IMUtoCAM[1]->Rot();
    pose.t_calib1 = state->_calib_IMUtoCAM[1]->pos();
    pose.R_calib2 = state->_calib_IMUtoCAM[2]->Rot();
    pose.t_calib2 = state->_calib_IMUtoCAM[2]->pos();
    pose.velocity = state->_imu->vel();
    pose.bias_a   = state->_imu->bias_a();
    pose.bias_g   = state->_imu->bias_g();

    ++ FrameId;
    Frame PerVio(pose.timestamp, FrameId);

    float Ng = GYR_N;
    float Na = ACC_N;
    float Ngw = GYR_W;
    float Naw = ACC_W;
    const float sf = sqrt(mImuFreq);
    Sophus::SE3<float> Tcb(pose.R_calib0.cast<float>(), pose.t_calib0.cast<float>());
    Sophus::SE3f Tbc = Tcb.inverse();
    PerVio.mImuCalib = *(new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf));
    Sophus::SE3<float> Tcb1(pose.R_calib1.cast<float>(), pose.t_calib1.cast<float>());
    Sophus::SE3f Tbc1 = Tcb1.inverse();
    PerVio.mImuCalib1 = *(new IMU::Calib(Tbc1,Ng*sf,Na*sf,Ngw/sf,Naw/sf));
    if(max_cameras == 3){
        Sophus::SE3<float> Tcb2(pose.R_calib2.cast<float>(), pose.t_calib2.cast<float>());
        Sophus::SE3f Tbc2 = Tcb2.inverse();
        PerVio.mImuCalib2 = *(new IMU::Calib(Tbc2,Ng*sf,Na*sf,Ngw/sf,Naw/sf));
    }
    
    PerVio.SetPose(pose);
    PerVio.old_pose = pose;
            
    PerVio.SetNewBiasInit(IMU::Bias(pose.bias_a(0),pose.bias_a(1),pose.bias_a(2),pose.bias_g(0),pose.bias_g(1),pose.bias_g(2)));
    PerVio.SetVelocity(Eigen::Vector3f(pose.velocity(0),pose.velocity(1),pose.velocity(2)));

    PerVio.mb = mb;
    float cam0fx = float(cam_intins[0]->get_value()(0));
    PerVio.mbf = mb * cam0fx;
    PerVio.SetGpsPose(pose.gps_data.cast<float>());

    Frame *pkf = new Frame(PerVio);
    framesMutex.lock();
    sysFrames[pose.timestamp] = pkf;
    framesMutex.unlock();
    // get feat
    std::vector<std::shared_ptr<ov_core::Feature>> feat;
    std::vector<std::shared_ptr<ov_core::Feature>> feats_lost, feats_marg, feats_maxtracks;
    feats_lost = feat_database->features_not_containing_newer(state->_timestamp,true,false);
    if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
        feats_marg = feat_database->features_containing(state->margtimestep(), false, false);
    }
    auto itm = feats_marg.begin();
    while (itm != feats_marg.end()) {
        // See if any of our camera's reached max track
        bool reached_max = false;
        for (const auto &cams : (*itm)->timestamps) {
            if ((int)cams.second.size() > state->_options.max_clone_size) {
                reached_max = true;
                break;
            }
        }
        // If max track, then add it to our possible slam feature list
        if (reached_max) {
            feats_maxtracks.push_back(feat_database->get_feature((*itm)->featid,true));
            feats_longtrack.push_back((*itm)->featid);
        } 
        itm++;      
    }
    auto itl = feats_longtrack.begin();
    while(itl != feats_longtrack.end()){
        std::shared_ptr<ov_core::Feature> feat_longtrack = feat_database->get_feature((*itl),true);
        if(feat_longtrack != nullptr)
            {feats_maxtracks.push_back(feat_longtrack);itl++;}
        else
            itl = feats_longtrack.erase(itl);
    }
    feat.insert(feat.end(),feats_lost.begin(),feats_lost.end());
    feat.insert(feat.end(),feats_maxtracks.begin(),feats_maxtracks.end());
    // take feat into optimize
    auto it = feat.begin();
    while(it != feat.end()){
        if(!(*it)->triangule){
            it = feat.erase(it);
            continue;
        }
        if((*it)->timestamps.size()>1){
            for(size_t q = 0;q < (*it)->timestamps[0].size();q++){
                if(std::find((*it)->timestamps[1].begin(),(*it)->timestamps[1].end(),(*it)->timestamps[0].at(q)) != (*it)->timestamps[1].end())
                {
                    feat_map[(*it)->timestamps[0].at(q)].cam0++;
                    feat_map[(*it)->timestamps[0].at(q)].cam1++;
                    feat_map[(*it)->timestamps[0].at(q)].stereo++;
                    feat_map[(*it)->timestamps[0].at(q)].observe++;
                }
                else
                {
                    feat_map[(*it)->timestamps[0].at(q)].cam0++;
                    feat_map[(*it)->timestamps[0].at(q)].mono++;
                    feat_map[(*it)->timestamps[0].at(q)].observe++;
                }
            }
        }
        else{
            for(size_t q = 0;q < (*it)->timestamps.begin()->second.size();q++){
                if((*it)->timestamps.begin()->first == 0)feat_map[(*it)->timestamps.begin()->second.at(q)].cam0++;
                if((*it)->timestamps.begin()->first == 2)feat_map[(*it)->timestamps.begin()->second.at(q)].cam2++;
                feat_map[(*it)->timestamps.begin()->second.at(q)].mono++;
                feat_map[(*it)->timestamps.begin()->second.at(q)].observe++;
            }
        }
        it++;
    }
    PRINT_INFO(GREEN "[opt]feat size %d\n" RESET,feat.size());
    for(size_t i = 0;i<feat.size();i++){
        size_t id = feat.at(i)->featid;
        Eigen::Vector3f pos(feat.at(i)->p_FinG.cast<float>());

        auto mpit = sysKeyPoints.find(id);
        if(mpit == sysKeyPoints.end() && !feat.at(i)->triangule) continue;
        KeyPoint mp(pos, id, int(feat.at(i)->anchor_cam_id));
        
        for(auto ii = feat.at(i)->timestamps.begin(); ii != feat.at(i)->timestamps.end(); ++ii){
            int camid = ii->first;
            for(int j = 0; j < ii->second.size(); ++j){
                double f_time = ii->second[j];
                auto fit = sysFrames.find(f_time);
                if(fit == sysFrames.end()) continue;
                framesMutex.lock();

                vector<double> uvs = {feat.at(i)->uvs.find(camid)->second[j](0), feat.at(i)->uvs.find(camid)->second[j](1)};
                vector<double> uvs_norm = {feat.at(i)->uvs_norm.find(camid)->second[j](0), feat.at(i)->uvs_norm.find(camid)->second[j](1)};

                if(mpit == sysKeyPoints.end()){
                    KeyPoint *pmp = new KeyPoint(mp);
                    pointsMutex.lock();
                    sysKeyPoints[id] = pmp;
                    pointsMutex.unlock();
                    mp.AddObservation(f_time);
                }else{
                    mpit->second->AddObservation(f_time);
                }
                std::vector<double> newFeat{uvs[0],uvs[1],uvs_norm[0],uvs_norm[1]};
                fit->second->AddFeatures(id, newFeat, camid);
                framesMutex.unlock();
            }
        }
    }
    feat.clear();
    feats_lost.clear(); 
    feats_marg.clear();
    feats_maxtracks.clear();
    quePoseMutex.lock();
    quePose.push(pose);
    quePoseMutex.unlock();
  }
void globalBA(){
    while(1){
        if(esi_has_end) break;
        unique_lock<mutex> lck(wait_mutex);
        wait_condition_variable.wait(lck);

        TicToc t_global;
        std::string filename = outputPath + "/f_time.txt";
        stringstream ss1;
        ss1 << filename;
        ofstream fout1(ss1.str(), ios::app);
        fout1.setf(ios::fixed, ios::floatfield);
        fout1.precision(5);
        fout1 << "global optimize begin" <<endl;

        map<double,Frame *> mGlobalFrames;
        unordered_map<double,Frame *> lFixedFrames;
        unordered_map<size_t, KeyPoint*> mGlobalPoints;
        vector<EdgeMono*> vpEdgesMono;
        vector<EdgeStereo*> vpEdgesStereo;
        vector<pair<Frame *, KeyPoint *>> vToConsider, vToConsiderStereo;
        vector<EdgeMonoNew*> vpEdgesMonoNew;
        vector<EdgeStereoNew*> vpEdgesStereoNew;
        vector<pair<Frame *, KeyPoint *>> vToConsiderNew, vToConsiderStereoNew;
        unordered_set<size_t> NoToOptimize;
        double monoInfo = 1.0 / 1.5; 
        double stereoInfo = 1.0 / 1.6;
        double thirdInfo = 0.5;
        int startIndex = sysKeyFrames.rbegin()->second->mnId;
        const unsigned long iniMPid = startIndex + 10;


        fout1 << "global optimize begin1" <<endl;
        int numFrame = 5;
        int framescnt = 0;
        Frame* midFrame = nullptr;
        for(auto fit = stepGlobalFrames.rbegin(); numFrame > 0; --numFrame, ++fit){
            mGlobalFrames[fit->first] = fit->second[4];
            if(framescnt == 2) midFrame = fit->second[4];
            ++framescnt;
            for(int i = 0; i < fit->second.size(); ++i){
                Frame* pf = fit->second[i];
                unordered_map<size_t, std::vector<double>> feats = pf->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = pf->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = pf->GetFeatures(2);

                for(auto tmpit = feats.begin(); tmpit != feats.end(); ++tmpit){
                    mGlobalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
                for(auto tmpit = feats1.begin(); tmpit != feats1.end(); ++tmpit){
                    mGlobalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
                for(auto tmpit = feats2.begin(); tmpit != feats2.end(); ++tmpit){
                    mGlobalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
            }
        }


        fout1 << "global optimize begin2" <<endl;
        for(auto it = mGlobalPoints.begin(); it != mGlobalPoints.end(); ++it){
            KeyPoint *pMP = it->second;
            int obs_num = 0;
            unordered_set<double> observes = it->second->GetObservation();
            if(observes.size() != 0){
                for(auto it1 = observes.begin(); it1 != observes.end(); ++it1){
                    auto iit = sysKeyFrames.find(*it1);
                    if(iit != sysKeyFrames.end()) ++obs_num;
                }
            }
            if(obs_num <= 1) NoToOptimize.insert(pMP->mnId);
        }

        for(auto frameIter = mGlobalFrames.begin(); frameIter != mGlobalFrames.end(); ++frameIter)
        {
            Frame *pf = frameIter->second;
            VertexPose * VP = new VertexPose(pf, cam_intins);
            VP->setId(pf->mnId);
            globalOptimizer.addVertex(VP);
        }
        for(auto it = mGlobalPoints.begin(); it != mGlobalPoints.end(); ++it){
            KeyPoint *pMP = it->second;
            VertexSBAPointXYZ* vPoint = new VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            size_t id = pMP->mnId +iniMPid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            unordered_set<double> observes = it->second->GetObservation();

            if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
            globalOptimizer.addVertex(vPoint);

            for(auto it1 = observes.begin(); it1 != observes.end(); ++it1){
                auto iit = sysKeyFrames.find(*it1);
                if(iit == sysKeyFrames.end()) continue;
                Frame *curFrame = iit->second;
                if(curFrame->leftPrevKF){
                    Frame *pkf = curFrame->leftPrevKF;
                    if(mGlobalFrames.find(pkf->mTimeStamp) == mGlobalFrames.end() && lFixedFrames.find(pkf->mTimeStamp) == lFixedFrames.end()){
                        lFixedFrames[pkf->mTimeStamp] = pkf;
                        VertexPose * VP = new VertexPose(pkf, cam_intins);
                        VP->setId(pkf->mnId);
                        VP->setFixed(true);
                        globalOptimizer.addVertex(VP);
                    }
                    // 这里开始做curFrame和leftPKF之间的边,pkf是中心帧，curFrame是当前帧（不进行优化）
                    ov_core::Pose_and_calib pose_cur = curFrame->GetPose();
                    Eigen::Matrix3d Rc1w = pose_cur.R_calib0 * pose_cur.R;
                    Eigen::Vector3d twc01 = pose_cur.t - Rc1w.transpose() * pose_cur.t_calib0;
                    Eigen::Vector3d tc1w = - Rc1w * twc01;

                    ov_core::Pose_and_calib pose_pkf = pkf->GetPose();
                    Eigen::Matrix3d Rcw = pose_pkf.R_calib0 * pose_pkf.R;
                    Eigen::Vector3d twc0 = pose_pkf.t - Rcw.transpose() * pose_pkf.t_calib0;
                    Eigen::Vector3d tcw = - Rcw * twc0;

                    Eigen::Matrix3d Rc1c = Rc1w * Rcw.transpose();
                    Eigen::Vector3d tc1c = tc1w - Rc1c * tcw;
                    unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
                    unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
                    unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

                    auto fit = feats.find(pMP->mnId);
                    auto fit1 = feats1.find(pMP->mnId);
                    auto fit2 = feats2.find(pMP->mnId);
                    if(fit != feats.end() && fit1 != feats1.end()){
                        Eigen::Matrix<double, 3, 1> obs;
                        EdgeStereoNew *e = new EdgeStereoNew(0, Rc1c, tc1c);
                        obs << fit->second[0],fit->second[1],fit1->second[0];

                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(obs);
                        e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        globalOptimizer.addEdge(e);
                        vpEdgesStereoNew.push_back(e);
                        vToConsiderStereoNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});

                        if(fit2 != feats2.end()){
                            EdgeMonoNew *e1 = new EdgeMonoNew(2, Rc1c, tc1c);
                            e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                            e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                            e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                            e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e1->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);
                            globalOptimizer.addEdge(e1);
                            vpEdgesMonoNew.push_back(e1);
                            vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                        }
                        continue;
                    }
                    if(fit != feats.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(0, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e);
                        vpEdgesMonoNew.push_back(e);
                        vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                    if(fit1 != feats1.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(1, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e);
                        vpEdgesMonoNew.push_back(e);
                        vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                    if(fit2 != feats2.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(2, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e);
                        vpEdgesMonoNew.push_back(e);
                        vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                }
                if(curFrame->rightPrevKF){
                    Frame *pkf = curFrame->rightPrevKF;
                    if(mGlobalFrames.find(pkf->mTimeStamp) == mGlobalFrames.end() && lFixedFrames.find(pkf->mTimeStamp) == lFixedFrames.end()){
                        lFixedFrames[pkf->mTimeStamp] = pkf;
                        VertexPose * VP = new VertexPose(pkf, cam_intins);
                        VP->setId(pkf->mnId);
                        VP->setFixed(true);
                        globalOptimizer.addVertex(VP);
                    }
                    // 这里开始做curFrame和rightPKF之间的边
                    ov_core::Pose_and_calib pose_cur = curFrame->GetPose();
                    Eigen::Matrix3d Rc1w = pose_cur.R_calib0 * pose_cur.R;
                    Eigen::Vector3d twc01 = pose_cur.t - Rc1w.transpose() * pose_cur.t_calib0;
                    Eigen::Vector3d tc1w = - Rc1w * twc01;

                    ov_core::Pose_and_calib pose_pkf = pkf->GetPose();
                    Eigen::Matrix3d Rcw = pose_pkf.R_calib0 * pose_pkf.R;
                    Eigen::Vector3d twc0 = pose_pkf.t - Rcw.transpose() * pose_pkf.t_calib0;
                    Eigen::Vector3d tcw = - Rcw * twc0;

                    Eigen::Matrix3d Rc1c = Rc1w * Rcw.transpose();
                    Eigen::Vector3d tc1c = tc1w - Rc1c * tcw;
                    unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
                    unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
                    unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

                    auto fit = feats.find(pMP->mnId);
                    auto fit1 = feats1.find(pMP->mnId);
                    auto fit2 = feats2.find(pMP->mnId);
                    if(fit != feats.end() && fit1 != feats1.end()){
                        Eigen::Matrix<double, 3, 1> obs;
                        EdgeStereoNew *e = new EdgeStereoNew(0, Rc1c, tc1c);
                        obs << fit->second[0],fit->second[1],fit1->second[0];

                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(obs);
                        e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        globalOptimizer.addEdge(e);
                        vpEdgesStereoNew.push_back(e);
                        vToConsiderStereoNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});

                        if(fit2 != feats2.end()){
                            EdgeMonoNew *e1 = new EdgeMonoNew(2, Rc1c, tc1c);
                            e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                            e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                            e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                            e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e1->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);
                            globalOptimizer.addEdge(e1);
                            vpEdgesMonoNew.push_back(e1);
                            vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                        }
                        continue;
                    }
                    if(fit != feats.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(0, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e);
                        vpEdgesMonoNew.push_back(e);
                        vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                    if(fit1 != feats1.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(1, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e);
                        vpEdgesMonoNew.push_back(e);
                        vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                    if(fit2 != feats2.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(2, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e);
                        vpEdgesMonoNew.push_back(e);
                        vToConsiderNew.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                }
                if(curFrame->leftPrevKF || curFrame->rightPrevKF) continue;
                // 没被continue的话，就是中心帧，进行下面的添加边操作

                unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

                auto fit = feats.find(pMP->mnId);
                auto fit1 = feats1.find(pMP->mnId);
                auto fit2 = feats2.find(pMP->mnId);
                if(fit != feats.end() && fit1 != feats1.end()){
                    Eigen::Matrix<double, 3, 1> obs;
                    EdgeStereo *e = new EdgeStereo(0);
                    obs << fit->second[0],fit->second[1],fit1->second[0];

                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                    e->setMeasurement(obs);

                    e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);
                    globalOptimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vToConsiderStereo.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});

                    if(fit2 != feats2.end()){
                        EdgeMono* e1 = new EdgeMono(2);
                        e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(curFrame->mnId)));
                        e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                        e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                        e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e1->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer.addEdge(e1);
                        vpEdgesMono.push_back(e1);
                        vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                    continue;
                }
                if(fit != feats.end()){
                    EdgeMono* e = new EdgeMono(0);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                    e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));
                    
                    e->setInformation(Eigen::Matrix2d::Identity() * monoInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    globalOptimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
                if(fit1 != feats1.end()){
                    EdgeMono* e = new EdgeMono(1);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                    e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                    e->setInformation(Eigen::Matrix2d::Identity() * monoInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    globalOptimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
                if(fit2 != feats2.end()){
                    EdgeMono* e = new EdgeMono(2);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer.vertex(id)));
                    e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                    e->setInformation(Eigen::Matrix2d::Identity() * thirdInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    globalOptimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
            }
        }
        fout1 << "global optimize begin3" <<endl;
        
        globalOptimizer.initializeOptimization();
        globalOptimizer.optimize(15);

        /*/
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            EdgeMono* e = vpEdgesMono[i];
            if(e->chi2()>chi2Mono2){
                Frame *eraseFrame = vToConsider[i].first;
                KeyPoint *erasePoint = vToConsider[i].second;

                eraseFrame->eraseFeatures(erasePoint->mnId, e->cam_idx);
                unordered_map<size_t, std::vector<double>> feats = eraseFrame->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = eraseFrame->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = eraseFrame->GetFeatures(2);
                if(feats.find(erasePoint->mnId)==feats.end()&&feats1.find(erasePoint->mnId)==feats1.end()&&feats2.find(erasePoint->mnId)==feats2.end()){
                    erasePoint->eraseObservation(eraseFrame->mTimeStamp);
                }
            }
        }
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            EdgeStereo* e = vpEdgesStereo[i];

            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
            if(e->chi2()>chi2Stereo2)
            {
                Frame *eraseFrame = vToConsiderStereo[i].first;
                KeyPoint *erasePoint = vToConsiderStereo[i].second;
                
                eraseFrame->eraseFeatures(erasePoint->mnId, 0);
                eraseFrame->eraseFeatures(erasePoint->mnId, 1);
                unordered_map<size_t, std::vector<double>> feats = eraseFrame->GetFeatures(2);
                if(feats.find(erasePoint->mnId)==feats.end()){
                    erasePoint->eraseObservation(eraseFrame->mTimeStamp);
                }
            }
        }
    
        for(size_t i=0, iend=vpEdgesMonoNew.size(); i<iend;i++)
        {
            EdgeMonoNew* e = vpEdgesMonoNew[i];
            if(e->chi2()>chi2Mono2){
                Frame *eraseFrame = vToConsiderNew[i].first;
                KeyPoint *erasePoint = vToConsiderNew[i].second;

                eraseFrame->eraseFeatures(erasePoint->mnId, e->cam_idx);
                unordered_map<size_t, std::vector<double>> feats = eraseFrame->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = eraseFrame->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = eraseFrame->GetFeatures(2);
                if(feats.find(erasePoint->mnId)==feats.end()&&feats1.find(erasePoint->mnId)==feats1.end()&&feats2.find(erasePoint->mnId)==feats2.end()){
                    erasePoint->eraseObservation(eraseFrame->mTimeStamp);
                }
            }
        }
        for(size_t i=0, iend=vpEdgesStereoNew.size(); i<iend;i++)
        {
            EdgeStereoNew* e = vpEdgesStereoNew[i];

            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
            if(e->chi2()>chi2Stereo2)
            {
                Frame *eraseFrame = vToConsiderStereoNew[i].first;
                KeyPoint *erasePoint = vToConsiderStereoNew[i].second;
                
                eraseFrame->eraseFeatures(erasePoint->mnId, 0);
                eraseFrame->eraseFeatures(erasePoint->mnId, 1);
                unordered_map<size_t, std::vector<double>> feats = eraseFrame->GetFeatures(2);
                if(feats.find(erasePoint->mnId)==feats.end()){
                    erasePoint->eraseObservation(eraseFrame->mTimeStamp);
                }
            }
        }
        */

        framescnt = 0;
        vector<Frame *> perLocal;
        unordered_set<double> havePushBack;
        for(auto itt = mGlobalFrames.begin(); itt != mGlobalFrames.end(); ++itt){
            Frame *pKFi = itt->second;
            ov_core::Pose_and_calib pose = pKFi->GetPose();
            ov_core::Pose_and_calib oldpose = pKFi->GetPose();
            VertexPose* VP = static_cast<VertexPose*>(globalOptimizer.vertex(pKFi->mnId));
            pose.R = pose.R_calib0.transpose() * VP->estimate().Rcw[0];
            pose.t = VP->estimate().Rcw[0].transpose()*pose.t_calib0 - VP->estimate().Rcw[0].transpose()*VP->estimate().tcw[0];
            // pKFi->SetPose(pose);
            // pose.R===Rbw,  pose.t===twb
            Eigen::Matrix3d Rcw = pose.R_calib0 * pose.R;
            Eigen::Vector3d twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
            Eigen::Vector3d tcw = - Rcw * twc0;
            g2o::SE3Quat Sjw(Rcw,tcw);
            // 
            Rcw = oldpose.R_calib0 * oldpose.R;
            twc0 = oldpose.t - Rcw.transpose() * oldpose.t_calib0;
            tcw = - Rcw * twc0;
            g2o::SE3Quat Siw(Rcw,tcw);
            g2o::SE3Quat Swi = Siw.inverse();
            g2o::SE3Quat Sji = Sjw * Swi;

            if(framescnt < 2){
                for(int i = 0; i < stepGlobalFrames[pKFi->mTimeStamp].size(); ++i){
                    stepGlobalFrames[pKFi->mTimeStamp][i]->rightPrevKF1 = midFrame;
                    if(!havePushBack.count(stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp)){
                        havePushBack.insert(stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp);
                        perLocal.push_back(stepGlobalFrames[pKFi->mTimeStamp][i]);

                        pose = stepGlobalFrames[pKFi->mTimeStamp][i]->GetPose();
                        Rcw = pose.R_calib0 * pose.R;
                        twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
                        tcw = - Rcw * twc0;
                        g2o::SE3Quat S_iw(Rcw,tcw);
                        g2o::SE3Quat S_jw = Sji * S_iw;
                        Eigen::Matrix3d eigR = S_jw.rotation().toRotationMatrix();
                        Eigen::Vector3d eigt = S_jw.translation();
                        // 这里分别是Rcw，tcw
                        pose.R = pose.R_calib0.transpose() * eigR;
                        pose.t = eigR.transpose()*pose.t_calib0 - eigR.transpose()*eigt;
                        stepGlobalFrames[pKFi->mTimeStamp][i]->SetPose(pose);
                    }
                }
            }else if(framescnt > 2){
                for(int i = 0; i < stepGlobalFrames[pKFi->mTimeStamp].size(); ++i){
                    stepGlobalFrames[pKFi->mTimeStamp][i]->leftPrevKF1 = midFrame;
                    if(!havePushBack.count(stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp)){
                        havePushBack.insert(stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp);
                        perLocal.push_back(stepGlobalFrames[pKFi->mTimeStamp][i]);

                        pose = stepGlobalFrames[pKFi->mTimeStamp][i]->GetPose();
                        Rcw = pose.R_calib0 * pose.R;
                        twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
                        tcw = - Rcw * twc0;
                        g2o::SE3Quat S_iw(Rcw,tcw);
                        g2o::SE3Quat S_jw = Sji * S_iw;
                        Eigen::Matrix3d eigR = S_jw.rotation().toRotationMatrix();
                        Eigen::Vector3d eigt = S_jw.translation();
                        // 这里分别是Rcw，tcw
                        pose.R = pose.R_calib0.transpose() * eigR;
                        pose.t = eigR.transpose()*pose.t_calib0 - eigR.transpose()*eigt;
                        stepGlobalFrames[pKFi->mTimeStamp][i]->SetPose(pose);
                    }
                }
            }else{
                bool flag = false;
                for(int i = 0; i < stepGlobalFrames[pKFi->mTimeStamp].size(); ++i){
                    if(!havePushBack.count(stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp)){
                        havePushBack.insert(stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp);
                        perLocal.push_back(stepGlobalFrames[pKFi->mTimeStamp][i]);

                        pose = stepGlobalFrames[pKFi->mTimeStamp][i]->GetPose();
                        Rcw = pose.R_calib0 * pose.R;
                        twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
                        tcw = - Rcw * twc0;
                        g2o::SE3Quat S_iw(Rcw,tcw);
                        g2o::SE3Quat S_jw = Sji * S_iw;
                        Eigen::Matrix3d eigR = S_jw.rotation().toRotationMatrix();
                        Eigen::Vector3d eigt = S_jw.translation();
                        // 这里分别是Rcw，tcw
                        pose.R = pose.R_calib0.transpose() * eigR;
                        pose.t = eigR.transpose()*pose.t_calib0 - eigR.transpose()*eigt;
                        stepGlobalFrames[pKFi->mTimeStamp][i]->SetPose(pose);
                    }
                    if(midFrame->mTimeStamp == stepGlobalFrames[pKFi->mTimeStamp][i]->mTimeStamp){
                        flag = true;
                        continue;
                    }
                    if(!flag) stepGlobalFrames[pKFi->mTimeStamp][i]->rightPrevKF1 = midFrame;
                    else stepGlobalFrames[pKFi->mTimeStamp][i]->leftPrevKF1 = midFrame;
                }
            }
            framescnt++;
        }
        wait_mutex1.lock();
        stepGlobalFrames1[midFrame->mTimeStamp] = perLocal;
        wait_mutex1.unlock();

        for(auto it = mGlobalPoints.begin(); it != mGlobalPoints.end(); ++it){
            KeyPoint *pMP = it->second;
            if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
            VertexSBAPointXYZ* vPoint = static_cast<VertexSBAPointXYZ*>(globalOptimizer.vertex(pMP->mnId+iniMPid+1));
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
        }
        
        globalOptimizer.clear();
        fout1<< " t_global:" << t_global.toc() ;
        fout1 <<endl;
        fout1.close();
    }
}

void globalBA1(){
    while(1){
        if(esi_has_end) break;
        unique_lock<mutex> lck(wait_mutex1);
        wait_condition_variable1.wait(lck);

        TicToc t_global;
        std::string filename = outputPath + "/f_time.txt";
        stringstream ss1;
        ss1 << filename;
        ofstream fout1(ss1.str(), ios::app);
        fout1.setf(ios::fixed, ios::floatfield);
        fout1.precision(5);
        fout1 << "global1 optimize begin" <<endl;

        map<double,Frame *> mGlobalFrames;
        unordered_map<double,Frame *> lFixedFrames;
        unordered_map<size_t, KeyPoint*> mGlobalPoints;
        vector<EdgeMono*> vpEdgesMono;
        vector<EdgeStereo*> vpEdgesStereo;
        vector<pair<Frame *, KeyPoint *>> vToConsider, vToConsiderStereo;
        unordered_set<size_t> NoToOptimize;
        double monoInfo = 1.0 / 1.5; 
        double stereoInfo = 1.0 / 1.6;
        double thirdInfo = 0.5;
        int startIndex = sysKeyFrames.rbegin()->second->mnId;
        const unsigned long iniMPid = startIndex + 10;

        int numFrame = 5;
        for(auto fit = stepGlobalFrames1.rbegin(); numFrame > 0; --numFrame, ++fit){
            mGlobalFrames[fit->first] = sysKeyFrames.find(fit->first)->second;
            fout1 << " " << fit->second.size() << " "<< fit->first << " ";

            for(int i = 0; i < fit->second.size(); ++i){
                Frame* pf = fit->second[i];
                unordered_map<size_t, std::vector<double>> feats = pf->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = pf->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = pf->GetFeatures(2);

                for(auto tmpit = feats.begin(); tmpit != feats.end(); ++tmpit){
                    mGlobalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
                for(auto tmpit = feats1.begin(); tmpit != feats1.end(); ++tmpit){
                    mGlobalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
                for(auto tmpit = feats2.begin(); tmpit != feats2.end(); ++tmpit){
                    mGlobalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
            }
        }

        for(auto it = mGlobalPoints.begin(); it != mGlobalPoints.end(); ++it){
            KeyPoint *pMP = it->second;
            int obs_num = 0;
            unordered_set<double> observes = it->second->GetObservation();
            if(observes.size() != 0){
                for(auto it1 = observes.begin(); it1 != observes.end(); ++it1){
                    auto iit = sysKeyFrames.find(*it1);
                    if(iit != sysKeyFrames.end()) ++obs_num;
                }
            }
            if(obs_num <= 1) NoToOptimize.insert(pMP->mnId);
        }

        for(auto frameIter = mGlobalFrames.begin(); frameIter != mGlobalFrames.end(); ++frameIter)
        {
            Frame *pf = frameIter->second;
            VertexPose * VP = new VertexPose(pf, cam_intins);
            VP->setId(pf->mnId);
            globalOptimizer1.addVertex(VP);
        }
        for(auto it = mGlobalPoints.begin(); it != mGlobalPoints.end(); ++it){
            KeyPoint *pMP = it->second;
            VertexSBAPointXYZ* vPoint = new VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            size_t id = pMP->mnId +iniMPid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            unordered_set<double> observes = it->second->GetObservation();

            if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
            globalOptimizer1.addVertex(vPoint);

            for(auto it1 = observes.begin(); it1 != observes.end(); ++it1){
                auto iit = sysKeyFrames.find(*it1);
                if(iit == sysKeyFrames.end()) continue;
                Frame *curFrame = iit->second;
                if(curFrame->leftPrevKF1){
                    Frame *pkf = curFrame->leftPrevKF1;
                    if(mGlobalFrames.find(pkf->mTimeStamp) == mGlobalFrames.end() && lFixedFrames.find(pkf->mTimeStamp) == lFixedFrames.end()){
                        lFixedFrames[pkf->mTimeStamp] = pkf;
                        VertexPose * VP = new VertexPose(pkf, cam_intins);
                        VP->setId(pkf->mnId);
                        VP->setFixed(true);
                        globalOptimizer1.addVertex(VP);
                    }
                    // 这里开始做curFrame和leftPKF之间的边,pkf是中心帧，curFrame是当前帧（不进行优化）
                    ov_core::Pose_and_calib pose_cur = curFrame->GetPose();
                    Eigen::Matrix3d Rc1w = pose_cur.R_calib0 * pose_cur.R;
                    Eigen::Vector3d twc01 = pose_cur.t - Rc1w.transpose() * pose_cur.t_calib0;
                    Eigen::Vector3d tc1w = - Rc1w * twc01;

                    ov_core::Pose_and_calib pose_pkf = pkf->GetPose();
                    Eigen::Matrix3d Rcw = pose_pkf.R_calib0 * pose_pkf.R;
                    Eigen::Vector3d twc0 = pose_pkf.t - Rcw.transpose() * pose_pkf.t_calib0;
                    Eigen::Vector3d tcw = - Rcw * twc0;

                    Eigen::Matrix3d Rc1c = Rc1w * Rcw.transpose();
                    Eigen::Vector3d tc1c = tc1w - Rc1c * tcw;
                    unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
                    unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
                    unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

                    auto fit = feats.find(pMP->mnId);
                    auto fit1 = feats1.find(pMP->mnId);
                    auto fit2 = feats2.find(pMP->mnId);
                    if(fit != feats.end() && fit1 != feats1.end()){
                        Eigen::Matrix<double, 3, 1> obs;
                        EdgeStereoNew *e = new EdgeStereoNew(0, Rc1c, tc1c);
                        obs << fit->second[0],fit->second[1],fit1->second[0];

                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(obs);
                        e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        globalOptimizer1.addEdge(e);

                        if(fit2 != feats2.end()){
                            EdgeMonoNew *e1 = new EdgeMonoNew(2, Rc1c, tc1c);
                            e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                            e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                            e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                            e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e1->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);
                            globalOptimizer1.addEdge(e1);
                        }
                        continue;
                    }
                    if(fit != feats.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(0, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e);
                    }
                    if(fit1 != feats1.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(1, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e);
                    }
                    if(fit2 != feats2.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(2, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e);
                    }
                }
                if(curFrame->rightPrevKF1){
                    Frame *pkf = curFrame->rightPrevKF1;
                    if(mGlobalFrames.find(pkf->mTimeStamp) == mGlobalFrames.end() && lFixedFrames.find(pkf->mTimeStamp) == lFixedFrames.end()){
                        lFixedFrames[pkf->mTimeStamp] = pkf;
                        VertexPose * VP = new VertexPose(pkf, cam_intins);
                        VP->setId(pkf->mnId);
                        VP->setFixed(true);
                        globalOptimizer1.addVertex(VP);
                    }
                    // 这里开始做curFrame和rightPKF之间的边
                    ov_core::Pose_and_calib pose_cur = curFrame->GetPose();
                    Eigen::Matrix3d Rc1w = pose_cur.R_calib0 * pose_cur.R;
                    Eigen::Vector3d twc01 = pose_cur.t - Rc1w.transpose() * pose_cur.t_calib0;
                    Eigen::Vector3d tc1w = - Rc1w * twc01;

                    ov_core::Pose_and_calib pose_pkf = pkf->GetPose();
                    Eigen::Matrix3d Rcw = pose_pkf.R_calib0 * pose_pkf.R;
                    Eigen::Vector3d twc0 = pose_pkf.t - Rcw.transpose() * pose_pkf.t_calib0;
                    Eigen::Vector3d tcw = - Rcw * twc0;

                    Eigen::Matrix3d Rc1c = Rc1w * Rcw.transpose();
                    Eigen::Vector3d tc1c = tc1w - Rc1c * tcw;
                    unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
                    unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
                    unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

                    auto fit = feats.find(pMP->mnId);
                    auto fit1 = feats1.find(pMP->mnId);
                    auto fit2 = feats2.find(pMP->mnId);
                    if(fit != feats.end() && fit1 != feats1.end()){
                        Eigen::Matrix<double, 3, 1> obs;
                        EdgeStereoNew *e = new EdgeStereoNew(0, Rc1c, tc1c);
                        obs << fit->second[0],fit->second[1],fit1->second[0];

                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(obs);
                        e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        globalOptimizer1.addEdge(e);

                        if(fit2 != feats2.end()){
                            EdgeMonoNew *e1 = new EdgeMonoNew(2, Rc1c, tc1c);
                            e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                            e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                            e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                            e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e1->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);
                            globalOptimizer1.addEdge(e1);
                        }
                        continue;
                    }
                    if(fit != feats.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(0, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e);
                    }
                    if(fit1 != feats1.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(1, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e);
                    }
                    if(fit2 != feats2.end()){
                        EdgeMonoNew *e = new EdgeMonoNew(2, Rc1c, tc1c);
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(pkf->mnId)));
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                        e->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e);
                    }
                }
                if(curFrame->leftPrevKF1 || curFrame->rightPrevKF1) continue;
                // 没被continue的话，就是中心帧，进行下面的添加边操作

                unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

                auto fit = feats.find(pMP->mnId);
                auto fit1 = feats1.find(pMP->mnId);
                auto fit2 = feats2.find(pMP->mnId);
                if(fit != feats.end() && fit1 != feats1.end()){
                    Eigen::Matrix<double, 3, 1> obs;
                    EdgeStereo *e = new EdgeStereo(0);
                    obs << fit->second[0],fit->second[1],fit1->second[0];

                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                    e->setMeasurement(obs);

                    e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);
                    globalOptimizer1.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vToConsiderStereo.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});

                    if(fit2 != feats2.end()){
                        EdgeMono* e1 = new EdgeMono(2);
                        e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(curFrame->mnId)));
                        e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                        e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                        e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e1->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        globalOptimizer1.addEdge(e1);
                        vpEdgesMono.push_back(e1);
                        vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                    }
                    continue;
                }
                if(fit != feats.end()){
                    EdgeMono* e = new EdgeMono(0);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                    e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));
                    
                    e->setInformation(Eigen::Matrix2d::Identity() * monoInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    globalOptimizer1.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
                if(fit1 != feats1.end()){
                    EdgeMono* e = new EdgeMono(1);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                    e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                    e->setInformation(Eigen::Matrix2d::Identity() * monoInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    globalOptimizer1.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
                if(fit2 != feats2.end()){
                    EdgeMono* e = new EdgeMono(2);
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(curFrame->mnId)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(globalOptimizer1.vertex(id)));
                    e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                    e->setInformation(Eigen::Matrix2d::Identity() * thirdInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    globalOptimizer1.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
            }
        }
        fout1 << "global optimize begin3" <<endl;
        
        globalOptimizer1.initializeOptimization();
        globalOptimizer1.optimize(15);

        unordered_set<double> havePushBack;
        for(auto itt = mGlobalFrames.begin(); itt != mGlobalFrames.end(); ++itt){
            Frame *pKFi = itt->second;
            ov_core::Pose_and_calib pose = pKFi->GetPose();
            ov_core::Pose_and_calib oldpose = pKFi->GetPose();
            VertexPose* VP = static_cast<VertexPose*>(globalOptimizer1.vertex(pKFi->mnId));
            pose.R = pose.R_calib0.transpose() * VP->estimate().Rcw[0];
            pose.t = VP->estimate().Rcw[0].transpose()*pose.t_calib0 - VP->estimate().Rcw[0].transpose()*VP->estimate().tcw[0];
            // pKFi->SetPose(pose);
            // pose.R===Rbw,  pose.t===twb
            Eigen::Matrix3d Rcw = pose.R_calib0 * pose.R;
            Eigen::Vector3d twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
            Eigen::Vector3d tcw = - Rcw * twc0;
            g2o::SE3Quat Sjw(Rcw,tcw);
            // 
            Rcw = oldpose.R_calib0 * oldpose.R;
            twc0 = oldpose.t - Rcw.transpose() * oldpose.t_calib0;
            tcw = - Rcw * twc0;
            g2o::SE3Quat Siw(Rcw,tcw);
            g2o::SE3Quat Swi = Siw.inverse();
            g2o::SE3Quat Sji = Sjw * Swi;

            for(int i = 0; i < stepGlobalFrames1[pKFi->mTimeStamp].size(); ++i){
                if(!havePushBack.count(stepGlobalFrames1[pKFi->mTimeStamp][i]->mTimeStamp)){
                    havePushBack.insert(stepGlobalFrames1[pKFi->mTimeStamp][i]->mTimeStamp);

                    pose = stepGlobalFrames1[pKFi->mTimeStamp][i]->GetPose();
                    Rcw = pose.R_calib0 * pose.R;
                    twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
                    tcw = - Rcw * twc0;
                    g2o::SE3Quat S_iw(Rcw,tcw);
                    g2o::SE3Quat S_jw = Sji * S_iw;
                    Eigen::Matrix3d eigR = S_jw.rotation().toRotationMatrix();
                    Eigen::Vector3d eigt = S_jw.translation();
                    // 这里分别是Rcw，tcw
                    pose.R = pose.R_calib0.transpose() * eigR;
                    pose.t = eigR.transpose()*pose.t_calib0 - eigR.transpose()*eigt;
                    stepGlobalFrames1[pKFi->mTimeStamp][i]->SetPose(pose);
                }
            }

        }

        for(auto it = mGlobalPoints.begin(); it != mGlobalPoints.end(); ++it){
            KeyPoint *pMP = it->second;
            if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
            VertexSBAPointXYZ* vPoint = static_cast<VertexSBAPointXYZ*>(globalOptimizer1.vertex(pMP->mnId+iniMPid+1));
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
        }
        
        globalOptimizer1.clear();
        fout1<< " t1_global:" << t_global.toc() ;
        fout1 <<endl;
        fout1.close();
    }
}


  /// @brief 获取特征点
  void get_feat_from_vio(){
    while(1){
        if(esi_has_end){
         ///这时前端已结束，可以将最后quePose中剩下的位姿和对应特征点都取出来做最后一次优化，不过前端结束了也可以选择不做
            break;
        }
        double marg_time = cur_marg_time;
        if(quePose.size() < 5 || quePose.front().timestamp > marg_time){
            std::chrono::milliseconds tSleep(2);
            std::this_thread::sleep_for(tSleep);
            continue;
        }

        ov_core::Pose_and_calib pose;
        quePoseMutex.lock();
        pose = quePose.front();
        quePose.pop();
        quePoseMutex.unlock();

        double timestamp = pose.timestamp;
        framesMutex.lock();
        Frame *pkf = sysFrames.find(timestamp)->second;
        framesMutex.unlock();

        if(sysKeyFrames.size() > 0){
            auto frameIt = sysKeyFrames.rbegin();
            pkf->mPrevKF = frameIt->second;
        }

        wait_keyframes_nums ++;
        if(sysKeyFrames.size() == 0){
            sysKeyFrames[pose.timestamp] = pkf;
            continue;
        }

        if(pkf->GetFeatures(0).size() < 5) continue;

        int localobs = 150;
        if(sysKeyFrames.size() > 10){
            unordered_map<size_t, KeyPoint*> lLocalPoints;
            TicToc timeForObs;
            auto ii = sysKeyFrames.rbegin();
            for(int k = 0; k < 10; ++k, ++ii){
                unordered_map<size_t, std::vector<double>> feats = ii->second->GetFeatures(0);
                unordered_map<size_t, std::vector<double>> feats1 = ii->second->GetFeatures(1);
                unordered_map<size_t, std::vector<double>> feats2 = ii->second->GetFeatures(2);

                for(auto tmpit = feats.begin(); tmpit != feats.end(); ++tmpit){
                    if(sysKeyPoints.find(tmpit->first) != sysKeyPoints.end())
                        lLocalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
                for(auto tmpit = feats1.begin(); tmpit != feats1.end(); ++tmpit){
                    if(sysKeyPoints.find(tmpit->first) != sysKeyPoints.end())
                        lLocalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
                for(auto tmpit = feats2.begin(); tmpit != feats2.end(); ++tmpit){
                    if(sysKeyPoints.find(tmpit->first) != sysKeyPoints.end())
                        lLocalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
                }
            }

            unordered_map<size_t, std::vector<double>> feats = pkf->GetFeatures(0);
            unordered_map<size_t, std::vector<double>> feats1 = pkf->GetFeatures(1);
            unordered_map<size_t, std::vector<double>> feats2 = pkf->GetFeatures(2);
            
            for(auto tmpit = feats.begin(); tmpit != feats.end(); ++tmpit){
                auto to = lLocalPoints.find(tmpit->first);
                if(to != lLocalPoints.end()){
                    ++localobs;
                    to = lLocalPoints.erase(to);
                }
            }
            for(auto tmpit = feats1.begin(); tmpit != feats1.end(); ++tmpit){
                auto to = lLocalPoints.find(tmpit->first);
                if(to != lLocalPoints.end()){
                    ++localobs;
                    to = lLocalPoints.erase(to);
                }
            }
            for(auto tmpit = feats2.begin(); tmpit != feats2.end(); ++tmpit){
                if(lLocalPoints.find(tmpit->first) != lLocalPoints.end()) ++localobs;
            }
        }
        if(localobs < 120) wait_keyframes_nums = 10;
        if(wait_keyframes_nums >= 5){
            framesMutex.lock();
            ov_core::Pose_and_calib pose_last = sysKeyFrames.rbegin()->second->GetPose();
            framesMutex.unlock();
            double trans = sqrt(pow(pose_last.t(0)-pose.t(0),2) + pow(pose_last.t(1)-pose.t(1),2));

            if(sysKeyFrames.size()<=10 && trans >= 0.1  || trans >= 0.1 && localobs > 150 || wait_keyframes_nums>9 ){
                if(localobs > 150 && last_localobs > 150) wait_keyframes_nums = 0;
                else wait_keyframes_nums = 10;
                last_localobs = localobs;

                framesMutex.lock();
                sysKeyFrames[pose.timestamp] = pkf;
                framesMutex.unlock();

                // IMU preintegration
                vector<IMU::Point> vImuMeas;
                m_buf.lock();
                if(!imu_buf.empty())
                {
                    // Load imu measurements from buffer
                    vImuMeas.clear();
                    while(!imu_buf.empty() && imu_buf.front().timestamp <= timestamp)
                    {
                        double t = imu_buf.front().timestamp;
                        if(t <= pkf->mPrevKF->mTimeStamp){
                            imu_buf.pop();
                            continue;
                        }
                        cv::Point3f acc(imu_buf.front().am(0), imu_buf.front().am(1), imu_buf.front().am(2));
                        cv::Point3f gyr(imu_buf.front().wm(0), imu_buf.front().wm(1), imu_buf.front().wm(2));
                        vImuMeas.push_back(IMU::Point(acc,gyr,t));
                        imu_buf.pop();
                    }
                }
                m_buf.unlock();

                for(int i_imu = 0; i_imu < vImuMeas.size(); ++i_imu){
                    mlQueueImuData.push_back(vImuMeas[i_imu]);
                }
                PreintegrateIMU(*pkf);

                ++localBAflag;
                if(localBAflag >= (10 - overlapCnt)){
                    localBAflag = 0;
                    if(sysKeyFrames.size() > 20){
                        localBA_new(10);
                        // ++globalBAflag;
                        // if(globalBAflag >= 3){
                        //     globalBAflag = 0;
                        //     if(stepGlobalFrames.size() > 5){
                        //         wait_condition_variable.notify_one();
                        //         // ++globalBAflag1;
                        //         // if(globalBAflag1 >= 3){
                        //         //     globalBAflag1 = 0;
                        //         //     if(stepGlobalFrames1.size() > 5){
                        //         //         wait_condition_variable1.notify_one();
                        //         //     }
                        //         // }
                        //     }
                        // }
                    }
                }
            }
        }
    }
};


  void feed_imu(const ov_core::ImuData &message) {
    m_buf.lock();
    imu_buf.push(message);
    m_buf.unlock();
  }

void essentialOptimize(map<double,Frame *> &lLocalFrames, unordered_map<size_t, KeyPoint*> &lLocalPoints, const unordered_set<size_t> &NoToOptimize)
{
    // step1. 设置变量
    const Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
    // 未经过优化的keyframe的pose
    map<int, g2o::SE3Quat> vScw;
    // 经过优化的keyframe的pose
    map<int, g2o::SE3Quat> vCorrectedSwc;
    // step2. 将地图中所有keyframe的pose作为顶点添加到优化器
    int fixCount = 0;
    int startIndex = lLocalFrames.rbegin()->second->mnId;
    for(auto frameIter1 = lLocalFrames.begin(); frameIter1 != lLocalFrames.end(); ++frameIter1)
    {
        Frame* pKFi = frameIter1->second;
        g2o::VertexSE3Expmap* VSim3 = new g2o::VertexSE3Expmap();
        const int nIDi = pKFi->mnId;
        ov_core::Pose_and_calib pose = pKFi->GetPose();
        // ov_core::Pose_and_calib pose = pKFi->old_pose;
        Eigen::Matrix3d Rcw = pose.R_calib0 * pose.R;
        Eigen::Vector3d twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
        Eigen::Vector3d tcw = - Rcw * twc0;
        g2o::SE3Quat Siw(Rcw,tcw);
        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);
        // 这里需要加上一个，就是重叠的帧就不进行优化
        // if(fixCount < overlapCnt) VSim3->setFixed(true);
        // ++fixCount;
        if(pKFi->has_optimize) VSim3->setFixed(true);

        VSim3->setId(nIDi);
        essentialGraphOptimizer.addVertex(VSim3);

        bool bFixed = false;
    }
    
    // step3. 加上边，这里把所有的顶点之间的边都加进去
    for(auto frameIter1 = lLocalFrames.begin(); frameIter1 != lLocalFrames.end(); ++frameIter1)
    {
        Frame* pKFi = frameIter1->second;
        if(pKFi->has_optimize){
            const int nIDi = pKFi->mnId;

            ov_core::Pose_and_calib pose = pKFi->old_pose;
            Eigen::Matrix3d Rcw = pose.R_calib0 * pose.R;
            Eigen::Vector3d twc0 = pose.t - Rcw.transpose() * pose.t_calib0;
            Eigen::Vector3d tcw = - Rcw * twc0;
            g2o::SE3Quat Siw(Rcw,tcw);
            // g2o::SE3Quat Siw = vScw[nIDi];

            g2o::SE3Quat Swi = Siw.inverse();
            for(auto it = lLocalFrames.begin(); it != lLocalFrames.end(); ++it){
                Frame* pKF = it->second;
                if(!pKF->has_optimize){
                    g2o::SE3Quat Slw = vScw[pKF->mnId];

                    g2o::SE3Quat Sli = Slw * Swi;
                    g2o::EdgeSE3Expmap* el = new g2o::EdgeSE3Expmap();

                    el->setInformation(g2o::EdgeSE3Expmap::InformationType::Identity());
                    el->setMeasurement(Sli);
                    el->vertices()[0] = essentialGraphOptimizer.vertex(nIDi);
                    el->vertices()[1] = essentialGraphOptimizer.vertex(pKF->mnId);
                    essentialGraphOptimizer.addEdge(el);

                    el->computeError();
                    Eigen::Matrix<double, 6, 1> error = el->error();
                    cout << error.transpose() <<endl;
                }
            }
        }
    }
    // step4. 开始优化
    essentialGraphOptimizer.initializeOptimization();
    essentialGraphOptimizer.optimize(20);
    // step5. 设定优化后的位姿
    for(auto frameIter1 = lLocalFrames.begin(); frameIter1 != lLocalFrames.end(); ++frameIter1)
    {
        Frame* pKFi = frameIter1->second;
        const int nIDi = pKFi->mnId;

        g2o::VertexSE3Expmap* VSim3 = static_cast<g2o::VertexSE3Expmap*>(essentialGraphOptimizer.vertex(nIDi));
        g2o::SE3Quat CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        // 这里分别是Rcw，tcw

        ov_core::Pose_and_calib pose = pKFi->GetPose();
        pose.R = pose.R_calib0.transpose() * eigR;
        pose.t = eigR.transpose()*pose.t_calib0 - eigR.transpose()*eigt;
        pKFi->SetPose(pose);
    }
    // step6. 优化得到关键帧的位姿后，MapPoints根据参考帧优化前后的相对关系调整自己的位置
    for(auto it = lLocalPoints.begin(); it != lLocalPoints.end(); ++it){
        KeyPoint *pMP = it->second;
        if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
        unordered_set<double> observes = it->second->GetObservation();
        if(lastOptPoints.find(pMP->mnId) == lastOptPoints.end()){
            unordered_set<double> observes = it->second->GetObservation();
            bool flag = true;
            for(auto frameIter1 = lLocalFrames.begin(); frameIter1 != lLocalFrames.end(); ++frameIter1)
            {
                Frame* pKFi = frameIter1->second;
                if(pKFi->has_optimize) continue;
                const int nIDr = pKFi->mnId;
                if(observes.find(pKFi->mTimeStamp) == observes.end()) continue;

                // 得到MapPoint参考关键帧优化前的位姿
                g2o::SE3Quat Srw = vScw[nIDr];
                // 得到MapPoint参考关键帧优化后的位姿
                g2o::SE3Quat correctedSwr = vCorrectedSwc[nIDr];

                Eigen::Vector3d p3d = pMP->GetWorldPos().cast<double>();
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(p3d));
                pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                flag = false;
                break;
            }
            if(flag && !pMP->has_optimize){
                Frame* pKFi = lLocalFrames.rbegin()->second;
                const int nIDr = pKFi->mnId;

                // 得到MapPoint参考关键帧优化前的位姿
                g2o::SE3Quat Srw = vScw[nIDr];
                // 得到MapPoint参考关键帧优化后的位姿
                g2o::SE3Quat correctedSwr = vCorrectedSwc[nIDr];

                Eigen::Vector3d p3d = pMP->GetWorldPos().cast<double>();
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(p3d));
                pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());
            }
        }
        
    }
    essentialGraphOptimizer.clear();
}

void localBA_new(int windowNum)
{
    TicToc t_local;
    map<double,Frame *> lLocalFrames;
    unordered_map<double,Frame *> lFixedFrames;
    unordered_map<size_t, KeyPoint*> lLocalPoints;
    unordered_map<size_t, KeyPoint*> lLocalPointsToOpt;
    vector<EdgeMono*> vpEdgesMono;
    vector<EdgeStereo*> vpEdgesStereo;
    vector<pair<Frame *, KeyPoint *>> vToConsider, vToConsiderStereo;
    unordered_set<size_t> NoToOptimize;
    double monoInfo = 1.0 / 1.5; 
    double stereoInfo = 1.0 / 1.6;
    double thirdInfo = 0.5;

    if(sysKeyFrames.size() < windowNum+1) return;
    framesMutex.lock();
    auto frameIter = sysKeyFrames.end();
    --frameIter;
    int startIndex = frameIter->second->mnId;
    for(int i = 0; i < windowNum; ++i, --frameIter){
        Frame *pf = frameIter->second;
        lLocalFrames[frameIter->first] = pf;
    }
    framesMutex.unlock();

    int nums = 0;
    int framescnt = 0;
    Frame* midFrame = nullptr;
    for(auto it = lLocalFrames.begin();it != lLocalFrames.end(); ++it){
        Frame *pf = it->second;
        if(framescnt == 4) midFrame = pf;
        framescnt++;
        unordered_map<size_t, std::vector<double>> feats = pf->GetFeatures(0);
        unordered_map<size_t, std::vector<double>> feats1 = pf->GetFeatures(1);
        unordered_map<size_t, std::vector<double>> feats2 = pf->GetFeatures(2);
        nums += feats.size();
        nums += feats2.size();
        for(auto tmpit = feats.begin(); tmpit != feats.end(); ++tmpit){
            lLocalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
        }
        for(auto tmpit = feats1.begin(); tmpit != feats1.end(); ++tmpit){
            lLocalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
        }
        for(auto tmpit = feats2.begin(); tmpit != feats2.end(); ++tmpit){
            lLocalPoints[tmpit->first] = sysKeyPoints.find(tmpit->first)->second;
        }
    }
    cout << "1111111111111"<<endl;

    // std::string filename22 = "/home/lonely/workspace/catkin_opvins_muitiple/src/open_vins-develop_v2.5/ov_msckf/traj/localpoints.txt";
    // stringstream ss22;
    // ss22 << filename22;
    // ofstream fout22(ss22.str(), ios::app);
    // fout22.setf(ios::fixed, ios::floatfield);
    // fout22.precision(5);
    // fout22 << lLocalPoints.size() <<  "------------------"<<endl;
    for(auto it = lLocalPoints.begin(); it != lLocalPoints.end(); ++it){
        KeyPoint *pMP = it->second;

        int obs_num = 0, obs_w_num = 0;
        unordered_set<double> observes = it->second->GetObservation();
        if(observes.size() != 0){
            for(auto it1 = observes.begin(); it1 != observes.end(); ++it1){
                auto iit = sysKeyFrames.find(*it1);
                if(iit != sysKeyFrames.end()) ++obs_num;
                if(lLocalFrames.find(*it1) != lLocalFrames.end()) ++obs_w_num;
                
            }
        }
        if(obs_num <= 1 || obs_w_num < 1) NoToOptimize.insert(pMP->mnId);
    }

    
   // --------------essential graph------------------------
    if(has_optimized){
        essentialOptimize(lLocalFrames, lLocalPoints, NoToOptimize);
    }
    // --------------essential graph  end------------------------
    
    for(auto frameIter = lLocalFrames.begin(); frameIter != lLocalFrames.end(); ++frameIter)
    {
        Frame *pf = frameIter->second;
        VertexPose * VP = new VertexPose(pf, cam_intins);
        VP->setId(pf->mnId);
        //第一次进行优化的第一帧固定
        if(!has_optimized && frameIter == lLocalFrames.begin()) VP->setFixed(true);

        localOptimizer.addVertex(VP);
        bool bFixed = true;

        if(pf->mbImuPreintegrated)
        {
            VertexVelocity* VV = new VertexVelocity(pf);
            VV->setId(startIndex+3*(pf->mnId)+1);
            VV->setFixed(bFixed);
            localOptimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pf);
            VG->setId(startIndex+3*(pf->mnId)+2);
            VG->setFixed(bFixed);
            localOptimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pf);
            VA->setId(startIndex+3*(pf->mnId)+3);
            VA->setFixed(bFixed);
            localOptimizer.addVertex(VA);   
        }
    }
    
    // IMU links
    for(auto frameIter1 = lLocalFrames.begin(); frameIter1 != lLocalFrames.end(); ++frameIter1)
    {
        Frame* pKFi = frameIter1->second;
        if(!pKFi->mPrevKF) continue;

        if(pKFi->mPrevKF && pKFi->mnId<=startIndex)
        {
            if(pKFi->mPrevKF->mnId>startIndex)
                continue;
            if(pKFi->mbImuPreintegrated && pKFi->mPrevKF->mbImuPreintegrated)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                // pKFi->mpImuPreintegrated->SetNewBias(pKFi->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = localOptimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = localOptimizer.vertex(startIndex+3*(pKFi->mPrevKF->mnId)+1);

                g2o::HyperGraph::Vertex* VG1;
                g2o::HyperGraph::Vertex* VA1;
                g2o::HyperGraph::Vertex* VG2;
                g2o::HyperGraph::Vertex* VA2;

                VG1 = localOptimizer.vertex(startIndex+3*(pKFi->mPrevKF->mnId)+2);
                VA1 = localOptimizer.vertex(startIndex+3*(pKFi->mPrevKF->mnId)+3);
                VG2 = localOptimizer.vertex(startIndex+3*(pKFi->mnId)+2);
                VA2 = localOptimizer.vertex(startIndex+3*(pKFi->mnId)+3);

                g2o::HyperGraph::Vertex* VP2 =  localOptimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = localOptimizer.vertex(startIndex+3*(pKFi->mnId)+1);

                if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                {
                    cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                    continue;
                }
                
                EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                rki->setDelta(sqrt(16.92));

                localOptimizer.addEdge(ei);
            }
            else
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
        }
    }
    
    
    const unsigned long iniMPid = startIndex * 4 + 5;
    int obsnums = 0, obss = 0;
    vector<pair<size_t, double>> obssv;
    for(auto it = lLocalPoints.begin(); it != lLocalPoints.end(); ++it){
        KeyPoint *pMP = it->second;
        VertexSBAPointXYZ* vPoint = new VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        size_t id = pMP->mnId +iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        unordered_set<double> observes = it->second->GetObservation();

        if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
        localOptimizer.addVertex(vPoint);

        for(auto it1 = observes.begin(); it1 != observes.end(); ++it1){
            if(lLocalFrames.find(*it1) == lLocalFrames.end() && lFixedFrames.find(*it1) == lFixedFrames.end()){
                auto iit = sysKeyFrames.find(*it1);
                if(iit == sysKeyFrames.end()) continue;
                Frame *fixPf = iit->second;
                lFixedFrames[*it1] = fixPf;
                VertexPose * VP = new VertexPose(fixPf, cam_intins);
                VP->setId(fixPf->mnId);
                VP->setFixed(true);
                localOptimizer.addVertex(VP);
            }
            ++obsnums;
            Frame *curFrame;
            if(lLocalFrames.find(*it1) != lLocalFrames.end()){
                curFrame = lLocalFrames.find(*it1)->second;
            }else{
                curFrame = lFixedFrames.find(*it1)->second;
            }
            unordered_map<size_t, std::vector<double>> feats = curFrame->GetFeatures(0);
            unordered_map<size_t, std::vector<double>> feats1 = curFrame->GetFeatures(1);
            unordered_map<size_t, std::vector<double>> feats2 = curFrame->GetFeatures(2);

            auto fit = feats.find(pMP->mnId);
            auto fit1 = feats1.find(pMP->mnId);
            auto fit2 = feats2.find(pMP->mnId);
            if(fit != feats.end() && fit1 != feats1.end()){
                Eigen::Matrix<double, 3, 1> obs;
                EdgeStereo *e = new EdgeStereo(0);
                obs << fit->second[0],fit->second[1],fit1->second[0];

                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(curFrame->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(id)));
                e->setMeasurement(obs);

                e->setInformation(Eigen::Matrix3d::Identity() * stereoInfo);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberStereo);
                localOptimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vToConsiderStereo.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});

                if(fit2 != feats2.end()){
                    EdgeMono* e1 = new EdgeMono(2);
                    e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(curFrame->mnId)));
                    e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(id)));
                    e1->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                    e1->setInformation(Eigen::Matrix2d::Identity()*thirdInfo);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e1->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    localOptimizer.addEdge(e1);
                    vpEdgesMono.push_back(e1);
                    vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
                }
                continue;
            }
            

            if(fit != feats.end()){
                EdgeMono* e = new EdgeMono(0);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(curFrame->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(id)));
                e->setMeasurement(Eigen::Vector2d(fit->second[0],fit->second[1]));
                
                e->setInformation(Eigen::Matrix2d::Identity() * monoInfo);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);
                localOptimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
            }
            if(fit1 != feats1.end()){
                EdgeMono* e = new EdgeMono(1);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(curFrame->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(id)));
                e->setMeasurement(Eigen::Vector2d(fit1->second[0],fit1->second[1]));

                e->setInformation(Eigen::Matrix2d::Identity() * monoInfo);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);
                localOptimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
            }
            if(fit2 != feats2.end()){
                EdgeMono* e = new EdgeMono(2);
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(curFrame->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(localOptimizer.vertex(id)));
                e->setMeasurement(Eigen::Vector2d(fit2->second[0],fit2->second[1]));

                e->setInformation(Eigen::Matrix2d::Identity() * thirdInfo);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);
                localOptimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vToConsider.push_back(pair<Frame *, KeyPoint *>{curFrame, pMP});
            }
        }
    }
    cout << "cur size: " << vpEdgesMono.size() << " " << lLocalPointsToOpt.size() <<endl;
    localOptimizer.initializeOptimization();
    int res = localOptimizer.optimize(15);
    int count = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>chi2Mono2 || !e->isDepthPositive())
        {
            e->setLevel(1);// 不优化
            ++count;
        }
        e->setRobustKernel(0);// 不使用核函数
    }
    int count1 = 0;
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>chi2Stereo2)
        {
            e->setLevel(1);// 不优化
            ++count1;
        }
        e->setRobustKernel(0);// 不使用核函数

    }
    cout << "e->chi(): " << "/" << count << "/" << vpEdgesMono.size() << " vpEdgesStereo:"<< count1 << "/" << vpEdgesStereo.size() <<endl;
    
    localOptimizer.initializeOptimization();
    localOptimizer.optimize(5);
    
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        if(e->chi2()>chi2Mono2){
            Frame *eraseFrame = vToConsider[i].first;
            KeyPoint *erasePoint = vToConsider[i].second;

            eraseFrame->eraseFeatures(erasePoint->mnId, e->cam_idx);
            unordered_map<size_t, std::vector<double>> feats = eraseFrame->GetFeatures(0);
            unordered_map<size_t, std::vector<double>> feats1 = eraseFrame->GetFeatures(1);
            unordered_map<size_t, std::vector<double>> feats2 = eraseFrame->GetFeatures(2);
            if(feats.find(erasePoint->mnId)==feats.end()&&feats1.find(erasePoint->mnId)==feats1.end()&&feats2.find(erasePoint->mnId)==feats2.end()){
                erasePoint->eraseObservation(eraseFrame->mTimeStamp);
            }
        }
    }
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>chi2Stereo2)
        {
            Frame *eraseFrame = vToConsiderStereo[i].first;
            KeyPoint *erasePoint = vToConsiderStereo[i].second;
            
            eraseFrame->eraseFeatures(erasePoint->mnId, 0);
            eraseFrame->eraseFeatures(erasePoint->mnId, 1);
            unordered_map<size_t, std::vector<double>> feats = eraseFrame->GetFeatures(2);
            if(feats.find(erasePoint->mnId)==feats.end()){
                erasePoint->eraseObservation(eraseFrame->mTimeStamp);
            }
        }
    }
    
    framescnt = 0;
    vector<Frame *> perLocal;
    for(auto itt = lLocalFrames.begin(); itt != lLocalFrames.end(); ++itt){
        Frame *pKFi = itt->second;
        ov_core::Pose_and_calib pose = pKFi->GetPose();
        VertexPose* VP = static_cast<VertexPose*>(localOptimizer.vertex(pKFi->mnId));
        pose.R = pose.R_calib0.transpose() * VP->estimate().Rcw[0];
        pose.t = VP->estimate().Rcw[0].transpose()*pose.t_calib0 - VP->estimate().Rcw[0].transpose()*VP->estimate().tcw[0];
        pKFi->SetPose(pose);

        pKFi->has_optimize = true;
        // do for global-BA
        if(framescnt < 4){
            pKFi->rightPrevKF = midFrame;
        }else if(framescnt > 4){
            pKFi->leftPrevKF = midFrame;
        }
        perLocal.push_back(pKFi);
        framescnt ++;

        // VertexVelocity* VV = static_cast<VertexVelocity*>(localOptimizer.vertex(startIndex+3*(pKFi->mnId)+1));
        // pKFi->SetVelocity(VV->estimate().cast<float>());
        // VertexGyroBias* VG = static_cast<VertexGyroBias*>(localOptimizer.vertex(startIndex+3*(pKFi->mnId)+2));
        // VertexAccBias* VA = static_cast<VertexAccBias*>(localOptimizer.vertex(startIndex+3*(pKFi->mnId)+3));
        // Vector6d vb;
        // vb << VG->estimate(), VA->estimate();
        // IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
        // pKFi->SetNewBias(b);
    }
    wait_mutex.lock();
    stepGlobalFrames[midFrame->mTimeStamp] = perLocal;
    wait_mutex.unlock();

    lastOptPoints.clear();
    for(auto it = lLocalPoints.begin(); it != lLocalPoints.end(); ++it){
        KeyPoint *pMP = it->second;
        if(NoToOptimize.find(pMP->mnId) != NoToOptimize.end()) continue;
        VertexSBAPointXYZ* vPoint = static_cast<VertexSBAPointXYZ*>(localOptimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        lastOptPoints.insert(pMP->mnId);
        pMP->has_optimize = true;
    }
    
    localOptimizer.clear();
    has_optimized = true;
}


void PreintegrateIMU(Frame &mCurrentFrame)
{
    if(!mCurrentFrame.mPrevKF)
    {
        mCurrentFrame.setIntegrated();
        return;
    }
    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        mCurrentFrame.setIntegrated();
        return;
    }
    while(true)
    {
        bool bSleep = false;
        {
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t < mCurrentFrame.mPrevKF->mTimeStamp - 0.001l)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-0.001l)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    // 得到两帧间的imu数据放入mvImuFromLastFrame中,得到后面预积分的处理数据
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }
    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mCurrentFrame.GetImuBias(),mCurrentFrame.mImuCalib);//*mpImuCalib);//mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mPrevKF->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mPrevKF->mTimeStamp;
        }
        else if(i<(n-1))
        {
            
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mPrevKF->mTimeStamp;
        }
        if (!pImuPreintegratedFromLastFrame)
            cout << "pImuPreintegratedFromLastFrame does not exist" << endl;
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);

    }

    // 记录当前预积分的图像帧
    mCurrentFrame.mpImuPreintegrated = pImuPreintegratedFromLastFrame;

    mCurrentFrame.setIntegrated();
    cout << "finish integrate!" <<endl;
}


void set_output(std::string path){
        outputPath = path;
        boost::filesystem::path dir(outputPath.c_str());
        // If it exists, then delete it
        if (!boost::filesystem::exists(outputPath))boost::filesystem::create_directories(dir);
        std::string filename = outputPath + "/final.txt";
        PRINT_INFO("opt: %s\n", filename.c_str());
        if (boost::filesystem::exists(filename)) {
            PRINT_INFO("Output file exists, deleting old file....\n");
            boost::filesystem::remove(filename);
        }
        filename = outputPath + "/f_time.txt";
        if (boost::filesystem::exists(filename)) {
            PRINT_INFO("Output file exists, deleting old file....\n");
            boost::filesystem::remove(filename);
        }
        filename = outputPath + "/feats_info.txt";
        if (boost::filesystem::exists(filename)) {
            PRINT_INFO("Output file exists, deleting old file....\n");
            boost::filesystem::remove(filename);
        }
    }

void esi_end(){
    // print key frame pose
    std::string filename = outputPath + "/final.txt";
    stringstream ss1;
    ss1 << filename;
    ofstream fout1(ss1.str(), ios::app);
    framesMutex.lock();
    for(auto out_it = sysKeyFrames.begin(); out_it != sysKeyFrames.end(); out_it++){
        ov_core::Pose_and_calib pose_out = out_it->second->GetPose();
        Eigen::Matrix<double, 4, 1> quat = ov_core::rot_2_quat(pose_out.R);// R需要transpose吗
        fout1.setf(ios::fixed, ios::floatfield);
        fout1.precision(5);
        fout1 << out_it->second->mTimeStamp << " ";
        fout1.precision(6);
        fout1<< pose_out.t(0) << " "
            << pose_out.t(1) << " "
            << pose_out.t(2) << " "
            << quat(0) << " "
            << quat(1) << " "
            << quat(2) << " "
            << quat(3) <<endl;
    }
    fout1.close();
    framesMutex.unlock();
    std::cout << "----" << sysKeyFrames.size() <<std::endl;
    // print feat info
    filename = outputPath + "/feats_info.txt";
    std::ofstream outfile;
    outfile.open(filename,std::ios::app);
    outfile<< "#timestamp(s) cam0 cam1 cam2 mono stereo observe" << std::endl;
    outfile.precision(5);
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    for(auto it = feat_map.begin();it!=feat_map.end();it++){
        outfile<<it->first<<" "<<it->second.cam0<<" "<<it->second.cam1<<" "<<it->second.cam2<<" "<<it->second.mono<<" "
            <<it->second.stereo<<" "<<it->second.observe<<std::endl;
    }
    outfile.close();
    // front end has end
    esi_has_end = true;
}

 std::shared_ptr<FeatureDatabase> get_feature_database() {return feat_database;}

public:
  /// front_end has end or not
  bool esi_has_end = false;
  /// feat_information
  std::shared_ptr<ov_core::FeatureDatabase> feat_database;
  std::map<double,ov_core::Feat_in_Cam> feat_map;
  std::vector<size_t> feats_longtrack;
  /// output path to print information
  std::string outputPath;

  // by lonely
  int max_cameras = 2;
  // 优化重叠个数
  int overlapCnt = 3;
  bool has_optimized = false;
  int last_localobs = 150;

  int max_clone = 0;
  queue<ov_core::Pose_and_calib> quePose;
  mutex quePoseMutex;
  map<double, Frame *> sysFrames;
  map<double, Frame *> sysKeyFrames, sysDKeyFrames;
  unordered_map<size_t, KeyPoint *> sysKeyPoints;
  size_t FrameId = 0;

  double ACC_N, ACC_W;
  double GYR_N, GYR_W;
  double mImuFreq = 200;
  Eigen::Vector3d G{0.0, 0.0, 9.81};
  float mb;

  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cam_intins;
  mutex framesMutex;
  mutex pointsMutex;
  queue<int> queGlobalFlag;
  mutex keyframeMutex;
  mutex globalMutex;
  queue<ov_core::ImuData> imu_buf;
  mutex m_buf;
  std::list<IMU::Point> mlQueueImuData;
  std::vector<IMU::Point> mvImuFromLastFrame;

  const float thHuberMono = sqrt(5.991);
  const float chi2Mono2 = 5.991;
  const float thHuberStereo = sqrt(7.815);
  const float chi2Stereo2 = 7.815;
  g2o::SparseOptimizer localOptimizer, globalOptimizer, globalOptimizer1, essentialGraphOptimizer;
  int num_fial = 0;
  int wait_keyframes_nums = 0;
  int globalBAflag = 0, localBAflag = 0, globalBAflag1 = 0;
  map<double, vector<Frame *>> stepGlobalFrames, stepGlobalFrames1;
  mutex wait_mutex, wait_mutex1;
  condition_variable wait_condition_variable, wait_condition_variable1;

  unordered_set<size_t> lastOptPoints;
  double cur_marg_time = 0.0;
};

} // namespace ov_msckf

#endif // OV_MSCKF_OPTIMIZER_H