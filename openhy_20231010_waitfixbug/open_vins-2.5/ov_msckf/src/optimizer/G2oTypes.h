/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef G2OTYPES_H
#define G2OTYPES_H

#include  <g2o/core/base_vertex.h>
#include  <g2o/core/base_binary_edge.h>
#include  <g2o/types/sba/types_sba.h>
#include  <g2o/core/base_multi_edge.h>
#include  <g2o/core/base_unary_edge.h>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "Frame.h"

#include "Converter.h"
#include <math.h>
#include <cstdlib>

#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "utils/sensor_data.h"
#include <iostream>

class Frame;
class GeometricCamera;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);

template<typename T = double>
Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) {
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}


class ImuCamPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuCamPose(){}
    ImuCamPose(Frame* pF, std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cam_intins);

    void SetParam(const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc,
                  const std::vector<Eigen::Vector3d> &_tbc, const double &_bf);

    void Update(const double *pu); // update in the imu reference
    void UpdateW(const double *pu); // update in the world reference
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw, int cam_idx=0) const; // Mono
    Eigen::Vector3d ProjectStereo(const Eigen::Vector3d &Xw, int cam_idx=0) const; // Stereo
    Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D,int cam_id) const;
    bool isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx=0) const;

public:
    // For IMU
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    // For set of cameras
    std::vector<Eigen::Matrix3d> Rcw;
    std::vector<Eigen::Vector3d> tcw;
    std::vector<Eigen::Matrix3d> Rcb, Rbc;
    std::vector<Eigen::Vector3d> tcb, tbc;
    double bf;
    std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cam_ins;

    // For posegraph 4DoF
    Eigen::Matrix3d Rwb0;
    Eigen::Matrix3d DR;

    int its;
};

// Optimizable parameters are IMU pose
class VertexPose : public g2o::BaseVertex<6,ImuCamPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){}
    VertexPose(Frame* pF, std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cam_intins){
        setEstimate(ImuCamPose(pF, cam_intins));
    }


    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

class VertexSBAPointXYZ : public g2o::BaseVertex<3,Eigen::Vector3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    VertexSBAPointXYZ(){};
    ~VertexSBAPointXYZ(){};
 
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        Eigen::Vector3d p(0,0,0);
      _estimate = p;
    }

    virtual void oplusImpl(const double* update)
    {
      Eigen::Map<const Vector3d> v(update);
      _estimate += v;
    }
};

class VertexVelocity : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity(){}
    VertexVelocity(Frame* pF);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uv);
    }
};

class VertexGyroBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias(){}
    VertexGyroBias(Frame* pF);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate()+ubg);
    }
};


class VertexAccBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias(){}
    VertexAccBias(Frame* pF);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uba);
    }
};



class EdgeMono : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,VertexSBAPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMono(int cam_idx_): cam_idx(cam_idx_){
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexSBAPointXYZ* VPoint = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(VPoint->estimate(),cam_idx);
    }


    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexSBAPointXYZ* VPoint = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        return VPose->estimate().isDepthPositive(VPoint->estimate(),cam_idx);
    }

    Eigen::Matrix<double,2,9> GetJacobian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

public:
    const int cam_idx;
};


class EdgeMonoNew : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,VertexSBAPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMonoNew(int cam_idx_, Eigen::Matrix3d Rc1c_, Eigen::Vector3d tc1c_): 
    cam_idx(cam_idx_), Rc1c(Rc1c_), tc1c(tc1c_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexSBAPointXYZ* VPoint = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        // _error = obs - VPose->estimate().Project(VPoint->estimate(),cam_idx);
        Eigen::Vector3d pc1 = Rc1c * (VPose->estimate().Rcw[cam_idx] * VPoint->estimate() + VPose->estimate().tcw[cam_idx]) + tc1c;
        const Eigen::Vector2f Xnorm(pc1(0)/pc1(2),pc1(1)/pc1(2));

        Eigen::Vector2f p_uv;
        for(auto i = VPose->estimate().cam_ins.begin();i != VPose->estimate().cam_ins.end();i++){
            if(i->first==cam_idx)p_uv = i->second->distort_f(Xnorm);
        }
        Eigen::Vector2d uvs(p_uv(0),p_uv(1));
        _error = obs - uvs;
    }


    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexSBAPointXYZ* VPoint = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        // return VPose->estimate().isDepthPositive(VPoint->estimate(),cam_idx);
        Eigen::Vector3d pc1 = Rc1c * (VPose->estimate().Rcw[cam_idx] * VPoint->estimate() + VPose->estimate().tcw[cam_idx]) + tc1c;
        return pc1(2) > 0.0;
    }

    Eigen::Matrix<double,2,9> GetJacobian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

public:
    const int cam_idx;
    const Eigen::Matrix3d Rc1c;
    const Eigen::Vector3d tc1c;
};

class EdgeStereo : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,g2o::VertexSBAPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 代码中都是0
    EdgeStereo(int cam_idx_=0): cam_idx(cam_idx_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - VPose->estimate().ProjectStereo(VPoint->estimate(),cam_idx);
    }


    virtual void linearizeOplus();

    Eigen::Matrix<double,3,9> GetJacobian(){
        linearizeOplus();
        Eigen::Matrix<double,3,9> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,6>(0,3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,9> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

public:
    const int cam_idx;
};

class EdgeStereoNew : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,g2o::VertexSBAPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeStereoNew(int cam_idx_, Eigen::Matrix3d Rc1c_, Eigen::Vector3d tc1c_): 
    cam_idx(cam_idx_), Rc1c(Rc1c_), tc1c(tc1c_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector3d obs(_measurement);
        // _error = obs - VPose->estimate().ProjectStereo(VPoint->estimate(),cam_idx);

        Eigen::Vector3d pc;

        Eigen::Vector3d Pc1 = Rc1c * (VPose->estimate().Rcw[cam_idx] * VPoint->estimate() + VPose->estimate().tcw[cam_idx]) + tc1c;
        const Eigen::Vector2f Xnorm(Pc1(0)/Pc1(2),Pc1(1)/Pc1(2));

        Eigen::Vector2f p_uv;
        for(auto i = VPose->estimate().cam_ins.begin();i != VPose->estimate().cam_ins.end();i++){
            if(i->first==cam_idx)p_uv = i->second->distort_f(Xnorm);
        }
        Eigen::Vector2d uvs(p_uv(0),p_uv(1));

        double invZ = 1/Pc1(2);
        pc.head(2) = uvs;
        pc(2) = pc(0) - VPose->estimate().bf*invZ;
        _error = obs - pc;
    }
    


    virtual void linearizeOplus();

    Eigen::Matrix<double,3,9> GetJacobian(){
        linearizeOplus();
        Eigen::Matrix<double,3,9> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,6>(0,3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,9> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

public:
    const int cam_idx;
    const Eigen::Matrix3d Rc1c;
    const Eigen::Vector3d tc1c;
};

class EdgeGPS : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGPS(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - VPose->estimate().twb;
    }


};


/*

class EdgeEssentialSE3 : public BaseBinaryEdge<6, SE3Quat, VertexPose, VertexPose>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeEssentialSE3(){}
//   virtual bool read(std::istream& is);
//   virtual bool write(std::ostream& os) const;

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

  void computeError()
  {
    const VertexPose* v1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexPose* v2 = static_cast<const VertexPose*>(_vertices[1]);

    SE3Quat C(_measurement);
    SE3Quat v1_est(v1->estimate().Rcw[0], v1->estimate().tcw[0]);
    SE3Quat v2_est(v2->estimate().Rcw[0], v2->estimate().tcw[0]);
    SE3Quat error_=C*v1_est*v2_est.inverse();
    _error = error_.log();
  }

  virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& , OptimizableGraph::Vertex* ) { return 1.;}

//   virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* )
//   {
//     VertexPose* v1 = static_cast<VertexPose*>(_vertices[0]);
//     VertexPose* v2 = static_cast<VertexPose*>(_vertices[1]);
//     SE3Quat v1_est(v1->estimate().Rcw[0], v1->estimate().tcw[0]);
//     SE3Quat v2_est(v2->estimate().Rcw[0], v2->estimate().tcw[0]);
//     if (from.count(v1) > 0){
//         v2->setEstimate()
//     }
//         v2->setEstimate(measurement()*v1->estimate());
//     else
//         v1->setEstimate(measurement().inverse()*v2->estimate());
//   }
};
*/

class EdgeInertial : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInertial(IMU::Preintegrated* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double,24,24> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,24> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,18,18> GetHessianNoPose1(){
        linearizeOplus();
        Eigen::Matrix<double,9,18> J;
        J.block<9,3>(0,0) = _jacobianOplus[1];
        J.block<9,3>(0,3) = _jacobianOplus[2];
        J.block<9,3>(0,6) = _jacobianOplus[3];
        J.block<9,6>(0,9) = _jacobianOplus[4];
        J.block<9,3>(0,15) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,9,9> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,6>(0,0) = _jacobianOplus[4];
        J.block<9,3>(0,6) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    IMU::Preintegrated* mpInt;
    const double dt;
    Eigen::Vector3d g;
};


#endif // G2OTYPES_H
