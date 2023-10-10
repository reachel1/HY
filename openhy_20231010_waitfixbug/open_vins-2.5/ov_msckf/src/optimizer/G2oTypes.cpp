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

#include "G2oTypes.h"


ImuCamPose::ImuCamPose(Frame *pKF, std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cam_intins)
{
    ov_core::Pose_and_calib pose = pKF->GetPose();
    // Load IMU pose
    twb = pose.t;
    Rwb = pose.R.transpose();

    // Load camera poses
    int num_cams = cam_intins.size();

    tcw.resize(num_cams);
    Rcw.resize(num_cams);
    tcb.resize(num_cams);
    Rcb.resize(num_cams);
    Rbc.resize(num_cams);
    tbc.resize(num_cams);

    // Left camera
    tcb[0] = pose.t_calib0;
    Rcb[0] = pose.R_calib0;
    Rbc[0] = Rcb[0].transpose();
    tbc[0] = -Rbc[0] * tcb[0];
    Rcw[0] = pose.R_calib0 * pose.R;
    Eigen::Vector3d twc0 = pose.t - Rcw[0].transpose() * pose.t_calib0;
    tcw[0] = -Rcw[0] * twc0;
    //cam intrin
    for(auto i = cam_intins.begin();i != cam_intins.end();i++){cam_ins[i->first] = i->second;}
    bf = pKF->mbf;

    if(num_cams>1)
    {
        tcb[1] = pose.t_calib1;
        Rcb[1] = pose.R_calib1;
        Rbc[1] = Rcb[1].transpose();
        tbc[1] = -Rbc[1] * tcb[1];
        Rcw[1] = pose.R_calib1 * pose.R;
        Eigen::Vector3d twc1 = pose.t - Rcw[1].transpose() * pose.t_calib1;
        tcw[1] = -Rcw[1] * twc1;
    }
    if(num_cams>2)
    {
        tcb[2] = pose.t_calib2;
        Rcb[2] = pose.R_calib2;
        Rbc[2] = Rcb[2].transpose();
        tbc[2] = -Rbc[2] * tcb[2];
        Rcw[2] = pose.R_calib2 * pose.R;
        Eigen::Vector3d twc2 = pose.t - Rcw[2].transpose() * pose.t_calib2;
        tcw[2] = -Rcw[2] * twc2;
    }
    // For posegraph 4DoF
    Rwb0 = Rwb;
    DR.setIdentity();
}


void ImuCamPose::SetParam(const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc,
              const std::vector<Eigen::Vector3d> &_tbc, const double &_bf)
{
    Rbc = _Rbc;
    tbc = _tbc;
    Rcw = _Rcw;
    tcw = _tcw;
    const int num_cams = Rbc.size();
    Rcb.resize(num_cams);
    tcb.resize(num_cams);

    for(int i=0; i<tcb.size(); i++)
    {
        Rcb[i] = Rbc[i].transpose();
        tcb[i] = -Rcb[i]*tbc[i];
    }
    Rwb = Rcw[0].transpose()*Rcb[0];
    twb = Rcw[0].transpose()*(tcb[0]-tcw[0]);

    bf = _bf;
}

///-------------------------
// Eigen::Vector2d ImuCamPose::Project(const Eigen::Vector3d &Xw, int cam_idx) const
// {
//     Eigen::Vector3d Xc = Rcw[cam_idx] * Xw + tcw[cam_idx];

//     return pCamera[cam_idx]->project(Xc);
// }

Eigen::Vector2d ImuCamPose::Project(const Eigen::Vector3d &Xw, int cam_idx) const
{
    Eigen::Vector3d Xc = Rcw[cam_idx] * Xw + tcw[cam_idx];

    const Eigen::Vector2f Xnorm(Xc(0)/Xc(2),Xc(1)/Xc(2));

    Eigen::Vector2f p_uv;
    for(auto i = cam_ins.begin();i != cam_ins.end();i++){
        if(i->first==cam_idx)p_uv = i->second->distort_f(Xnorm);
    }
    Eigen::Vector2d uvs(p_uv(0),p_uv(1));
    return uvs;

}

Eigen::Vector3d ImuCamPose::ProjectStereo(const Eigen::Vector3d &Xw, int cam_idx) const
{
    Eigen::Vector3d pc;

    Eigen::Vector3d Xc = Rcw[cam_idx] * Xw + tcw[cam_idx];

    const Eigen::Vector2f Xnorm(Xc(0)/Xc(2),Xc(1)/Xc(2));

    Eigen::Vector2f p_uv;
    for(auto i = cam_ins.begin();i != cam_ins.end();i++){
        if(i->first==cam_idx)p_uv = i->second->distort_f(Xnorm);
    }
    Eigen::Vector2d uvs(p_uv(0),p_uv(1));

    double invZ = 1/Xc(2);
    pc.head(2) = uvs;
    pc(2) = pc(0) - bf*invZ;
    return pc;
}

Eigen::Matrix<double, 2, 3> ImuCamPose::projectJac(const Eigen::Vector3d &v3D,int cam_id) const
{
    double x2 = v3D[0] * v3D[0], y2 = v3D[1] * v3D[1], z2 = v3D[2] * v3D[2];
    double r2 = x2 + y2;
    double r = sqrt(r2);
    double r3 = r2 * r;
    double theta = atan2(r, v3D[2]);

    double theta2 = theta * theta, theta3 = theta2 * theta;
    double theta4 = theta2 * theta2, theta5 = theta4 * theta;
    double theta6 = theta2 * theta4, theta7 = theta6 * theta;
    double theta8 = theta4 * theta4, theta9 = theta8 * theta;
    Eigen::Matrix<double, 8, 1> cam_value;
    for(auto i = cam_ins.begin();i != cam_ins.end();i++){
        if(i->first==cam_id)cam_value = i->second->get_value();
    }
    double f = theta + theta3 * cam_value(4) + theta5 * cam_value(5) + theta7 * cam_value(6) +
                theta9 * cam_value(7);
    double fd = 1 + 3 * cam_value(4) * theta2 + 5 * cam_value(5) * theta4 + 7 * cam_value(6) * theta6 +
                9 * cam_value(7) * theta8;

    Eigen::Matrix<double, 2, 3> JacGood;
    JacGood(0, 0) = cam_value(0) * (fd * v3D[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
    JacGood(1, 0) =
            cam_value(1) * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) - f * v3D[1] * v3D[0] / r3);

    JacGood(0, 1) =
            cam_value(0) * (fd * v3D[2] * v3D[1] * v3D[0] / (r2 * (r2 + z2)) - f * v3D[1] * v3D[0] / r3);
    JacGood(1, 1) = cam_value(1) * (fd * v3D[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

    JacGood(0, 2) = -cam_value(0) * fd * v3D[0] / (r2 + z2);
    JacGood(1, 2) = -cam_value(1) * fd * v3D[1] / (r2 + z2);

    return JacGood;
}


//----------------++
bool ImuCamPose::isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx) const
{
    // return (Rcw[cam_idx].row(2) * Xw + tcw[cam_idx](2)) > 0.0;
    Eigen::Vector3d p_inc = Rcw[cam_idx] *Xw + tcw[cam_idx];
    return p_inc(2)>0.0;
}

void ImuCamPose::Update(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];

    // Update body pose
    twb += Rwb * ut;
    Rwb = Rwb * ExpSO3(ur);

    // Normalize rotation after 5 updates
    its++;
    if(its>=3)
    {
        NormalizeRotation(Rwb);
        its=0;
    }

    // Update camera poses
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw * twb;

    for(int i=0; i<cam_ins.size(); i++)
    {
        Rcw[i] = Rcb[i] * Rbw;
        tcw[i] = Rcb[i] * tbw + tcb[i];
    }

}

void ImuCamPose::UpdateW(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];


    const Eigen::Matrix3d dR = ExpSO3(ur);
    DR = dR * DR;
    Rwb = DR * Rwb0;
    // Update body pose
    twb += ut;

    // Normalize rotation after 5 updates
    its++;
    if(its>=5)
    {
        DR(0,2) = 0.0;
        DR(1,2) = 0.0;
        DR(2,0) = 0.0;
        DR(2,1) = 0.0;
        NormalizeRotation(DR);
        its = 0;
    }

    // Update camera pose
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw * twb;

    for(int i=0; i<cam_ins.size(); i++)
    {
        Rcw[i] = Rcb[i] * Rbw;
        tcw[i] = Rcb[i] * tbw+tcb[i];
    }
}

bool VertexPose::read(std::istream& is)
{  
    return false;
}

bool VertexPose::write(std::ostream& os) const
{
    return false;
}

bool  VertexSBAPointXYZ::read(std::istream& is){return false;}

bool  VertexSBAPointXYZ::write(std::ostream& os) const{return false;}

void EdgeMono::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const VertexSBAPointXYZ* VPoint = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw*VPoint->estimate() + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];

    // const Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().pCamera[cam_idx]->projectJac(Xc);
    const Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().projectJac(Xc, cam_idx);
    _jacobianOplusXi = -proj_jac * Rcw;

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);

    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jac * Rcb * SE3deriv; // TODO optimize this product
}

void EdgeMonoNew::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const VertexSBAPointXYZ* VPoint = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw*VPoint->estimate() + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];
    const Eigen::Vector3d Xc1 = Rc1c * Xc + tc1c;

    const Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().projectJac(Xc1, cam_idx);
    _jacobianOplusXi = -proj_jac * Rc1c * Rcw;

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);

    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jac * Rc1c * Rcb * SE3deriv; // TODO optimize this product
}


void EdgeStereo::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw*VPoint->estimate() + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];
    const double bf = VPose->estimate().bf;
    const double inv_z2 = 1.0/(Xc(2)*Xc(2));

    Eigen::Matrix<double,3,3> proj_jac;
    // proj_jac.block<2,3>(0,0) = VPose->estimate().pCamera[cam_idx]->projectJac(Xc);
    proj_jac.block<2,3>(0,0) = VPose->estimate().projectJac(Xc, cam_idx);
    proj_jac.block<1,3>(2,0) = proj_jac.block<1,3>(0,0);
    proj_jac(2,2) += bf*inv_z2;

    _jacobianOplusXi = -proj_jac * Rcw;

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);

    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jac * Rcb * SE3deriv;
}

void EdgeStereoNew::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw*VPoint->estimate() + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];
    const Eigen::Vector3d Xc1 = Rc1c * Xc + tc1c;

    const double bf = VPose->estimate().bf;
    const double inv_z2 = 1.0/(Xc1(2)*Xc1(2));

    Eigen::Matrix<double,3,3> proj_jac;
    proj_jac.block<2,3>(0,0) = VPose->estimate().projectJac(Xc1, cam_idx);
    proj_jac.block<1,3>(2,0) = proj_jac.block<1,3>(0,0);
    proj_jac(2,2) += bf*inv_z2;

    _jacobianOplusXi = -proj_jac * Rc1c * Rcw;

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);

    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jac * Rc1c * Rcb * SE3deriv;
}

VertexVelocity::VertexVelocity(Frame* pKF)
{
    setEstimate(pKF->GetVelocity().cast<double>());
}

VertexGyroBias::VertexGyroBias(Frame *pKF)
{
    setEstimate(pKF->GetGyroBias().cast<double>());
}

VertexAccBias::VertexAccBias(Frame *pKF)
{
    setEstimate(pKF->GetAccBias().cast<double>());
}

EdgeInertial::EdgeInertial(IMU::Preintegrated *pInt):JRg(pInt->JRg.cast<double>()),
    JVg(pInt->JVg.cast<double>()), JPg(pInt->JPg.cast<double>()), JVa(pInt->JVa.cast<double>()),
    JPa(pInt->JPa.cast<double>()), mpInt(pInt), dt(pInt->dT)
{
    // This edge links 6 vertices
    resize(6);
    g << 0, 0, -IMU::GRAVITY_VALUE;

    Matrix9d Info = pInt->C.block<9,9>(0,0).cast<double>().inverse();
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
    for(int i=0;i<9;i++)
        if(eigs[i]<1e-12)
            eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}


void EdgeInertial::computeError()
{
    // TODO Maybe Reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1).cast<double>();
    const Eigen::Vector3d dV = mpInt->GetDeltaVelocity(b1).cast<double>();
    const Eigen::Vector3d dP = mpInt->GetDeltaPosition(b1).cast<double>();

    const Eigen::Vector3d er = LogSO3(dR.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(VV2->estimate() - VV1->estimate() - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb
                                                               - VV1->estimate()*dt - g*dt*dt/2) - dP;

    _error << er, ev, ep;
}

void EdgeInertial::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2= static_cast<const VertexVelocity*>(_vertices[5]);
    const IMU::Bias b1(VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2],VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2]);
    const IMU::Bias db = mpInt->GetDeltaBias(b1);
    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;

    const Eigen::Matrix3d dR = mpInt->GetDeltaRotation(b1).cast<double>();
    const Eigen::Matrix3d eR = dR.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = LogSO3(eR);
    const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
     // rotation
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1; // OK
    // std::cout << "test for debug1:"<<std::endl;
    // std::cout << g*dt<<std::endl;
    // std::cout << "test for debug2:"<<std::endl;
    // std::cout << VV2->estimate()<<std::endl;
    // std::cout << "test for debug3:"<<std::endl;
    // std::cout << VV1->estimate()<<std::endl;
    // std::cout << "test for debug4:"<<std::endl;
    // std::cout << Rbw1<<std::endl;
    // std::cout << "test for debug5:"<<std::endl;
    // std::cout << VP2->estimate().twb<<std::endl;
    // std::cout << "test for debug6:"<<std::endl;
    // std::cout << VP1->estimate().twb<<std::endl;
    // std::cout << "test for debug7:"<<std::endl;
    // std::cout << dt<<std::endl;
    _jacobianOplus[0].block<3,3>(3,0) = Sophus::SO3d::hat(Rbw1*(VV2->estimate() - VV1->estimate() - g*dt)); // OK
    _jacobianOplus[0].block<3,3>(6,0) = Sophus::SO3d::hat(Rbw1*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt - 0.5*g*dt*dt)); // OK
    // translation
    _jacobianOplus[0].block<3,3>(6,3) = -Eigen::Matrix3d::Identity(); // OK

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -Rbw1; // OK
    _jacobianOplus[1].block<3,3>(6,0) = -Rbw1*dt; // OK

    // Jacobians wrt Gyro 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*RightJacobianSO3(JRg*dbg)*JRg; // OK
    _jacobianOplus[2].block<3,3>(3,0) = -JVg; // OK
    _jacobianOplus[2].block<3,3>(6,0) = -JPg; // OK

    // Jacobians wrt Accelerometer 1
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa; // OK
    _jacobianOplus[3].block<3,3>(6,0) = -JPa; // OK

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3,3>(0,0) = invJr; // OK
    // translation
    _jacobianOplus[4].block<3,3>(6,3) = Rbw1*Rwb2; // OK

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = Rbw1; // OK
}


// SO3 FUNCTIONS
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    return ExpSO3(w[0],w[1],w[2]);
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return NormalizeRotation(res);
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return NormalizeRotation(res);
    }
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1],w[2], 0.0, -w[0],-w[1],  w[0], 0.0;
    return W;
}
