#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <string>

#include "calc/ResultTrajectory.h"
#include "alignment/AlignUtils.h"
#include "alignment/AlignTrajectory.h"
#include "utils/colors.h"
#include "utils/print.h"

#ifdef HAVE_PYTHONLIBS

// import the c++ wrapper for matplot lib
// https://github.com/lava/matplotlib-cpp
// sudo apt-get install python-matplotlib python-numpy python2.7-dev
#include "plot/matplotlibcpp.h"

// Will plot the xy 3d position of the pose trajectories
void plot_xy_positions(const std::string &name, const std::string &color, const std::vector<Eigen::Matrix<double, 7, 1>> &poses) {

  // Paramters for our line
  std::map<std::string, std::string> params;
  params.insert({"label", name});
  params.insert({"linestyle", "-"});
  params.insert({"color", color});

  // Create vectors of our x and y axis
  std::vector<double> x, y;
  for (size_t i = 0; i < poses.size(); i++) {
    x.push_back(poses.at(i)(0));
    y.push_back(poses.at(i)(1));
  }

  // Finally plot
  matplotlibcpp::plot(x, y, params);
}

// Will plot the z 3d position of the pose trajectories
void plot_z_positions(const std::string &name, const std::string &color, const std::vector<double> &times,
                      const std::vector<Eigen::Matrix<double, 7, 1>> &poses) {

  // Paramters for our line
  std::map<std::string, std::string> params;
  params.insert({"label", name});
  params.insert({"linestyle", "-"});
  params.insert({"color", color});

  // Create vectors of our x and y axis
  std::vector<double> time, z;
  for (size_t i = 0; i < poses.size(); i++) {
    time.push_back(times.at(i));
    z.push_back(poses.at(i)(2));
  }

  // Finally plot
  matplotlibcpp::plot(time, z, params);
}

#endif



int main(int argc, char **argv) {

  // Verbosity setting
  ov_core::Printer::setPrintLevel("INFO");

  // Ensure we have a path
  if (argc < 4) {
    PRINT_ERROR(RED "ERROR: Please specify a align mode, groudtruth, and algorithm run file\n" RESET);
    PRINT_ERROR(RED "ERROR: ./rpe_estimate <file_gt.txt> <file_est.txt> alignmode\n" RESET);
    PRINT_ERROR(RED "ERROR: rosrun ov_eval rpe_estimate <file_gt.txt> <file_est.txt> alignmode\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Load it!
  boost::filesystem::path path_gt(argv[1]);
  boost::filesystem::path path(argv[2]);
  std::string alignment_method = argv[3];
  std::vector<double> time_gt,time_est;
  std::vector<Eigen::Matrix<double, 7, 1>> pose_gt,pose_est;
  std::vector<std::string> names;
  std::vector<std::vector<double>> times;
  std::vector<std::vector<Eigen::Matrix<double, 7, 1>>> poses;
  std::vector<Eigen::Matrix3d> cov_ori, cov_pos,gt_cov_ori,gt_cov_pos;
  ov_eval::Loader::load_data(argv[1], time_gt, pose_gt, gt_cov_ori, gt_cov_pos);
  ov_eval::Loader::load_data(argv[2], time_est, pose_est, cov_ori, cov_pos);
  // Print its length and stats and save ori pose and time for plot
  double length = ov_eval::Loader::get_total_length(pose_gt);
  PRINT_INFO("[COMPGT]: %d poses in %s => length of %.2f meters\n", (int)time_gt.size(), path_gt.stem().string().c_str(), length);
  length = ov_eval::Loader::get_total_length(pose_est);
  PRINT_INFO("[COMPES]: %d poses in %s => length of %.2f meters\n", (int)time_est.size(), path.stem().string().c_str(), length);
  names.push_back((path_gt.stem().string()+"_gt").c_str());
  names.push_back(path.stem().string().c_str());
  times.push_back(time_gt);
  times.push_back(time_est);
  poses.push_back(pose_gt);
  poses.push_back(pose_est);
  // Intersect timestamps
  // gt timestamps to est timestamps
  ov_eval::AlignUtils::perform_association(0, 0.02, time_est, time_gt, pose_est, pose_gt, cov_ori, cov_pos, gt_cov_ori, gt_cov_pos);
  //ov_eval::AlignUtils::perform_association(0, 0.02, time_gt, time_est, pose_gt, pose_est, gt_cov_ori, gt_cov_pos, cov_ori, cov_pos);
  assert(pose_gt.size() == pose_est.size());
  assert(time_gt.size() == time_est.size());
  // Create our trajectory object
  std::vector<double> segments = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0};
  for(const double &distance : segments){
    std::vector<std::vector<Eigen::Matrix<double, 7, 1>>> gtpose_seg, esipose_seg;
    std::vector<std::vector<double>> gttime_seg, esitime_seg;
    std::vector<Eigen::Matrix<double, 7, 1>> est_poses_aignedtoGT_seg;
    std::vector<std::vector<double>> time_align;
    std::vector<std::vector<Eigen::Matrix<double, 7, 1>>> pose_align;
    //seg by length
    ov_eval::Loader::pose_seg_by_length(time_gt, time_est, pose_gt, pose_est, distance, gtpose_seg, esipose_seg, gttime_seg, esitime_seg);
    assert(gtpose_seg.size() == esipose_seg.size());
    assert(gttime_seg.size() == esitime_seg.size());
    assert(gtpose_seg.size() == esitime_seg.size());
    std::vector<ov_eval::ResultTrajectory> trajs;
    for(int i = 0;i < gtpose_seg.size();i++){
      // Perform alignment of the trajectories
      Eigen::Matrix3d R_ESTtoGT, R_GTtoEST;
      Eigen::Vector3d t_ESTinGT, t_GTinEST;
      double s_ESTtoGT, s_GTtoEST;
      ov_eval::AlignTrajectory::align_trajectory(esipose_seg.at(i), gtpose_seg.at(i), R_ESTtoGT, t_ESTinGT, s_ESTtoGT, alignment_method);
      ov_eval::AlignTrajectory::align_trajectory(gtpose_seg.at(i), esipose_seg.at(i), R_GTtoEST, t_GTinEST, s_GTtoEST, alignment_method);
      // Debug print to the user
      Eigen::Vector4d q_ESTtoGT = ov_core::rot_2_quat(R_ESTtoGT);
      Eigen::Vector4d q_GTtoEST = ov_core::rot_2_quat(R_GTtoEST);
      // Finally lets calculate the aligned trajectories
      std::vector<Eigen::Matrix<double, 7, 1>> est_poses_aignedtoGT;
      std::vector<Eigen::Matrix<double, 7, 1>> gt_poses_aignedtoEST;
      for (size_t j = 0; j < gttime_seg.at(i).size(); j++) {
        Eigen::Matrix<double, 7, 1> pose_ESTinGT, pose_GTinEST;
        pose_ESTinGT.block(0, 0, 3, 1) = s_ESTtoGT * R_ESTtoGT * esipose_seg.at(i).at(j).block(0, 0, 3, 1) + t_ESTinGT;
        pose_ESTinGT.block(3, 0, 4, 1) = ov_core::quat_multiply(esipose_seg.at(i).at(j).block(3, 0, 4, 1), ov_core::Inv(q_ESTtoGT));
        pose_GTinEST.block(0, 0, 3, 1) = s_GTtoEST * R_GTtoEST * gtpose_seg.at(i).at(j).block(0, 0, 3, 1) + t_GTinEST;
        pose_GTinEST.block(3, 0, 4, 1) = ov_core::quat_multiply(gtpose_seg.at(i).at(j).block(3, 0, 4, 1), ov_core::Inv(q_GTtoEST));
        est_poses_aignedtoGT.push_back(pose_ESTinGT);
        gt_poses_aignedtoEST.push_back(pose_GTinEST);
      }
      ov_eval::ResultTrajectory traj_part(esitime_seg.at(i), gttime_seg.at(i),
          esipose_seg.at(i), gtpose_seg.at(i), est_poses_aignedtoGT, gt_poses_aignedtoEST);
      trajs.push_back(traj_part);
      est_poses_aignedtoGT_seg.insert(est_poses_aignedtoGT_seg.end(),est_poses_aignedtoGT.begin(),est_poses_aignedtoGT.end());
    }
    time_align.push_back(time_gt);
    time_align.push_back(time_est);
    pose_align.push_back(pose_gt);
    pose_align.push_back(est_poses_aignedtoGT_seg);

    //plot
    #ifdef HAVE_PYTHONLIBS
      std::vector<std::string> colors = {"black", "blue", "red", "green", "cyan", "magenta"};
      // Plot this figure
      matplotlibcpp::figure_size(1000, 750);

      // Plot the position trajectories
      for (size_t m = 0; m < time_align.size(); m++) {
        plot_xy_positions(names.at(m), colors.at(m), pose_align.at(m));
      }

      // Display to the user
      matplotlibcpp::xlabel("x-axis (m)");
      matplotlibcpp::ylabel("y-axis (m)");
      matplotlibcpp::legend();
      matplotlibcpp::title("align" + std::to_string((int)distance));
      matplotlibcpp::show(false);
      matplotlibcpp::save("align" + std::to_string((int)distance) + ".png");
    #endif
    //cal RPE and APE for every traj_part
    double RPE_rmse = 0,APE_rmse = 0;
    int RPE_size = 0,APE_size = 0;
    double RPE_all2 = 0,APE_all2 = 0;
    PRINT_INFO("======================================\n");
    PRINT_INFO("Length %d APE and RPE\n",(int)distance);
    PRINT_INFO("======================================\n");
    int part_num = 0;
    for(auto &part : trajs){
      std::vector<double> segments = {distance};
      std::map<double, std::pair<ov_eval::Statistics, ov_eval::Statistics>> error_rpe;
      part.calculate_rpe_ep(segments, error_rpe);//cal rpe just for two endpoints
      // Calculate
      ov_eval::Statistics error_ori, error_pos;
      part.calculate_ate_2d(error_ori, error_pos);
      // Print it
      auto seg = error_rpe.begin();
      PRINT_INFO("part %d [APE]rmse_e = %.3f - [RPE]ori_e = %.3f | pos_e = %.3f | pos_rpe/seg = %.3f%\n", part_num, error_pos.rmse, 
        seg->second.first.mean,seg->second.second.mean, seg->second.second.mean/seg->first*100);
      APE_all2 += pow(error_pos.rmse,2);
      RPE_all2 += pow(seg->second.second.mean,2);
      APE_size++;
      RPE_size++;
      part_num++;
    }
    PRINT_INFO("\n");
    PRINT_INFO("seg %d [APE]rmse = %.3f - [RPE]rmse = %.3f rmse/seg = %.3f%(%d parts)\n", (int)distance, sqrt(APE_all2/APE_size), 
        sqrt(RPE_all2/RPE_size), sqrt(RPE_all2/RPE_size)/distance*100, part_num);
  }
#ifdef HAVE_PYTHONLIBS

  // Colors that we are plotting
  std::vector<std::string> colors = {"black", "blue", "red", "green", "cyan", "magenta"};
  // assert(algo_rpe.size() <= colors.size()*linestyle.size());

  // Plot this figure
  matplotlibcpp::figure_size(1000, 750);

  // Plot the position trajectories
  for (size_t i = 0; i < times.size(); i++) {
    plot_xy_positions(names.at(i), colors.at(i), poses.at(i));
  }

  // Display to the user
  matplotlibcpp::xlabel("x-axis (m)");
  matplotlibcpp::ylabel("y-axis (m)");
  matplotlibcpp::legend();
  matplotlibcpp::title("ori");
  matplotlibcpp::save("ori.png");
  matplotlibcpp::show(true);
  

#endif

}