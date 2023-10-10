#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <string>

#include "calc/ResultTrajectory.h"
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
  if (argc < 3) {
    PRINT_ERROR(RED "ERROR: ./aligenape_estimate <file_gt.txt> <file_est.txt>\n" RESET);
    PRINT_ERROR(RED "ERROR: rosrun ov_eval aligenape_estimate <file_gt.txt> <file_est.txt>\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Load it!
  boost::filesystem::path path_gt(argv[1]);
  boost::filesystem::path path(argv[2]);
  std::vector<double> timegt,timeest;
  std::vector<Eigen::Matrix<double, 7, 1>> posegt,poseest;
  std::vector<std::string> names;
  std::vector<std::vector<double>> times;
  std::vector<std::vector<Eigen::Matrix<double, 7, 1>>> poses;
  std::vector<Eigen::Matrix3d> cov_ori, cov_pos;
  ov_eval::Loader::load_data(argv[1], timegt, posegt, cov_ori, cov_pos);
  ov_eval::Loader::load_data(argv[2], timeest, poseest, cov_ori, cov_pos);
  // Print its length and stats
  double length = ov_eval::Loader::get_total_length(posegt);
  PRINT_INFO("[COMPGT]: %d poses in %s => length of %.2f meters\n", (int)timegt.size(), path_gt.stem().string().c_str(), length);
  length = ov_eval::Loader::get_total_length(poseest);
  PRINT_INFO("[COMPES]: %d poses in %s => length of %.2f meters\n", (int)timeest.size(), path.stem().string().c_str(), length);
  names.push_back((path_gt.stem().string()+"_gt").c_str());
  names.push_back(path.stem().string().c_str());
  times.push_back(timegt);
  times.push_back(timeest);
  poses.push_back(posegt);
  poses.push_back(poseest);
  // Create our trajectory object
  ov_eval::ResultTrajectory traj(argv[2], argv[1], "none");

  //===========================================================
  // Relative pose error
  //===========================================================

  // Calculate
  std::vector<double> segments = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0};
  std::map<double, std::pair<ov_eval::Statistics, ov_eval::Statistics>> error_ate;
  traj.calculate_ate_alignbegin(segments, error_ate);

  // Print it
  PRINT_INFO("======================================\n");
  PRINT_INFO("Absolute Trajectory Error\n");
  PRINT_INFO("======================================\n");
  for (const auto &seg : error_ate) {
    PRINT_INFO("seg %d - rmse_pos = %.3f | mean_pos = %.3f  - mean_pos/seg = %.3f%(%d samples)\n", (int)seg.first, seg.second.second.rmse,
               seg.second.second.mean, seg.second.second.mean/seg.first*100, (int)seg.second.second.values.size());
    // PRINT_DEBUG("seg %d - std_ori  = %.3f | std_pos  = %.3f\n",(int)seg.first,seg.second.first.std,seg.second.second.std);
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
  matplotlibcpp::show(true);
  //matplotlibcpp::save("ori.png");

#endif

}