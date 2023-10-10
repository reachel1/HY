#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "ros/ROS1Visualizer.h"
#include "utils/dataset_reader.h"

#include "optimizer/Optimizer.h"

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
std::shared_ptr<ROS1Visualizer> viz;

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Launch our ros node
  ros::init(argc, argv, "run_serial_msckf");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);
  std::string output = "unset_path_to_output";
  nh->param<std::string>("bag_name", output, output);
  output = ros::package::getPath("ov_data") + "/optimizer/" + output;
  // Load the config
  std::cout<<config_path<<std::endl;
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node_handler(nh);

  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  sys = std::make_shared<VioManager>(params);
  viz = std::make_shared<ROS1Visualizer>(nh, sys);
  //set rect for image
  std::string stereo_calib = "unset_path_to_stereo_calib.yaml";
  nh->param<std::string>("stereo_calib", stereo_calib, stereo_calib);
  viz->set_calib(stereo_calib);
  viz->setup_subscribers(parser);
  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  sys->opt->set_output(output);
  // std::thread thsys5(&Optimizer::get_feat_from_vio,sys->opt);

  // init with groundtruth and 
  // Eigen::Matrix<double,17,1> imustate;
  // imustate<<1695455412.61693,  0.999845, -0.000064, 0.017014, 0.004627, 0, 0, 0 ,0, 0, 0 , -0.001, -0.0015, 0.0005, 0.1, 0.15, -0.01;
  // sys->initialize_with_gt(imustate);
  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  sys->opt->esi_end();
  // thsys5.join();
  // Final visualization
  viz->visualize_final();

  ros::shutdown();
  // Done!
  return EXIT_SUCCESS;
  
}