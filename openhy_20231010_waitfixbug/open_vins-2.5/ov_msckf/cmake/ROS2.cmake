cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ov_core REQUIRED)
find_package(ov_init REQUIRED)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    message(FATAL_ERROR "Build with ROS1.cmake if you don't have ROS.")
endif ()
add_definitions(-DROS_AVAILABLE=2)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)
list(APPEND ament_libraries
        rclcpp
        tf2_ros
        std_msgs
        geometry_msgs
        sensor_msgs
        nav_msgs
        cv_bridge
        image_transport
        ov_core
        ov_init
)

##################################################
# Make the shared library
##################################################

list(APPEND library_source_files
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/core/VioManager.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp
)
list(APPEND library_source_files
        src/ros/ROS2Visualizer.cpp
)
add_library(ov_msckf_lib SHARED ${library_source_files})
ament_target_dependencies(ov_msckf_lib ${ament_libraries})
target_link_libraries(ov_msckf_lib ${thirdparty_libraries})
target_include_directories(ov_msckf_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/)
install(TARGETS ov_msckf_lib
        LIBRARY         DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME         DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
ament_export_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
ament_export_libraries(ov_msckf_lib)

##################################################
# Make binary files!
##################################################

# TODO: UPGRADE THIS TO ROS2 AS ANOTHER FILE!!
#add_executable(ros2_serial_msckf src/ros2_serial_msckf.cpp)
#target_link_libraries(ros2_serial_msckf ov_msckf_lib ${thirdparty_libraries})

add_executable(run_subscribe_msckf src/run_subscribe_msckf.cpp)
ament_target_dependencies(run_subscribe_msckf ${ament_libraries})
target_link_libraries(run_subscribe_msckf ov_msckf_lib ${thirdparty_libraries})
install(TARGETS run_subscribe_msckf DESTINATION lib/${PROJECT_NAME})

add_executable(run_simulation src/run_simulation.cpp)
ament_target_dependencies(run_simulation ${ament_libraries})
target_link_libraries(run_simulation ov_msckf_lib ${thirdparty_libraries})
install(TARGETS run_simulation DESTINATION lib/${PROJECT_NAME})

add_executable(test_sim_meas src/test_sim_meas.cpp)
ament_target_dependencies(test_sim_meas ${ament_libraries})
target_link_libraries(test_sim_meas ov_msckf_lib ${thirdparty_libraries})
install(TARGETS test_sim_meas DESTINATION lib/${PROJECT_NAME})

add_executable(test_sim_repeat src/test_sim_repeat.cpp)
ament_target_dependencies(test_sim_repeat ${ament_libraries})
target_link_libraries(test_sim_repeat ov_msckf_lib ${thirdparty_libraries})
install(TARGETS test_sim_repeat DESTINATION lib/${PROJECT_NAME})

# Install launch and config directories
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY ../config/ DESTINATION share/${PROJECT_NAME}/)

# finally define this as the package
ament_package()
