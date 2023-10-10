cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(catkin QUIET COMPONENTS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge ov_core ov_init novatel_oem7_msgs)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge ov_core ov_init novatel_oem7_msgs
            INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/
            LIBRARIES ov_msckf_lib
    )
else ()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
endif ()

find_package(Cholmod QUIET)
find_package(G2O QUIET)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/../thirdparty/geographiclib/include/
        ${G2O_INCLUDE_DIRS} 
        ${PROJECT_SOURCE_DIR}/../thirdparty/Sophus
        ${CSPARSE_INCLUDE_DIR}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/../thirdparty/geographiclib/build/src/libGeographic.so
        ${PROJECT_SOURCE_DIR}/../thirdparty/geographiclib/build/src/libGeographic.so.17
        ${PROJECT_SOURCE_DIR}/../thirdparty/geographiclib/build/src/libGeographic.so.17.1.2
        ${Sophus_LIBRARIES}
        g2o_types_icp g2o_types_slam2d
        g2o_core  g2o_solver_structure_only
        g2o_types_sba g2o_types_slam3d 
        g2o_solver_dense g2o_stuff
        g2o_types_sclam2d g2o_solver_pcg
        g2o_types_data g2o_types_sim3
        ${CSPARSE_LIBRARY}
)

# If we are not building with ROS then we need to manually link to its headers
# This isn't that elegant of a way, but this at least allows for building without ROS
# If we had a root cmake we could do this: https://stackoverflow.com/a/11217008/7718197
# But since we don't we need to basically build all the cpp / h files explicitly :(
if (NOT catkin_FOUND OR NOT ENABLE_ROS)

    message(WARNING "MANUALLY LINKING TO OV_CORE LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_core/src/)
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/cam/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/cpi/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/feat/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/plot/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/sim/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/track/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/types/*.cpp")
    list(APPEND library_source_files ${ov_core_files})
    file(GLOB_RECURSE ov_core_files "${CMAKE_SOURCE_DIR}/../ov_core/src/utils/*.cpp")
    list(APPEND library_source_files ${ov_core_files})

    message(WARNING "MANUALLY LINKING TO OV_INIT LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_init/src/)
    file(GLOB_RECURSE ov_init_files "${CMAKE_SOURCE_DIR}/../ov_init/src/*.[hc]pp")
    list(APPEND library_source_files ${ov_init_files})

endif ()

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
        src/optimizer/Converter.cpp
        src/optimizer/ImuTypes.cpp
        src/optimizer/Frame.cpp
        src/optimizer/KeyPoint.cpp
        src/optimizer/G2oTypes.cpp
)
if (catkin_FOUND AND ENABLE_ROS)
    list(APPEND library_source_files
            src/ros/ROS1Visualizer.cpp
    )
endif ()
add_library(ov_msckf_lib SHARED ${library_source_files})
target_link_libraries(ov_msckf_lib ${thirdparty_libraries})
target_include_directories(ov_msckf_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/)
install(TARGETS ov_msckf_lib
        LIBRARY         DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME         DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

##################################################
# Make binary files!
##################################################

if (catkin_FOUND AND ENABLE_ROS)

    add_executable(ros1_serial_msckf src/ros1_serial_msckf.cpp)
    target_link_libraries(ros1_serial_msckf ov_msckf_lib ${thirdparty_libraries})

    add_executable(ros1_rt_msckf src/ros1_rt_msckf.cpp)
    target_link_libraries(ros1_rt_msckf ov_msckf_lib ${thirdparty_libraries})

endif ()




