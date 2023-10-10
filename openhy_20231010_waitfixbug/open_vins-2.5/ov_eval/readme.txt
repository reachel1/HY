请使用新的ov_eval
评估相对误差使用rpe_estimate,其指令为
rosrun ov_eval rpe_estimate src/open_vins-2.5/ov_data/trajgt/yuanqu04.txt(gt_path)  src/open_vins-2.5/ov_data/traj/yuanqu04.txt(esi_path)  'none'
绘制多个轨迹使用plot_trajectories,其指令为
rosrun ov_eval plot_trajectories 'none' src/open_vins-2.5/ov_data/trajgt/yuanqu02.txt(est_path1)  src/open_vins-2.5/ov_data/traj/yuanqu02.txt(est_path2)  src/open_vins-2.5/ov_data/trajgps/yuanqu02.txt(est_path3)
