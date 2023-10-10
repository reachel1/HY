/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2021 Patrick Geneva
 * Copyright (C) 2021 Guoquan Huang
 * Copyright (C) 2021 OpenVINS Contributors
 * Copyright (C) 2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "TrackKLT.h"
#include "utils/print.h"

using namespace ov_core;

void TrackKLT::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  size_t num_images = message.images.size();
  if (num_images == 1) {
    feed_monocular_mid(message, 2);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo_new(message, 0, 1);
  } else if (num_images == 3 && use_stereo) {
    // feed_threestereo(message, 0 , 1, 2);
    feed_stereo_new(message, 0, 1);
    feed_monocular_mid(message, 2);
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackKLT::feed_monocular(const CameraData &message, size_t msg_id) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::unique_lock<std::mutex> lck(mtx_feeds.at(cam_id));

  // Histogram equalize
  cv::Mat img, mask;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id), img);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id), img);
  } else {
    img = message.images.at(msg_id);
  }
  mask = message.masks.at(msg_id);

  // Extract the new image pyramid
  std::vector<cv::Mat> imgpyr;
  cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id].empty()) {
    // Detect new features
    perform_detection_monocular(imgpyr, mask, pts_last[cam_id], ids_last[cam_id]);
    // Save the current image and pyramid
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_last[cam_id], ids_last[cam_id]);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_last[cam_id];

  // Lets track temporally
  perform_matching(img_pyramid_last[cam_id], imgpyr, pts_last[cam_id], pts_left_new, cam_id, cam_id, mask_ll);
  assert(pts_left_new.size() == ids_last[cam_id].size());
  rT4 = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
        (int)pts_left_new.at(i).pt.y >= img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track, and also tracked from left to right
    if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_last[cam_id][i]);
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Move forward in time
  img_last[cam_id] = img;
  img_pyramid_last[cam_id] = imgpyr;
  img_mask_last[cam_id] = mask;
  pts_last[cam_id] = good_left;
  ids_last[cam_id] = good_ids_left;
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for pyramid\n",(rT2-rT1).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for detection\n",(rT3-rT2).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for temporal klt\n",(rT4-rT3).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n",(rT5-rT4).total_microseconds() * 1e-6,
  // (int)good_left.size()); PRINT_DEBUG("[TIME-KLT]: %.4f seconds for total\n",(rT5-rT1).total_microseconds() * 1e-6);
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::unique_lock<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::unique_lock<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Histogram equalize images
  cv::Mat img_left, img_right, mask_left, mask_right;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id_left), img_left);
    cv::equalizeHist(message.images.at(msg_id_right), img_right);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id_left), img_left);
    clahe->apply(message.images.at(msg_id_right), img_right);
  } else {
    img_left = message.images.at(msg_id_left);
    img_right = message.images.at(msg_id_right);
  }
  mask_left = message.masks.at(msg_id_left);
  mask_right = message.masks.at(msg_id_right);

  // Extract image pyramids
  std::vector<cv::Mat> imgpyr_left, imgpyr_right;
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    cv::buildOpticalFlowPyramid(is_left ? img_left : img_right, is_left ? imgpyr_left : imgpyr_right, win_size, pyr_levels);
                  }
                }));
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    // Track into the new image
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, pts_last[cam_id_left],
                             pts_last[cam_id_right], ids_last[cam_id_left], ids_last[cam_id_right]);
    // Save the current image and pyramid
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_last[cam_id_left], pts_last[cam_id_right],
                           ids_last[cam_id_left], ids_last[cam_id_right]);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_last[cam_id_left];
  std::vector<cv::KeyPoint> pts_right_new = pts_last[cam_id_right];

  // Lets track temporally
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     pts_last[is_left ? cam_id_left : cam_id_right], is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr);
                  }
                }));
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // left to right matching
  // TODO: we should probably still do this to reject outliers
  // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
  // std::vector<uchar> mask_lr;
  // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
  rT5 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty() && mask_rr.empty()) {
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;
    // See if we have the same feature in the right
    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_last[cam_id_right].size(); n++) {
      if (ids_last[cam_id_left].at(i) == ids_last[cam_id_right].at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_last[cam_id_left].at(i));
      good_ids_right.push_back(ids_last[cam_id_right].at(index_right));
      // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_last[cam_id_left].at(i), ids_last[cam_id_right].at(index_right));
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_last[cam_id_left].at(i));
      // PRINT_DEBUG("adding to left - %u \n", ids_last[cam_id_left].at(i));
    }
  }

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_last[cam_id_right].at(i)) != good_ids_right.end());
    // If it has not already been added as a good feature, add it as a mono track
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_last[cam_id_right].at(i));
      // PRINT_DEBUG("adding to right - %u \n", ids_last[cam_id_right].at(i));
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  // Move forward in time
  img_last[cam_id_left] = img_left;
  img_last[cam_id_right] = img_right;
  img_pyramid_last[cam_id_left] = imgpyr_left;
  img_pyramid_last[cam_id_right] = imgpyr_right;
  img_mask_last[cam_id_left] = mask_left;
  img_mask_last[cam_id_right] = mask_right;
  pts_last[cam_id_left] = good_left;
  pts_last[cam_id_right] = good_right;
  ids_last[cam_id_left] = good_ids_left;
  ids_last[cam_id_right] = good_ids_right;
  rT6 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for pyramid\n",(rT2-rT1).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for detection\n",(rT3-rT2).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for temporal klt\n",(rT4-rT3).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for stereo klt\n",(rT5-rT4).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n",(rT6-rT5).total_microseconds() * 1e-6,
  // (int)good_left.size()); PRINT_DEBUG("[TIME-KLT]: %.4f seconds for total\n",(rT6-rT1).total_microseconds() * 1e-6);
}

void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size((int)((float)img0pyr.at(0).cols / (float)min_px_dist), (int)((float)img0pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d = cv::Mat::zeros(size, CV_8UC1);
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size.width || y_grid < 0 || y_grid >= size.height || x < 0 || x >= img0pyr.at(0).cols || y < 0 ||
        y >= img0pyr.at(0).rows) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  int num_featsneeded = num_features - (int)pts0.size();

  // If we don't need any features, just return
  if (num_featsneeded < std::min(75, (int)(0.2 * num_features)))
    return;

  // Extract our features (use fast with griding)
  std::vector<cv::KeyPoint> pts0_ext;
  Grider_FAST::perform_griding(img0pyr.at(0), mask0, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size.width || y_grid < 0 || y_grid >= size.height)
      continue;
    // See if there is a point at this location
    if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size0((int)((float)img0pyr.at(0).cols / (float)min_px_dist), (int)((float)img0pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_0 = cv::Mat::zeros(size0, CV_8UC1);
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size0.width || y_grid < 0 || y_grid >= size0.height || x < 0 || x >= img0pyr.at(0).cols || y < 0 ||
        y >= img0pyr.at(0).rows) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_0.at<uint8_t>(y_grid, x_grid) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_0.at<uint8_t>(y_grid, x_grid) = 255;
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  int num_featsneeded_0 = num_features - (int)pts0.size();

  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  if (num_featsneeded_0 > std::min(75, (int)(0.2 * num_features))) {

    // Extract our features (use fast with griding)
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_FAST::perform_griding(img0pyr.at(0), mask0, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size0.width || y_grid < 0 || y_grid >= size0.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      grid_2d_0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

    // TODO: Project points from the left frame into the right frame
    // TODO: This will not work for large baseline systems.....
    // TODO: If we had some depth estimates we could do a better projection
    // TODO: Or project and search along the epipolar line??
    std::vector<cv::KeyPoint> kpts1_new;
    std::vector<cv::Point2f> pts1_new;
    kpts1_new = kpts0_new;
    pts1_new = pts0_new;

    // If we have points, do KLT tracking to get the valid projections into the right image
    if (!pts0_new.empty()) {

      // Do our KLT tracking from the left to the right frame of reference
      // Note: we have a pretty big window size here since our projection might be bad
      // Note: but this might cause failure in cases of repeated textures (eg. checkerboard)
      std::vector<uchar> mask;
      // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
      std::vector<float> error;
      cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01);
      cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);

      // Loop through and record only ones that are valid
      for (size_t i = 0; i < pts0_new.size(); i++) {

        // Check that it is in bounds
        if ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
            (int)pts0_new.at(i).y >= img0pyr.at(0).rows) {
          continue;
        }
        if ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
            (int)pts1_new.at(i).y >= img1pyr.at(0).rows) {
          continue;
        }

        // Check to see if it there is already a feature in the right image at this location
        //  1) If this is not already in the right image, then we should treat it as a stereo
        //  2) Otherwise we will treat this as just a monocular track of the feature
        // TODO: we should check to see if we can combine this new feature and the one in the right
        // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
        if (mask[i] == 1) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          kpts1_new.at(i).pt = pts1_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          pts1.push_back(kpts1_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
          ids1.push_back(temp);
        } else {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
        }
      }
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_1 = cv::Mat::zeros(size1, CV_8UC1);
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size1.width || y_grid < 0 || y_grid >= size1.height || x < 0 || x >= img1pyr.at(0).cols || y < 0 ||
        y >= img1pyr.at(0).rows) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this is a stereo point
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    if (grid_2d_1.at<uint8_t>(y_grid, x_grid) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_1.at<uint8_t>(y_grid, x_grid) = 255;
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (num_featsneeded_1 > std::min(75, (int)(0.2 * num_features))) {

    // Extract our features (use fast with griding)
    std::vector<cv::KeyPoint> pts1_ext;
    Grider_FAST::perform_griding(img1pyr.at(0), mask1, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size1.width || y_grid < 0 || y_grid >= size1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) {

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1, pts0_back;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts0_back.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt, mask_klt_back;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
  cv::calcOpticalFlowPyrLK(img1pyr, img0pyr, pts1, pts0_back, mask_klt_back, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
  
  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  /*
  std::vector<cv::Point2f> pts0_n, pts1_n,pts0_n_back;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts0_n_back.push_back(camera_calib.at(id0)->undistort_cv(pts0_back.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc,mask_rsc_back;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);

  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 1 / max_focallength, 0.999, mask_rsc);
  cv::findFundamentalMat(pts1_n, pts0_n_back, cv::FM_RANSAC, 1 / max_focallength, 0.999, mask_rsc_back);
  */
  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i]  && i < mask_klt_back.size() && mask_klt_back[i] 
              && fabs(pts0_back.at(i).x - pts0.at(i).x) < 0.5 && fabs(pts0_back.at(i).y - pts0.at(i).y) < 0.5
              && std::floor(pts0_back.at(i).x) == std::floor(pts0.at(i).x)  && std::floor(pts0_back.at(i).y) == std::floor(pts0.at(i).y) 
              ) ? 1 : 0);
    //&& pts0_back.at(i).x == pts0.at(i).x && pts0_back.at(i).y == pts0.at(i).y
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}



void TrackKLT::feed_threestereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right,size_t msg_id_mid){

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  size_t cam_id_mid = message.sensor_ids.at(msg_id_mid);
  std::unique_lock<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::unique_lock<std::mutex> lck2(mtx_feeds.at(cam_id_right));
  std::unique_lock<std::mutex> lck3(mtx_feeds.at(cam_id_mid));

  // Histogram equalize images
  cv::Mat img_left, img_right, img_mid, mask_left, mask_right, mask_mid;
  std::map<size_t,cv::Mat> img_map;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id_left), img_left);
    cv::equalizeHist(message.images.at(msg_id_right), img_right);
    cv::equalizeHist(message.images.at(msg_id_mid), img_mid);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id_left), img_left);
    clahe->apply(message.images.at(msg_id_right), img_right);
    clahe->apply(message.images.at(msg_id_right), img_mid);
  } else {
    img_left = message.images.at(msg_id_left);
    img_right = message.images.at(msg_id_right);
    img_mid = message.images.at(msg_id_mid);
  }
  mask_left = message.masks.at(msg_id_left);
  mask_right = message.masks.at(msg_id_right);
  mask_mid = message.masks.at(msg_id_mid);
  img_map[msg_id_left] = img_left;
  img_map[msg_id_right] = img_right;
  img_map[msg_id_mid] = img_mid;

  // Extract image pyramids
  std::map<size_t,std::vector<cv::Mat>> imgpyr_map;
  for (int i = 0; i < 3; i++) {
                    //bool is_left = (i == 0);
                    std::vector<cv::Mat> pyr;
                    cv::buildOpticalFlowPyramid(img_map[i], pyr, win_size, pyr_levels);
                    imgpyr_map[i] = pyr;
                  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_mid].empty()) {
    // Track into the new image
    perform_detection(imgpyr_map[cam_id_left], cam_id_left,pts_last[cam_id_left],ids_last[cam_id_left],min_px_dist);
    perform_detection(imgpyr_map[cam_id_mid], cam_id_mid,pts_last[cam_id_mid],ids_last[cam_id_mid],min_px_dist_mid);
    // Save the current image and pyramid
    img_last = img_map;
    img_pyramid_last = imgpyr_map;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    img_mask_last[cam_id_mid] = mask_mid;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  perform_detection(imgpyr_map[cam_id_left], cam_id_left,pts_last[cam_id_left],ids_last[cam_id_left],min_px_dist);
  perform_detection(imgpyr_map[cam_id_mid], cam_id_mid,pts_last[cam_id_mid],ids_last[cam_id_mid],min_px_dist_mid);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Let do track on mid img
  // Our return success masks, and predicted new features
  std::vector<uchar> mask_mm;
  std::vector<cv::KeyPoint> pts_mid_new = pts_last[cam_id_mid];

  // Lets track temporally
  perform_matching(img_pyramid_last[cam_id_mid], imgpyr_map[cam_id_mid], pts_last[cam_id_mid], pts_mid_new, cam_id_mid, cam_id_mid, mask_mm);
  assert(pts_mid_new.size() == ids_last[cam_id_mid].size());
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_mid;
  std::vector<size_t> good_ids_mid;

  // Loop through all mid points
  for (size_t i = 0; i < pts_mid_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_mid_new.at(i).pt.x < 0 || pts_mid_new.at(i).pt.y < 0 || (int)pts_mid_new.at(i).pt.x >= img_mid.cols ||
        (int)pts_mid_new.at(i).pt.y >= img_mid.rows)
      continue;
    // Mask for mid cam
    if(mask_mid.at<uint8_t>((int)pts_mid_new.at(i).pt.y, (int)pts_mid_new.at(i).pt.x) > 1)
      continue;
    // If it is a good track, and also tracked back
    if (mask_mm[i]) {
      good_mid.push_back(pts_mid_new[i]);
      good_ids_mid.push_back(ids_last[cam_id_mid][i]);
    }
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Mono track(mono for left cam)
  // Let do track on mid img
  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_newmono = pts_last[cam_id_left];

  // Lets track temporally
  perform_matching(img_pyramid_last[cam_id_left], imgpyr_map[cam_id_left], pts_last[cam_id_left], pts_left_newmono, cam_id_left, cam_id_left, mask_ll);
  
  // Stereo track(ALL is stereo pairs)
  // Let do track on left img to right img
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;
  std::vector<uchar> mask_lr;
  std::vector<cv::KeyPoint> pts_left_new = pts_last[cam_id_left];
  std::vector<cv::KeyPoint> pts_right_new = pts_left_new;
  perform_matching_stereo(imgpyr_map[cam_id_left], imgpyr_map[cam_id_right], img_pyramid_last[cam_id_left],
        img_pyramid_last[cam_id_right], pts_left_new, pts_right_new,
        pts_last[cam_id_left], pts_last[cam_id_right], cam_id_left, cam_id_right, mask_lr);
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 ||
        (int)pts_right_new.at(i).pt.x >= img_right.cols || (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_lr[i]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(i));
      good_ids_left.push_back(ids_last[cam_id_left].at(i));
      good_ids_right.push_back(ids_last[cam_id_left].at(i));
    } 
    else if(mask_ll[i]){
      good_left.push_back(pts_left_newmono.at(i));
      good_ids_left.push_back(ids_last[cam_id_left].at(i));
    }
  }
  // Update our feature database, with theses new observations
  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }
  for (size_t i = 0; i < good_mid.size(); i++) {
    cv::Point2f npt_m = camera_calib.at(cam_id_mid)->undistort_cv(good_mid.at(i).pt);
    database->update_feature(good_ids_mid.at(i), message.timestamp, cam_id_mid, good_mid.at(i).pt.x, good_mid.at(i).pt.y, npt_m.x,
                             npt_m.y);
  }

  // Move forward in time
  img_last = img_map;
  img_pyramid_last = imgpyr_map;
  img_mask_last[cam_id_left] = mask_left;
  img_mask_last[cam_id_right] = mask_right;
  img_mask_last[cam_id_mid] = mask_mid;
  pts_last[cam_id_left] = good_left;
  pts_last[cam_id_right] = good_right;
  pts_last[cam_id_mid] = good_mid;
  ids_last[cam_id_left] = good_ids_left;
  ids_last[cam_id_right] = good_ids_right;
  ids_last[cam_id_mid] = good_ids_mid;

  rT6 = boost::posix_time::microsec_clock::local_time();

}


void TrackKLT::perform_detection(const std::vector<cv::Mat> &imgpyr0,
    size_t cam_id,std::vector<cv::KeyPoint> &pts0,std::vector<size_t> &ids0,int min_px_dist_feat){
      
  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size0((int)((float)imgpyr0.at(0).cols / (float)min_px_dist_feat), (int)((float)imgpyr0.at(0).rows / (float)min_px_dist_feat));
  // Points occupancy
  cv::Mat grid_2d_0 = cv::Mat::zeros(size0, CV_8UC1);
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist_feat);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist_feat);
    if (x_grid < 0 || x_grid >= size0.width || y_grid < 0 || y_grid >= size0.height || x < 0 || 
          x >= imgpyr0.at(0).cols || y < 0 ||y >= imgpyr0.at(0).rows) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_0.at<uint8_t>(y_grid, x_grid) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_0.at<uint8_t>(y_grid, x_grid) = 255;
    it0++;
    it1++;
  }
  // Add points
  std::vector<cv::KeyPoint> pts0_ext,pts0_atnv; // alternative
  Grider_FAST::FAST_detection(imgpyr0.at(0), pts0_ext, num_features, min_px_dist_feat, threshold_max, threshold_min, true);
  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  std::sort(pts0_ext.begin(), pts0_ext.end(), Grider_FAST::compare_response);
  auto it2 = pts0_ext.begin();
  while(it2 != pts0_ext.end()) {
    // Check that it is in bounds
    cv::KeyPoint kpt = *it2;
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist_feat);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist_feat);
    if (x_grid < 0 || x_grid >= size0.width || y_grid < 0 || y_grid >= size0.height){
      it2++;
      continue;
    }
    // See if there is a point at this location
    if (grid_2d_0.at<uint8_t>(y_grid, x_grid) > 127){
      pts0_atnv.push_back(kpt);
      it2++;
      continue;
    }
    // Else lets add it!
    grid_2d_0.at<uint8_t>(y_grid, x_grid) = 255;
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    it2++;
  }
  
  // add some points
  int feat_need_num = num_features - kpts0_new.size() - pts0.size();
  PRINT_INFO(BLUE "[detect]detect num %d feat need num %d\n" RESET,pts0_ext.size(),feat_need_num);
  if(feat_need_num > 0){
    std::sort(pts0_atnv.begin(), pts0_atnv.end(), Grider_FAST::compare_response);
    auto it3 = pts0_atnv.begin();
    while(it3 != pts0_atnv.end() && feat_need_num > 0) {
      // Check that it is in bounds
      cv::KeyPoint kpt = *it3;
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist_feat);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist_feat);
      if (x_grid < 0 || x_grid >= size0.width || y_grid < 0 || y_grid >= size0.height){
        it3++;
        continue;
      }
      // Else lets add it!
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
      it3++;
      feat_need_num--;
    }
  }
  
  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}


void TrackKLT::perform_matching_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, 
  const std::vector<cv::Mat> &img0pyr_last,const std::vector<cv::Mat> &img1pyr_last,
  std::vector<cv::KeyPoint> &pts0,std::vector<cv::KeyPoint> &pts1, 
  std::vector<cv::KeyPoint> &pts0_last,std::vector<cv::KeyPoint> &pts1_last, 
  size_t id0, size_t id1, std::vector<uchar> &mask_out){

  assert(pts0.size() == pts0_last.size());
  assert(pts0.size() == pts1.size());
  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> kpts0_new = pts0_last;
  perform_matching(img0pyr_last, img0pyr, pts0_last, kpts0_new, id0, id0, mask_ll);
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : kpts0_new) {
      pts0_new.push_back(kpt.pt);
  }
  //perform_matching(imgpyr_map[cam_id_left],img_pyramid_last[cam_id_left],pts_left_new,pts_left_last_back,cam_id_left, cam_id_left,mask_back_ll);
  // TODO: Project points from the left frame into the right frame
  // TODO: This will not work for large baseline systems.....
  // TODO: If we had some depth estimates we could do a better projection
  // TODO: Or project and search along the epipolar line??

  std::vector<cv::Point2f> pts1_new;
  pts1_new = pts0_new;

  // If we have points, do KLT tracking to get the valid projections into the right image
  // Do our KLT tracking from the left to the right frame of reference
  // Note: we have a pretty big window size here since our projection might be bad
  // Note: but this might cause failure in cases of repeated textures (eg. checkerboard)
  std::vector<uchar> mask_lr;
  // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask_lr, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);
  // Track match from right to last right frame
  std::vector<uchar> mask_rr_back;
  std::vector<cv::KeyPoint> kpts1_new,kpts1_old;
  kpts1_new.clear();
  for (auto &pt : pts1_new) {
    cv::KeyPoint kpt;
    kpt.pt = pt;
    kpts1_new.push_back(kpt);
  }
  kpts1_old = kpts1_new;
  perform_matching(img1pyr, img1pyr_last, kpts1_new, kpts1_old, id1, id1, mask_rr_back);
  // Do our KLT tracking from the right to the left frame of last
  std::vector<cv::Point2f> pts1_old,pts0_old;
  for (auto &kpt : kpts1_old) {
      pts1_old.push_back(kpt.pt);
  }
  pts0_old = pts1_old;
  std::vector<uchar> mask_rl_back;
  // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
  std::vector<float> error_back;
  cv::TermCriteria term_crit_back = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01);
  cv::calcOpticalFlowPyrLK(img1pyr_last, img0pyr_last, pts1_old, pts0_old, mask_rl_back, error_back, 
      win_size, pyr_levels, term_crit_back, cv::OPTFLOW_USE_INITIAL_FLOW);
  assert(kpts0_new.size() == kpts1_new.size());
  // Loop through and record only ones that are valid
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // Check to see if it there is already a feature in the right image at this location
    //  1) If this is not already in the right image, then we should treat it as a stereo
    // TODO: we should check to see if we can combine this new feature and the one in the right
    // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
    if (mask_ll[i] == 1 && mask_lr[i] == 1 && mask_rr_back[i] == 1 && mask_rl_back[i] == 1 &&
          std::floor(pts0_old.at(i).x) == std::floor(pts0_last.at(i).pt.x) && 
          std::floor(pts0_old.at(i).y) == std::floor(pts0_last.at(i).pt.y) &&
          fabs(pts1_new.at(i).y - pts0_new.at(i).y) < 10 && 
          fabs(pts0_old.at(i).x - pts0_last.at(i).pt.x) < 0.5 && 
          fabs(pts0_old.at(i).y - pts0_last.at(i).pt.y) < 0.5){
        // update the uv coordinates
        auto mask = (uchar) 1;
        mask_out.push_back(mask);
    }
    else{
        auto mask = (uchar) 0;
        mask_out.push_back(mask);
    }
    pts0.at(i) = kpts0_new.at(i);
    pts1.at(i) = kpts1_new.at(i);
  }

}


void TrackKLT::feed_monocular_mid(const CameraData &message, size_t msg_id) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::unique_lock<std::mutex> lck(mtx_feeds.at(cam_id));

  // Histogram equalize
  cv::Mat img, mask;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id), img);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id), img);
  } else {
    img = message.images.at(msg_id);
  }
  mask = message.masks.at(msg_id);

  // Extract the new image pyramid
  std::vector<cv::Mat> imgpyr;
  cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id].empty()) {
    // Detect new features
    // perform_detection_monocular(imgpyr, mask, pts_last[cam_id], ids_last[cam_id]);
    perform_detection(imgpyr, cam_id, pts_last[cam_id], ids_last[cam_id], min_px_dist_mid);
    // Save the current image and pyramid
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  // perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_last[cam_id], ids_last[cam_id]);
  perform_detection(img_pyramid_last[cam_id], cam_id, pts_last[cam_id], ids_last[cam_id], min_px_dist_mid);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_mm;
  std::vector<cv::KeyPoint> pts_new = pts_last[cam_id];

  // Lets track temporally
  perform_matching(img_pyramid_last[cam_id], imgpyr, pts_last[cam_id], pts_new, cam_id, cam_id, mask_mm);
  assert(pts_new.size() == ids_last[cam_id].size());
  rT4 = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_mm.empty()) {
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_mid;
  std::vector<size_t> good_ids_mid;

  // Loop through all mid points
  for (size_t i = 0; i < pts_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_new.at(i).pt.x < 0 || pts_new.at(i).pt.y < 0 || (int)pts_new.at(i).pt.x >= img.cols ||
        (int)pts_new.at(i).pt.y >= img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_new.at(i).pt.y, (int)pts_new.at(i).pt.x) > 127)
      continue;
    // If it is a good mid track
    if (mask_mm[i]) {
      good_mid.push_back(pts_new[i]);
      good_ids_mid.push_back(ids_last[cam_id][i]);
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_mid.size(); i++) {
    cv::Point2f npt_m = camera_calib.at(cam_id)->undistort_cv(good_mid.at(i).pt);
    mid_database->update_feature(good_ids_mid.at(i), message.timestamp, cam_id, good_mid.at(i).pt.x, good_mid.at(i).pt.y, npt_m.x, npt_m.y);
  }

  // Move forward in time
  img_last[cam_id] = img;
  img_pyramid_last[cam_id] = imgpyr;
  img_mask_last[cam_id] = mask;
  pts_last[cam_id] = good_mid;
  ids_last[cam_id] = good_ids_mid;
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for pyramid\n",(rT2-rT1).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for detection\n",(rT3-rT2).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for temporal klt\n",(rT4-rT3).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n",(rT5-rT4).total_microseconds() * 1e-6,
  // (int)good_mid.size()); PRINT_DEBUG("[TIME-KLT]: %.4f seconds for total\n",(rT5-rT1).total_microseconds() * 1e-6);
}
