// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lidar_transfusion/transfusion_trt.hpp"

#include "lidar_transfusion/preprocess/preprocess_kernel.hpp"
#include "lidar_transfusion/transfusion_config.hpp"

#include <tier4_autoware_utils/math/constants.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace lidar_transfusion
{

TransfusionTRT::TransfusionTRT(
  const NetworkParam & network_param, const DensificationParam & densification_param,
  const TransfusionConfig & config)
: config_(config)
{
  network_trt_ptr_ = std::make_unique<NetworkTRT>(config_);

  network_trt_ptr_->init(
    network_param.onnx_path(), network_param.engine_path(), network_param.trt_precision());
  vg_ptr_ = std::make_unique<VoxelGenerator>(densification_param, config_, stream_);
  stop_watch_ptr_ = std::make_unique<tier4_autoware_utils::StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("processing/inner");
  initPtr();

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

TransfusionTRT::~TransfusionTRT()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

void TransfusionTRT::initPtr()
{
  // point cloud to voxels
  voxel_features_size_ =
    config_.max_voxels_ * config_.max_num_points_per_pillar_ * config_.num_point_feature_size_;
  voxel_num_size_ = config_.max_voxels_;
  voxel_idxs_size_ = config_.max_voxels_ * config_.num_point_values_;

  // output of TRT -- input of post-process
  cls_size_ = config_.num_proposals_ * config_.num_classes_;
  box_size_ = config_.num_proposals_ * config_.num_box_values_;
  dir_cls_size_ = config_.num_proposals_ * 2;  // x, y
  cls_output_d_ = cuda::make_unique<float[]>(cls_size_);
  box_output_d_ = cuda::make_unique<float[]>(box_size_);
  dir_cls_output_d_ = cuda::make_unique<float[]>(dir_cls_size_);

  params_input_d_ = cuda::make_unique<unsigned int>();
  voxel_features_d_ = cuda::make_unique<float[]>(voxel_features_size_);
  voxel_num_d_ = cuda::make_unique<unsigned int[]>(voxel_num_size_);
  voxel_idxs_d_ = cuda::make_unique<unsigned int[]>(voxel_idxs_size_);
  points_d_ = cuda::make_unique<float[]>(config_.cloud_capacity_ * config_.num_point_feature_size_);
  pre_ptr_ = std::make_unique<PreprocessCuda>(config_, stream_);
  post_ptr_ = std::make_unique<PostprocessCuda>(config_, stream_);
}

bool TransfusionTRT::detect(
  const sensor_msgs::msg::PointCloud2 & msg, const tf2_ros::Buffer & tf_buffer,
  std::vector<Box3D> & det_boxes3d, std::unordered_map<std::string, double> & proc_timing)
{
  stop_watch_ptr_->toc("processing/inner", true);
  if (!preprocess(msg, tf_buffer)) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_transfusion"), "Fail to preprocess and skip to detect.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/preprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!inference()) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_transfusion"), "Fail to inference and skip to detect.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/inference_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!postprocess(det_boxes3d)) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_transfusion"), "Fail to postprocess and skip to detect.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/postprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  return true;
}

bool TransfusionTRT::preprocess(
  const sensor_msgs::msg::PointCloud2 & msg, const tf2_ros::Buffer & tf_buffer)
{
  if (!vg_ptr_->enqueuePointCloud(msg, tf_buffer)) {
    return false;
  }

  cuda::clear_async(cls_output_d_.get(), cls_size_, stream_);
  cuda::clear_async(box_output_d_.get(), box_size_, stream_);
  cuda::clear_async(dir_cls_output_d_.get(), dir_cls_size_, stream_);
  cuda::clear_async(voxel_features_d_.get(), voxel_features_size_, stream_);
  cuda::clear_async(voxel_num_d_.get(), voxel_num_size_, stream_);
  cuda::clear_async(voxel_idxs_d_.get(), voxel_idxs_size_, stream_);
  cuda::clear_async(params_input_d_.get(), 1, stream_);
  cuda::clear_async(
    points_d_.get(), config_.cloud_capacity_ * config_.num_point_feature_size_, stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  const auto count = vg_ptr_->generateSweepPoints(msg, points_d_);
  RCLCPP_DEBUG_STREAM(rclcpp::get_logger("lidar_transfusion"), "Generated sweep points: " << count);

  pre_ptr_->generateVoxels(
    points_d_.get(), count, params_input_d_.get(), voxel_features_d_.get(), voxel_num_d_.get(),
    voxel_idxs_d_.get());
  unsigned int params_input;
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &params_input, params_input_d_.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (params_input < config_.min_voxel_size_) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_transfusion"),
      "Too few voxels (" << params_input << ") for actual optimization profile ("
                         << config_.min_voxel_size_ << ")");
    return false;
  }

  if (params_input > config_.max_voxels_) {
    params_input = config_.max_voxels_;
  }
  RCLCPP_DEBUG_STREAM(
    rclcpp::get_logger("lidar_transfusion"), "Generated input voxels: " << params_input);

  network_trt_ptr_->context->setTensorAddress(
    network_trt_ptr_->getTensorName(NetworkIO::voxels), voxel_features_d_.get());
  network_trt_ptr_->context->setInputShape(
    network_trt_ptr_->getTensorName(NetworkIO::voxels),
    nvinfer1::Dims3{
      static_cast<int32_t>(params_input), static_cast<int32_t>(config_.max_num_points_per_pillar_),
      static_cast<int32_t>(config_.num_point_feature_size_)});
  network_trt_ptr_->context->setTensorAddress(
    network_trt_ptr_->getTensorName(NetworkIO::num_points), voxel_num_d_.get());
  network_trt_ptr_->context->setInputShape(
    network_trt_ptr_->getTensorName(NetworkIO::num_points),
    nvinfer1::Dims{1, {static_cast<int32_t>(params_input)}});
  network_trt_ptr_->context->setTensorAddress(
    network_trt_ptr_->getTensorName(NetworkIO::coors), voxel_idxs_d_.get());
  network_trt_ptr_->context->setInputShape(
    network_trt_ptr_->getTensorName(NetworkIO::coors),
    nvinfer1::Dims2{
      static_cast<int32_t>(params_input), static_cast<int32_t>(config_.num_point_values_)});
  network_trt_ptr_->context->setTensorAddress(
    network_trt_ptr_->getTensorName(NetworkIO::cls_score), cls_output_d_.get());
  network_trt_ptr_->context->setTensorAddress(
    network_trt_ptr_->getTensorName(NetworkIO::bbox_pred), box_output_d_.get());
  network_trt_ptr_->context->setTensorAddress(
    network_trt_ptr_->getTensorName(NetworkIO::dir_pred), dir_cls_output_d_.get());
  return true;
}

bool TransfusionTRT::inference()
{
  auto status = network_trt_ptr_->context->enqueueV3(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (!status) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_transfusion"), "Fail to enqueue and skip to detect.");
    return false;
  }
  return true;
}

bool TransfusionTRT::postprocess(std::vector<Box3D> & det_boxes3d)
{
  CHECK_CUDA_ERROR(post_ptr_->generateDetectedBoxes3D_launch(
    cls_output_d_.get(), box_output_d_.get(), dir_cls_output_d_.get(), det_boxes3d, stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  return true;
}

}  //  namespace lidar_transfusion
