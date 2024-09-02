// Copyright 2023 TIER IV, Inc.
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

#include "image_projection_based_fusion/segmentation_pointcloud_fusion/node.hpp"

#include "image_projection_based_fusion/utils/geometry.hpp"
#include "image_projection_based_fusion/utils/utils.hpp"

#include <perception_utils/run_length_encoder.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#endif

namespace image_projection_based_fusion
{
SegmentPointCloudFusionNode::SegmentPointCloudFusionNode(const rclcpp::NodeOptions & options)
: FusionNode<PointCloud2, PointCloud2, Image>("segmentation_pointcloud_fusion", options)
{
  filter_distance_threshold_ = declare_parameter<float>("filter_distance_threshold");
  for (auto & item : filter_semantic_label_target_list_) {
    item.second = declare_parameter<bool>("filter_semantic_label_target." + item.first);
  }
  for (const auto & item : filter_semantic_label_target_list_) {
    RCLCPP_INFO(
      this->get_logger(), "filter_semantic_label_target: %s %d", item.first.c_str(), item.second);
  }
  is_publish_debug_mask_ = declare_parameter<bool>("is_publish_debug_mask");
  pub_debug_mask_ptr_ = image_transport::create_publisher(this, "~/debug/mask");
}

void SegmentPointCloudFusionNode::preprocess(__attribute__((unused)) PointCloud2 & pointcloud_msg)
{
  return;
}

void SegmentPointCloudFusionNode::postprocess(__attribute__((unused)) PointCloud2 & pointcloud_msg)
{
  return;
}
void SegmentPointCloudFusionNode::fuseOnSingleImage(
  const PointCloud2 & input_pointcloud_msg, __attribute__((unused)) const std::size_t image_id,
  [[maybe_unused]] const Image & input_mask, __attribute__((unused)) const CameraInfo & camera_info,
  __attribute__((unused)) PointCloud2 & output_cloud)
{
  if (input_pointcloud_msg.data.empty()) {
    return;
  }
  if (input_mask.height == 0 || input_mask.width == 0) {
    return;
  }
  std::vector<uint8_t> mask_data(input_mask.data.begin(), input_mask.data.end());
  cv::Mat mask = perception_utils::runLengthDecoder(mask_data, input_mask.height, input_mask.width);

  // publish debug mask
  if (is_publish_debug_mask_) {
    sensor_msgs::msg::Image::SharedPtr debug_mask_msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", mask).toImageMsg();
    debug_mask_msg->header = input_mask.header;
    pub_debug_mask_ptr_.publish(debug_mask_msg);
  }
  const int orig_width = camera_info.width;
  const int orig_height = camera_info.height;
  // resize mask to the same size as the camera image
  cv::resize(mask, mask, cv::Size(orig_width, orig_height), 0, 0, cv::INTER_NEAREST);
  Eigen::Matrix4d projection;
  projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2), camera_info.p.at(3),
    camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6), camera_info.p.at(7),
    camera_info.p.at(8), camera_info.p.at(9), camera_info.p.at(10), camera_info.p.at(11), 0.0, 0.0,
    0.0, 1.0;
  geometry_msgs::msg::TransformStamped transform_stamped;
  // transform pointcloud from frame id to camera optical frame id
  {
    const auto transform_stamped_optional = getTransformStamped(
      tf_buffer_, input_mask.header.frame_id, input_pointcloud_msg.header.frame_id,
      input_pointcloud_msg.header.stamp);
    if (!transform_stamped_optional) {
      return;
    }
    transform_stamped = transform_stamped_optional.value();
  }

  PointCloud2 transformed_cloud;
  tf2::doTransform(input_pointcloud_msg, transformed_cloud, transform_stamped);

  int point_step = input_pointcloud_msg.point_step;
  int x_offset = input_pointcloud_msg.fields[pcl::getFieldIndex(input_pointcloud_msg, "x")].offset;
  int y_offset = input_pointcloud_msg.fields[pcl::getFieldIndex(input_pointcloud_msg, "y")].offset;
  int z_offset = input_pointcloud_msg.fields[pcl::getFieldIndex(input_pointcloud_msg, "z")].offset;
  size_t output_pointcloud_size = 0;
  output_cloud.data.clear();
  output_cloud.data.resize(input_pointcloud_msg.data.size());
  output_cloud.fields = input_pointcloud_msg.fields;
  output_cloud.header = input_pointcloud_msg.header;
  output_cloud.height = input_pointcloud_msg.height;
  output_cloud.point_step = input_pointcloud_msg.point_step;
  output_cloud.is_bigendian = input_pointcloud_msg.is_bigendian;
  output_cloud.is_dense = input_pointcloud_msg.is_dense;
  for (size_t global_offset = 0; global_offset < transformed_cloud.data.size();
       global_offset += point_step) {
    float transformed_x =
      *reinterpret_cast<float *>(&transformed_cloud.data[global_offset + x_offset]);
    float transformed_y =
      *reinterpret_cast<float *>(&transformed_cloud.data[global_offset + y_offset]);
    float transformed_z =
      *reinterpret_cast<float *>(&transformed_cloud.data[global_offset + z_offset]);
    // skip filtering pointcloud behind the camera or too far from camera
    if (transformed_z <= 0.0 || transformed_z > filter_distance_threshold_) {
      copyPointCloud(
        input_pointcloud_msg, point_step, global_offset, output_cloud, output_pointcloud_size);
      continue;
    }

    Eigen::Vector4d projected_point =
      projection * Eigen::Vector4d(transformed_x, transformed_y, transformed_z, 1.0);
    Eigen::Vector2d normalized_projected_point = Eigen::Vector2d(
      projected_point.x() / projected_point.z(), projected_point.y() / projected_point.z());

    bool is_inside_image =
      normalized_projected_point.x() > 0 && normalized_projected_point.x() < camera_info.width &&
      normalized_projected_point.y() > 0 && normalized_projected_point.y() < camera_info.height;
    if (!is_inside_image) {
      copyPointCloud(
        input_pointcloud_msg, point_step, global_offset, output_cloud, output_pointcloud_size);
      continue;
    }

    // skip filtering pointcloud where semantic id out of the defined list
    uint8_t semantic_id = mask.at<uint8_t>(
      static_cast<uint16_t>(normalized_projected_point.y()),
      static_cast<uint16_t>(normalized_projected_point.x()));
    if (static_cast<size_t>(semantic_id) >= filter_semantic_label_target_list_.size()) {
      copyPointCloud(
        input_pointcloud_msg, point_step, global_offset, output_cloud, output_pointcloud_size);
      continue;
    }
    if (!filter_semantic_label_target_list_.at(semantic_id).second) {
      copyPointCloud(
        input_pointcloud_msg, point_step, global_offset, output_cloud, output_pointcloud_size);
    }
  }

  output_cloud.data.resize(output_pointcloud_size);
  output_cloud.row_step = output_pointcloud_size / output_cloud.height;
  output_cloud.width = output_pointcloud_size / output_cloud.point_step / output_cloud.height;
}

bool SegmentPointCloudFusionNode::out_of_scope(__attribute__((unused))
                                               const PointCloud2 & filtered_cloud)
{
  return false;
}
}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::SegmentPointCloudFusionNode)
