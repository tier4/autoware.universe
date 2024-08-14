// Copyright 2022 Tier IV, Inc.
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

#include "pointcloud_preprocessor/ring_filter/ring_filter.hpp"

namespace
{
sensor_msgs::msg::PointCloud2 filterPointCloudByRing(const sensor_msgs::msg::PointCloud2& input_cloud, int ring_interval) {
    // Prepare new pointcloud2 message
    sensor_msgs::msg::PointCloud2 output_cloud;
    output_cloud.header = input_cloud.header;
    output_cloud.height = input_cloud.height;
    output_cloud.is_bigendian = input_cloud.is_bigendian;
    output_cloud.is_dense = input_cloud.is_dense;
    output_cloud.point_step = input_cloud.point_step;
    output_cloud.row_step = input_cloud.row_step;

    // Copy fileds
    output_cloud.fields = input_cloud.fields;
    
    // Variable to copy required data
    std::vector<uint8_t> output_data;

    sensor_msgs::PointCloud2ConstIterator<uint16_t> ring_iter(input_cloud, "ring");

    for (sensor_msgs::PointCloud2ConstIterator<uint8_t> iter(input_cloud, "x"); iter != iter.end(); ++iter, ++ring_iter) {
        if ((*ring_iter % ring_interval) == 0) {
            // Add points according to the ring interval
            const uint8_t* point_ptr = &(*iter);
            output_data.insert(output_data.end(), point_ptr, point_ptr + input_cloud.point_step);
        }
    }

    // Set new data
    output_cloud.data = std::move(output_data);
    output_cloud.width = output_cloud.data.size() / output_cloud.point_step;

    return output_cloud;
}
}  // anonymous namespace

namespace pointcloud_preprocessor
{
RingFilterComponent::RingFilterComponent(
  const rclcpp::NodeOptions & node_options)
: Filter("RingFilter", node_options)
{
  ring_interval_ = static_cast<uint32_t>(declare_parameter("ring_interval", 2));
}

void RingFilterComponent::filter(
  const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output)
{
  if (indices) {
    RCLCPP_WARN(get_logger(), "Indices are not supported and will be ignored");
  }

  output = filterPointCloudByRing(*input, ring_interval_);
}

}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::RingFilterComponent)
