/*
 * Copyright 2022 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "csv.hpp"
#include "pointcloud_preprocessor/filter.hpp"

#include <pcl/common/impl/common.hpp>
#include <pcl_ros/transforms.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/assign.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/optional.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

//! For debug. Remove them later
#include <chrono>

namespace pointcloud_preprocessor
{

namespace debug
{
class Timer
{
public:
  Timer() : iteration_(0), sum_(0.0), past_(std::chrono::high_resolution_clock::now()) {}

  int lap()
  {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - past_).count();
    past_ = now;

    iteration_++;
    sum_ += microseconds * 1e-3;

    return microseconds;
  }

  double average() {
    return sum_ / iteration_;
  }

  std::size_t iteration() {
    return iteration_;
  }

private:
  std::size_t iteration_;
  double sum_;
  std::chrono::_V2::system_clock::time_point past_;
};
}  // namespace debug

enum class AreaType {
  DELETE_STATIC,  // Delete only static cloud
  DELETE_ALL,     // Delete static and dynamic cloud
  DELETE_OBJECT,  // Delete detected object bbox
};

using autoware_auto_perception_msgs::msg::PredictedObjects;

class MapAreaFilterComponent : public pointcloud_preprocessor::Filter
{
protected:
  void filter(
    const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
    PointCloud2 & output);
  void subscribe() override;
  void unsubscribe() override;

  bool do_filter_, csv_loaded_;

  /** \brief Parameter service callback result : needed to be hold */
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  /** \brief Parameter service callback */
  rcl_interfaces::msg::SetParametersResult paramCallback(const std::vector<rclcpp::Parameter> & p);

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr area_markers_pub_;
  rclcpp::Publisher<PredictedObjects>::SharedPtr filtered_objects_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr objects_cloud_sub_;
  rclcpp::Subscription<PredictedObjects>::SharedPtr objects_sub_;

private:
  std::shared_ptr<tf2_ros::Buffer> tf2_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  debug::Timer t_local_;
  debug::Timer t_local_2_;
  debug::Timer t_global_;

  std::string map_frame_;
  std::string base_link_frame_;
  typedef boost::geometry::model::d2::point_xy<float> PointXY;
  typedef boost::geometry::model::polygon<PointXY> Polygon2D;

  std::vector<AreaType> area_types_;
  std::vector<Polygon2D> area_polygons_;
  std::vector<PointXY> centroid_polygons_;

  visualization_msgs::msg::MarkerArray area_markers_msg_;

  geometry_msgs::msg::PoseStamped current_pose_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr objects_cloud_ptr_;
  boost::optional<PredictedObjects::ConstSharedPtr> objects_ptr_;

  rclcpp::TimerBase::SharedPtr timer_;

  double area_distance_check_;

  /***
   * Returns true if valid polygons were found in the CSV
   * @param file_name CSV to parse and load polygons
   * @return true if valid polygons were found in the CSV
   */
  bool load_areas_from_csv(const std::string & file_name);

  void filter_points_by_area(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & input,
    pcl::PointCloud<pcl::PointXYZ>::Ptr output, const std::size_t border);

  bool filter_objects_by_area(PredictedObjects & out_objects);

  void timer_callback();
  void pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr & pose_msg);
  void objects_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg);
  void objects_callback(const PredictedObjects::ConstSharedPtr & cloud_msg);

  void create_area_marker_msg();

  static bool transform_pointcloud(
    const sensor_msgs::msg::PointCloud2 & input, const tf2_ros::Buffer & tf2,
    const std::string & target_frame, sensor_msgs::msg::PointCloud2 & output);

public:
  explicit MapAreaFilterComponent(const rclcpp::NodeOptions & options);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace pointcloud_preprocessor
