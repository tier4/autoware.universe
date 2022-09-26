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

#include "pointcloud_preprocessor/outlier_filter/map_area_filter_nodelet.hpp"

namespace pointcloud_preprocessor
{

using std::placeholders::_1;

MapAreaFilterComponent::MapAreaFilterComponent(const rclcpp::NodeOptions & options)
: Filter("MapAreaFilter", options)
{
  // set initial parameters
  std::string map_area_csv;

  do_filter_ = static_cast<bool>(this->declare_parameter("do_filter", true));
  map_frame_ = static_cast<std::string>(this->declare_parameter("map_frame", "map"));
  base_link_frame_ =
    static_cast<std::string>(this->declare_parameter("base_link_frame", "base_link"));
  map_area_csv = static_cast<std::string>(this->declare_parameter("map_area_csv", ""));
  area_distance_check_ = static_cast<double>(this->declare_parameter("area_distance_check", 300));

  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "input/pose_topic", rclcpp::QoS(1),
    std::bind(&MapAreaFilterComponent::pose_callback, this, _1));
  objects_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "input/objects_cloud", rclcpp::QoS(1),
    std::bind(&MapAreaFilterComponent::objects_cloud_callback, this, _1));
  area_markers_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("output/debug", rclcpp::QoS(1));

  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&MapAreaFilterComponent::paramCallback, this, _1));

  RCLCPP_INFO_STREAM(this->get_logger(), "Loading CSV: " << map_area_csv);
  if (map_area_csv.empty() || !load_areas_from_csv(map_area_csv)) {
    RCLCPP_INFO_STREAM(
      this->get_logger(), "Invalid CSV File provided: '" << map_area_csv << "'. Not filtering");
  }
  RCLCPP_INFO_STREAM(this->get_logger(), "Areas: " << area_polygons_.size());

  rclcpp::Clock::SharedPtr clock = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tf2_.reset(new tf2_ros::Buffer(clock));
  tf2_listener_.reset(new tf2_ros::TransformListener(*tf2_));

  do_filter_ = true;
}

void MapAreaFilterComponent::subscribe() { Filter::subscribe(); }

void MapAreaFilterComponent::unsubscribe() { Filter::unsubscribe(); }

void MapAreaFilterComponent::publish_area_markers()
{
  visualization_msgs::msg::MarkerArray area_markers;
  size_t i = 0;

  for (const auto & polygon : area_polygons_) {
    visualization_msgs::msg::Marker area;
    area.ns = "filtered_area";
    area.header.frame_id = map_frame_;
    area.id = i;

    if (!delete_all_area_[i]) {
      area.color.r = 3. / 255.;
      area.color.g = 169. / 255.;
      area.color.g = 252. / 255.;
      area.color.a = 0.99;
    } else {
      area.color.r = 1.;
      area.color.g = 0.;
      area.color.g = 0.;
      area.color.a = 0.99;
    }

    area.scale.x = 0.1;
    area.scale.y = 0.1;
    area.scale.z = 0.1;
    area.type = visualization_msgs::msg::Marker::LINE_STRIP;
    area.action = visualization_msgs::msg::Marker::MODIFY;

    for (auto it = boost::begin(boost::geometry::exterior_ring(polygon)),
              end = boost::end(boost::geometry::exterior_ring(polygon));
         it != end; ++it) {
      geometry_msgs::msg::Point point;
      point.x = it->x();
      point.y = it->y();
      area.points.emplace_back(point);
    }
    area_markers.markers.emplace_back(area);
    i++;
  }

  area_markers_pub_->publish(area_markers);
}

void MapAreaFilterComponent::pose_callback(
  const geometry_msgs::msg::PoseStamped::ConstSharedPtr & pose_msg)
{
  std::scoped_lock lock(mutex_);
  current_pose_ = *pose_msg;
}

void MapAreaFilterComponent::objects_cloud_callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg)
{
  std::scoped_lock lock(mutex_);
  objects_cloud_ptr_ = cloud_msg;
}

bool MapAreaFilterComponent::load_areas_from_csv(const std::string & file_name)
{
  csv::CSVFormat format;
  format.no_header();
  format.variable_columns(csv::VariableColumnPolicy::KEEP);

  csv::CSVReader reader(file_name, format);
  for (csv::CSVRow & row : reader) {
    std::vector<PointXY> row_points;
    Polygon2D current_polygon;
    size_t i = 0, j = -1;
    float current_x = 0.f;
    for (csv::CSVField & field : row) {
      if (i == 0) {  // first column contains type of area.
        if (field.get<int>() == 1) {
          delete_all_area_.push_back(true);
        } else {
          delete_all_area_.push_back(false);
        }
        i++;
        j++;
        continue;
      }
      if (j % 2) {
        row_points.emplace_back(PointXY(current_x, field.get<float>()));
      } else {
        current_x = field.get<float>();
      }
      j++;
    }
    if (!row_points.empty()) {
      boost::geometry::assign_points(current_polygon, row_points);
      if (!boost::geometry::is_valid(current_polygon)) {
        boost::geometry::correct(current_polygon);
      }
      RCLCPP_INFO_STREAM(
        this->get_logger(), "Polygon in row: " << boost::geometry::dsv(current_polygon)
                                               << " has an area of "
                                               << boost::geometry::area(current_polygon));

      if (boost::geometry::area(current_polygon) > 0) {
        PointXY centroid(0.f, 0.f);
        boost::geometry::centroid(current_polygon, centroid);
        centroid_polygons_.emplace_back(centroid);
        area_polygons_.emplace_back(current_polygon);
      } else {
        RCLCPP_WARN_STREAM(
          this->get_logger(), "Ignoring invalid polygon:" << boost::geometry::dsv(current_polygon));
      }
    } else {
      RCLCPP_WARN_STREAM(this->get_logger(), "Invalid point in CSV:" << row.to_json());
    }
  }
  csv_loaded_ = true;
  publish_area_markers();

  return !area_polygons_.empty();
}

bool MapAreaFilterComponent::transformPointcloud(
  const sensor_msgs::msg::PointCloud2 & input, const tf2_ros::Buffer & tf2,
  const std::string & target_frame, sensor_msgs::msg::PointCloud2 & output)
{
  try {
    geometry_msgs::msg::TransformStamped tf_stamped;
    tf_stamped = tf2.lookupTransform(
      target_frame, input.header.frame_id, input.header.stamp, rclcpp::Duration::from_seconds(0.5));

    Eigen::Matrix4f tf_matrix = tf2::transformToEigen(tf_stamped.transform).matrix().cast<float>();
    pcl_ros::transformPointCloud(tf_matrix, input, output);
    output.header.stamp = input.header.stamp;
    output.header.frame_id = target_frame;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("map_area_filter"), ex.what());
    return false;
  }
  return true;
}

void MapAreaFilterComponent::filter_points_by_area(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & input,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output)
{
  int area_i = 0;

  std::vector<bool> area_check(area_polygons_.size(), false);

  for ([[maybe_unused]] const auto & area : area_polygons_) {
    const auto & centroid = centroid_polygons_[area_i];
    double distance = sqrt(
      (centroid.x() - current_pose_.pose.position.x) *
        (centroid.x() - current_pose_.pose.position.x) +
      (centroid.y() - current_pose_.pose.position.y) *
        (centroid.y() - current_pose_.pose.position.y));

    area_check[area_i++] = (distance <= area_distance_check_);
  }  // for polygons

  for (const auto & point : input->points) {
    bool found = false;
    area_i = 0;
    for (const auto & check : area_check) {
      if (check) {
        PointXY bpoint(point.x, point.y);
        if (boost::geometry::within(bpoint, area_polygons_[area_i])) {
          if (delete_all_area_[area_i]) {
            found = true;
          } else if (!point.r && !point.g && !point.b) {  // if black (no class)
            found = true;
          }
          break;
        }
      }
      area_i++;
    }  // for areas
    if (!found) {
      output->points.emplace_back(point);
    }
  }  // for points in cloud
}

void MapAreaFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);

  if (!csv_loaded_) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1, "Areas CSV Not yet loaded");
  }
  if (!do_filter_ || area_polygons_.empty()) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1, "Not filtering: Check empty areas or Parameters.");
    output = *input;
    return;
  }

  // transform cloud to filter to map frame
  sensor_msgs::msg::PointCloud2 map_frame_cloud;
  if (!transformPointcloud(*input, *tf2_, map_frame_, map_frame_cloud)) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1,
      "Cannot transform cloud to " << map_frame_ << " not filtering.");
    output = *input;
    return;
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_frame_cloud_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(map_frame_cloud, *map_frame_cloud_pcl);

  // transform known object cloud to map frame
  sensor_msgs::msg::PointCloud2 objects_frame_cloud;
  if (objects_cloud_ptr_ != nullptr) {
    if (!transformPointcloud(*objects_cloud_ptr_, *tf2_, map_frame_, objects_frame_cloud)) {
      RCLCPP_ERROR_STREAM_THROTTLE(
        this->get_logger(), *this->get_clock(), 1,
        "Cannot transform objects cloud to " << map_frame_
                                             << " not filtering. Not including object cloud");
    }
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects_cloud_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(objects_frame_cloud, *objects_cloud_pcl);

  // if known object cloud contains points add to the filtered cloud
  if (!objects_cloud_pcl->points.empty()) {
    *map_frame_cloud_pcl += *objects_cloud_pcl;
  }

  // create filtered cloud container
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_frame_cloud_pcl_filtered(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  map_frame_cloud_pcl_filtered->points.clear();

  filter_points_by_area(map_frame_cloud_pcl, map_frame_cloud_pcl_filtered);

  pcl::toROSMsg(*map_frame_cloud_pcl_filtered, output);
  output.header = input->header;
  output.header.frame_id = map_frame_;

  if (!transformPointcloud(output, *tf2_, input->header.frame_id, output)) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1,
      "Cannot transform back cloud to " << input->header.frame_id << " not filtering.");
    output = *input;
    return;
  }
  output.header = input->header;
}

rcl_interfaces::msg::SetParametersResult MapAreaFilterComponent::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  std::scoped_lock lock(mutex_);

  if (get_param(p, "do_filter", do_filter_)) {
    RCLCPP_DEBUG(
      this->get_logger(), "Setting new whether to apply filter: %s.",
      do_filter_ ? "true" : "false");
  }
  if (get_param(p, "map_frame", map_frame_)) {
    RCLCPP_DEBUG(this->get_logger(), "Setting new map frame to: %s.", map_frame_.c_str());
  }
  if (get_param(p, "base_link_frame", base_link_frame_)) {
    RCLCPP_DEBUG(
      this->get_logger(), "Setting new base link frame to: %s.", base_link_frame_.c_str());
  }
  if (get_param(p, "area_distance_check", area_distance_check_)) {
    RCLCPP_DEBUG(
      this->get_logger(), "Setting new area distance check to: %.2lf.", area_distance_check_);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::MapAreaFilterComponent)
