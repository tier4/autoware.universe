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
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

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
    "input/objects_cloud", rclcpp::SensorDataQoS(),
    std::bind(&MapAreaFilterComponent::objects_cloud_callback, this, _1));
  objects_sub_ = this->create_subscription<PredictedObjects>(
    "input/objects", rclcpp::QoS(10),
    std::bind(&MapAreaFilterComponent::objects_callback, this, _1));
  filtered_objects_pub_ =
    this->create_publisher<PredictedObjects>("output/objects", rclcpp::QoS(10));
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

  using namespace std::chrono_literals;
  timer_ = this->create_wall_timer(1s, std::bind(&MapAreaFilterComponent::timer_callback, this));

  do_filter_ = true;
}

void MapAreaFilterComponent::subscribe() { Filter::subscribe(); }

void MapAreaFilterComponent::unsubscribe() { Filter::unsubscribe(); }

void MapAreaFilterComponent::create_area_marker_msg()
{
  size_t i = 0;

  for (const auto & polygon : area_polygons_) {
    visualization_msgs::msg::Marker area;
    area.ns = "filtered_area";
    area.header.frame_id = map_frame_;
    area.id = i;

    if (area_types_[i] == AreaType::DELETE_ALL) {
      area.color.r = 1.;
      area.color.g = 0.;
      area.color.b = 0.;
      area.color.a = 0.5;
    } else if (area_types_[i] == AreaType::DELETE_STATIC) {
      area.color.r = 0.;
      area.color.g = 1.;
      area.color.b = 0.;
      area.color.a = 0.5;
    } else if (area_types_[i] == AreaType::DELETE_OBJECT) {
      area.color.r = 0.;
      area.color.g = 0.;
      area.color.b = 1.;
      area.color.a = 0.5;
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
    area_markers_msg_.markers.emplace_back(area);
    i++;
  }
}

void MapAreaFilterComponent::timer_callback() { area_markers_pub_->publish(area_markers_msg_); }

void MapAreaFilterComponent::pose_callback(
  const geometry_msgs::msg::PoseStamped::ConstSharedPtr & msg)
{
  std::scoped_lock lock(mutex_);
  current_pose_ = *msg;
}

void MapAreaFilterComponent::objects_cloud_callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  std::scoped_lock lock(mutex_);
  objects_cloud_ptr_ = msg;
}

void MapAreaFilterComponent::objects_callback(const PredictedObjects::ConstSharedPtr & msg)
{
  std::scoped_lock lock(mutex_);
  objects_ptr_ = msg;
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
        if (field.get<int>() == 0) {
          area_types_.emplace_back(AreaType::DELETE_STATIC);
        } else if (field.get<int>() == 1) {
          area_types_.emplace_back(AreaType::DELETE_ALL);
        } else if (field.get<int>() == 2) {
          area_types_.emplace_back(AreaType::DELETE_OBJECT);
        } else {
          RCLCPP_WARN_STREAM(
            this->get_logger(),
            "Invalid area type specified: " << field.get<int>() << " in CSV:" << row.to_json());
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
      RCLCPP_INFO_STREAM(  // Verbose output
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
  create_area_marker_msg();

  return !area_polygons_.empty();
}

bool MapAreaFilterComponent::transform_pointcloud(
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
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & input, pcl::PointCloud<pcl::PointXYZ>::Ptr output,
  const std::size_t border)
{
  t_local_2_.lap();

  const auto polygon_size = area_polygons_.size();

  std::vector<bool> area_check(polygon_size, false);
  for (std::size_t area_i = 0; area_i < polygon_size; area_i++) {
    const auto & centroid = centroid_polygons_[area_i];
    double distance = sqrt(
      (centroid.x() - current_pose_.pose.position.x) *
        (centroid.x() - current_pose_.pose.position.x) +
      (centroid.y() - current_pose_.pose.position.y) *
        (centroid.y() - current_pose_.pose.position.y));

    area_check[area_i++] = (distance <= area_distance_check_);
  }  // for polygons

  std::cout << ">>> >>> 各area_polygonsの重心を求める操作 [us]:                             "
            << t_local_2_.lap() << std::endl;

  const auto point_size = input->points.size();
  std::vector<bool> within(point_size, true);
#pragma omp parallel for
  for (std::size_t point_i = 0; point_i < point_size; ++point_i) {
    const auto point = input->points[point_i];
    bool _within = false;
    for (std::size_t area_i = 0; area_i < polygon_size; area_i++) {
      if (!area_check[area_i]) continue;

      if (boost::geometry::within(PointXY(point.x, point.y), area_polygons_[area_i])) {
        if (area_types_[area_i] == AreaType::DELETE_ALL) {
          _within = true;
        } else if (area_types_[area_i] == AreaType::DELETE_STATIC) {
          if (point_i >= border) _within = true;
        }
      }
    }

    if (!_within) within[point_i] = false;
  }

  for (std::size_t point_i = 0; point_i < point_size; ++point_i) {
    if (!within[point_i]) {
      const auto point = input->points[point_i];
      output->points.emplace_back(pcl::PointXYZ(point.x, point.y, point.z));
    }
  }

  /* ========================== */

  // int area_i = 0;
  // std::size_t  point_i = 0;
  // for (const auto & point : input->points) {
  //   bool within = false;
  //   area_i = 0;
  //   for (const auto & check : area_check) {
  //     if (check) {
  //       if (boost::geometry::within(PointXY(point.x, point.y), area_polygons_[area_i])) {
  //         if (area_types_[area_i] == AreaType::DELETE_ALL) {
  //           within = true;
  //           break;
  //         } else if (area_types_[area_i] == AreaType::DELETE_STATIC) {
  //           if (point_i >= border) {
  //             within = true;
  //           }
  //         }
  //       }
  //     }
  //     area_i++;
  //   }  // for areas
  //   if (!within) {
  //     output->points.emplace_back(pcl::PointXYZ(point.x, point.y, point.z));
  //   }
  //   point_i++;
  }

  std::cout << ">>> >>> 各点群がarea_polygonsの範囲内に存在するかを求める操作 [us]:         "
            << t_local_2_.lap() << std::endl;
}

bool MapAreaFilterComponent::filter_objects_by_area(PredictedObjects & out_objects)
{
  if (!objects_ptr_) {
    return false;
  }

  if (objects_ptr_.get() == nullptr) {
    return false;
  }

  PredictedObjects in_objects;
  in_objects = *objects_ptr_.get();
  out_objects.header = in_objects.header;

  for (const auto & object : in_objects.objects) {
    const auto pos = object.kinematics.initial_pose_with_covariance.pose.position;

    bool within = false;
    for (std::size_t area_i = 0, size = area_polygons_.size(); area_i < size; ++area_i) {
      if (area_types_[area_i] != AreaType::DELETE_OBJECT) continue;

      if ((boost::geometry::within(PointXY(pos.x, pos.y), area_polygons_[area_i]))) {
        within = true;
      }
    }
    if (!within) {
      out_objects.objects.emplace_back(object);
    }
  }

  objects_ptr_ = boost::none;

  return true;
}

void MapAreaFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  t_local_.lap();
  t_global_.lap();
  std::cout << "======================================================================="
            << std::endl;

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
  if (!transform_pointcloud(*input, *tf2_, map_frame_, map_frame_cloud)) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1,
      "Cannot transform cloud to " << map_frame_ << " not filtering.");
    output = *input;
    return;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_frame_cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(map_frame_cloud, *map_frame_cloud_pcl);
  std::cout << ">>> フィルタリングする点群をbase_link座標系からmap座標系へ座標変換 [us]:    "
            << t_local_.lap() << std::endl;

  // transform known object cloud to map frame
  sensor_msgs::msg::PointCloud2 objects_frame_cloud;
  if (objects_cloud_ptr_ != nullptr) {
    if (!transform_pointcloud(*objects_cloud_ptr_, *tf2_, map_frame_, objects_frame_cloud)) {
      RCLCPP_ERROR_STREAM_THROTTLE(
        this->get_logger(), *this->get_clock(), 1,
        "Cannot transform objects cloud to " << map_frame_
                                             << " not filtering. Not including object cloud");
    }
  }
  std::cout << ">>> 物体検出で抽出した点群をbase_link座標系からmap座標系へ座標変換 [us]:    "
            << t_local_.lap() << std::endl;

  /* ========================== */

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects_cloud_pcl_temp(
  //   new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::fromROSMsg(objects_frame_cloud, *objects_cloud_pcl_temp);

  // // if known object cloud contains points add to the filtered cloud
  // if (!objects_cloud_pcl_temp->points.empty()) {
  //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects_cloud_pcl(new
  //   pcl::PointCloud<pcl::PointXYZRGB>); for (const auto & point : objects_cloud_pcl_temp->points)
  //   {
  //     objects_cloud_pcl->emplace_back(pcl::PointXYZRGB(point.x, point.y, point.z, 255, 255,
  //     255));
  //   }
  //   *map_frame_cloud_pcl += *objects_cloud_pcl;
  // }

  /* ========================== */

  pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(objects_frame_cloud, *objects_cloud_pcl);

  // if known object cloud contains points add to the filtered cloud
  const std::size_t border = map_frame_cloud_pcl->size();
  if (!objects_cloud_pcl->points.empty()) *map_frame_cloud_pcl += *objects_cloud_pcl;

  /* ========================== */

  std::cout << ">>> 物体検出で抽出した点群をpcl::PointCloud<pcl::PointXYZRGB>型へ変換 [us]: "
            << t_local_.lap() << std::endl;

  // create filtered cloud container
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_frame_cloud_pcl_filtered(
    new pcl::PointCloud<pcl::PointXYZ>);
  filter_points_by_area(map_frame_cloud_pcl, map_frame_cloud_pcl_filtered, border);
  std::cout << ">>> CSVで指定したエリアの点群をフィルタリングする処理 [us]:                 "
            << t_local_.lap() << std::endl;

  pcl::toROSMsg(*map_frame_cloud_pcl_filtered, output);
  output.header = input->header;
  output.header.frame_id = map_frame_;

  if (!transform_pointcloud(output, *tf2_, input->header.frame_id, output)) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1,
      "Cannot transform back cloud to " << input->header.frame_id << " not filtering.");
    output = *input;
    return;
  }
  output.header = input->header;
  std::cout << ">>> フィルタリングした点群をmap座標系からbase_link座標系へ座標変換 [us]:    "
            << t_local_.lap() << std::endl;

  PredictedObjects out_objects;
  if (filter_objects_by_area(out_objects)) filtered_objects_pub_->publish(out_objects);
  std::cout << ">>> CSVで指定したエリアの物体検出結果をフィルタリングする処理 [us]:         "
            << t_local_.lap() << std::endl;
  std::cout << "-----------------------------------------------------------------------"
            << std::endl;
  std::cout << ">>> 全体の処理にかかった時間 [us]:                                          "
            << t_global_.lap() << std::endl;
  std::cout << ">>> これまでの処理時間の平均 (iteration: " << t_global_.iteration()
            << ") [ms]: " << t_global_.average() << std::endl;
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
      this->get_logger(), "Setting new area check distance to: %.2lf.", area_distance_check_);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::MapAreaFilterComponent)
