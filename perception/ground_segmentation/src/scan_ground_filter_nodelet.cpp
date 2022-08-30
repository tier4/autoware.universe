// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#include "ground_segmentation/scan_ground_filter_nodelet.hpp"

#include <pcl_ros/transforms.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/normalization.hpp>
#include <tier4_autoware_utils/math/unit_conversion.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <memory>
#include <string>
#include <vector>

namespace ground_segmentation
{
using pointcloud_preprocessor::get_param;
using tier4_autoware_utils::calcDistance3d;
using tier4_autoware_utils::deg2rad;
using tier4_autoware_utils::normalizeDegree;
using tier4_autoware_utils::normalizeRadian;
using vehicle_info_util::VehicleInfoUtil;

ScanGroundFilterComponent::ScanGroundFilterComponent(const rclcpp::NodeOptions & options)
: Filter("ScanGroundFilter", options)
{
  // set initial parameters
  {
    base_link_shift_ = static_cast<float>(declare_parameter("base_link_shift", 0.0));
    non_ground_height_threshold_ =
      static_cast<float>(declare_parameter("non_ground_height_threshold", 0.15));
    num_prev_grid_slope_refer_ =
      static_cast<int>(declare_parameter("num_prev_grid_slope_refer", 2));
    division_mode_distance_threshold_ =
      static_cast<float>(declare_parameter("division_mode_distance_threshold", 5.0));
    max_height_detection_range_ =
      static_cast<float>(declare_parameter("max_height_detection_range", 2.5));
    min_height_detection_range_ =
      static_cast<float>(declare_parameter("min_height_detection_range", 0.0));

    vertical_grid_resolution_distance_ =
      static_cast<float>(declare_parameter("vertical_grid_resolution_distance", 0.1));
    num_gnd_grids_reference_ = static_cast<int>(declare_parameter("num_gnd_grids_reference", 10));
    base_frame_ = declare_parameter("base_frame", "base_link");
    global_slope_max_angle_rad_ = deg2rad(declare_parameter("global_slope_max_angle_deg", 8.0));

    radial_divider_angle_rad_ = deg2rad(declare_parameter("radial_divider_angle_deg", 1.0));
    split_points_distance_tolerance_ = declare_parameter("split_points_distance_tolerance", 0.2);
    split_height_distance_ = declare_parameter("split_height_distance", 0.2);
    use_virtual_ground_point_ = declare_parameter("use_virtual_ground_point", true);
    radial_dividers_num_ = std::ceil(2.0 * M_PI / radial_divider_angle_rad_);
    vehicle_info_ = VehicleInfoUtil(*this).getVehicleInfo();

    division_mode_grid_id_threshold =
      division_mode_distance_threshold_ /
      vertical_grid_resolution_distance_;  // changing the mode of grid division
    division_mode_angle_rad_threshold =
      std::atan2(division_mode_distance_threshold_, virtual_lidar_height);
    vertical_grid_resolution_angle_rad_ =
      std::atan2(division_mode_distance_threshold_, virtual_lidar_height) -
      std::atan2(
        division_mode_distance_threshold_ - vertical_grid_resolution_distance_,
        virtual_lidar_height);
  }
  ground_pcl_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "debug/ground_pointcloud", rclcpp::SensorDataQoS());
  unknown_pcl_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "debug/unknown_pointcloud", rclcpp::SensorDataQoS());
  under_ground_pcl_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "debug/underground_pointcloud", rclcpp::SensorDataQoS());
  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&ScanGroundFilterComponent::onParameter, this, _1));
}

bool ScanGroundFilterComponent::transformPointCloud(
  const std::string & in_target_frame, const PointCloud2ConstPtr & in_cloud_ptr,
  const PointCloud2::SharedPtr & out_cloud_ptr)
{
  if (in_target_frame == in_cloud_ptr->header.frame_id) {
    *out_cloud_ptr = *in_cloud_ptr;
    return true;
  }

  geometry_msgs::msg::TransformStamped transform_stamped;
  try {
    transform_stamped = tf_buffer_.lookupTransform(
      in_target_frame, in_cloud_ptr->header.frame_id, in_cloud_ptr->header.stamp,
      rclcpp::Duration::from_seconds(1.0));
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(get_logger(), ex.what());
    return false;
  }
  Eigen::Matrix4f mat = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
  pcl_ros::transformPointCloud(mat, *in_cloud_ptr, *out_cloud_ptr);
  out_cloud_ptr->header.frame_id = in_target_frame;
  return true;
}

void ScanGroundFilterComponent::convertPointcloud(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
  std::vector<PointCloudRefVector> & out_radial_ordered_points)
{
  out_radial_ordered_points.resize(radial_dividers_num_);
  PointRef current_point;
  uint16_t back_steps_num = 1;

  vertical_grid_resolution_angle_rad_ =
    normalizeRadian(std::atan2(
      division_mode_distance_threshold_ + vertical_grid_resolution_distance_,
      virtual_lidar_height)) -
    normalizeRadian(std::atan2(division_mode_distance_threshold_, virtual_lidar_height));
  for (size_t i = 0; i < in_cloud->points.size(); ++i) {
    auto x{in_cloud->points[i].x - base_link_shift_};  // base on front wheel center
    // auto y{in_cloud->points[i].y};
    auto radius{static_cast<float>(std::hypot(x, in_cloud->points[i].y))};
    auto theta{normalizeRadian(std::atan2(x, in_cloud->points[i].y), 0.0)};

    // divide by angle
    auto gama{normalizeRadian(std::atan2(radius, virtual_lidar_height), 0.0f)};
    auto radial_div{
      static_cast<size_t>(std::floor(normalizeDegree(theta / radial_divider_angle_rad_, 0.0)))};
    uint16_t grid_id = 0;
    float grid_radius = 0.0f;
    if (radius <= division_mode_distance_threshold_) {
      grid_id = static_cast<uint16_t>(radius / vertical_grid_resolution_distance_);
      grid_radius =
        static_cast<float>((grid_id - back_steps_num) * vertical_grid_resolution_distance_);
    } else {
      grid_id = division_mode_grid_id_threshold +
                (gama - division_mode_angle_rad_threshold) / vertical_grid_resolution_angle_rad_;
      if (grid_id <= division_mode_grid_id_threshold + back_steps_num) {
        grid_radius =
          static_cast<float>((grid_id - back_steps_num) * vertical_grid_resolution_distance_);
      } else {
        grid_radius =
          std::tan(
            gama - static_cast<float>(back_steps_num) * vertical_grid_resolution_angle_rad_) *
          virtual_lidar_height;
      }
    }
    current_point.grid_id = grid_id;
    current_point.grid_radius = grid_radius;
    current_point.radius = radius;
    current_point.theta = theta;
    current_point.radial_div = radial_div;
    current_point.point_state = PointLabel::INIT;
    current_point.orig_index = i;
    current_point.orig_point = &in_cloud->points[i];

    // radial divisions
    out_radial_ordered_points[radial_div].emplace_back(current_point);
  }

  // sort by distance
  for (size_t i = 0; i < radial_dividers_num_; ++i) {
    std::sort(
      out_radial_ordered_points[i].begin(), out_radial_ordered_points[i].end(),
      [](const PointRef & a, const PointRef & b) { return a.radius < b.radius; });
  }
}

void ScanGroundFilterComponent::calcVirtualGroundOrigin(pcl::PointXYZ & point)
{
  point.x = vehicle_info_.wheel_base_m;
  point.y = 0;
  point.z = 0;
}
void ScanGroundFilterComponent::classifyPointCloud(
  std::vector<PointCloudRefVector> & in_radial_ordered_clouds,
  pcl::PointIndices & out_no_ground_indices, pcl::PointIndices & out_ground_indices,
  pcl::PointIndices & out_unknown_indices, pcl::PointIndices & out_underground_indices)
{
  out_no_ground_indices.indices.clear();
  out_ground_indices.indices.clear();
  out_unknown_indices.indices.clear();
  out_underground_indices.indices.clear();
  std::cout << " predict_curr_gnd_heigh : ";
  for (size_t i = 0; i < in_radial_ordered_clouds.size(); i++) {
    // initialize the gnd_point(0,0,0)
    // float prev_gnd_height = 0.0f;
    // float prev_gn_grid_radius = 0.0f;  // origin of previous gnd point cloud
    // float approx_prev_local_gnd_slope_rad = 0.0f;
    // uint16_t prev_gnd_grid_id = 0;
    PointsCentroid ground_cluster, non_ground_cluster;
    ground_cluster.initialize();
    non_ground_cluster.initialize();
    std::vector<float> prev_gnd_grid_aver_height_list;
    std::vector<float> prev_gnd_grid_radius_list;
    std::vector<uint16_t> prev_gnd_grid_id_list;
    std::vector<float> prev_gnd_grid_max_height_list;

    // check empty ray:
    if (in_radial_ordered_clouds[i].size() == 0) {
      continue;
    }

    // check the first point in ray
    auto * p = &in_radial_ordered_clouds[i][0];  // in_radial_ordered_clouds[i].front();
    PointRef * prev_p;
    prev_p = &in_radial_ordered_clouds[i][0];  // for checking the distance to prev point
    // float distance_to_prev = calcDistance3d(p->orig_point,prev_p->orig_point);

    for (int j = p->grid_id - 1 - num_gnd_grids_reference_; j < p->grid_id; j++) {
      prev_gnd_grid_aver_height_list.push_back(0.0f);
      prev_gnd_grid_radius_list.push_back(j * vertical_grid_resolution_distance_);
      prev_gnd_grid_id_list.push_back(j);
      prev_gnd_grid_max_height_list.push_back(0.0f);
    }
    bool initilized_flg = false;

    float dist_to_front = vehicle_info_.front_overhang_m + vehicle_info_.wheel_base_m / 2.0f; 
    for (size_t j = 0; j < in_radial_ordered_clouds[i].size(); j++) {
      p = &in_radial_ordered_clouds[i][j];
      if (p->grid_id > prev_p->grid_id) {
        // check if the prev grid have ground point cloud:
        if (ground_cluster.getAverageRadius() > 0.0) {
          prev_gnd_grid_aver_height_list.push_back(ground_cluster.getAverageHeight());
          prev_gnd_grid_max_height_list.push_back(ground_cluster.getMaxheight());
          prev_gnd_grid_id_list.push_back(prev_p->grid_id);
          prev_gnd_grid_radius_list.push_back(ground_cluster.getAverageRadius());
          ground_cluster.initialize();
        }
      }
      float global_slope_curr_p = 0.0f;
      // float local_slope_curr_p = 0.0f;
      global_slope_curr_p = std::atan2(p->orig_point->z, p->radius);
      // chec if the current point cloud is new grid:
      // local_slope_curr_p = std::atan2(
      //   p->orig_point->z - prev_gnd_grid_aver_height_list.back(),
      //   p->radius - prev_gnd_grid_radius_list.back());
      if ((initilized_flg == false) || (p->radius < dist_to_front + vertical_grid_resolution_distance_)) {
        // add condition for suddent slope, but it lose ability to detect 20cm object near by
        if (
          (p->orig_point->z > non_ground_height_threshold_ + std::tan(DEG2RAD(5.0)) * (dist_to_front + vertical_grid_resolution_distance_)) ) {
          out_no_ground_indices.indices.push_back(p->orig_index);
          prev_p = p;
          initilized_flg = true;

        } else if (
          (abs(p->orig_point->z) < non_ground_height_threshold_ + std::tan(DEG2RAD(5.0)) * (dist_to_front + vertical_grid_resolution_distance_)))  {
          out_ground_indices.indices.push_back(p->orig_index);
          ground_cluster.addPoint(p->radius, p->orig_point->z);
          prev_p = p;
          initilized_flg = true;
        } else {
          // nothing
        }
      } else {
        if (global_slope_curr_p > global_slope_max_angle_rad_){
          out_no_ground_indices.indices.push_back(p->orig_index);
        } else{

          float predict_next_gnd_heigh = 0.0f;
          float app_curr_gnd_slope = 0.0f;
          float mid_ref_gnd_height = 0.0f;
          float mid_ref_gnd_radius = 0.0f;
          for (int i_ref = num_gnd_grids_reference_ + 1; i_ref > 1; i_ref--){
            mid_ref_gnd_height += *(prev_gnd_grid_aver_height_list.end() - i_ref);
            mid_ref_gnd_radius += *(prev_gnd_grid_radius_list.end() - i_ref);
          }
          mid_ref_gnd_height /= static_cast<float>(num_gnd_grids_reference_);
          mid_ref_gnd_radius /= static_cast<float>(num_gnd_grids_reference_);
          app_curr_gnd_slope = std::atan2(prev_gnd_grid_aver_height_list.back() - mid_ref_gnd_height,
                                          prev_gnd_grid_radius_list.back() - mid_ref_gnd_radius);
          
          predict_next_gnd_heigh = std::tan(app_curr_gnd_slope) * (p->radius - mid_ref_gnd_radius) + mid_ref_gnd_height;
          float gnd_z_threshold = std::tan(DEG2RAD(5.0)) * (p->radius - prev_gnd_grid_radius_list.back());

          if ((p->grid_id < *(prev_gnd_grid_id_list.end() - num_gnd_grids_reference_) + num_gnd_grids_reference_ + 3 ) ||  
            (p->radius - prev_gnd_grid_radius_list.back() <  num_gnd_grids_reference_ * vertical_grid_resolution_distance_))
          {
            //checking by last some gnd grids

            if((p->orig_point->z - predict_next_gnd_heigh) <= non_ground_height_threshold_ + gnd_z_threshold &&
                p->orig_point->z - predict_next_gnd_heigh >= -gnd_z_threshold){
              out_ground_indices.indices.push_back(p->orig_index);
              if (abs(p->orig_point->z - predict_next_gnd_heigh) <  gnd_z_threshold)
              {ground_cluster.addPoint(p->radius,p->orig_point->z);}
            }else if (p->orig_point->z - predict_next_gnd_heigh > non_ground_height_threshold_ + gnd_z_threshold){
              out_no_ground_indices.indices.push_back(p->orig_index);
            }

          }else {
            //checking by reference only the last gnd grid
            float local_slope_p = std::atan2(p->orig_point->z - prev_gnd_grid_aver_height_list.back(), 
                                              p->radius - prev_gnd_grid_radius_list.back());
            if ((abs(local_slope_p - app_curr_gnd_slope) < global_slope_max_angle_rad_)){
              out_ground_indices.indices.push_back(p->orig_index);
              ground_cluster.addPoint(p->radius, p->orig_point->z);
            }
            else if ( local_slope_p - app_curr_gnd_slope > global_slope_max_angle_rad_){
              out_no_ground_indices.indices.push_back(p->orig_index);
            }
          }
        }
        




        // if (global_slope_curr_p >= global_slope_max_angle_rad_) {
        //   out_no_ground_indices.indices.push_back(p->orig_index);
        // } else {
        //   float reference_aveg_gnd_height = 0.0f;
        //   float reference_aveg_gnd_radius = 0.0f;
        //   float predict_curr_gnd_heigh = 0.0f;
        //   float predict_curr_gnd_heigh_max = 0.0f;
        //   float predict_curr_local_slope = 0.0f;
        //   float ref_origin_height =
        //     *(prev_gnd_grid_aver_height_list.end() - num_gnd_grids_reference_);
        //   float ref_origin_height_max =
        //     *(prev_gnd_grid_max_height_list.end() - num_gnd_grids_reference_);
        //   float ref_origin_radius = *(prev_gnd_grid_radius_list.end() - num_gnd_grids_reference_);
        //   for (int ind = num_gnd_grids_reference_ - 1; ind > 0; ind--) {
        //     reference_aveg_gnd_height += *(prev_gnd_grid_aver_height_list.end() - ind);
        //     reference_aveg_gnd_radius += *(prev_gnd_grid_radius_list.end() - ind);
        //     float ind_height = *(prev_gnd_grid_aver_height_list.end() - ind) - ref_origin_height;
        //     float ind_height_max =
        //       *(prev_gnd_grid_max_height_list.end() - ind) - ref_origin_height_max;
        //     float ind_radius = *(prev_gnd_grid_radius_list.end() - ind) - ref_origin_radius;
        //     predict_curr_gnd_heigh += ind_height / ind_radius;
        //     predict_curr_gnd_heigh_max += ind_height_max / ind_radius;
        //   }
        //   reference_aveg_gnd_height /= (num_gnd_grids_reference_ - 1);
        //   reference_aveg_gnd_radius /= (num_gnd_grids_reference_ - 1);
        //   local_slope_curr_p = std::atan(
        //     (p->orig_point->z - reference_aveg_gnd_height) /
        //     (p->radius - reference_aveg_gnd_radius));

        //   predict_curr_gnd_heigh *=
        //     (p->radius - ref_origin_radius) / (num_gnd_grids_reference_ - 1);
        //   predict_curr_gnd_heigh += ref_origin_height;
        //   predict_curr_gnd_heigh_max *=
        //     (p->radius - ref_origin_radius) / (num_gnd_grids_reference_ - 1);
        //   predict_curr_gnd_heigh_max += ref_origin_height_max;

        //   predict_curr_local_slope = std::atan(
        //     (predict_curr_gnd_heigh - reference_aveg_gnd_height) /
        //     (p->radius - reference_aveg_gnd_radius));
        //   // local slope reference to prev aproximation
        //   local_slope_max_angle_rad_ = std::atan(
        //     (0.15 + std::tan(deg2rad(5.0)) * vertical_grid_resolution_distance_) /
        //     (vertical_grid_resolution_distance_ * 2.0f));
        //   float posible_next_gnd_change =
        //     (p->radius - reference_aveg_gnd_radius) * std::tan(deg2rad(5.0));
        //   // local_slope_max_angle_rad_ =
        //   // std::max(local_slope_max_angle_rad_,global_slope_max_angle_rad_);
        //   if (i == 300 && p->grid_id > prev_p->grid_id) {
        //     std::cout << " : " << p->radius << " : " << ref_origin_radius << " : "
        //               << ref_origin_height << " : " << predict_curr_gnd_heigh << " : "
        //               << RAD2DEG(predict_curr_local_slope) << " : " << RAD2DEG(local_slope_curr_p)
        //               << " : " << RAD2DEG(local_slope_max_angle_rad_);
        //   }
        //   if (
        //     p->grid_id <= *(prev_gnd_grid_id_list.end() - num_gnd_grids_reference_) +
        //                     num_gnd_grids_reference_ + 3) {
        //     if (
        //       (p->orig_point->z - predict_curr_gnd_heigh <
        //        non_ground_height_threshold_ + posible_next_gnd_change) ||
        //       (p->orig_point->z - predict_curr_gnd_heigh > -posible_next_gnd_change)) {
        //       ground_cluster.addPoint(p->radius, p->orig_point->z);
        //       out_ground_indices.indices.push_back(p->orig_index);
        //     } else if ((p->orig_point->z - predict_curr_gnd_heigh >
        //                 non_ground_height_threshold_ + posible_next_gnd_change)) {
        //       out_no_ground_indices.indices.push_back(p->orig_index);
        //     } else if (
        //       ((p->orig_point->z - predict_curr_gnd_heigh) <= -non_ground_height_threshold_) &&
        //       (local_slope_curr_p <= -local_slope_max_angle_rad_)) {
        //       out_underground_indices.indices.push_back(p->orig_index);
        //     } else {
        //       out_unknown_indices.indices.push_back(p->orig_index);
        //     }
        //   } else {
        //     local_slope_curr_p = std::atan2(
        //       p->orig_point->z - prev_gnd_grid_aver_height_list.back(),
        //       p->radius - prev_gnd_grid_radius_list.back());
        //     if (abs(local_slope_curr_p) < global_slope_max_angle_rad_) {
        //       ground_cluster.addPoint(p->radius, p->orig_point->z);
        //       out_ground_indices.indices.push_back(p->orig_index);
        //     } else {
        //       out_no_ground_indices.indices.push_back(p->orig_index);
        //     }
        //   }
        // }
      }
      prev_p = p;
    }
    // estimate the height from predicted current ground and compare with threshold

    // estimate the local slope to previous virtual gnd and compare with
  }
  std::cout << "    \n";
}

void ScanGroundFilterComponent::classifyPointCloud(
  std::vector<PointCloudRefVector> & in_radial_ordered_clouds,
  pcl::PointIndices & out_no_ground_indices)
{
  out_no_ground_indices.indices.clear();

  const pcl::PointXYZ init_ground_point(0, 0, 0);
  pcl::PointXYZ virtual_ground_point(0, 0, 0);
  calcVirtualGroundOrigin(virtual_ground_point);

  // point classification algorithm
  // sweep through each radial division
  for (size_t i = 0; i < in_radial_ordered_clouds.size(); i++) {
    float prev_gnd_radius = 0.0f;
    float prev_gnd_slope = 0.0f;
    float points_distance = 0.0f;
    PointsCentroid ground_cluster, non_ground_cluster;
    float local_slope = 0.0f;
    PointLabel prev_point_label = PointLabel::INIT;
    pcl::PointXYZ prev_gnd_point(0, 0, 0);
    // loop through each point in the radial div
    for (size_t j = 0; j < in_radial_ordered_clouds[i].size(); j++) {
      const float global_slope_max_angle = global_slope_max_angle_rad_;
      const float local_slope_max_angle = local_slope_max_angle_rad_;
      auto * p = &in_radial_ordered_clouds[i][j];
      auto * p_prev = &in_radial_ordered_clouds[i][j - 1];

      if (j == 0) {
        bool is_front_side = (p->orig_point->x > virtual_ground_point.x);
        if (use_virtual_ground_point_ && is_front_side) {
          prev_gnd_point = virtual_ground_point;
        } else {
          prev_gnd_point = init_ground_point;
        }
        prev_gnd_radius = std::hypot(prev_gnd_point.x, prev_gnd_point.y);
        prev_gnd_slope = 0.0f;
        ground_cluster.initialize();
        non_ground_cluster.initialize();
        points_distance = calcDistance3d(*p->orig_point, prev_gnd_point);
      } else {
        points_distance = calcDistance3d(*p->orig_point, *p_prev->orig_point);
      }

      float radius_distance_from_gnd = p->radius - prev_gnd_radius;
      float height_from_gnd = p->orig_point->z - prev_gnd_point.z;
      float height_from_obj = p->orig_point->z - non_ground_cluster.getAverageHeight();
      bool calculate_slope = false;
      bool is_point_close_to_prev =
        (points_distance <
         (p->radius * radial_divider_angle_rad_ + split_points_distance_tolerance_));

      float global_slope = std::atan2(p->orig_point->z, p->radius);
      // check points which is far enough from previous point
      if (global_slope > global_slope_max_angle) {
        p->point_state = PointLabel::NON_GROUND;
        calculate_slope = false;
      } else if (
        (prev_point_label == PointLabel::NON_GROUND) &&
        (abs(height_from_obj) >= split_height_distance_)) {
        calculate_slope = true;
      } else if (is_point_close_to_prev && abs(height_from_gnd) < split_height_distance_) {
        // close to the previous point, set point follow label
        p->point_state = PointLabel::POINT_FOLLOW;
        calculate_slope = false;
      } else {
        calculate_slope = true;
      }
      if (is_point_close_to_prev) {
        height_from_gnd = p->orig_point->z - ground_cluster.getAverageHeight();
        radius_distance_from_gnd = p->radius - ground_cluster.getAverageRadius();
      }
      if (calculate_slope) {
        // far from the previous point
        local_slope = std::atan2(height_from_gnd, radius_distance_from_gnd);
        if (local_slope - prev_gnd_slope > local_slope_max_angle) {
          // the point is outside of the local slope threshold
          p->point_state = PointLabel::NON_GROUND;
        } else {
          p->point_state = PointLabel::GROUND;
        }
      }

      if (p->point_state == PointLabel::GROUND) {
        ground_cluster.initialize();
        non_ground_cluster.initialize();
      }
      if (p->point_state == PointLabel::NON_GROUND) {
        out_no_ground_indices.indices.push_back(p->orig_index);
      } else if (  // NOLINT
        (prev_point_label == PointLabel::NON_GROUND) &&
        (p->point_state == PointLabel::POINT_FOLLOW)) {
        p->point_state = PointLabel::NON_GROUND;
        out_no_ground_indices.indices.push_back(p->orig_index);
      } else if (  // NOLINT
        (prev_point_label == PointLabel::GROUND) && (p->point_state == PointLabel::POINT_FOLLOW)) {
        p->point_state = PointLabel::GROUND;
      } else {
      }

      // update the ground state
      prev_point_label = p->point_state;
      if (p->point_state == PointLabel::GROUND) {
        prev_gnd_radius = p->radius;
        prev_gnd_point = pcl::PointXYZ(p->orig_point->x, p->orig_point->y, p->orig_point->z);
        ground_cluster.addPoint(p->radius, p->orig_point->z);
        prev_gnd_slope = ground_cluster.getAverageSlope();
      }
      // update the non ground state
      if (p->point_state == PointLabel::NON_GROUND) {
        non_ground_cluster.addPoint(p->radius, p->orig_point->z);
      }
    }
  }
}

void ScanGroundFilterComponent::extractObjectPoints(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, const pcl::PointIndices & in_indices,
  pcl::PointCloud<pcl::PointXYZ>::Ptr out_object_cloud_ptr)
{
  for (const auto & i : in_indices.indices) {
    out_object_cloud_ptr->points.emplace_back(in_cloud_ptr->points[i]);
  }
}

void ScanGroundFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  auto input_transformed_ptr = std::make_shared<PointCloud2>();
  bool succeeded = transformPointCloud(base_frame_, input, input_transformed_ptr);
  sensor_frame_ = input->header.frame_id;
  if (!succeeded) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "Failed transform from " << base_frame_ << " to " << input->header.frame_id);
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*input_transformed_ptr, *current_sensor_cloud_ptr);

  std::vector<PointCloudRefVector> radial_ordered_points;

  convertPointcloud(current_sensor_cloud_ptr, radial_ordered_points);

  pcl::PointIndices no_ground_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  no_ground_cloud_ptr->points.reserve(current_sensor_cloud_ptr->points.size());

  pcl::PointIndices ground_pcl_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  ground_cloud_ptr->points.reserve(current_sensor_cloud_ptr->points.size());

  pcl::PointIndices unknown_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr unknown_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  unknown_cloud_ptr->points.reserve(current_sensor_cloud_ptr->points.size());

  pcl::PointIndices underground_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr underground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  underground_cloud_ptr->points.reserve(current_sensor_cloud_ptr->points.size());

  // classifyPointCloud(radial_ordered_points, no_ground_indices);
  classifyPointCloud(
    radial_ordered_points, no_ground_indices, ground_pcl_indices, unknown_indices,
    underground_indices);

  extractObjectPoints(current_sensor_cloud_ptr, no_ground_indices, no_ground_cloud_ptr);
  extractObjectPoints(current_sensor_cloud_ptr, ground_pcl_indices, ground_cloud_ptr);
  extractObjectPoints(current_sensor_cloud_ptr, unknown_indices, unknown_cloud_ptr);
  extractObjectPoints(current_sensor_cloud_ptr, underground_indices, underground_cloud_ptr);

  sensor_msgs::msg::PointCloud2 ground_pcl_msg;
  pcl::toROSMsg(*ground_cloud_ptr, ground_pcl_msg);
  ground_pcl_msg.header = input->header;
  ground_pcl_pub_->publish(ground_pcl_msg);

  sensor_msgs::msg::PointCloud2 unknown_pcl_msg;
  pcl::toROSMsg(*unknown_cloud_ptr, unknown_pcl_msg);
  unknown_pcl_msg.header = input->header;
  unknown_pcl_pub_->publish(unknown_pcl_msg);

  sensor_msgs::msg::PointCloud2 underground_msg;
  pcl::toROSMsg(*underground_cloud_ptr, underground_msg);
  underground_msg.header = input->header;
  under_ground_pcl_pub_->publish(underground_msg);

  auto no_ground_cloud_msg_ptr = std::make_shared<PointCloud2>();
  pcl::toROSMsg(*no_ground_cloud_ptr, *no_ground_cloud_msg_ptr);

  no_ground_cloud_msg_ptr->header.stamp = input->header.stamp;
  no_ground_cloud_msg_ptr->header.frame_id = base_frame_;
  output = *no_ground_cloud_msg_ptr;
}

rcl_interfaces::msg::SetParametersResult ScanGroundFilterComponent::onParameter(
  const std::vector<rclcpp::Parameter> & p)
{
  if (get_param(p, "base_frame", base_frame_)) {
    RCLCPP_DEBUG_STREAM(get_logger(), "Setting base_frame to: " << base_frame_);
  }
  double global_slope_max_angle_deg{get_parameter("global_slope_max_angle_deg").as_double()};
  if (get_param(p, "global_slope_max_angle_deg", global_slope_max_angle_deg)) {
    global_slope_max_angle_rad_ = deg2rad(global_slope_max_angle_deg);
    RCLCPP_DEBUG(
      get_logger(), "Setting global_slope_max_angle_rad to: %f.", global_slope_max_angle_rad_);
  }
  double local_slope_max_angle_deg{get_parameter("local_slope_max_angle_deg").as_double()};
  if (get_param(p, "local_slope_max_angle_deg", local_slope_max_angle_deg)) {
    local_slope_max_angle_rad_ = deg2rad(local_slope_max_angle_deg);
    RCLCPP_DEBUG(
      get_logger(), "Setting local_slope_max_angle_rad to: %f.", local_slope_max_angle_rad_);
  }
  double radial_divider_angle_deg{get_parameter("radial_divider_angle_deg").as_double()};
  if (get_param(p, "radial_divider_angle_deg", radial_divider_angle_deg)) {
    radial_divider_angle_rad_ = deg2rad(radial_divider_angle_deg);
    radial_dividers_num_ = std::ceil(2.0 * M_PI / radial_divider_angle_rad_);
    RCLCPP_DEBUG(
      get_logger(), "Setting radial_divider_angle_rad to: %f.", radial_divider_angle_rad_);
    RCLCPP_DEBUG(get_logger(), "Setting radial_dividers_num to: %zu.", radial_dividers_num_);
  }
  if (get_param(p, "split_points_distance_tolerance", split_points_distance_tolerance_)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting split_points_distance_tolerance to: %f.",
      split_points_distance_tolerance_);
  }
  if (get_param(p, "split_height_distance", split_height_distance_)) {
    RCLCPP_DEBUG(get_logger(), "Setting split_height_distance to: %f.", split_height_distance_);
  }
  if (get_param(p, "use_virtual_ground_point", use_virtual_ground_point_)) {
    RCLCPP_DEBUG_STREAM(
      get_logger(),
      "Setting use_virtual_ground_point to: " << std::boolalpha << use_virtual_ground_point_);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

}  // namespace ground_segmentation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ground_segmentation::ScanGroundFilterComponent)
