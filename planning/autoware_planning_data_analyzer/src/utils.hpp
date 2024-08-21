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

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "autoware/motion_utils/trajectory/interpolation.hpp"
#include "autoware/motion_utils/trajectory/trajectory.hpp"
#include "autoware_frenet_planner/frenet_planner.hpp"
#include "autoware_path_sampler/prepare_inputs.hpp"
#include "autoware_path_sampler/utils/trajectory_utils.hpp"
#include "data_structs.hpp"
#include "type_alias.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::behavior_analyzer::utils
{
Point vector2point(const geometry_msgs::msg::Vector3 & v)
{
  return autoware::universe_utils::createPoint(v.x, v.y, v.z);
}

tf2::Vector3 from_msg(const Point & p)
{
  return tf2::Vector3(p.x, p.y, p.z);
}

tf2::Vector3 get_velocity_in_world_coordinate(const PredictedObjectKinematics & kinematics)
{
  const auto pose = kinematics.initial_pose_with_covariance.pose;
  const auto v_local = kinematics.initial_twist_with_covariance.twist.linear;
  const auto v_world = autoware::universe_utils::transformPoint(vector2point(v_local), pose);

  return from_msg(v_world) - from_msg(pose.position);
}

tf2::Vector3 get_velocity_in_world_coordinate(const Odometry & odometry)
{
  const auto pose = odometry.pose.pose;
  const auto v_local = odometry.twist.twist.linear;
  const auto v_world = autoware::universe_utils::transformPoint(vector2point(v_local), pose);

  return from_msg(v_world) - from_msg(pose.position);
}

tf2::Vector3 get_velocity_in_world_coordinate(const TrajectoryPoint & point)
{
  const auto pose = point.pose;
  const auto v_local =
    geometry_msgs::build<Vector3>().x(point.longitudinal_velocity_mps).y(0.0).z(0.0);
  const auto v_world = autoware::universe_utils::transformPoint(vector2point(v_local), pose);

  return from_msg(v_world) - from_msg(pose.position);
}

double time_to_collision(
  const PredictedObjects & objects, const Pose & p_ego, const tf2::Vector3 & v_ego)
{
  if (objects.objects.empty()) {
    return std::numeric_limits<double>::max();
  }

  std::vector<double> time_to_collisions(objects.objects.size());

  for (const auto & object : objects.objects) {
    const auto p_object = object.kinematics.initial_pose_with_covariance.pose;
    const auto v_ego2object =
      autoware::universe_utils::point2tfVector(p_ego.position, p_object.position);

    const auto v_object = get_velocity_in_world_coordinate(object.kinematics);
    const auto v_relative = tf2::tf2Dot(v_ego2object.normalized(), v_ego) -
                            tf2::tf2Dot(v_ego2object.normalized(), v_object);

    time_to_collisions.push_back(v_ego2object.length() / v_relative);
  }

  const auto itr = std::remove_if(
    time_to_collisions.begin(), time_to_collisions.end(),
    [](const auto & value) { return value < 1e-3; });
  time_to_collisions.erase(itr, time_to_collisions.end());

  std::sort(time_to_collisions.begin(), time_to_collisions.end());

  return time_to_collisions.front();
}

auto convertToTrajectoryPoints(
  const autoware::sampler_common::Trajectory & trajectory,
  const vehicle_info_utils::VehicleInfo & vehicle_info) -> std::vector<TrajectoryPoint>
{
  std::vector<TrajectoryPoint> traj_points;
  for (auto i = 0UL; i < trajectory.points.size(); ++i) {
    TrajectoryPoint p;
    p.pose.position.x = trajectory.points[i].x();
    p.pose.position.y = trajectory.points[i].y();
    auto quat = tf2::Quaternion();
    quat.setRPY(0.0, 0.0, trajectory.yaws[i]);
    p.pose.orientation.w = quat.w();
    p.pose.orientation.x = quat.x();
    p.pose.orientation.y = quat.y();
    p.pose.orientation.z = quat.z();
    p.longitudinal_velocity_mps = trajectory.longitudinal_velocities.at(i);
    p.lateral_velocity_mps = trajectory.lateral_velocities.at(i);
    p.front_wheel_angle_rad = vehicle_info.wheel_base_m * trajectory.curvatures.at(i);
    traj_points.push_back(p);
  }
  return traj_points;
}

template <class T>
auto convertToFrenetPoint(const T & points, const Point & search_point_geom, const size_t seg_idx)
  -> FrenetPoint
{
  FrenetPoint frenet_point;

  const double longitudinal_length =
    autoware::motion_utils::calcLongitudinalOffsetToSegment(points, seg_idx, search_point_geom);
  frenet_point.length =
    autoware::motion_utils::calcSignedArcLength(points, 0, seg_idx) + longitudinal_length;
  frenet_point.distance =
    autoware::motion_utils::calcLateralOffset(points, search_point_geom, seg_idx);

  return frenet_point;
}

auto prepareSamplingParameters(
  const autoware::sampler_common::State & initial_state, const double base_length,
  const autoware::sampler_common::transform::Spline2D & path_spline, const double trajectory_length,
  const TargetStateParameters & parameters) -> autoware::frenet_planner::SamplingParameters
{
  autoware::frenet_planner::SamplingParameters sampling_parameters;

  // calculate target lateral positions
  sampling_parameters.resolution = 0.5;
  const auto max_s = path_spline.lastS();
  autoware::frenet_planner::SamplingParameter p;
  p.target_duration = 10.0;
  for (const auto lon_position : parameters.lon_positions) {
    p.target_state.position.s = std::min(
      max_s, path_spline.frenet(initial_state.pose).s +
               std::max(0.0, lon_position * trajectory_length - base_length));
    for (const auto lon_velocity : parameters.lon_velocities) {
      p.target_state.longitudinal_velocity = lon_velocity;
      for (const auto lon_acceleration : parameters.lon_accelerations) {
        p.target_state.longitudinal_acceleration = lon_acceleration;
        for (const auto lat_position : parameters.lat_positions) {
          p.target_state.position.d = lat_position;
          for (const auto lat_velocity : parameters.lat_velocities) {
            p.target_state.lateral_velocity = lat_velocity;
            for (const auto lat_acceleration : parameters.lat_accelerations) {
              p.target_state.lateral_acceleration = lat_acceleration;
              sampling_parameters.parameters.push_back(p);
            }
          }
        }
      }
    }
    if (p.target_state.position.s == max_s) break;
  }
  return sampling_parameters;
}

auto resampling(
  const Trajectory & trajectory, const Pose & p_ego, const size_t resample_num,
  const double time_resolution) -> std::vector<TrajectoryPoint>
{
  const auto ego_seg_idx = autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
    trajectory.points, p_ego, 10.0, M_PI_2);

  std::vector<TrajectoryPoint> output;
  const auto vehicle_pose_frenet =
    convertToFrenetPoint(trajectory.points, p_ego.position, ego_seg_idx);

  double length = 0.0;
  for (size_t i = 0; i < resample_num; i++) {
    const auto pose = autoware::motion_utils::calcInterpolatedPose(
      trajectory.points, vehicle_pose_frenet.length + length);
    const auto p_trajectory = autoware::motion_utils::calcInterpolatedPoint(trajectory, pose);
    output.push_back(p_trajectory);

    const auto pred_accel = p_trajectory.acceleration_mps2;
    const auto pred_velocity = p_trajectory.longitudinal_velocity_mps;

    length +=
      pred_velocity * time_resolution + 0.5 * pred_accel * time_resolution * time_resolution;
  }

  return output;
}

auto sampling(
  const Trajectory & trajectory, const Pose & p_ego, const double v_ego, const double a_ego,
  const vehicle_info_utils::VehicleInfo & vehicle_info,
  const std::shared_ptr<Parameters> & parameters) -> std::vector<std::vector<TrajectoryPoint>>
{
  const auto reference_trajectory =
    autoware::path_sampler::preparePathSpline(trajectory.points, true);

  autoware::sampler_common::State current_state;
  current_state.pose = {p_ego.position.x, p_ego.position.y};
  current_state.heading = tf2::getYaw(p_ego.orientation);

  current_state.frenet = reference_trajectory.frenet(current_state.pose);
  // current_state.pose = reference_trajectory.cartesian(current_state.frenet.s);
  current_state.heading = reference_trajectory.yaw(current_state.frenet.s);
  current_state.curvature = reference_trajectory.curvature(current_state.frenet.s);

  const auto trajectory_length = autoware::motion_utils::calcArcLength(trajectory.points);
  const auto sampling_parameters = prepareSamplingParameters(
    current_state, 0.0, reference_trajectory, trajectory_length, parameters->target_state);

  autoware::frenet_planner::FrenetState initial_frenet_state;
  initial_frenet_state.position = reference_trajectory.frenet(current_state.pose);
  initial_frenet_state.longitudinal_velocity = v_ego;
  initial_frenet_state.longitudinal_acceleration = a_ego;
  const auto s = initial_frenet_state.position.s;
  const auto d = initial_frenet_state.position.d;
  // Calculate Velocity and acceleration parametrized over arc length
  // From appendix I of Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet
  // Frame
  const auto frenet_yaw = current_state.heading - reference_trajectory.yaw(s);
  const auto path_curvature = reference_trajectory.curvature(s);
  const auto delta_s = 0.001;
  initial_frenet_state.lateral_velocity = (1 - path_curvature * d) * std::tan(frenet_yaw);
  const auto path_curvature_deriv =
    (reference_trajectory.curvature(s + delta_s) - path_curvature) / delta_s;
  const auto cos_yaw = std::cos(frenet_yaw);
  if (cos_yaw == 0.0) {
    initial_frenet_state.lateral_acceleration = 0.0;
  } else {
    initial_frenet_state.lateral_acceleration =
      -(path_curvature_deriv * d + path_curvature * initial_frenet_state.lateral_velocity) *
        std::tan(frenet_yaw) +
      ((1 - path_curvature * d) / (cos_yaw * cos_yaw)) *
        (current_state.curvature * ((1 - path_curvature * d) / cos_yaw) - path_curvature);
  }

  const auto sampling_frenet_trajectories = autoware::frenet_planner::generateTrajectories(
    reference_trajectory, initial_frenet_state, sampling_parameters);

  std::vector<std::vector<TrajectoryPoint>> output;
  output.reserve(sampling_frenet_trajectories.size());

  for (const auto & trajectory : sampling_frenet_trajectories) {
    output.push_back(convertToTrajectoryPoints(
      trajectory.resampleTimeFromZero(parameters->time_resolution), vehicle_info));
  }

  return output;
}
}  // namespace autoware::behavior_analyzer::utils

#endif  // UTILS_HPP_
