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

#include "data_structs.hpp"

#include "utils.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::behavior_analyzer
{

template <>
bool Buffer<SteeringReport>::is_ready() const
{
  if (msgs.empty()) {
    return false;
  }

  return rclcpp::Time(msgs.back().stamp).nanoseconds() -
           rclcpp::Time(msgs.front().stamp).nanoseconds() >
         BUFFER_TIME;
}

template <>
bool Buffer<TFMessage>::is_ready() const
{
  if (msgs.empty()) {
    return false;
  }

  if (msgs.front().transforms.empty()) {
    return false;
  }

  if (msgs.back().transforms.empty()) {
    return false;
  }

  return rclcpp::Time(msgs.back().transforms.front().header.stamp).nanoseconds() -
           rclcpp::Time(msgs.front().transforms.front().header.stamp).nanoseconds() >
         BUFFER_TIME;
}

template <>
void Buffer<SteeringReport>::remove_old_data(const rcutils_time_point_value_t now)
{
  if (msgs.empty()) {
    return;
  }

  const auto itr = std::remove_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
    return rclcpp::Time(msg.stamp).nanoseconds() < now;
  });
  msgs.erase(itr, msgs.end());
}

template <>
void Buffer<TFMessage>::remove_old_data(const rcutils_time_point_value_t now)
{
  if (msgs.empty()) {
    return;
  }

  const auto itr = std::remove_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
    return rclcpp::Time(msg.transforms.front().header.stamp).nanoseconds() < now;
  });
  msgs.erase(itr, msgs.end());
}

template <>
auto Buffer<SteeringReport>::get(const rcutils_time_point_value_t now) const
  -> std::optional<SteeringReport>
{
  const auto itr = std::find_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
    return rclcpp::Time(msg.stamp).nanoseconds() > now;
  });

  if (itr == msgs.end()) {
    return std::nullopt;
  }

  return *itr;
}

template <>
auto Buffer<TFMessage>::get(const rcutils_time_point_value_t now) const -> std::optional<TFMessage>
{
  const auto itr = std::find_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
    return rclcpp::Time(msg.transforms.front().header.stamp).nanoseconds() > now;
  });

  if (itr == msgs.end()) {
    return std::nullopt;
  }

  return *itr;
}

CommonData::CommonData(
  const std::shared_ptr<TrimmedData> & trimmed_data,
  const vehicle_info_utils::VehicleInfo & vehicle_info, const size_t resample_num,
  const double time_resolution, const std::string & tag)
: vehicle_info{vehicle_info}, resample_num(resample_num), tag{tag}
{
  for (size_t i = 0; i < resample_num; i++) {
    const auto opt_objects =
      trimmed_data->buf_objects.get(trimmed_data->timestamp + 1e9 * time_resolution * i);
    if (!opt_objects.has_value()) {
      break;
    }
    objects_history.push_back(opt_objects.value());
  }
}

void CommonData::calculate()
{
  std::vector<double> lateral_accel_values;
  std::vector<double> minimum_ttc_values;
  std::vector<double> longitudinal_jerk_values;
  std::vector<double> travel_distance_values;

  for (size_t i = 0; i < resample_num - 1; i++) {
    lateral_accel_values.push_back(lateral_accel(i));
    longitudinal_jerk_values.push_back(longitudinal_jerk(i));
    minimum_ttc_values.push_back(minimum_ttc(i));
    travel_distance_values.push_back(travel_distance(i));
  }

  {
    lateral_accel_values.push_back(lateral_accel(resample_num - 1));
    longitudinal_jerk_values.push_back(0.0);
    minimum_ttc_values.push_back(minimum_ttc(resample_num - 1));
    travel_distance_values.push_back(travel_distance(resample_num - 1));
  }

  values.emplace(METRIC::LATERAL_ACCEL, lateral_accel_values);
  values.emplace(METRIC::LONGITUDINAL_JERK, longitudinal_jerk_values);
  values.emplace(METRIC::MINIMUM_TTC, minimum_ttc_values);
  values.emplace(METRIC::TRAVEL_DISTANCE, travel_distance_values);

  scores.emplace(SCORE::LONGITUDINAL_COMFORTABILITY, longitudinal_comfortability());
  scores.emplace(SCORE::LATERAL_COMFORTABILITY, lateral_comfortability());
  scores.emplace(SCORE::EFFICIENCY, efficiency());
  scores.emplace(SCORE::SAFETY, safety());
}

double CommonData::longitudinal_comfortability() const
{
  constexpr double TIME_FACTOR = 0.8;

  double score = 0.0;

  const auto min = 0.0;
  const auto max = 0.5;
  const auto normalize = [&min, &max](const double value) {
    return (max - std::clamp(value, min, max)) / (max - min);
  };

  for (size_t i = 0; i < resample_num; i++) {
    score +=
      normalize(std::pow(TIME_FACTOR, i) * std::abs(values.at(METRIC::LONGITUDINAL_JERK).at(i)));
  }

  return score / resample_num;
}

double CommonData::lateral_comfortability() const
{
  constexpr double TIME_FACTOR = 0.8;

  double score = 0.0;

  const auto min = 0.0;
  const auto max = 0.5;
  const auto normalize = [&min, &max](const double value) {
    return (max - std::clamp(value, min, max)) / (max - min);
  };

  for (size_t i = 0; i < resample_num; i++) {
    score += normalize(std::pow(TIME_FACTOR, i) * std::abs(values.at(METRIC::LATERAL_ACCEL).at(i)));
  }

  return score / resample_num;
}

double CommonData::efficiency() const
{
  constexpr double TIME_FACTOR = 0.8;

  double score = 0.0;

  const auto min = 0.0;
  const auto max = 20.0;
  const auto normalize = [&min, &max](const double value) {
    return std::clamp(value, min, max) / (max - min);
  };

  for (size_t i = 0; i < resample_num; i++) {
    score += normalize(std::pow(TIME_FACTOR, i) * values.at(METRIC::TRAVEL_DISTANCE).at(i) / 0.5);
  }

  return score / resample_num;
}

double CommonData::safety() const
{
  constexpr double TIME_FACTOR = 0.8;

  double score = 0.0;

  const auto min = 0.0;
  const auto max = 5.0;
  const auto normalize = [&min, &max](const double value) {
    return std::clamp(value, min, max) / (max - min);
  };

  for (size_t i = 0; i < resample_num; i++) {
    score += normalize(std::pow(TIME_FACTOR, i) * values.at(METRIC::MINIMUM_TTC).at(i));
  }

  return score / resample_num;
}

double CommonData::total() const
{
  constexpr double w0 = 1.0;
  constexpr double w1 = 1.0;
  constexpr double w2 = 1.0;
  constexpr double w3 = 1.0;
  return w0 * scores.at(SCORE::LATERAL_COMFORTABILITY) +
         w1 * scores.at(SCORE::LONGITUDINAL_COMFORTABILITY) + w2 * scores.at(SCORE::EFFICIENCY) +
         w3 * scores.at(SCORE::SAFETY);
}

ManualDrivingData::ManualDrivingData(
  const std::shared_ptr<TrimmedData> & trimmed_data,
  const vehicle_info_utils::VehicleInfo & vehicle_info, const size_t resample_num,
  const double time_resolution)
: CommonData(trimmed_data, vehicle_info, resample_num, time_resolution, "manual")
{
  for (size_t i = 0; i < resample_num; i++) {
    const auto opt_odometry =
      trimmed_data->buf_odometry.get(trimmed_data->timestamp + 1e9 * time_resolution * i);
    if (!opt_odometry.has_value()) {
      break;
    }
    odometry_history.push_back(opt_odometry.value());

    const auto opt_accel =
      trimmed_data->buf_accel.get(trimmed_data->timestamp + 1e9 * time_resolution * i);
    if (!opt_accel.has_value()) {
      break;
    }
    accel_history.push_back(opt_accel.value());

    const auto opt_steer =
      trimmed_data->buf_steer.get(trimmed_data->timestamp + 1e9 * time_resolution * i);
    if (!opt_steer.has_value()) {
      break;
    }
    steer_history.push_back(opt_steer.value());
  }

  calculate();
}

double ManualDrivingData::lateral_accel(const size_t idx) const
{
  const auto radius =
    vehicle_info.wheel_base_m / std::tan(steer_history.at(idx).steering_tire_angle);
  const auto speed = odometry_history.at(idx).twist.twist.linear.x;
  return speed * speed / radius;
}

double ManualDrivingData::longitudinal_jerk(const size_t idx) const
{
  const double dt = rclcpp::Time(accel_history.at(idx + 1).header.stamp).nanoseconds() -
                    rclcpp::Time(accel_history.at(idx).header.stamp).nanoseconds();

  return 1e9 *
         (accel_history.at(idx + 1).accel.accel.linear.x -
          accel_history.at(idx).accel.accel.linear.x) /
         dt;
}

double ManualDrivingData::minimum_ttc(const size_t idx) const
{
  const auto p_ego = odometry_history.at(idx).pose.pose;
  const auto v_ego = utils::get_velocity_in_world_coordinate(odometry_history.at(idx));

  return utils::time_to_collision(objects_history.at(idx), p_ego, v_ego);
}

double ManualDrivingData::travel_distance(const size_t idx) const
{
  double distance = 0.0;
  for (size_t i = 0L; i < idx; i++) {
    distance += autoware::universe_utils::calcDistance3d(
      odometry_history.at(i + 1).pose.pose, odometry_history.at(i).pose.pose);
  }
  return distance;
}

TrajectoryData::TrajectoryData(
  const std::shared_ptr<TrimmedData> & trimmed_data,
  const vehicle_info_utils::VehicleInfo & vehicle_info, const size_t resample_num,
  const double time_resolution, const std::string & tag,
  const std::vector<TrajectoryPoint> & points)
: CommonData(trimmed_data, vehicle_info, resample_num, time_resolution, tag), points{points}
{
  calculate();
}

double TrajectoryData::lateral_accel(const size_t idx) const
{
  const auto radius = vehicle_info.wheel_base_m / std::tan(points.at(idx).front_wheel_angle_rad);
  const auto speed = points.at(idx).longitudinal_velocity_mps;
  return speed * speed / radius;
}

double TrajectoryData::longitudinal_jerk(const size_t idx) const
{
  return (points.at(idx + 1).acceleration_mps2 - points.at(idx).acceleration_mps2) / 0.5;
}

double TrajectoryData::minimum_ttc(const size_t idx) const
{
  const auto p_ego = points.at(idx).pose;
  const auto v_ego = utils::get_velocity_in_world_coordinate(points.at(idx));

  return utils::time_to_collision(objects_history.at(idx), p_ego, v_ego);
}

double TrajectoryData::travel_distance(const size_t idx) const
{
  return autoware::motion_utils::calcSignedArcLength(points, 0L, idx);
}

bool TrajectoryData::feasible() const
{
  const auto condition = [](const auto & p) { return p.longitudinal_velocity_mps > 0.0; };
  return std::all_of(points.begin(), points.end(), condition);
}

SamplingTrajectoryData::SamplingTrajectoryData(
  const std::shared_ptr<TrimmedData> & trimmed_data,
  const vehicle_info_utils::VehicleInfo & vehicle_info,
  const std::shared_ptr<Parameters> & parameters)
{
  const auto opt_odometry = trimmed_data->buf_odometry.get(trimmed_data->timestamp);
  if (!opt_odometry.has_value()) {
    throw std::logic_error("data is not enough.");
  }

  const auto opt_accel = trimmed_data->buf_accel.get(trimmed_data->timestamp);
  if (!opt_accel.has_value()) {
    throw std::logic_error("data is not enough.");
  }

  const auto opt_trajectory = trimmed_data->buf_trajectory.get(trimmed_data->timestamp);
  if (!opt_trajectory.has_value()) {
    throw std::logic_error("data is not enough.");
  }
  data.emplace_back(
    trimmed_data, vehicle_info, parameters->resample_num, parameters->time_resolution, "autoware",
    utils::resampling(
      opt_trajectory.value(), opt_odometry.value().pose.pose, parameters->resample_num,
      parameters->time_resolution));

  for (const auto & sample : utils::sampling(
         opt_trajectory.value(), opt_odometry.value().pose.pose,
         opt_odometry.value().twist.twist.linear.x, opt_accel.value().accel.accel.linear.x,
         vehicle_info, parameters)) {
    data.emplace_back(
      trimmed_data, vehicle_info, parameters->resample_num, parameters->time_resolution, "frenet",
      sample);
  }

  std::sort(
    data.begin(), data.end(), [](const auto & a, const auto & b) { return a.total() > b.total(); });

  std::remove_if(data.begin(), data.end(), [](const auto & d) { return !d.feasible(); });
}
}  // namespace autoware::behavior_analyzer
