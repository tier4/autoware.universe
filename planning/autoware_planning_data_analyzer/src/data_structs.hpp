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

#ifndef DATA_STRUCTS_HPP_
#define DATA_STRUCTS_HPP_

#include "type_alias.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::behavior_analyzer
{

enum class METRIC {
  LATERAL_ACCEL = 0,
  LONGITUDINAL_ACCEL = 1,
  LONGITUDINAL_JERK = 2,
  TRAVEL_DISTANCE = 3,
  MINIMUM_TTC = 4,
  SIZE
};

enum class SCORE {
  LATERAL_COMFORTABILITY = 0,
  LONGITUDINAL_COMFORTABILITY = 1,
  EFFICIENCY = 2,
  SAFETY = 3,
  SIZE
};

struct FrenetPoint
{
  double length{0.0};    // longitudinal
  double distance{0.0};  // lateral
};

template <class T>
struct Buffer
{
  std::vector<T> msgs;

  const double BUFFER_TIME = 20.0 * 1e9;

  bool is_ready() const
  {
    if (msgs.empty()) {
      return false;
    }

    return rclcpp::Time(msgs.back().header.stamp).nanoseconds() -
             rclcpp::Time(msgs.front().header.stamp).nanoseconds() >
           BUFFER_TIME;
  }

  void remove_old_data(const rcutils_time_point_value_t now)
  {
    const auto itr = std::remove_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
      return rclcpp::Time(msg.header.stamp).nanoseconds() < now;
    });
    msgs.erase(itr, msgs.end());
  }

  void append(const T & msg) { msgs.push_back(msg); }

  auto get() const -> T { return msgs.front(); }

  auto get(const rcutils_time_point_value_t now) const -> std::optional<T>
  {
    const auto itr = std::find_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
      return rclcpp::Time(msg.header.stamp).nanoseconds() > now;
    });

    if (itr == msgs.end()) {
      return std::nullopt;
    }

    return *itr;
  }

  auto get_all_data() const -> std::vector<T> { return msgs; }
};

template <>
bool Buffer<SteeringReport>::is_ready() const;

template <>
bool Buffer<TFMessage>::is_ready() const;

template <>
void Buffer<SteeringReport>::remove_old_data(const rcutils_time_point_value_t now);

template <>
void Buffer<TFMessage>::remove_old_data(const rcutils_time_point_value_t now);

template <>
auto Buffer<SteeringReport>::get(const rcutils_time_point_value_t now) const
  -> std::optional<SteeringReport>;

template <>
auto Buffer<TFMessage>::get(const rcutils_time_point_value_t now) const -> std::optional<TFMessage>;

struct TrimmedData
{
  explicit TrimmedData(const rcutils_time_point_value_t timestamp) : timestamp{timestamp} {}

  Buffer<TFMessage> buf_tf;
  Buffer<Odometry> buf_odometry;
  Buffer<PredictedObjects> buf_objects;
  Buffer<AccelWithCovarianceStamped> buf_accel;
  Buffer<SteeringReport> buf_steer;
  Buffer<Trajectory> buf_trajectory;

  rcutils_time_point_value_t timestamp;

  void update(const rcutils_time_point_value_t dt)
  {
    timestamp += dt;
    remove_old_data();
  }

  void remove_old_data()
  {
    buf_tf.remove_old_data(timestamp);
    buf_odometry.remove_old_data(timestamp);
    buf_objects.remove_old_data(timestamp);
    buf_accel.remove_old_data(timestamp);
    buf_steer.remove_old_data(timestamp);
    buf_trajectory.remove_old_data(timestamp);
  }

  bool is_ready() const
  {
    return buf_tf.is_ready() && buf_objects.is_ready() && buf_odometry.is_ready() &&
           buf_accel.is_ready() && buf_steer.is_ready() && buf_trajectory.is_ready();
  }
};

struct CommonData
{
  CommonData(
    const std::shared_ptr<TrimmedData> & trimmed_data,
    const vehicle_info_utils::VehicleInfo & vehicle_info, const size_t resample_num,
    const double time_resolution, const std::string & tag);

  void calculate();

  double longitudinal_comfortability() const;

  double lateral_comfortability() const;

  double efficiency() const;

  double safety() const;

  double total() const;

  virtual double lateral_accel(const size_t idx) const = 0;

  virtual double longitudinal_jerk(const size_t idx) const = 0;

  virtual double minimum_ttc(const size_t idx) const = 0;

  virtual double travel_distance(const size_t idx) const = 0;

  virtual bool feasible() const = 0;

  std::vector<PredictedObjects> objects_history;

  std::unordered_map<METRIC, std::vector<double>> values;
  std::unordered_map<SCORE, double> scores;

  vehicle_info_utils::VehicleInfo vehicle_info;

  size_t resample_num;

  std::string tag{""};
};

struct ManualDrivingData : CommonData
{
  ManualDrivingData(
    const std::shared_ptr<TrimmedData> & trimmed_data,
    const vehicle_info_utils::VehicleInfo & vehicle_info, const size_t resample_num,
    const double time_resolution);

  double lateral_accel(const size_t idx) const override;

  double longitudinal_jerk(const size_t idx) const override;

  double minimum_ttc(const size_t idx) const override;

  double travel_distance(const size_t idx) const override;

  bool feasible() const override { return true; }

  std::vector<Odometry> odometry_history;
  std::vector<AccelWithCovarianceStamped> accel_history;
  std::vector<SteeringReport> steer_history;
};

struct TrajectoryData : CommonData
{
  TrajectoryData(
    const std::shared_ptr<TrimmedData> & trimmed_data,
    const vehicle_info_utils::VehicleInfo & vehicle_info, const size_t resample_num,
    const double time_resolution, const std::string & tag,
    const std::vector<TrajectoryPoint> & points);

  double lateral_accel(const size_t idx) const override;

  double longitudinal_jerk(const size_t idx) const override;

  double minimum_ttc(const size_t idx) const override;

  double travel_distance(const size_t idx) const override;

  bool feasible() const override;

  std::vector<TrajectoryPoint> points;
};

struct TargetStateParameters
{
  std::vector<double> lat_positions{};
  std::vector<double> lat_velocities{};
  std::vector<double> lat_accelerations{};
  std::vector<double> lon_positions{};
  std::vector<double> lon_velocities{};
  std::vector<double> lon_accelerations{};
};

struct Parameters
{
  size_t resample_num{20};
  double time_resolution{0.5};
  TargetStateParameters target_state{};
};

struct SamplingTrajectoryData
{
  SamplingTrajectoryData(
    const std::shared_ptr<TrimmedData> & trimmed_data,
    const vehicle_info_utils::VehicleInfo & vehicle_info,
    const std::shared_ptr<Parameters> & parameters);

  auto best() const -> TrajectoryData { return data.front(); }

  auto autoware() const -> TrajectoryData
  {
    const auto itr = std::find_if(data.begin(), data.end(), [](const auto & trajectory) {
      return trajectory.tag == "autoware";
    });
    return *itr;
  }

  std::vector<TrajectoryData> data;
};

struct DataSet
{
  DataSet(
    const std::shared_ptr<TrimmedData> & trimmed_data,
    const vehicle_info_utils::VehicleInfo & vehicle_info,
    const std::shared_ptr<Parameters> & parameters)
  : manual{ManualDrivingData(
      trimmed_data, vehicle_info, parameters->resample_num, parameters->time_resolution)},
    sampling{SamplingTrajectoryData(trimmed_data, vehicle_info, parameters)}
  {
  }

  ManualDrivingData manual;
  SamplingTrajectoryData sampling;
};

}  // namespace autoware::behavior_analyzer

#endif  // DATA_STRUCTS_HPP_
