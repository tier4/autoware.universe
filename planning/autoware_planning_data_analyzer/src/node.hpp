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

#ifndef NODE_HPP_
#define NODE_HPP_

#include "autoware/motion_utils/trajectory/interpolation.hpp"
#include "autoware/motion_utils/trajectory/trajectory.hpp"
#include "autoware/universe_utils/ros/polling_subscriber.hpp"
#include "autoware_frenet_planner/frenet_planner.hpp"
#include "autoware_path_sampler/prepare_inputs.hpp"
#include "autoware_path_sampler/utils/trajectory_utils.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "type_alias.hpp"

#include <autoware/route_handler/route_handler.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>
#include <magic_enum.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::behavior_analyzer
{

enum class METRICS {
  MANUAL_LATERAL_ACCEL = 0,
  MANUAL_LONGITUDINAL_ACCEL = 1,
  MANUAL_LONGITUDINAL_JERK = 2,
  MANUAL_TRAVEL_DISTANCE = 3,
  MANUAL_MINIMUM_TTC = 4,
  SYSTEM_LATERAL_ACCEL = 5,
  SYSTEM_LONGITUDINAL_ACCEL = 6,
  SYSTEM_LONGITUDINAL_JERK = 7,
  SYSTEM_TRAVEL_DISTANCE = 8,
  SYSTEM_MINIMUM_TTC = 9,
  SIZE
};

enum class REWARD {
  MANUAL_LATERAL_COMFORTABILITY = 0,
  MANUAL_LONGITUDINAL_COMFORTABILITY = 1,
  MANUAL_EFFICIENCY = 2,
  MANUAL_SAFETY = 3,
  SYSTEM_LATERAL_COMFORTABILITY = 4,
  SYSTEM_LONGITUDINAL_COMFORTABILITY = 5,
  SYSTEM_EFFICIENCY = 6,
  SYSTEM_SAFETY = 7,
  SIZE
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

  T get() const { return msgs.front(); }

  std::optional<T> get(const rcutils_time_point_value_t now) const
  {
    const auto itr = std::find_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
      return rclcpp::Time(msg.header.stamp).nanoseconds() > now;
    });

    if (itr == msgs.end()) {
      return std::nullopt;
    }

    return *itr;
  }

  std::vector<T> get_all_data() const { return msgs; }
};

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
std::optional<SteeringReport> Buffer<SteeringReport>::get(
  const rcutils_time_point_value_t now) const
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
std::optional<TFMessage> Buffer<TFMessage>::get(const rcutils_time_point_value_t now) const
{
  const auto itr = std::find_if(msgs.begin(), msgs.end(), [&now, this](const auto & msg) {
    return rclcpp::Time(msg.transforms.front().header.stamp).nanoseconds() > now;
  });

  if (itr == msgs.end()) {
    return std::nullopt;
  }

  return *itr;
}

struct Data
{
  TFMessage tf;
  Odometry odometry;
  PredictedObjects objects;
  AccelWithCovarianceStamped accel;
  SteeringReport steer;
  Trajectory trajectory;
  TrajectoryPoint predicted_point;

  std::vector<std::vector<TrajectoryPoint>> candidate_trajectories;

  std::unordered_map<METRICS, double> values;
};

struct FrenetPoint
{
  double length{0.0};    // longitudinal
  double distance{0.0};  // lateral
};

// data conversions
template <class T>
FrenetPoint convertToFrenetPoint(
  const T & points, const Point & search_point_geom, const size_t seg_idx)
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

autoware::frenet_planner::SamplingParameters prepareSamplingParameters(
  const autoware::sampler_common::State & initial_state, const double base_length,
  const autoware::sampler_common::transform::Spline2D & path_spline, const double trajectory_length)
{
  autoware::frenet_planner::SamplingParameters sampling_parameters;

  // calculate target lateral positions
  std::vector<double> target_lateral_positions = {-4.5, -2.5, 0.0, 2.5, 4.5};
  sampling_parameters.resolution = 0.5;
  const auto max_s = path_spline.lastS();
  autoware::frenet_planner::SamplingParameter p;
  p.target_duration = 10.0;
  for (const auto target_length : {trajectory_length}) {
    p.target_state.position.s = std::min(
      max_s, path_spline.frenet(initial_state.pose).s + std::max(0.0, target_length - base_length));
    for (const auto target_longitudinal_velocity : {5.56, 11.1}) {
      p.target_state.longitudinal_velocity = target_longitudinal_velocity;
      for (const auto target_longitudinal_acceleration : {0.0}) {
        p.target_state.longitudinal_acceleration = target_longitudinal_acceleration;
        for (const auto target_lateral_position : target_lateral_positions) {
          p.target_state.position.d = target_lateral_position;
          for (const auto target_lat_vel : {0.0}) {
            p.target_state.lateral_velocity = target_lat_vel;
            for (const auto target_lat_acc : {-0.2, -0.1, 0.0, 0.1, 0.2}) {
              p.target_state.lateral_acceleration = target_lat_acc;
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

struct DataSet
{
  explicit DataSet(const rcutils_time_point_value_t timestamp) : timestamp{timestamp} {}

  Buffer<TFMessage> buf_tf;
  Buffer<Odometry> buf_odometry;
  Buffer<PredictedObjects> buf_objects;
  Buffer<AccelWithCovarianceStamped> buf_accel;
  Buffer<SteeringReport> buf_steer;
  Buffer<Trajectory> buf_trajectory;

  rcutils_time_point_value_t timestamp;

  std::vector<TrajectoryPoint> predict(const Data & data)
  {
    const double time_horizon = 10.0;
    const double time_resolution = 0.5;

    const auto current_pose = data.odometry.pose.pose;

    const auto points = data.trajectory.points;

    const auto ego_seg_idx =
      autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
        points, current_pose, 10.0, M_PI_2);

    std::vector<TrajectoryPoint> predicted_path;
    const auto vehicle_pose_frenet =
      convertToFrenetPoint(points, current_pose.position, ego_seg_idx);

    double length = 0.0;
    for (double t = 0.0; t < time_horizon; t += time_resolution) {
      const auto pose =
        autoware::motion_utils::calcInterpolatedPose(points, vehicle_pose_frenet.length + length);
      const auto p_trajectory =
        autoware::motion_utils::calcInterpolatedPoint(data.trajectory, pose);
      predicted_path.push_back(p_trajectory);

      const auto pred_accel = p_trajectory.acceleration_mps2;
      const auto pred_velocity = p_trajectory.longitudinal_velocity_mps;

      length +=
        pred_velocity * time_resolution + 0.5 * pred_accel * time_resolution * time_resolution;
    }

    return predicted_path;
  }

  std::vector<autoware::frenet_planner::Trajectory> sampling(const Data & data)
  {
    const auto reference_trajectory =
      autoware::path_sampler::preparePathSpline(data.trajectory.points, true);

    autoware::sampler_common::State current_state;
    current_state.pose = {data.odometry.pose.pose.position.x, data.odometry.pose.pose.position.y};
    current_state.heading = tf2::getYaw(data.odometry.pose.pose.orientation);

    current_state.frenet = reference_trajectory.frenet(current_state.pose);
    current_state.pose = reference_trajectory.cartesian(current_state.frenet.s);
    current_state.heading = reference_trajectory.yaw(current_state.frenet.s);
    current_state.curvature = reference_trajectory.curvature(current_state.frenet.s);

    const auto trajectory_length = autoware::motion_utils::calcArcLength(data.trajectory.points);
    const auto sampling_parameters =
      prepareSamplingParameters(current_state, 0.0, reference_trajectory, trajectory_length);

    autoware::frenet_planner::FrenetState initial_frenet_state;
    initial_frenet_state.position = reference_trajectory.frenet(current_state.pose);
    initial_frenet_state.longitudinal_velocity = data.odometry.twist.twist.linear.x;
    initial_frenet_state.longitudinal_acceleration = data.accel.accel.accel.linear.x;
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

    return autoware::frenet_planner::generateTrajectories(
      reference_trajectory, initial_frenet_state, sampling_parameters);
  }

  std::vector<Data> extract(const double time_horizon, const double time_resolution)
  {
    std::vector<Data> extract_data;

    for (size_t t = timestamp; t < timestamp + time_horizon * 1e9; t += time_resolution * 1e9) {
      Data data;

      const auto tf = buf_tf.get(t);
      if (!tf.has_value()) {
        break;
      }
      data.tf = tf.value();

      const auto odometry = buf_odometry.get(t);
      if (!odometry.has_value()) {
        break;
      }
      data.odometry = odometry.value();

      const auto objects = buf_objects.get(t);
      if (!objects.has_value()) {
        break;
      }
      data.objects = objects.value();

      const auto accel = buf_accel.get(t);
      if (!accel.has_value()) {
        break;
      }
      data.accel = accel.value();

      const auto steer = buf_steer.get(t);
      if (!steer.has_value()) {
        break;
      }
      data.steer = steer.value();

      const auto trajectory = buf_trajectory.get(t);
      if (!trajectory.has_value()) {
        break;
      }
      data.trajectory = trajectory.value();

      extract_data.push_back(data);
    }

    const auto trajectory_points = predict(extract_data.front());
    if (trajectory_points.size() != extract_data.size()) {
      throw std::logic_error("there is an inconsistency among data.");
    }

    const auto candidate_frenet_trajectories = sampling(extract_data.front());
    for (const auto & frenet_trajectory : candidate_frenet_trajectories) {
      extract_data.front().candidate_trajectories.push_back(
        autoware::path_sampler::trajectory_utils::convertToTrajectoryPoints(frenet_trajectory));
    }

    for (size_t i = 0; i < extract_data.size(); ++i) {
      extract_data.at(i).predicted_point = trajectory_points.at(i);
    }

    return extract_data;
  }

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

class BehaviorAnalyzerNode : public rclcpp::Node
{
public:
  explicit BehaviorAnalyzerNode(const rclcpp::NodeOptions & node_options);

private:
  void on_timer();

  void update(std::shared_ptr<DataSet> & data_set) const;

  void process(std::vector<Data> & extract_data) const;

  void visualize(const std::vector<Data> & extract_data) const;

  void play(const SetBool::Request::SharedPtr req, SetBool::Response::SharedPtr res);

  void rewind(const Trigger::Request::SharedPtr req, Trigger::Response::SharedPtr res);

  void fill(std::vector<Data> & extract_data) const;

  auto manual_all_ttc(const Data & data) const -> std::vector<double>;

  auto system_all_ttc(const Data & data) const -> std::vector<double>;

  double manual_lateral_accel(const Data & data) const;

  double system_lateral_accel(const Data & data) const;

  double manual_longitudinal_jerk(const Data & front_data, const Data & back_data) const;

  double system_longitudinal_jerk(const Data & front_data, const Data & back_data) const;

  double manual_travel_distance(const Data & front_data, const Data & back_data) const;

  double system_travel_distance(const Data & front_data, const Data & back_data) const;

  auto metrics(const std::vector<Data> & extract_data) const -> Float32MultiArrayStamped;

  auto reward(const std::vector<Data> & extract_data) const -> Float32MultiArrayStamped;

  auto longitudinal_comfortability(const std::vector<Data> & extract_data) const
    -> std::pair<double, double>;

  auto lateral_comfortability(const std::vector<Data> & extract_data) const
    -> std::pair<double, double>;

  auto efficiency(const std::vector<Data> & extract_data) const -> std::pair<double, double>;

  auto safety(const std::vector<Data> & extract_data) const -> std::pair<double, double>;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_marker_;
  rclcpp::Publisher<Odometry>::SharedPtr pub_odometry_;
  rclcpp::Publisher<PredictedObjects>::SharedPtr pub_objects_;
  rclcpp::Publisher<Trajectory>::SharedPtr pub_trajectory_;
  rclcpp::Publisher<TFMessage>::SharedPtr pub_tf_;
  rclcpp::Publisher<Float32MultiArrayStamped>::SharedPtr pub_metrics_;
  rclcpp::Publisher<Float32MultiArrayStamped>::SharedPtr pub_cost_;
  rclcpp::Service<SetBool>::SharedPtr srv_play_;
  rclcpp::Service<Trigger>::SharedPtr srv_rewind_;

  vehicle_info_utils::VehicleInfo vehicle_info_;

  std::shared_ptr<DataSet> data_set_;

  mutable rosbag2_cpp::Reader reader_;

  bool is_ready_{false};
};
}  // namespace autoware::behavior_analyzer

#endif  // NODE_HPP_
