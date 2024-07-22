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
#include "rosbag2_cpp/reader.hpp"
#include "type_alias.hpp"

#include <autoware/route_handler/route_handler.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::path_selector
{

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

  std::unordered_map<std::string, double> metrics;
};

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

class PathSelectorNode : public rclcpp::Node
{
public:
  explicit PathSelectorNode(const rclcpp::NodeOptions & node_options);

private:
  void on_timer();

  void update(std::shared_ptr<DataSet> & data_set) const;

  void process(std::vector<Data> & extract_data) const;

  void visualize(const std::vector<Data> & extract_data) const;

  void play(const SetBool::Request::SharedPtr req, SetBool::Response::SharedPtr res);

  auto all_ttc(const Data & data) const -> std::vector<double>;

  double lateral_accel(const Data & data) const;

  double longitudinal_jerk(const Data & front_data, const Data & back_data) const;

  double travel_distance(const Data & front_data, const Data & back_data) const;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_marker_;
  rclcpp::Publisher<Odometry>::SharedPtr pub_odometry_;
  rclcpp::Publisher<PredictedObjects>::SharedPtr pub_objects_;
  rclcpp::Publisher<Trajectory>::SharedPtr pub_trajectory_;
  rclcpp::Publisher<TFMessage>::SharedPtr pub_tf_;
  rclcpp::Publisher<Float32MultiArrayStamped>::SharedPtr pub_metrics_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_bag_;

  vehicle_info_utils::VehicleInfo vehicle_info_;

  std::shared_ptr<DataSet> data_set_;

  mutable rosbag2_cpp::Reader reader_;

  bool is_ready_{false};
};
}  // namespace autoware::path_selector

#endif  // NODE_HPP_
