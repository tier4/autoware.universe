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

#include "node.hpp"

#include <autoware/universe_utils/geometry/boost_geometry.hpp>
#include <autoware/universe_utils/geometry/boost_polygon_utils.hpp>
#include <autoware/universe_utils/ros/marker_helper.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/union.hpp>

namespace autoware::behavior_analyzer
{
using autoware::universe_utils::createDefaultMarker;
using autoware::universe_utils::createMarkerColor;
using autoware::universe_utils::createMarkerScale;
using autoware::universe_utils::Point2d;
using autoware::universe_utils::Polygon2d;

namespace
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
}  // namespace

BehaviorAnalyzerNode::BehaviorAnalyzerNode(const rclcpp::NodeOptions & node_options)
: Node("path_selector_node", node_options)
{
  using namespace std::literals::chrono_literals;
  timer_ = rclcpp::create_timer(
    this, get_clock(), 100ms, std::bind(&BehaviorAnalyzerNode::on_timer, this));

  vehicle_info_ = autoware::vehicle_info_utils::VehicleInfoUtils(*this).getVehicleInfo();

  pub_marker_ = create_publisher<MarkerArray>("~/marker", 1);
  pub_odometry_ = create_publisher<Odometry>("/localization/kinematic_state", rclcpp::QoS(1));
  pub_objects_ =
    create_publisher<PredictedObjects>("/perception/object_recognition/objects", rclcpp::QoS(1));
  pub_trajectory_ =
    create_publisher<Trajectory>("/planning/scenario_planning/trajectory", rclcpp::QoS(1));
  pub_tf_ = create_publisher<TFMessage>("/tf", rclcpp::QoS(1));

  pub_metrics_ = create_publisher<Float32MultiArrayStamped>("~/metrics", rclcpp::QoS{1});
  pub_cost_ = create_publisher<Float32MultiArrayStamped>("~/cost", rclcpp::QoS{1});

  srv_play_ = this->create_service<SetBool>(
    "play",
    std::bind(&BehaviorAnalyzerNode::play, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  srv_rewind_ = this->create_service<Trigger>(
    "rewind",
    std::bind(&BehaviorAnalyzerNode::rewind, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  reader_.open(declare_parameter<std::string>("bag_path"));

  data_set_ = std::make_shared<DataSet>(
    duration_cast<nanoseconds>(reader_.get_metadata().starting_time.time_since_epoch()).count());
}

void BehaviorAnalyzerNode::update(std::shared_ptr<DataSet> & data_set) const
{
  rosbag2_storage::StorageFilter filter;
  filter.topics.emplace_back("/tf");
  filter.topics.emplace_back("/localization/kinematic_state");
  filter.topics.emplace_back("/localization/acceleration");
  filter.topics.emplace_back("/perception/object_recognition/objects");
  filter.topics.emplace_back("/vehicle/status/steering_status");
  filter.topics.emplace_back("/planning/scenario_planning/trajectory");
  reader_.set_filter(filter);

  if (!reader_.has_next()) {
    return;
  }

  data_set->update(0.1 * 1e9);

  while (reader_.has_next()) {
    const auto next_data = reader_.read_next();
    rclcpp::SerializedMessage serialized_msg(*next_data->serialized_data);

    if (data_set->is_ready()) {
      break;
    }

    if (next_data->topic_name == "/tf") {
      rclcpp::Serialization<TFMessage> serializer;
      const auto deserialized_message = std::make_shared<TFMessage>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      data_set->buf_tf.append(*deserialized_message);
    }

    if (next_data->topic_name == "/localization/kinematic_state") {
      rclcpp::Serialization<Odometry> serializer;
      const auto deserialized_message = std::make_shared<Odometry>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      data_set->buf_odometry.append(*deserialized_message);
    }

    if (next_data->topic_name == "/localization/acceleration") {
      rclcpp::Serialization<AccelWithCovarianceStamped> serializer;
      const auto deserialized_message = std::make_shared<AccelWithCovarianceStamped>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      data_set->buf_accel.append(*deserialized_message);
    }

    if (next_data->topic_name == "/perception/object_recognition/objects") {
      rclcpp::Serialization<PredictedObjects> serializer;
      const auto deserialized_message = std::make_shared<PredictedObjects>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      data_set->buf_objects.append(*deserialized_message);
    }

    if (next_data->topic_name == "/vehicle/status/steering_status") {
      rclcpp::Serialization<SteeringReport> serializer;
      const auto deserialized_message = std::make_shared<SteeringReport>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      data_set->buf_steer.append(*deserialized_message);
    }

    if (next_data->topic_name == "/planning/scenario_planning/trajectory") {
      rclcpp::Serialization<Trajectory> serializer;
      const auto deserialized_message = std::make_shared<Trajectory>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      data_set->buf_trajectory.append(*deserialized_message);
    }
  }
}

void BehaviorAnalyzerNode::play(
  const SetBool::Request::SharedPtr req, SetBool::Response::SharedPtr res)
{
  is_ready_ = req->data;
  if (is_ready_) {
    RCLCPP_INFO(get_logger(), "start evaluation.");
  } else {
    RCLCPP_INFO(get_logger(), "stop evaluation.");
  }
  res->success = true;
}

void BehaviorAnalyzerNode::rewind(
  [[maybe_unused]] const Trigger::Request::SharedPtr req, Trigger::Response::SharedPtr res)
{
  reader_.seek(0);

  data_set_.reset();
  data_set_ = std::make_shared<DataSet>(
    duration_cast<nanoseconds>(reader_.get_metadata().starting_time.time_since_epoch()).count());

  res->success = true;
}

auto BehaviorAnalyzerNode::manual_all_ttc(const Data & data) const -> std::vector<double>
{
  if (data.objects.objects.empty()) {
    return {};
  }

  const auto p_ego = data.odometry.pose.pose;
  const auto v_ego = get_velocity_in_world_coordinate(data.odometry);

  std::vector<double> ttc(data.objects.objects.size());

  for (const auto & object : data.objects.objects) {
    const auto p_object = object.kinematics.initial_pose_with_covariance.pose;
    const auto v_ego2object =
      autoware::universe_utils::point2tfVector(p_ego.position, p_object.position);

    const auto v_object = get_velocity_in_world_coordinate(object.kinematics);
    const auto v_relative = tf2::tf2Dot(v_ego2object.normalized(), v_ego) -
                            tf2::tf2Dot(v_ego2object.normalized(), v_object);

    ttc.push_back(v_ego2object.length() / v_relative);
  }

  const auto itr =
    std::remove_if(ttc.begin(), ttc.end(), [](const auto & value) { return value < 1e-3; });
  ttc.erase(itr, ttc.end());

  return ttc;
}

auto BehaviorAnalyzerNode::system_all_ttc(const Data & data) const -> std::vector<double>
{
  if (data.objects.objects.empty()) {
    return {};
  }

  const auto p_ego = data.predicted_point.pose;
  const auto v_ego = get_velocity_in_world_coordinate(data.predicted_point);

  std::vector<double> ttc(data.objects.objects.size());

  for (const auto & object : data.objects.objects) {
    const auto p_object = object.kinematics.initial_pose_with_covariance.pose;
    const auto v_ego2object =
      autoware::universe_utils::point2tfVector(p_ego.position, p_object.position);

    const auto v_object = get_velocity_in_world_coordinate(object.kinematics);
    const auto v_relative = tf2::tf2Dot(v_ego2object.normalized(), v_ego) -
                            tf2::tf2Dot(v_ego2object.normalized(), v_object);

    ttc.push_back(v_ego2object.length() / v_relative);
  }

  const auto itr =
    std::remove_if(ttc.begin(), ttc.end(), [](const auto & value) { return value < 1e-3; });
  ttc.erase(itr, ttc.end());

  return ttc;
}

double BehaviorAnalyzerNode::manual_lateral_accel(const Data & data) const
{
  const auto radius = vehicle_info_.wheel_base_m / std::tan(data.steer.steering_tire_angle);
  const auto speed = data.odometry.twist.twist.linear.x;
  return speed * speed / radius;
}

double BehaviorAnalyzerNode::system_lateral_accel(const Data & data) const
{
  const auto radius =
    vehicle_info_.wheel_base_m / std::tan(data.predicted_point.front_wheel_angle_rad);
  const auto speed = data.predicted_point.longitudinal_velocity_mps;
  return speed * speed / radius;
}

double BehaviorAnalyzerNode::manual_longitudinal_jerk(
  const Data & front_data, const Data & back_data) const
{
  const auto & front_accel = front_data.accel;
  const auto & back_accel = back_data.accel;

  const double dt = rclcpp::Time(back_accel.header.stamp).nanoseconds() -
                    rclcpp::Time(front_accel.header.stamp).nanoseconds();

  return 1e9 * (back_accel.accel.accel.linear.x - front_accel.accel.accel.linear.x) / dt;
}

double BehaviorAnalyzerNode::system_longitudinal_jerk(
  const Data & front_data, const Data & back_data) const
{
  return (back_data.predicted_point.acceleration_mps2 -
          front_data.predicted_point.acceleration_mps2) /
         0.5;
}

double BehaviorAnalyzerNode::manual_travel_distance(
  const Data & front_data, const Data & back_data) const
{
  const auto travel_distance = autoware::universe_utils::calcDistance3d(
    front_data.odometry.pose.pose, back_data.odometry.pose.pose);
  return travel_distance + front_data.values.at(METRICS::MANUAL_TRAVEL_DISTANCE);
}

double BehaviorAnalyzerNode::system_travel_distance(
  const Data & front_data, const Data & back_data) const
{
  const auto travel_distance = autoware::universe_utils::calcDistance3d(
    front_data.predicted_point.pose, back_data.predicted_point.pose);
  return travel_distance + front_data.values.at(METRICS::SYSTEM_TRAVEL_DISTANCE);
}

void BehaviorAnalyzerNode::process(std::vector<Data> & extract_data) const
{
  if (extract_data.empty()) {
    return;
  }

  fill(extract_data);

  pub_tf_->publish(extract_data.front().tf);

  pub_objects_->publish(extract_data.front().objects);

  pub_trajectory_->publish(extract_data.front().trajectory);

  pub_metrics_->publish(metrics(extract_data));

  pub_cost_->publish(reward(extract_data));

  visualize(extract_data);
}

void BehaviorAnalyzerNode::fill(std::vector<Data> & extract_data) const
{
  {
    // travel distance
    {
      extract_data.front().values.emplace(METRICS::MANUAL_TRAVEL_DISTANCE, 0.0);
      extract_data.front().values.emplace(METRICS::SYSTEM_TRAVEL_DISTANCE, 0.0);
    }
  }

  for (size_t i = 0; i < extract_data.size() - 1; i++) {
    // longitudinal acceleration
    {
      extract_data.at(i).values.emplace(
        METRICS::MANUAL_LONGITUDINAL_ACCEL, extract_data.at(i).accel.accel.accel.linear.x);
    }

    // lateral acceleration
    {
      extract_data.at(i).values.emplace(
        METRICS::MANUAL_LATERAL_ACCEL, manual_lateral_accel(extract_data.at(i)));
      extract_data.at(i).values.emplace(
        METRICS::SYSTEM_LATERAL_ACCEL, system_lateral_accel(extract_data.at(i)));
    }

    // minimum ttc
    {
      std::vector<double> ttc = manual_all_ttc(extract_data.at(i));
      std::sort(ttc.begin(), ttc.end());
      if (!ttc.empty()) {
        extract_data.at(i).values.emplace(METRICS::MANUAL_MINIMUM_TTC, ttc.front());
      }
    }
    {
      std::vector<double> ttc = system_all_ttc(extract_data.at(i));
      std::sort(ttc.begin(), ttc.end());
      if (!ttc.empty()) {
        extract_data.at(i).values.emplace(METRICS::SYSTEM_MINIMUM_TTC, ttc.front());
      }
    }

    // longitudinal jerk
    {
      extract_data.at(i).values.emplace(
        METRICS::MANUAL_LONGITUDINAL_JERK,
        manual_longitudinal_jerk(extract_data.at(i), extract_data.at(i + 1)));
      extract_data.at(i).values.emplace(
        METRICS::SYSTEM_LONGITUDINAL_JERK,
        system_longitudinal_jerk(extract_data.at(i), extract_data.at(i + 1)));
    }

    // travel distance
    {
      extract_data.at(i + 1).values.emplace(
        METRICS::MANUAL_TRAVEL_DISTANCE,
        manual_travel_distance(extract_data.at(i), extract_data.at(i + 1)));
      extract_data.at(i + 1).values.emplace(
        METRICS::SYSTEM_TRAVEL_DISTANCE,
        system_travel_distance(extract_data.at(i), extract_data.at(i + 1)));
    }
  }

  {
    // acceleration
    {
      extract_data.back().values.emplace(
        METRICS::MANUAL_LONGITUDINAL_ACCEL, extract_data.back().accel.accel.accel.linear.x);
    }

    // lateral acceleration
    {
      extract_data.back().values.emplace(
        METRICS::MANUAL_LATERAL_ACCEL, manual_lateral_accel(extract_data.back()));
      extract_data.back().values.emplace(
        METRICS::SYSTEM_LATERAL_ACCEL, system_lateral_accel(extract_data.back()));
    }

    // minimum ttc
    {
      std::vector<double> ttc = manual_all_ttc(extract_data.back());
      std::sort(ttc.begin(), ttc.end());
      if (!ttc.empty()) {
        extract_data.back().values.emplace(METRICS::MANUAL_MINIMUM_TTC, ttc.front());
      }
    }
    {
      std::vector<double> ttc = system_all_ttc(extract_data.back());
      std::sort(ttc.begin(), ttc.end());
      if (!ttc.empty()) {
        extract_data.back().values.emplace(METRICS::SYSTEM_MINIMUM_TTC, ttc.front());
      }
    }

    // longitudinal jerk
    {
      extract_data.back().values.emplace(METRICS::MANUAL_LONGITUDINAL_JERK, 0.0);
      extract_data.back().values.emplace(METRICS::SYSTEM_LONGITUDINAL_JERK, 0.0);
    }
  }
}

auto BehaviorAnalyzerNode::metrics(const std::vector<Data> & extract_data) const
  -> Float32MultiArrayStamped
{
  Float32MultiArrayStamped msg{};

  msg.stamp = now();
  msg.data.resize(static_cast<size_t>(METRICS::SIZE) * extract_data.size());

  const auto set_metrics = [&msg, &extract_data](const auto metrics_type, const auto idx) {
    const auto offset = static_cast<size_t>(metrics_type) * extract_data.size();
    if (extract_data.at(idx).values.count(metrics_type) != 0) {
      msg.data.at(offset + idx) = extract_data.at(idx).values.at(metrics_type);
    }
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    // comfortability
    set_metrics(METRICS::MANUAL_LATERAL_ACCEL, i);
    set_metrics(METRICS::SYSTEM_LATERAL_ACCEL, i);

    set_metrics(METRICS::MANUAL_LONGITUDINAL_ACCEL, i);
    set_metrics(METRICS::SYSTEM_LONGITUDINAL_ACCEL, i);

    set_metrics(METRICS::MANUAL_LONGITUDINAL_JERK, i);
    set_metrics(METRICS::SYSTEM_LONGITUDINAL_JERK, i);

    // efficiency
    set_metrics(METRICS::MANUAL_TRAVEL_DISTANCE, i);
    set_metrics(METRICS::SYSTEM_TRAVEL_DISTANCE, i);

    // minimum ttc
    set_metrics(METRICS::MANUAL_MINIMUM_TTC, i);
    set_metrics(METRICS::SYSTEM_MINIMUM_TTC, i);
  }

  return msg;
}

auto BehaviorAnalyzerNode::reward(const std::vector<Data> & extract_data) const
  -> Float32MultiArrayStamped
{
  Float32MultiArrayStamped msg{};

  msg.stamp = now();
  msg.data.resize(static_cast<size_t>(REWARD::SIZE));

  const auto set_reward = [&msg](const auto reward_type, const auto value) {
    msg.data.at(static_cast<size_t>(reward_type)) = value;
  };

  {
    const auto [manual, system] = longitudinal_comfortability(extract_data);
    set_reward(REWARD::MANUAL_LONGITUDINAL_COMFORTABILITY, manual);
    set_reward(REWARD::SYSTEM_LONGITUDINAL_COMFORTABILITY, system);
  }

  {
    const auto [manual, system] = lateral_comfortability(extract_data);
    set_reward(REWARD::MANUAL_LATERAL_COMFORTABILITY, manual);
    set_reward(REWARD::SYSTEM_LATERAL_COMFORTABILITY, system);
  }

  {
    const auto [manual, system] = efficiency(extract_data);
    set_reward(REWARD::MANUAL_EFFICIENCY, manual);
    set_reward(REWARD::SYSTEM_EFFICIENCY, system);
  }

  {
    const auto [manual, system] = safety(extract_data);
    set_reward(REWARD::MANUAL_SAFETY, manual);
    set_reward(REWARD::SYSTEM_SAFETY, system);
  }

  return msg;
}

auto BehaviorAnalyzerNode::safety(const std::vector<Data> & extract_data) const
  -> std::pair<double, double>
{
  constexpr double TIME_FACTOR = 0.8;

  double manual_cost = 0.0;
  double system_cost = 0.0;

  const auto min = 0.0;
  const auto max = 5.0;
  const auto normalize = [&min, &max](const double value) {
    return std::clamp(value, min, max) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).values.count(METRICS::MANUAL_MINIMUM_TTC) != 0) {
      manual_cost += normalize(
        std::pow(TIME_FACTOR, i) * extract_data.at(i).values.at(METRICS::MANUAL_MINIMUM_TTC));
    }
    if (extract_data.at(i).values.count(METRICS::SYSTEM_MINIMUM_TTC) != 0) {
      system_cost += normalize(
        std::pow(TIME_FACTOR, i) * extract_data.at(i).values.at(METRICS::SYSTEM_MINIMUM_TTC));
    }
  }

  return {manual_cost / extract_data.size(), system_cost / extract_data.size()};
}

auto BehaviorAnalyzerNode::longitudinal_comfortability(const std::vector<Data> & extract_data) const
  -> std::pair<double, double>
{
  constexpr double TIME_FACTOR = 0.8;

  double manual_cost = 0.0;
  double system_cost = 0.0;

  const auto min = 0.0;
  const auto max = 0.5;
  const auto normalize = [&min, &max](const double value) {
    return (max - std::clamp(value, min, max)) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).values.count(METRICS::MANUAL_LONGITUDINAL_JERK) != 0) {
      manual_cost += normalize(
        std::pow(TIME_FACTOR, i) *
        std::abs(extract_data.at(i).values.at(METRICS::MANUAL_LONGITUDINAL_JERK)));
    }
    if (extract_data.at(i).values.count(METRICS::SYSTEM_LONGITUDINAL_JERK) != 0) {
      system_cost += normalize(
        std::pow(TIME_FACTOR, i) *
        std::abs(extract_data.at(i).values.at(METRICS::SYSTEM_LONGITUDINAL_JERK)));
    }
  }

  return {manual_cost / extract_data.size(), system_cost / extract_data.size()};
}

auto BehaviorAnalyzerNode::lateral_comfortability(const std::vector<Data> & extract_data) const
  -> std::pair<double, double>
{
  constexpr double TIME_FACTOR = 0.8;

  double manual_cost = 0.0;
  double system_cost = 0.0;

  const auto min = 0.0;
  const auto max = 0.5;
  const auto normalize = [&min, &max](const double value) {
    return (max - std::clamp(value, min, max)) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).values.count(METRICS::MANUAL_LATERAL_ACCEL) != 0) {
      manual_cost += normalize(
        std::pow(TIME_FACTOR, i) *
        std::abs(extract_data.at(i).values.at(METRICS::MANUAL_LATERAL_ACCEL)));
    }
    if (extract_data.at(i).values.count(METRICS::SYSTEM_LATERAL_ACCEL) != 0) {
      system_cost += normalize(
        std::pow(TIME_FACTOR, i) *
        std::abs(extract_data.at(i).values.at(METRICS::SYSTEM_LATERAL_ACCEL)));
    }
  }

  return {manual_cost / extract_data.size(), system_cost / extract_data.size()};
}

auto BehaviorAnalyzerNode::efficiency(const std::vector<Data> & extract_data) const
  -> std::pair<double, double>
{
  constexpr double TIME_FACTOR = 0.8;

  double manual_reward = 0.0;
  double system_reward = 0.0;

  const auto min = 0.0;
  const auto max = 20.0;
  const auto normalize = [&min, &max](const double value) {
    return std::clamp(value, min, max) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).values.count(METRICS::MANUAL_TRAVEL_DISTANCE) != 0) {
      manual_reward += normalize(
        std::pow(TIME_FACTOR, i) * extract_data.at(i).values.at(METRICS::MANUAL_TRAVEL_DISTANCE) /
        0.5);
    }
    if (extract_data.at(i).values.count(METRICS::SYSTEM_TRAVEL_DISTANCE) != 0) {
      system_reward += normalize(
        std::pow(TIME_FACTOR, i) * extract_data.at(i).values.at(METRICS::SYSTEM_TRAVEL_DISTANCE) /
        0.5);
    }
  }

  return {manual_reward / extract_data.size(), system_reward / extract_data.size()};
}

void BehaviorAnalyzerNode::visualize(const std::vector<Data> & extract_data) const
{
  MarkerArray msg;

  size_t i = 0;
  for (const auto & data : extract_data) {
    {
      Marker marker = createDefaultMarker(
        "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "future_poses", i++, Marker::ARROW,
        createMarkerScale(0.7, 0.3, 0.3), createMarkerColor(1.0, 0.0, 0.0, 0.999));
      marker.pose = data.odometry.pose.pose;
      msg.markers.push_back(marker);
    }

    {
      Marker marker = createDefaultMarker(
        "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "predicted_poses", i++, Marker::ARROW,
        createMarkerScale(0.7, 0.3, 0.3), createMarkerColor(1.0, 1.0, 0.0, 0.999));
      marker.pose = data.predicted_point.pose;
      msg.markers.push_back(marker);
    }
  }

  for (const auto & points : extract_data.front().candidate_trajectories) {
    Marker marker = createDefaultMarker(
      "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "candidate_poses", i++, Marker::LINE_STRIP,
      createMarkerScale(0.05, 0.0, 0.0), createMarkerColor(0.0, 0.0, 1.0, 0.999));
    for (const auto & point : points) {
      marker.points.push_back(point.pose.position);
    }
    msg.markers.push_back(marker);
  }

  pub_marker_->publish(msg);
}

void BehaviorAnalyzerNode::on_timer()
{
  if (!is_ready_) {
    return;
  }

  update(data_set_);

  auto extract_data = data_set_->extract(10.0, 0.5);

  process(extract_data);
}
}  // namespace autoware::behavior_analyzer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::behavior_analyzer::BehaviorAnalyzerNode)
