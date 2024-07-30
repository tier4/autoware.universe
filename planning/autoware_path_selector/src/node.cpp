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

namespace autoware::path_selector
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
}  // namespace

PathSelectorNode::PathSelectorNode(const rclcpp::NodeOptions & node_options)
: Node("path_selector_node", node_options)
{
  using namespace std::literals::chrono_literals;
  timer_ =
    rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&PathSelectorNode::on_timer, this));

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
    "play", std::bind(&PathSelectorNode::play, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  srv_rewind_ = this->create_service<Trigger>(
    "rewind",
    std::bind(&PathSelectorNode::rewind, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  reader_.open(declare_parameter<std::string>("bag_path"));

  data_set_ = std::make_shared<DataSet>(
    duration_cast<nanoseconds>(reader_.get_metadata().starting_time.time_since_epoch()).count());
}

void PathSelectorNode::update(std::shared_ptr<DataSet> & data_set) const
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

void PathSelectorNode::play(const SetBool::Request::SharedPtr req, SetBool::Response::SharedPtr res)
{
  is_ready_ = req->data;
  if (is_ready_) {
    RCLCPP_INFO(get_logger(), "start evaluation.");
  } else {
    RCLCPP_INFO(get_logger(), "stop evaluation.");
  }
  res->success = true;
}

void PathSelectorNode::rewind(
  [[maybe_unused]] const Trigger::Request::SharedPtr req, Trigger::Response::SharedPtr res)
{
  reader_.seek(0);

  data_set_.reset();
  data_set_ = std::make_shared<DataSet>(
    duration_cast<nanoseconds>(reader_.get_metadata().starting_time.time_since_epoch()).count());

  res->success = true;
}

auto PathSelectorNode::all_ttc(const Data & data) const -> std::vector<double>
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

double PathSelectorNode::lateral_accel(const Data & data) const
{
  const auto radius = vehicle_info_.wheel_base_m / std::tan(data.steer.steering_tire_angle);
  const auto speed = data.odometry.twist.twist.linear.x;
  return speed * speed / radius;
}

double PathSelectorNode::longitudinal_jerk(const Data & front_data, const Data & back_data) const
{
  const auto & front_accel = front_data.accel;
  const auto & back_accel = back_data.accel;

  const double dt = rclcpp::Time(back_accel.header.stamp).nanoseconds() -
                    rclcpp::Time(front_accel.header.stamp).nanoseconds();

  return 1e9 * (back_accel.accel.accel.linear.x - front_accel.accel.accel.linear.x) / dt;
}

double PathSelectorNode::travel_distance(const Data & front_data, const Data & back_data) const
{
  const auto travel_distance = autoware::universe_utils::calcDistance3d(
    front_data.odometry.pose.pose, back_data.odometry.pose.pose);
  return travel_distance + front_data.metrics.at("travel_distance");
}

void PathSelectorNode::process(std::vector<Data> & extract_data) const
{
  if (extract_data.empty()) {
    return;
  }

  {
    // travel distance
    {
      extract_data.front().metrics.emplace("travel_distance", 0.0);
    }
  }

  for (size_t i = 0; i < extract_data.size() - 1; i++) {
    // longitudinal acceleration
    {
      extract_data.at(i).metrics.emplace(
        "lon_accel", extract_data.at(i).accel.accel.accel.linear.x);
    }

    // lateral acceleration
    {
      extract_data.at(i).metrics.emplace("lat_accel", lateral_accel(extract_data.at(i)));
    }

    // minimum ttc
    std::vector<double> ttc = all_ttc(extract_data.at(i));
    std::sort(ttc.begin(), ttc.end());
    if (!ttc.empty()) {
      extract_data.at(i).metrics.emplace("ttc_min", ttc.front());
    }

    // longitudinal jerk
    {
      extract_data.at(i).metrics.emplace(
        "lon_jerk", longitudinal_jerk(extract_data.at(i), extract_data.at(i + 1)));
    }

    // travel distance
    {
      extract_data.at(i + 1).metrics.emplace(
        "travel_distance", travel_distance(extract_data.at(i), extract_data.at(i + 1)));
    }
  }

  {
    // acceleration
    {
      extract_data.back().metrics.emplace(
        "lon_accel", extract_data.back().accel.accel.accel.linear.x);
    }

    // minimum ttc
    std::vector<double> ttc = all_ttc(extract_data.back());
    std::sort(ttc.begin(), ttc.end());
    if (!ttc.empty()) {
      extract_data.back().metrics.emplace("ttc_min", ttc.front());
    }

    // longitudinal jerk
    {
      extract_data.back().metrics.emplace("lon_jerk", 0.0);
    }
  }

  Float32MultiArrayStamped metrics{};

  constexpr size_t METRICS_NUM = 5;

  metrics.stamp = now();
  metrics.data.resize(METRICS_NUM * extract_data.size());
  metrics.layout.dim.resize(METRICS_NUM);

  metrics.layout.dim.at(0).label = "lon_accel";
  metrics.layout.dim.at(0).size = extract_data.size();
  metrics.layout.dim.at(1).label = "ttc_min";
  metrics.layout.dim.at(1).size = extract_data.size();
  metrics.layout.dim.at(2).label = "lon_jerk";
  metrics.layout.dim.at(2).size = extract_data.size();
  metrics.layout.dim.at(3).label = "travel_distance";
  metrics.layout.dim.at(3).size = extract_data.size();
  metrics.layout.dim.at(4).label = "lat_accel";
  metrics.layout.dim.at(4).size = extract_data.size();

  for (size_t i = 0; i < extract_data.size(); i++) {
    // acceleration
    if (extract_data.at(i).metrics.count("lon_accel") != 0) {
      metrics.data.at(0 * metrics.layout.dim.at(0).size + i) =
        extract_data.at(i).metrics.at("lon_accel");
    }

    // minimum ttc
    if (extract_data.at(i).metrics.count("ttc_min") != 0) {
      metrics.data.at(1 * metrics.layout.dim.at(1).size + i) =
        extract_data.at(i).metrics.at("ttc_min");
    }

    // longitudinal jerk
    if (extract_data.at(i).metrics.count("lon_jerk") != 0) {
      metrics.data.at(2 * metrics.layout.dim.at(2).size + i) =
        extract_data.at(i).metrics.at("lon_jerk");
    }

    // travel_distance
    if (extract_data.at(i).metrics.count("travel_distance") != 0) {
      metrics.data.at(3 * metrics.layout.dim.at(3).size + i) =
        extract_data.at(i).metrics.at("travel_distance");
    }

    // acceleration
    if (extract_data.at(i).metrics.count("lat_accel") != 0) {
      metrics.data.at(4 * metrics.layout.dim.at(0).size + i) =
        extract_data.at(i).metrics.at("lat_accel");
    }
  }

  Float32MultiArrayStamped cost{};

  constexpr size_t COST_NUM = 4;

  cost.stamp = now();
  cost.data.resize(COST_NUM);

  cost.data.at(0) = longitudinal_comfortability(extract_data);
  cost.data.at(1) = lateral_comfortability(extract_data);
  cost.data.at(2) = efficiency(extract_data);
  cost.data.at(3) = safety(extract_data);

  pub_tf_->publish(extract_data.front().tf);

  pub_objects_->publish(extract_data.front().objects);

  pub_trajectory_->publish(extract_data.front().trajectory);

  pub_metrics_->publish(metrics);

  pub_cost_->publish(cost);

  visualize(extract_data);
}

double PathSelectorNode::safety(const std::vector<Data> & extract_data) const
{
  constexpr double TIME_FACTOR = 0.8;

  double cost = 0.0;

  const auto min = 0.0;
  const auto max = 5.0;
  const auto normalize = [&min, &max](const double value) {
    return std::clamp(value, min, max) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).metrics.count("ttc_min") != 0) {
      cost += normalize(std::pow(TIME_FACTOR, i) * extract_data.at(i).metrics.at("ttc_min"));
    }
  }

  return cost / extract_data.size();
}

double PathSelectorNode::longitudinal_comfortability(const std::vector<Data> & extract_data) const
{
  constexpr double TIME_FACTOR = 0.8;

  double cost = 0.0;

  const auto min = 0.0;
  const auto max = 0.5;
  const auto normalize = [&min, &max](const double value) {
    return (max - std::clamp(value, min, max)) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).metrics.count("lon_jerk") != 0) {
      cost +=
        normalize(std::pow(TIME_FACTOR, i) * std::abs(extract_data.at(i).metrics.at("lon_jerk")));
    }
  }

  return cost / extract_data.size();
}

double PathSelectorNode::lateral_comfortability(const std::vector<Data> & extract_data) const
{
  constexpr double TIME_FACTOR = 0.8;

  double cost = 0.0;

  const auto min = 0.0;
  const auto max = 0.5;
  const auto normalize = [&min, &max](const double value) {
    return (max - std::clamp(value, min, max)) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).metrics.count("lat_accel") != 0) {
      cost +=
        normalize(std::pow(TIME_FACTOR, i) * std::abs(extract_data.at(i).metrics.at("lat_accel")));
    }
  }

  return cost / extract_data.size();
}

double PathSelectorNode::efficiency(const std::vector<Data> & extract_data) const
{
  constexpr double TIME_FACTOR = 0.8;

  double reward = 0.0;

  const auto min = 0.0;
  const auto max = 20.0;
  const auto normalize = [&min, &max](const double value) {
    return std::clamp(value, min, max) / (max - min);
  };

  for (size_t i = 0; i < extract_data.size(); i++) {
    if (extract_data.at(i).metrics.count("travel_distance") != 0) {
      reward += normalize(
        std::pow(TIME_FACTOR, i) * extract_data.at(i).metrics.at("travel_distance") / 0.5);
    }
  }

  return reward / extract_data.size();
}

void PathSelectorNode::visualize(const std::vector<Data> & extract_data) const
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

    if (data.metrics.count("ttc_min") != 0) {
      Marker marker = createDefaultMarker(
        "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "ttc", i++, Marker::TEXT_VIEW_FACING,
        createMarkerScale(1.0, 1.0, 1.0), createMarkerColor(1.0, 1.0, 1.0, 1.0));
      marker.pose = data.odometry.pose.pose;

      std::ostringstream string_stream;
      string_stream << data.metrics.at("ttc_min") << "[s]";
      marker.text = string_stream.str();
      msg.markers.push_back(marker);
    }
  }

  pub_marker_->publish(msg);
}

void PathSelectorNode::on_timer()
{
  if (!is_ready_) {
    return;
  }

  update(data_set_);

  auto extract_data = data_set_->extract(10.0, 0.5);

  process(extract_data);
}
}  // namespace autoware::path_selector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::path_selector::PathSelectorNode)
