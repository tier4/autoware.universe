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

#include <autoware/universe_utils/ros/marker_helper.hpp>

namespace autoware::behavior_analyzer
{
using autoware::universe_utils::createDefaultMarker;
using autoware::universe_utils::createMarkerColor;
using autoware::universe_utils::createMarkerScale;
using autoware::universe_utils::Point2d;
using autoware::universe_utils::Polygon2d;

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

  pub_manual_metrics_ =
    create_publisher<Float32MultiArrayStamped>("~/manual_metrics", rclcpp::QoS{1});
  pub_system_metrics_ =
    create_publisher<Float32MultiArrayStamped>("~/system_metrics", rclcpp::QoS{1});
  pub_manual_score_ = create_publisher<Float32MultiArrayStamped>("~/manual_score", rclcpp::QoS{1});
  pub_system_score_ = create_publisher<Float32MultiArrayStamped>("~/system_score", rclcpp::QoS{1});

  srv_play_ = this->create_service<SetBool>(
    "play",
    std::bind(&BehaviorAnalyzerNode::play, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  srv_rewind_ = this->create_service<Trigger>(
    "rewind",
    std::bind(&BehaviorAnalyzerNode::rewind, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  reader_.open(declare_parameter<std::string>("bag_path"));

  trimmed_data_ = std::make_shared<TrimmedData>(
    duration_cast<nanoseconds>(reader_.get_metadata().starting_time.time_since_epoch()).count());

  parameters_ = std::make_shared<Parameters>();
  parameters_->resample_num = declare_parameter<int>("resample_num");
  parameters_->time_resolution = declare_parameter<double>("time_resolution");
  parameters_->target_state.lat_positions =
    declare_parameter<std::vector<double>>("target_state.lateral_positions");
  parameters_->target_state.lat_velocities =
    declare_parameter<std::vector<double>>("target_state.lateral_velocities");
  parameters_->target_state.lat_accelerations =
    declare_parameter<std::vector<double>>("target_state.lateral_accelerations");
  parameters_->target_state.lon_positions =
    declare_parameter<std::vector<double>>("target_state.longitudinal_positions");
  parameters_->target_state.lon_velocities =
    declare_parameter<std::vector<double>>("target_state.longitudinal_velocities");
  parameters_->target_state.lon_accelerations =
    declare_parameter<std::vector<double>>("target_state.longitudinal_accelerations");
}

void BehaviorAnalyzerNode::update(std::shared_ptr<TrimmedData> & trimmed_data) const
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

  trimmed_data->update(0.1 * 1e9);

  while (reader_.has_next()) {
    const auto next_data = reader_.read_next();
    rclcpp::SerializedMessage serialized_msg(*next_data->serialized_data);

    if (trimmed_data->is_ready()) {
      break;
    }

    if (next_data->topic_name == "/tf") {
      rclcpp::Serialization<TFMessage> serializer;
      const auto deserialized_message = std::make_shared<TFMessage>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      trimmed_data->buf_tf.append(*deserialized_message);
    }

    if (next_data->topic_name == "/localization/kinematic_state") {
      rclcpp::Serialization<Odometry> serializer;
      const auto deserialized_message = std::make_shared<Odometry>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      trimmed_data->buf_odometry.append(*deserialized_message);
    }

    if (next_data->topic_name == "/localization/acceleration") {
      rclcpp::Serialization<AccelWithCovarianceStamped> serializer;
      const auto deserialized_message = std::make_shared<AccelWithCovarianceStamped>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      trimmed_data->buf_accel.append(*deserialized_message);
    }

    if (next_data->topic_name == "/perception/object_recognition/objects") {
      rclcpp::Serialization<PredictedObjects> serializer;
      const auto deserialized_message = std::make_shared<PredictedObjects>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      trimmed_data->buf_objects.append(*deserialized_message);
    }

    if (next_data->topic_name == "/vehicle/status/steering_status") {
      rclcpp::Serialization<SteeringReport> serializer;
      const auto deserialized_message = std::make_shared<SteeringReport>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      trimmed_data->buf_steer.append(*deserialized_message);
    }

    if (next_data->topic_name == "/planning/scenario_planning/trajectory") {
      rclcpp::Serialization<Trajectory> serializer;
      const auto deserialized_message = std::make_shared<Trajectory>();
      serializer.deserialize_message(&serialized_msg, deserialized_message.get());
      trimmed_data->buf_trajectory.append(*deserialized_message);
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

  trimmed_data_.reset();
  trimmed_data_ = std::make_shared<TrimmedData>(
    duration_cast<nanoseconds>(reader_.get_metadata().starting_time.time_since_epoch()).count());

  res->success = true;
}

void BehaviorAnalyzerNode::process(const std::shared_ptr<TrimmedData> & trimmed_data) const
{
  const auto data_set = std::make_shared<DataSet>(trimmed_data, vehicle_info_, parameters_);

  const auto opt_tf = trimmed_data->buf_tf.get(trimmed_data->timestamp);
  if (opt_tf.has_value()) {
    pub_tf_->publish(opt_tf.value());
  }

  const auto opt_objects = trimmed_data->buf_objects.get(trimmed_data->timestamp);
  if (opt_objects.has_value()) {
    pub_objects_->publish(opt_objects.value());
  }

  const auto opt_trajectory = trimmed_data->buf_trajectory.get(trimmed_data->timestamp);
  if (opt_trajectory.has_value()) {
    pub_trajectory_->publish(opt_trajectory.value());
  }

  metrics(data_set);

  score(data_set);

  visualize(data_set);
}

void BehaviorAnalyzerNode::metrics(const std::shared_ptr<DataSet> & data_set) const
{
  {
    Float32MultiArrayStamped msg{};

    msg.stamp = now();
    msg.data.resize(static_cast<size_t>(METRIC::SIZE) * data_set->manual.resample_num);

    const auto set_metrics = [&msg](const auto & data, const auto metric_type) {
      const auto offset = static_cast<size_t>(metric_type) * data.resample_num;
      const auto metric = data.values.at(metric_type);
      std::copy(metric.begin(), metric.end(), msg.data.begin() + offset);
    };

    set_metrics(data_set->manual, METRIC::LATERAL_ACCEL);
    set_metrics(data_set->manual, METRIC::LONGITUDINAL_JERK);
    set_metrics(data_set->manual, METRIC::TRAVEL_DISTANCE);
    set_metrics(data_set->manual, METRIC::MINIMUM_TTC);

    pub_manual_metrics_->publish(msg);
  }

  {
    Float32MultiArrayStamped msg{};

    msg.stamp = now();
    msg.data.resize(static_cast<size_t>(METRIC::SIZE) * data_set->manual.resample_num);

    const auto set_metrics = [&msg](const auto & data, const auto metric_type) {
      const auto offset = static_cast<size_t>(metric_type) * data.resample_num;
      const auto metric = data.values.at(metric_type);
      std::copy(metric.begin(), metric.end(), msg.data.begin() + offset);
    };

    set_metrics(data_set->sampling.autoware(), METRIC::LATERAL_ACCEL);
    set_metrics(data_set->sampling.autoware(), METRIC::LONGITUDINAL_JERK);
    set_metrics(data_set->sampling.autoware(), METRIC::TRAVEL_DISTANCE);
    set_metrics(data_set->sampling.autoware(), METRIC::MINIMUM_TTC);

    pub_system_metrics_->publish(msg);
  }
}

void BehaviorAnalyzerNode::score(const std::shared_ptr<DataSet> & data_set) const
{
  {
    Float32MultiArrayStamped msg{};

    msg.stamp = now();
    msg.data.resize(static_cast<size_t>(SCORE::SIZE));

    const auto set_reward = [&msg](const auto & data, const auto score_type) {
      msg.data.at(static_cast<size_t>(score_type)) = static_cast<float>(data.scores.at(score_type));
    };

    set_reward(data_set->manual, SCORE::LONGITUDINAL_COMFORTABILITY);
    set_reward(data_set->manual, SCORE::LATERAL_COMFORTABILITY);
    set_reward(data_set->manual, SCORE::EFFICIENCY);
    set_reward(data_set->manual, SCORE::SAFETY);

    pub_manual_score_->publish(msg);
  }

  {
    Float32MultiArrayStamped msg{};

    msg.stamp = now();
    msg.data.resize(static_cast<size_t>(SCORE::SIZE));

    const auto set_reward = [&msg](const auto & data, const auto score_type) {
      msg.data.at(static_cast<size_t>(score_type)) = static_cast<float>(data.scores.at(score_type));
    };

    set_reward(data_set->sampling.autoware(), SCORE::LONGITUDINAL_COMFORTABILITY);
    set_reward(data_set->sampling.autoware(), SCORE::LATERAL_COMFORTABILITY);
    set_reward(data_set->sampling.autoware(), SCORE::EFFICIENCY);
    set_reward(data_set->sampling.autoware(), SCORE::SAFETY);

    pub_system_score_->publish(msg);
  }
}

void BehaviorAnalyzerNode::visualize(const std::shared_ptr<DataSet> & data_set) const
{
  MarkerArray msg;

  size_t i = 0;
  for (const auto & point : data_set->manual.odometry_history) {
    Marker marker = createDefaultMarker(
      "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "manual", i++, Marker::ARROW,
      createMarkerScale(0.7, 0.3, 0.3), createMarkerColor(1.0, 0.0, 0.0, 0.999));
    marker.pose = point.pose.pose;
    msg.markers.push_back(marker);
  }

  for (const auto & point : data_set->sampling.autoware().points) {
    Marker marker = createDefaultMarker(
      "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "system", i++, Marker::ARROW,
      createMarkerScale(0.7, 0.3, 0.3), createMarkerColor(1.0, 1.0, 0.0, 0.999));
    marker.pose = point.pose;
    msg.markers.push_back(marker);
  }

  for (const auto & trajectory : data_set->sampling.data) {
    Marker marker = createDefaultMarker(
      "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "candidates", i++, Marker::LINE_STRIP,
      createMarkerScale(0.05, 0.0, 0.0), createMarkerColor(0.0, 0.0, 1.0, 0.999));
    if (!trajectory.feasible()) {
      for (const auto & point : trajectory.points) {
        marker.points.push_back(point.pose.position);
        marker.colors.push_back(createMarkerColor(0.1, 0.1, 0.1, 0.5));
      }
    } else {
      for (const auto & point : trajectory.points) {
        marker.points.push_back(point.pose.position);
        marker.colors.push_back(createMarkerColor(0.0, 0.0, 1.0, 0.999));
      }
    }
    msg.markers.push_back(marker);
  }

  {
    Marker marker = createDefaultMarker(
      "map", rclcpp::Clock{RCL_ROS_TIME}.now(), "best", i++, Marker::LINE_STRIP,
      createMarkerScale(0.2, 0.0, 0.0), createMarkerColor(1.0, 1.0, 1.0, 0.999));
    for (const auto & point : data_set->sampling.best().points) {
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

  update(trimmed_data_);

  process(trimmed_data_);
}
}  // namespace autoware::behavior_analyzer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::behavior_analyzer::BehaviorAnalyzerNode)
