// Copyright 2022 TIER IV, Inc.
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

#include "autoware/behavior_path_static_obstacle_avoidance_module/data_structs.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/type_alias.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include <autoware_perception_msgs/msg/object_classification.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace autoware::behavior_path_planner::static_obstacle_avoidance
{

using autoware::behavior_path_planner::AvoidanceParameters;
using autoware::behavior_path_planner::ObjectData;
using autoware::route_handler::Direction;
using autoware::universe_utils::createPoint;
using autoware::universe_utils::createQuaternion;
using autoware::universe_utils::createVector3;
using autoware::universe_utils::generateUUID;

PathWithLaneId generatePath(const geometry_msgs::msg::Pose & pose)
{
  constexpr double interval_distance = 1.0;

  PathWithLaneId traj;
  for (double s = 0.0; s <= 10.0 * interval_distance; s += interval_distance) {
    PathPointWithLaneId p;
    p.point.pose = pose;
    p.point.pose.position.x += s;
    p.point.longitudinal_velocity_mps = 20.0;  // m/s
    traj.points.push_back(p);
  }

  return traj;
}

TEST(TestUtils, isSameDirectionShift)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::isOnRight;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::isSameDirectionShift;

  constexpr double negative_shift_length = -1.0;
  constexpr double positive_shift_length = 1.0;

  {
    ObjectData object;
    object.direction = Direction::RIGHT;

    ASSERT_TRUE(isSameDirectionShift(isOnRight(object), negative_shift_length));
    ASSERT_FALSE(isSameDirectionShift(isOnRight(object), positive_shift_length));
  }

  {
    ObjectData object;
    object.direction = Direction::LEFT;
    ASSERT_TRUE(isSameDirectionShift(isOnRight(object), positive_shift_length));
    ASSERT_FALSE(isSameDirectionShift(isOnRight(object), negative_shift_length));
  }

  {
    ObjectData object;
    object.direction = Direction::NONE;
    EXPECT_THROW(isSameDirectionShift(isOnRight(object), positive_shift_length), std::logic_error);
  }
}

TEST(TestUtils, isShiftNecessary)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::isOnRight;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::isShiftNecessary;

  constexpr double negative_shift_length = -1.0;
  constexpr double positive_shift_length = 1.0;

  {
    ObjectData object;
    object.direction = Direction::RIGHT;

    ASSERT_TRUE(isShiftNecessary(isOnRight(object), positive_shift_length));
    ASSERT_FALSE(isShiftNecessary(isOnRight(object), negative_shift_length));
  }

  {
    ObjectData object;
    object.direction = Direction::LEFT;
    ASSERT_TRUE(isShiftNecessary(isOnRight(object), negative_shift_length));
    ASSERT_FALSE(isShiftNecessary(isOnRight(object), positive_shift_length));
  }

  {
    ObjectData object;
    object.direction = Direction::NONE;
    EXPECT_THROW(isShiftNecessary(isOnRight(object), positive_shift_length), std::logic_error);
  }
}

TEST(TestUtils, calcShiftLength)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::calcShiftLength;

  {
    constexpr bool is_on_right = true;
    constexpr double overhang = -1.0;
    constexpr double margin = 1.5;

    const auto output = calcShiftLength(is_on_right, overhang, margin);
    EXPECT_DOUBLE_EQ(output, 0.5);
  }

  {
    constexpr bool is_on_right = false;
    constexpr double overhang = -1.0;
    constexpr double margin = 1.5;

    const auto output = calcShiftLength(is_on_right, overhang, margin);
    EXPECT_DOUBLE_EQ(output, -2.5);
  }
}

TEST(TestUtils, insertDecelPoint)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::insertDecelPoint;

  // invalid target decel point
  {
    const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                             .position(createPoint(0.0, 0.0, 0.0))
                             .orientation(createQuaternion(0.0, 0.0, 0.0, 1.0));

    auto path = generatePath(edge_pose);

    const auto ego_position = createPoint(2.5, 0.5, 0.0);
    constexpr double offset = 100.0;
    constexpr double velocity = 1.0;

    std::optional<geometry_msgs::msg::Pose> p_out{std::nullopt};
    insertDecelPoint(ego_position, offset, velocity, path, p_out);

    EXPECT_FALSE(p_out.has_value());
    std::for_each(path.points.begin(), path.points.end(), [](const auto & p) {
      EXPECT_DOUBLE_EQ(p.point.longitudinal_velocity_mps, 20.0);
    });
  }

  // nominal case
  {
    const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                             .position(createPoint(0.0, 0.0, 0.0))
                             .orientation(createQuaternion(0.0, 0.0, 0.0, 1.0));

    auto path = generatePath(edge_pose);

    const auto ego_position = createPoint(3.5, 0.5, 0.0);
    constexpr double offset = 3.0;
    constexpr double velocity = 1.0;

    std::optional<geometry_msgs::msg::Pose> p_out{std::nullopt};
    insertDecelPoint(ego_position, offset, velocity, path, p_out);

    EXPECT_TRUE(p_out.has_value());
    EXPECT_DOUBLE_EQ(p_out.value().position.x, 6.5);
    EXPECT_DOUBLE_EQ(p_out.value().position.y, 0.0);
    EXPECT_DOUBLE_EQ(p_out.value().position.z, 0.0);
    for (size_t i = 7; i < path.points.size(); i++) {
      EXPECT_DOUBLE_EQ(path.points.at(i).point.longitudinal_velocity_mps, 1.0);
    }
  }
}

TEST(TestUtils, fillObjectMovingTime)
{
  using namespace std::literals::chrono_literals;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::fillObjectMovingTime;

  const auto create_params = [](const double th_speed, const double th_time) {
    ObjectParameter param{};
    param.moving_speed_threshold = th_speed;
    param.moving_time_threshold = th_time;
    return param;
  };

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_parameters.emplace(ObjectClassification::TRUCK, create_params(0.5, 1.0));

  const auto uuid = generateUUID();

  ObjectData old_object;
  old_object.object.object_id = uuid;
  old_object.last_move = rclcpp::Clock{RCL_ROS_TIME}.now();
  old_object.last_stop = rclcpp::Clock{RCL_ROS_TIME}.now();
  old_object.move_time = 0.0;
  old_object.object.classification.emplace_back(
    autoware_perception_msgs::build<ObjectClassification>()
      .label(ObjectClassification::TRUCK)
      .probability(1.0));
  old_object.object.kinematics.initial_twist_with_covariance.twist.linear =
    createVector3(0.0, 0.0, 0.0);

  rclcpp::sleep_for(500ms);

  // find new stop object
  {
    ObjectData new_object;
    new_object.object.object_id = generateUUID();
    new_object.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    new_object.object.kinematics.initial_twist_with_covariance.twist.linear =
      createVector3(0.49, 0.0, 0.0);

    ObjectDataArray buffer{};
    fillObjectMovingTime(new_object, buffer, parameters);
    EXPECT_NEAR(new_object.stop_time, 0.0, 1e-3);
    EXPECT_NEAR(new_object.move_time, 0.0, 1e-3);
    EXPECT_FALSE(buffer.empty());
  }

  // find new move object
  {
    ObjectData new_object;
    new_object.object.object_id = generateUUID();
    new_object.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    new_object.object.kinematics.initial_twist_with_covariance.twist.linear =
      createVector3(0.51, 0.0, 0.0);

    ObjectDataArray buffer{};
    fillObjectMovingTime(new_object, buffer, parameters);
    EXPECT_NEAR(new_object.stop_time, 0.0, 1e-3);
    EXPECT_TRUE(buffer.empty());
  }

  // stop to move (moving time < threshold)
  {
    ObjectData new_object;
    new_object.object.object_id = uuid;
    new_object.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    new_object.object.kinematics.initial_twist_with_covariance.twist.linear =
      createVector3(0.51, 0.0, 0.0);

    ObjectDataArray buffer{old_object};
    fillObjectMovingTime(new_object, buffer, parameters);

    EXPECT_NEAR(new_object.stop_time, 0.0, 1e-3);
    EXPECT_NEAR(new_object.move_time, 0.5, 1e-3);
    EXPECT_FALSE(buffer.empty());
  }

  // stop to stop
  {
    ObjectData new_object;
    new_object.object.object_id = uuid;
    new_object.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    new_object.object.kinematics.initial_twist_with_covariance.twist.linear =
      createVector3(0.49, 0.0, 0.0);

    ObjectDataArray buffer{old_object};
    fillObjectMovingTime(new_object, buffer, parameters);

    EXPECT_NEAR(new_object.stop_time, 0.5, 1e-3);
    EXPECT_NEAR(new_object.move_time, 0.0, 1e-3);
    EXPECT_FALSE(buffer.empty());
  }

  rclcpp::sleep_for(500ms);

  // stop to move (threshold < moving time)
  {
    ObjectData new_object;
    new_object.object.object_id = uuid;
    new_object.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    new_object.object.kinematics.initial_twist_with_covariance.twist.linear =
      createVector3(0.51, 0.0, 0.0);

    ObjectDataArray buffer{old_object};
    fillObjectMovingTime(new_object, buffer, parameters);

    EXPECT_NEAR(new_object.stop_time, 0.0, 1e-3);
    EXPECT_NEAR(new_object.move_time, 1.0, 1e-3);
    EXPECT_TRUE(buffer.empty());
  }
}

}  // namespace autoware::behavior_path_planner::static_obstacle_avoidance
