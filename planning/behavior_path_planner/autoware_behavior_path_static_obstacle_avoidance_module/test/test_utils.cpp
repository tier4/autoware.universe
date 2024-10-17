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

#include "../src/utils.cpp"  // NOLINT
#include "autoware/behavior_path_static_obstacle_avoidance_module/data_structs.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/type_alias.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"
#include "autoware/universe_utils/math/unit_conversion.hpp"

#include <autoware_perception_msgs/msg/object_classification.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace autoware::behavior_path_planner::static_obstacle_avoidance
{

using autoware::behavior_path_planner::AvoidanceParameters;
using autoware::behavior_path_planner::ObjectData;
using autoware::route_handler::Direction;
using autoware::universe_utils::createPoint;
using autoware::universe_utils::createQuaternionFromRPY;
using autoware::universe_utils::createVector3;
using autoware::universe_utils::deg2rad;
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

TEST(TestUtils, isMovingObject)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::filtering_utils::
    isMovingObject;

  const auto create_params = [](const double th_time) {
    ObjectParameter param{};
    param.moving_time_threshold = th_time;
    return param;
  };

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_parameters.emplace(ObjectClassification::TRUCK, create_params(1.0));

  {
    ObjectData object_data;
    object_data.move_time = 0.5;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    EXPECT_FALSE(isMovingObject(object_data, parameters));
  }

  {
    ObjectData object_data;
    object_data.move_time = 1.5;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    EXPECT_TRUE(isMovingObject(object_data, parameters));
  }
}

TEST(TestUtils, getObjectBehavior)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::filtering_utils::
    getObjectBehavior;

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_check_yaw_deviation = deg2rad(20);

  lanelet::LineString3d left_bound;
  lanelet::LineString3d right_bound;

  left_bound.push_back(lanelet::Point3d{lanelet::InvalId, -1, -1});
  left_bound.push_back(lanelet::Point3d{lanelet::InvalId, 0, -1});
  left_bound.push_back(lanelet::Point3d{lanelet::InvalId, 1, -1});
  right_bound.push_back(lanelet::Point3d{lanelet::InvalId, -1, 1});
  right_bound.push_back(lanelet::Point3d{lanelet::InvalId, 0, 1});
  right_bound.push_back(lanelet::Point3d{lanelet::InvalId, 1, 1});

  lanelet::Lanelet lanelet{lanelet::InvalId, left_bound, right_bound};

  lanelet::LineString3d centerline;
  centerline.push_back(lanelet::Point3d{lanelet::InvalId, -1, 0});
  centerline.push_back(lanelet::Point3d{lanelet::InvalId, 0, 0});
  centerline.push_back(lanelet::Point3d{lanelet::InvalId, 1, 0});

  lanelet.setCenterline(centerline);

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::LEFT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, 0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::NONE);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::RIGHT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, -0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(170)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::NONE);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::LEFT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, 0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(30)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::DEVIATING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::RIGHT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, -0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(30)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::MERGING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::LEFT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, 0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(-150)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::DEVIATING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::RIGHT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, -0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(-150)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::MERGING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::LEFT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, 0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(-30)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::MERGING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::RIGHT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, -0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(-30)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::DEVIATING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::LEFT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, 0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(150)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::MERGING);
  }

  {
    ObjectData object_data;
    object_data.overhang_lanelet = lanelet;
    object_data.direction = Direction::RIGHT;
    object_data.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(0.0, -0.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(150)));

    EXPECT_EQ(getObjectBehavior(object_data, parameters), ObjectData::Behavior::DEVIATING);
  }
}

TEST(TestUtils, isNoNeedAvoidanceBehavior)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    calcEnvelopeOverhangDistance;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::filtering_utils::
    isNoNeedAvoidanceBehavior;

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->lateral_execution_threshold = 0.5;

  // object is NOT avoidable. but there is possibility that the ego has to avoid it.
  {
    ObjectData object_data;
    object_data.avoid_margin = std::nullopt;

    EXPECT_FALSE(isNoNeedAvoidanceBehavior(object_data, parameters));
  }

  // don't have to avoid.
  {
    const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                             .position(createPoint(0.0, 0.0, 0.0))
                             .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    const auto path = generatePath(edge_pose);

    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, 3.6, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.avoid_margin = 2.0;
    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);
    object_data.overhang_points = calcEnvelopeOverhangDistance(object_data, path);

    EXPECT_TRUE(isNoNeedAvoidanceBehavior(object_data, parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::ENOUGH_LATERAL_DISTANCE);
  }

  // larger than execution threshold.
  {
    const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                             .position(createPoint(0.0, 0.0, 0.0))
                             .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    const auto path = generatePath(edge_pose);

    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, 3.4, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.avoid_margin = 2.0;
    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);
    object_data.overhang_points = calcEnvelopeOverhangDistance(object_data, path);

    EXPECT_TRUE(isNoNeedAvoidanceBehavior(object_data, parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::LESS_THAN_EXECUTION_THRESHOLD);
  }

  // larger than execution threshold.
  {
    const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                             .position(createPoint(0.0, 0.0, 0.0))
                             .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    const auto path = generatePath(edge_pose);

    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, 2.9, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.avoid_margin = 2.0;
    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);
    object_data.overhang_points = calcEnvelopeOverhangDistance(object_data, path);

    EXPECT_FALSE(isNoNeedAvoidanceBehavior(object_data, parameters));
  }
}

TEST(TestUtils, getAvoidMargin)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::filtering_utils::
    getAvoidMargin;

  const auto create_params = [](
                               const double soft, const double hard, const double hard_for_parked) {
    ObjectParameter param{};
    param.lateral_soft_margin = soft;
    param.lateral_hard_margin = hard;
    param.lateral_hard_margin_for_parked_vehicle = hard_for_parked;
    return param;
  };

  const auto planner_data = std::make_shared<PlannerData>();
  planner_data->parameters.vehicle_width = 2.0;

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_parameters.emplace(ObjectClassification::TRUCK, create_params(0.7, 0.2, 0.7));
  parameters->hard_drivable_bound_margin = 0.1;
  parameters->soft_drivable_bound_margin = 0.5;

  // wide road
  {
    ObjectData object_data;
    object_data.is_parked = false;
    object_data.distance_factor = 1.0;
    object_data.to_road_shoulder_distance = 5.0;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto output = getAvoidMargin(object_data, planner_data, parameters);
    ASSERT_TRUE(output.has_value());
    EXPECT_DOUBLE_EQ(output.value(), 1.9);
  }

  // narrow road (relax lateral soft margin)
  {
    ObjectData object_data;
    object_data.is_parked = false;
    object_data.distance_factor = 1.0;
    object_data.to_road_shoulder_distance = 3.0;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto output = getAvoidMargin(object_data, planner_data, parameters);
    ASSERT_TRUE(output.has_value());
    EXPECT_DOUBLE_EQ(output.value(), 1.5);
  }

  // narrow road (relax drivable bound margin)
  {
    ObjectData object_data;
    object_data.is_parked = false;
    object_data.distance_factor = 1.0;
    object_data.to_road_shoulder_distance = 2.5;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto output = getAvoidMargin(object_data, planner_data, parameters);
    ASSERT_TRUE(output.has_value());
    EXPECT_DOUBLE_EQ(output.value(), 1.2);
  }

  // road width is not enough.
  {
    ObjectData object_data;
    object_data.is_parked = true;
    object_data.distance_factor = 1.0;
    object_data.to_road_shoulder_distance = 2.5;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto output = getAvoidMargin(object_data, planner_data, parameters);
    EXPECT_FALSE(output.has_value());
  }
}

TEST(TestUtils, isSatisfiedWithCommonCondition)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::filtering_utils::
    isSatisfiedWithCommonCondition;

  constexpr double forward_detection_range = 5.5;

  const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                           .position(createPoint(0.0, 0.0, 0.0))
                           .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

  const auto path = generatePath(edge_pose);

  const auto create_params = [](const double th_time, const double is_target) {
    ObjectParameter param{};
    param.moving_time_threshold = th_time;
    param.is_avoidance_target = is_target;
    return param;
  };

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_parameters.emplace(ObjectClassification::UNKNOWN, create_params(0.5, false));
  parameters->object_parameters.emplace(ObjectClassification::TRUCK, create_params(0.5, true));
  parameters->object_check_backward_distance = 1.0;
  parameters->object_check_goal_distance = 2.0;

  // no configuration for this object.
  {
    ObjectData object_data;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::PEDESTRIAN)
        .probability(1.0));

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(0.0, 0.0, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::IS_NOT_TARGET_OBJECT);
  }

  // not target object.
  {
    ObjectData object_data;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::UNKNOWN)
        .probability(1.0));

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(0.0, 0.0, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::IS_NOT_TARGET_OBJECT);
  }

  // moving object.
  {
    ObjectData object_data;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.6;

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(0.0, 0.0, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::MOVING_OBJECT);
  }

  // object behind the ego.
  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.4;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(8.0, 0.5, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::FURTHER_THAN_THRESHOLD);
  }

  // farther than detection range.
  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(7.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.4;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(0.0, 0.0, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::FURTHER_THAN_THRESHOLD);
  }

  // farther than goal position.
  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(7.0, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.4;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(0.0, 0.0, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::FURTHER_THAN_GOAL);
  }

  // within detection range.
  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(4.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.4;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    EXPECT_FALSE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 6.4, createPoint(0.0, 0.0, 0.0), false,
      parameters));
    EXPECT_EQ(object_data.info, ObjectInfo::TOO_NEAR_TO_GOAL);
  }

  // within detection range.
  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(4.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.4;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    EXPECT_TRUE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 6.6, createPoint(0.0, 0.0, 0.0), false,
      parameters));
  }

  // within detection range.
  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(4.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    object_data.move_time = 0.4;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    EXPECT_TRUE(isSatisfiedWithCommonCondition(
      object_data, path, forward_detection_range, 4.0, createPoint(0.0, 0.0, 0.0), true,
      parameters));
  }
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
                             .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

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
                             .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

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

TEST(TestUtils, calcEnvelopeOverhangDistance)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    calcEnvelopeOverhangDistance;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;

  const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                           .position(createPoint(0.0, 0.0, 0.0))
                           .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

  auto path = generatePath(edge_pose);

  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -0.5, epsilon);

    const auto output = calcEnvelopeOverhangDistance(object_data, path);

    ASSERT_EQ(output.size(), 5);
    EXPECT_NEAR(output.at(0).first, -0.5, epsilon);
    EXPECT_NEAR(output.at(1).first, -0.5, epsilon);
    EXPECT_NEAR(output.at(2).first, -0.5, epsilon);
    EXPECT_NEAR(output.at(3).first, 2.5, epsilon);
    EXPECT_NEAR(output.at(4).first, 2.5, epsilon);
  }

  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, -1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::RIGHT;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -2.5, epsilon);

    const auto output = calcEnvelopeOverhangDistance(object_data, path);

    ASSERT_EQ(output.size(), 5);
    EXPECT_NEAR(output.at(0).first, 0.5, epsilon);
    EXPECT_NEAR(output.at(1).first, 0.5, epsilon);
    EXPECT_NEAR(output.at(2).first, -2.5, epsilon);
    EXPECT_NEAR(output.at(3).first, -2.5, epsilon);
    EXPECT_NEAR(output.at(4).first, -2.5, epsilon);
  }
}

TEST(TestUtils, createEnvelopePolygon)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;

  Polygon2d footprint;
  footprint.outer() = {
    Point2d{1.0, 0.0}, Point2d{3.0, 0.0}, Point2d{3.0, 1.0}, Point2d{1.0, 1.0}, Point2d{1.0, 0.0}};

  constexpr double margin = 0.353553390593273762;
  const auto pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                      .position(createPoint(0.0, 0.0, 0.0))
                      .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

  const auto output = createEnvelopePolygon(footprint, pose, margin);

  constexpr auto epsilon = 1e-6;
  ASSERT_EQ(output.outer().size(), 5);
  EXPECT_NEAR(output.outer().at(0).x(), 2.0, epsilon);
  EXPECT_NEAR(output.outer().at(0).y(), -1.5, epsilon);
  EXPECT_NEAR(output.outer().at(1).x(), 0.0, epsilon);
  EXPECT_NEAR(output.outer().at(1).y(), 0.5, epsilon);
  EXPECT_NEAR(output.outer().at(2).x(), 2.0, epsilon);
  EXPECT_NEAR(output.outer().at(2).y(), 2.5, epsilon);
  EXPECT_NEAR(output.outer().at(3).x(), 4.0, epsilon);
  EXPECT_NEAR(output.outer().at(3).y(), 0.5, epsilon);
  EXPECT_NEAR(output.outer().at(4).x(), 2.0, epsilon);
  EXPECT_NEAR(output.outer().at(4).y(), -1.5, epsilon);
}

TEST(TestUtils, generateObstaclePolygonsForDrivableArea)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    generateObstaclePolygonsForDrivableArea;

  const auto create_params = [](
                               const double envelope_buffer, const double hard, const double soft) {
    ObjectParameter param{};
    param.envelope_buffer_margin = envelope_buffer;
    param.lateral_soft_margin = soft;
    param.lateral_hard_margin = hard;
    return param;
  };

  constexpr double vehicle_width = 1.0;
  constexpr double envelope_buffer = 0.353553390593273762;
  constexpr double lateral_soft_margin = 0.707106781186547524;

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_parameters.emplace(
    ObjectClassification::TRUCK, create_params(envelope_buffer, 0.5, lateral_soft_margin));

  // empty
  {
    ObjectDataArray objects{};
    const auto output = generateObstaclePolygonsForDrivableArea(objects, parameters, vehicle_width);
    EXPECT_TRUE(output.empty());
  }

  // invalid margin
  {
    ObjectData object_data;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    const auto object_type = utils::getHighestProbLabel(object_data.object.classification);
    const auto object_parameter = parameters->object_parameters.at(object_type);
    object_data.avoid_margin = std::nullopt;

    Polygon2d footprint;
    footprint.outer() = {
      Point2d{1.0, 0.0}, Point2d{3.0, 0.0}, Point2d{3.0, 1.0}, Point2d{1.0, 1.0},
      Point2d{1.0, 0.0}};

    const auto pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                        .position(createPoint(0.0, 0.0, 0.0))
                        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    object_data.envelope_poly =
      createEnvelopePolygon(footprint, pose, object_parameter.envelope_buffer_margin);

    ObjectDataArray objects{object_data};
    const auto output = generateObstaclePolygonsForDrivableArea(objects, parameters, vehicle_width);
    EXPECT_TRUE(output.empty());
  }

  // invalid envelope polygon
  {
    ObjectData object_data;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    const auto object_type = utils::getHighestProbLabel(object_data.object.classification);
    const auto object_parameter = parameters->object_parameters.at(object_type);
    object_data.avoid_margin = object_parameter.lateral_soft_margin + 0.5 * vehicle_width;

    object_data.envelope_poly = {};

    ObjectDataArray objects{object_data};
    const auto output = generateObstaclePolygonsForDrivableArea(objects, parameters, vehicle_width);
    EXPECT_TRUE(output.empty());
  }

  // nominal case
  {
    ObjectData object_data;
    object_data.direction = Direction::RIGHT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));
    const auto object_type = utils::getHighestProbLabel(object_data.object.classification);
    const auto object_parameter = parameters->object_parameters.at(object_type);
    object_data.avoid_margin = object_parameter.lateral_soft_margin + 0.5 * vehicle_width;

    Polygon2d footprint;
    footprint.outer() = {
      Point2d{1.0, 0.0}, Point2d{3.0, 0.0}, Point2d{3.0, 1.0}, Point2d{1.0, 1.0},
      Point2d{1.0, 0.0}};

    const auto pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                        .position(createPoint(0.0, 0.0, 0.0))
                        .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    object_data.envelope_poly =
      createEnvelopePolygon(footprint, pose, object_parameter.envelope_buffer_margin);

    ObjectDataArray objects{object_data};
    const auto output = generateObstaclePolygonsForDrivableArea(objects, parameters, vehicle_width);
    EXPECT_FALSE(output.empty());

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(output.front().poly.outer().size(), 5);
    EXPECT_NEAR(output.front().poly.outer().at(0).x(), 2.0, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(0).y(), -2.0, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(1).x(), -0.5, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(1).y(), 0.5, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(2).x(), 2.0, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(2).y(), 3.0, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(3).x(), 4.5, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(3).y(), 0.5, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(4).x(), 2.0, epsilon);
    EXPECT_NEAR(output.front().poly.outer().at(4).y(), -2.0, epsilon);
  }
}

TEST(TestUtils, fillLongitudinalAndLengthByClosestEnvelopeFootprint)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    fillLongitudinalAndLengthByClosestEnvelopeFootprint;

  const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                           .position(createPoint(0.0, 0.0, 0.0))
                           .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

  auto path = generatePath(edge_pose);

  {
    const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                               .position(createPoint(2.5, 1.0, 0.0))
                               .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    object_data.object.kinematics.initial_pose_with_covariance.pose = object_pose;
    object_data.direction = Direction::LEFT;

    constexpr double margin = 0.0;
    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    object_data.envelope_poly = createEnvelopePolygon(object_data, pose, margin);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -0.5, epsilon);

    fillLongitudinalAndLengthByClosestEnvelopeFootprint(
      path, createPoint(0.0, 0.0, 0.0), object_data);
    EXPECT_NEAR(object_data.longitudinal, 1.0, epsilon);
    EXPECT_NEAR(object_data.length, 3.0, epsilon);
  }
}

TEST(TestUtils, fillObjectEnvelopePolygon)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    calcErrorEclipseLongRadius;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::createEnvelopePolygon;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    fillObjectEnvelopePolygon;

  const auto edge_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                           .position(createPoint(0.0, 0.0, 0.0))
                           .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

  auto path = generatePath(edge_pose);

  const auto create_params = [](const double envelope_buffer, const double th_error) {
    ObjectParameter param{};
    param.envelope_buffer_margin = envelope_buffer;
    param.th_error_eclipse_long_radius = th_error;
    return param;
  };

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_parameters.emplace(ObjectClassification::TRUCK, create_params(0.0, 2.0));

  const auto uuid = generateUUID();

  const auto object_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                             .position(createPoint(2.5, 1.0, 0.0))
                             .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

  ObjectData stored_object;
  stored_object.object.object_id = uuid;
  stored_object.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  stored_object.object.shape.dimensions.x = 2.8284271247461901;
  stored_object.object.shape.dimensions.y = 1.41421356237309505;
  // clang-format off
  stored_object.object.kinematics.initial_pose_with_covariance =
    geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(object_pose).covariance(
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  // clang-format on
  stored_object.error_eclipse_max =
    calcErrorEclipseLongRadius(stored_object.object.kinematics.initial_pose_with_covariance);
  stored_object.direction = Direction::LEFT;
  stored_object.object.classification.emplace_back(
    autoware_perception_msgs::build<ObjectClassification>()
      .label(ObjectClassification::TRUCK)
      .probability(1.0));

  const auto pose =
    path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
      .point.pose;

  constexpr double margin = 0.0;
  stored_object.envelope_poly = createEnvelopePolygon(stored_object, pose, margin);

  // new object.
  {
    ObjectData object_data;
    object_data.object.object_id = generateUUID();
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    // clang-format off
    object_data.object.kinematics.initial_pose_with_covariance =
      geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(object_pose).covariance(
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    // clang-format on
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    ObjectDataArray stored_objects{};

    fillObjectEnvelopePolygon(object_data, stored_objects, pose, parameters);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -0.5, epsilon);
  }

  // update envelope polygon by new pose.
  {
    const auto new_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                            .position(createPoint(3.0, 0.5, 0.0))
                            .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.object_id = uuid;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    // clang-format off
    object_data.object.kinematics.initial_pose_with_covariance =
      geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(new_pose).covariance(
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    // clang-format on
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, new_pose.position))
        .point.pose;

    ObjectDataArray stored_objects{stored_object};

    fillObjectEnvelopePolygon(object_data, stored_objects, pose, parameters);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -1.0, epsilon);
  }

  // use previous envelope polygon because new pose's error eclipse long radius is larger than
  // threshold. error eclipse long radius: 2.1213203435596
  {
    const auto new_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                            .position(createPoint(3.0, 0.5, 0.0))
                            .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.object_id = uuid;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    // clang-format off
    object_data.object.kinematics.initial_pose_with_covariance =
      geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(new_pose).covariance(
        {2.5, 2.0, 0.0, 0.0, 0.0, 0.0,
         2.0, 2.5, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    // clang-format on
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, new_pose.position))
        .point.pose;

    ObjectDataArray stored_objects{stored_object};

    fillObjectEnvelopePolygon(object_data, stored_objects, pose, parameters);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -0.5, epsilon);
  }

  // use new envelope polygon because new pose's error eclipse long radius is smaller than
  // threshold.
  {
    ObjectData huge_covariance_object;
    huge_covariance_object.object.object_id = uuid;
    huge_covariance_object.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    huge_covariance_object.object.shape.dimensions.x = 2.8284271247461901;
    huge_covariance_object.object.shape.dimensions.y = 1.41421356237309505;
    // clang-format off
    huge_covariance_object.object.kinematics.initial_pose_with_covariance =
      geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(object_pose).covariance(
          {5.0, 4.0, 0.0, 0.0, 0.0, 0.0,
           4.0, 5.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    // clang-format on
    huge_covariance_object.error_eclipse_max = calcErrorEclipseLongRadius(
      huge_covariance_object.object.kinematics.initial_pose_with_covariance);
    huge_covariance_object.direction = Direction::LEFT;
    huge_covariance_object.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    constexpr double margin = 0.0;
    huge_covariance_object.envelope_poly =
      createEnvelopePolygon(huge_covariance_object, pose, margin);

    const auto new_pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                            .position(createPoint(3.0, 0.5, 0.0))
                            .orientation(createQuaternionFromRPY(0.0, 0.0, deg2rad(45)));

    ObjectData object_data;
    object_data.object.object_id = uuid;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.8284271247461901;
    object_data.object.shape.dimensions.y = 1.41421356237309505;
    // clang-format off
    object_data.object.kinematics.initial_pose_with_covariance =
      geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(new_pose).covariance(
        {2.5, 2.0, 0.0, 0.0, 0.0, 0.0,
         2.0, 2.5, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    // clang-format on
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    ObjectDataArray stored_objects{huge_covariance_object};

    fillObjectEnvelopePolygon(object_data, stored_objects, pose, parameters);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -1.0, epsilon);
  }

  // use previous envelope polygon because the new one is within old one.
  {
    ObjectData object_data;
    object_data.object.object_id = uuid;
    object_data.object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    object_data.object.shape.dimensions.x = 2.0;
    object_data.object.shape.dimensions.y = 1.0;
    // clang-format off
    object_data.object.kinematics.initial_pose_with_covariance =
      geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(object_pose).covariance(
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    // clang-format on
    object_data.direction = Direction::LEFT;
    object_data.object.classification.emplace_back(
      autoware_perception_msgs::build<ObjectClassification>()
        .label(ObjectClassification::TRUCK)
        .probability(1.0));

    const auto pose =
      path.points.at(autoware::motion_utils::findNearestIndex(path.points, object_pose.position))
        .point.pose;

    ObjectDataArray stored_objects{stored_object};

    fillObjectEnvelopePolygon(object_data, stored_objects, pose, parameters);

    constexpr auto epsilon = 1e-6;
    ASSERT_EQ(object_data.envelope_poly.outer().size(), 5);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(0).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(1).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(2).y(), 2.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).x(), 4.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(3).y(), -0.5, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).x(), 1.0, epsilon);
    EXPECT_NEAR(object_data.envelope_poly.outer().at(4).y(), -0.5, epsilon);
  }
}

TEST(TestUtils, compensateLostTargetObjects)
{
  using namespace std::literals::chrono_literals;
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    compensateLostTargetObjects;

  const auto parameters = std::make_shared<AvoidanceParameters>();
  parameters->object_last_seen_threshold = 0.2;

  Odometry odometry;
  odometry.pose.pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                         .position(createPoint(0.0, 0.0, 0.0))
                         .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

  const auto planner_data = std::make_shared<PlannerData>();
  planner_data->self_odometry = std::make_shared<Odometry>(odometry);

  const auto uuid = generateUUID();
  const auto init_time = rclcpp::Clock{RCL_ROS_TIME}.now();

  ObjectData stored_object;
  stored_object.object.object_id = uuid;
  stored_object.last_seen = init_time;
  stored_object.object.kinematics.initial_pose_with_covariance.pose =
    geometry_msgs::build<geometry_msgs::msg::Pose>()
      .position(createPoint(1.0, 1.0, 0.0))
      .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

  rclcpp::sleep_for(100ms);

  // add stored objects.
  {
    const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

    ObjectDataArray stored_objects{};

    ObjectData new_object;
    new_object.object.object_id = generateUUID();
    new_object.last_seen = now;
    new_object.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(2.0, 5.0, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    AvoidancePlanningData avoidance_planning_data;
    avoidance_planning_data.target_objects = {new_object};
    avoidance_planning_data.other_objects = {};

    compensateLostTargetObjects(
      stored_objects, avoidance_planning_data, now, planner_data, parameters);
    ASSERT_FALSE(stored_objects.empty());
    EXPECT_EQ(stored_objects.front().object.object_id, new_object.object.object_id);
  }

  // compensate detection lost.
  {
    const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

    ObjectDataArray stored_objects{stored_object};

    AvoidancePlanningData avoidance_planning_data;
    avoidance_planning_data.target_objects = {};
    avoidance_planning_data.other_objects = {};

    compensateLostTargetObjects(
      stored_objects, avoidance_planning_data, now, planner_data, parameters);
    ASSERT_FALSE(avoidance_planning_data.target_objects.empty());
    EXPECT_EQ(
      avoidance_planning_data.target_objects.front().object.object_id,
      stored_object.object.object_id);
  }

  // update stored objects (same uuid).
  {
    const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

    ObjectDataArray stored_objects{stored_object};

    ObjectData detected_object;
    detected_object.object.object_id = uuid;
    detected_object.last_seen = now;
    detected_object.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(1.0, 1.0, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    AvoidancePlanningData avoidance_planning_data;
    avoidance_planning_data.target_objects = {detected_object};
    avoidance_planning_data.other_objects = {};

    compensateLostTargetObjects(
      stored_objects, avoidance_planning_data, now, planner_data, parameters);
    ASSERT_FALSE(stored_objects.empty());
    EXPECT_EQ(stored_objects.front().last_seen, detected_object.last_seen);
  }

  // update stored objects (detected near the stored object).
  {
    const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

    ObjectDataArray stored_objects{stored_object};

    ObjectData detected_object;
    detected_object.object.object_id = generateUUID();
    detected_object.last_seen = now;
    detected_object.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(1.1, 1.1, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    AvoidancePlanningData avoidance_planning_data;
    avoidance_planning_data.target_objects = {detected_object};
    avoidance_planning_data.other_objects = {};

    compensateLostTargetObjects(
      stored_objects, avoidance_planning_data, now, planner_data, parameters);
    ASSERT_FALSE(stored_objects.empty());
    EXPECT_EQ(stored_objects.front().last_seen, detected_object.last_seen);
  }

  // don't update stored object because there is no matched object.
  {
    const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

    ObjectDataArray stored_objects{stored_object};

    ObjectData detected_object;
    detected_object.object.object_id = generateUUID();
    detected_object.last_seen = now;
    detected_object.object.kinematics.initial_pose_with_covariance.pose =
      geometry_msgs::build<geometry_msgs::msg::Pose>()
        .position(createPoint(3.0, 3.0, 0.0))
        .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));

    AvoidancePlanningData avoidance_planning_data;
    avoidance_planning_data.target_objects = {detected_object};
    avoidance_planning_data.other_objects = {};

    compensateLostTargetObjects(
      stored_objects, avoidance_planning_data, now, planner_data, parameters);
    ASSERT_FALSE(stored_objects.empty());
    EXPECT_EQ(stored_objects.front().last_seen, init_time);
  }

  rclcpp::sleep_for(200ms);

  // don't compensate detection lost because time elapses more than threshold.
  {
    const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

    ObjectDataArray stored_objects{stored_object};

    AvoidancePlanningData avoidance_planning_data;
    avoidance_planning_data.target_objects = {};
    avoidance_planning_data.other_objects = {};

    compensateLostTargetObjects(
      stored_objects, avoidance_planning_data, now, planner_data, parameters);
    EXPECT_TRUE(avoidance_planning_data.target_objects.empty());
  }
}

TEST(TestUtils, calcErrorEclipseLongRadius)
{
  using autoware::behavior_path_planner::utils::static_obstacle_avoidance::
    calcErrorEclipseLongRadius;
  const auto pose = geometry_msgs::build<geometry_msgs::msg::Pose>()
                      .position(createPoint(3.0, 3.0, 0.0))
                      .orientation(createQuaternionFromRPY(0.0, 0.0, 0.0));
  // clang-format off
  const auto pose_with_covariance =
    geometry_msgs::build<geometry_msgs::msg::PoseWithCovariance>().pose(pose).covariance(
      {5.0, 4.0, 0.0, 0.0, 0.0, 0.0,
       4.0, 5.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  // clang-format on

  EXPECT_DOUBLE_EQ(calcErrorEclipseLongRadius(pose_with_covariance), 3.0);
}

}  // namespace autoware::behavior_path_planner::static_obstacle_avoidance
