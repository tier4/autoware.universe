// Copyright 2020 Tier IV, Inc.
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

#include <dummy_perception_publisher/random_objects_publisher.hpp>
#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
geometry_msgs::msg::Pose getRandomPose(const geometry_msgs::msg::Pose & base_pose)
{
  std::mt19937 random_generator;
  std::random_device seed_gen;
  random_generator.seed(seed_gen());

  std::normal_distribution<> pos_random(0.0, 100.0);
  std::normal_distribution<> yaw_random(0.0, 1.0);

  auto pose = tier4_autoware_utils::calcOffsetPose(
    base_pose, pos_random(random_generator), pos_random(random_generator), 0.0);
  pose.position.z = base_pose.position.z;
  pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw_random(random_generator));

  return pose;
}

geometry_msgs::msg::Pose createCenterPose(const lanelet::ConstLanelet & lanelet)
{
  const auto centerline = lanelet.centerline();
  const size_t middle_point_idx = std::floor(centerline.size() / 2.0);

  // get middle position of the lanelet
  geometry_msgs::msg::Point middle_pos;
  middle_pos.x = centerline[middle_point_idx].x();
  middle_pos.y = centerline[middle_point_idx].y();

  // get next middle position of the lanelet
  geometry_msgs::msg::Point next_middle_pos;
  next_middle_pos.x = centerline[middle_point_idx + 1].x();
  next_middle_pos.y = centerline[middle_point_idx + 1].y();

  // calculate middle pose
  geometry_msgs::msg::Pose middle_pose;
  middle_pose.position = middle_pos;
  const double yaw = tier4_autoware_utils::calcAzimuthAngle(middle_pos, next_middle_pos);
  middle_pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw);

  return middle_pose;
}

lanelet::ConstLanelet extractRandomLanelet(const lanelet::ConstLanelets & lanelets)
{
  std::uniform_int_distribution<> index_random(0, lanelets.size());
  std::mt19937 random_generator;
  const size_t idx = static_cast<size_t>(index_random(random_generator));
  return lanelets.at(idx);
}

template <typename T>
size_t findNearestSegmentIndex(const T & points, const geometry_msgs::msg::Pose & pose)
{
  const auto seg_idx = motion_utils::findNearestSegmentIndex(points, pose);
  if (seg_idx) {
    return *seg_idx;
  }
  return motion_utils::findNearestSegmentIndex(points, pose.position);
}

std::vector<geometry_msgs::msg::Point> convertToPoints(
  const lanelet::ConstLineString3d & centerline)
{
  std::vector<geometry_msgs::msg::Point> points;
  for (size_t i = 0; i < centerline.size(); ++i) {
    geometry_msgs::msg::Point point;
    point.x = centerline[i].basicPoint().x();
    point.y = centerline[i].basicPoint().y();
    point.z = centerline[i].basicPoint().z();
    points.push_back(point);
  }
  return points;
}

std::vector<geometry_msgs::msg::Pose> convertToPoses(const lanelet::ConstLineString3d & centerline)
{
  const auto points = convertToPoints(centerline);

  std::vector<geometry_msgs::msg::Pose> poses;
  for (size_t i = 0; i < points.size(); ++i) {
    geometry_msgs::msg::Pose pose;
    pose.position = points.at(i);

    const size_t prev_idx = i == 0 ? i : i - 1;
    const size_t next_idx = i == 0 ? i + 1 : i;
    const double yaw =
      tier4_autoware_utils::calcAzimuthAngle(points.at(next_idx), points.at(prev_idx));
    pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw);
    poses.push_back(pose);
  }
  return poses;
}
}  // namespace

RandomObjectsPublisher::RandomObjectsPublisher()
: Node("random_objects_publisher"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  path_sub_ = create_subscription<autoware_auto_planning_msgs::msg::Path>(
    "/planning/scenario_planning/behavior_planning/path", 1,
    [this](const autoware_auto_planning_msgs::msg::Path::ConstSharedPtr msg) { path_ptr_ = msg; });
  map_sub_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&RandomObjectsPublisher::onMap, this, std::placeholders::_1));

  detected_object_with_feature_pub_ =
    this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
      "/perception/object_recognition/detection/objects_with_feature", 1);
  pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/perception/obstacle_segmentation/pointcloud", 1);

  update_hz_ = 10.0;
  const auto period_ns = rclcpp::Rate(update_hz_).period();
  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&RandomObjectsPublisher::onTimer, this));
}

void RandomObjectsPublisher::onMap(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
{
  const auto lanelet_map_ptr = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *msg, lanelet_map_ptr, &traffic_rules_ptr_, &routing_graph_ptr_);
  const auto lanelets = lanelet::utils::query::laneletLayer(lanelet_map_ptr);
  lanelets_.insert(lanelets_.end(), lanelets.begin(), lanelets.end());
}

void RandomObjectsPublisher::onTimer()
{
  if (!path_ptr_) {
    return;
  }

  tf2::Transform tf_base_link2map;
  try {
    geometry_msgs::msg::TransformStamped ros_base_link2map;
    ros_base_link2map =
      tf_buffer_.lookupTransform("base_link", "map", now(), rclcpp::Duration::from_seconds(0.5));
    tf2::fromMsg(ros_base_link2map.transform, tf_base_link2map);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
    return;
  }

  std::cerr << vehicle_objects_.size() << std::endl;
  const size_t max_object_num = 10;
  if (vehicle_objects_.size() < max_object_num) {
    spawnVehicleObject();
  }

  step();
  validateObject();

  tier4_perception_msgs::msg::DetectedObjectsWithFeature objects_msg;
  objects_msg.header.frame_id = "base_link";
  objects_msg.header.stamp = now();
  for (const auto & object : vehicle_objects_) {
    tier4_perception_msgs::msg::DetectedObjectWithFeature feature_object;

    autoware_auto_perception_msgs::msg::ObjectClassification classification;
    classification.label = autoware_auto_perception_msgs::msg::ObjectClassification::CAR;
    feature_object.object.classification.push_back(classification);

    feature_object.object.shape.type = autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX;

    // pose
    geometry_msgs::msg::Transform ros_map2moved_object;
    ros_map2moved_object.translation.x = object.pose.position.x;
    ros_map2moved_object.translation.y = object.pose.position.y;
    ros_map2moved_object.translation.z = object.pose.position.z;
    ros_map2moved_object.rotation = object.pose.orientation;
    tf2::Transform tf_map2moved_object;
    tf2::fromMsg(ros_map2moved_object, tf_map2moved_object);

    const auto tf_base_link2moved_object = tf_base_link2map * tf_map2moved_object;

    tf2::toMsg(
      tf_base_link2moved_object, feature_object.object.kinematics.pose_with_covariance.pose);

    // velocity
    feature_object.object.kinematics.twist_with_covariance.twist.linear.x = object.velocity;
    objects_msg.feature_objects.push_back(feature_object);
  }
  detected_object_with_feature_pub_->publish(objects_msg);
  // pointcloud_pub_->publish();
}

void RandomObjectsPublisher::spawnVehicleObject()
{
  const auto random_pose = getRandomPose(path_ptr_->pose);
  lanelet::ConstLanelet start_lanelet;
  const bool is_found =
    lanelet::utils::query::getClosestLanelet(lanelets_, random_pose, &start_lanelet);
  if (!is_found) {
    return;
  }

  const auto initial_pose = createCenterPose(start_lanelet);
  const double initial_vel = 10.0;
  const auto vehicle_object = VehicleObject(start_lanelet, initial_pose, initial_vel);

  vehicle_objects_.push_back(vehicle_object);
}

void RandomObjectsPublisher::step()
{
  for (const auto & object : vehicle_objects_) {
    const double drive_length = object.velocity / update_hz_;

    double remained_length = drive_length;
    auto nearest_lanelet = object.nearest_lanelet;
    while (true) {
      const auto & centerline = convertToPoses(nearest_lanelet.centerline());
      const size_t seg_idx = findNearestSegmentIndex(centerline, object.pose);
      const auto offset_pose =
        motion_utils::calcLongitudinalOffsetPose(centerline, seg_idx, remained_length);
      if (offset_pose) {
      } else {
        const double total_length = motion_utils::calcArcLength(centerline);
        const double offset =
          motion_utils::calcSignedArcLength(centerline, 0, object.pose.position, seg_idx);
        remained_length = remained_length - (total_length - offset);

        const auto next_lanelets = routing_graph_ptr_->following(nearest_lanelet);
        if (next_lanelets.empty()) {
          std::cerr << "Lane ends" << std::endl;
          break;
        }
        nearest_lanelet = extractRandomLanelet(next_lanelets);
      }
    }
  }
}

void RandomObjectsPublisher::validateObject()
{
  for (size_t i = 0; i < vehicle_objects_.size(); ++i) {
    const auto & object = vehicle_objects_.at(i);
    const size_t seg_idx = findNearestSegmentIndex(path_ptr_->points, object.pose);
    const double lat_offset_to_path =
      motion_utils::calcLateralOffset(path_ptr_->points, object.pose.position, seg_idx);
    if (100.0 < std::abs(lat_offset_to_path)) {
      vehicle_objects_.erase(vehicle_objects_.begin() + i);
      break;  // TODO(murooka): remove this break
    }
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RandomObjectsPublisher>());
  rclcpp::shutdown();
  return 0;
}
