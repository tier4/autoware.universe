// Copyright 2023 TIER IV, Inc.
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

#ifndef PATH_SMOOTHING__PATH_SMOOTHER_NODE_HPP_
#define PATH_SMOOTHING__PATH_SMOOTHER_NODE_HPP_

#include "motion_utils/motion_utils.hpp"
#include "path_smoothing/common_structs.hpp"
#include "path_smoothing/elastic_band.hpp"
#include "path_smoothing/type_alias.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace path_smoothing
{
class PathSmootherNode : public rclcpp::Node
{
public:
  explicit PathSmootherNode(const rclcpp::NodeOptions & node_options);

protected:  // for the static_centerline_optimizer package
  // TODO(murooka) move this node to common
  class DrivingDirectionChecker
  {
  public:
    bool isDrivingForward(const std::vector<PathPoint> & path_points)
    {
      const auto is_driving_forward = motion_utils::isDrivingForward(path_points);
      is_driving_forward_ = is_driving_forward ? is_driving_forward.get() : is_driving_forward_;
      return is_driving_forward_;
    }

  private:
    bool is_driving_forward_{true};
  };
  DrivingDirectionChecker driving_direction_checker_{};

  // argument variables
  mutable std::shared_ptr<TimeKeeper> time_keeper_ptr_{nullptr};

  // flags for some functions
  bool enable_debug_info_;
  bool enable_outside_drivable_area_stop_;
  bool enable_smoothing_;
  bool enable_skip_optimization_;
  bool enable_reset_prev_optimization_;
  bool use_footprint_polygon_for_outside_drivable_area_check_;

  // algorithms
  std::shared_ptr<EBPathSmoother> eb_path_smoother_ptr_{nullptr};

  // parameters
  CommonParam common_param_{};
  EgoNearestParam ego_nearest_param_{};

  // variables for subscribers
  Odometry::SharedPtr ego_state_ptr_;

  // variables for previous information
  std::shared_ptr<std::vector<TrajectoryPoint>> prev_optimized_traj_points_ptr_;

  // interface publisher
  rclcpp::Publisher<Trajectory>::SharedPtr traj_pub_;

  // interface subscriber
  rclcpp::Subscription<Path>::SharedPtr path_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;

  // debug publisher
  rclcpp::Publisher<Trajectory>::SharedPtr debug_extended_traj_pub_;
  rclcpp::Publisher<StringStamped>::SharedPtr debug_calculation_time_pub_;

  // parameter callback
  rcl_interfaces::msg::SetParametersResult onParam(
    const std::vector<rclcpp::Parameter> & parameters);
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  // subscriber callback function
  void onPath(const Path::SharedPtr);

  // reset functions
  void initializePlanning();
  void resetPreviousData();

  // main functions
  bool isDataReady(const Path & path, rclcpp::Clock clock) const;
  PlannerData createPlannerData(const Path & path) const;
  std::vector<TrajectoryPoint> generateOptimizedTrajectory(const PlannerData & planner_data);
  std::vector<TrajectoryPoint> extendTrajectory(
    const std::vector<TrajectoryPoint> & traj_points,
    const std::vector<TrajectoryPoint> & optimized_points) const;

  // functions in generateOptimizedTrajectory
  std::vector<TrajectoryPoint> optimizeTrajectory(const PlannerData & planner_data);
  std::vector<TrajectoryPoint> getPrevOptimizedTrajectory(
    const std::vector<TrajectoryPoint> & traj_points) const;
  void applyInputVelocity(
    std::vector<TrajectoryPoint> & output_traj_points,
    const std::vector<TrajectoryPoint> & input_traj_points,
    const geometry_msgs::msg::Pose & ego_pose) const;
  void insertZeroVelocityOutsideDrivableArea(
    const PlannerData & planner_data, std::vector<TrajectoryPoint> & traj_points) const;
  void publishVirtualWall(const geometry_msgs::msg::Pose & stop_pose) const;
};
}  // namespace path_smoothing

#endif  // PATH_SMOOTHING__PATH_SMOOTHER_NODE_HPP_
