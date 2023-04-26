#ifndef NDT_SCAN_MATCHER__SAMPLING_SEARCH_HPP_
#define NDT_SCAN_MATCHER__SAMPLING_SEARCH_HPP_

#include "tf2_eigen/tf2_eigen.hpp"

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// this function checks transformation probability(TP) on trajectory.
// normally, converged position indicates maximum TP.
// however, in tunnel, another place indicates larger TP sometime due to small convergence area.
// this function find maxium TP place on trajectory and update initial position to recover to
// correct position.
struct Sampling_search
{
  using PointSource = pcl::PointXYZ;
  using PointTarget = pcl::PointXYZ;
  using NormalDistributionsTransform =
    pclomp::NormalDistributionsTransform<PointSource, PointTarget>;

  Sampling_search() { mc_slip_pose_ = Eigen::Matrix4f::Identity(4, 4); }

  void sampling_search(
    Eigen::Matrix4f result_pose_matrix, double peak_tp,
    std::shared_ptr<pcl::PointCloud<PointSource>> sensor_points_baselinkTF_ptr,
    const std::shared_ptr<NormalDistributionsTransform> & ndt_ptr)
  {
    // getMaxiteration
    int max_iter = ndt_ptr->getMaximumIterations();
    // setMaxiteration to 1
    ndt_ptr->setMaximumIterations(1);
    double mc_max_tp = peak_tp;

    auto output_cloud = std::make_shared<pcl::PointCloud<PointSource>>();
    // for (double dist = -2; dist <= 2; dist+=0.5) {
    double dist;
    for (int try_count = 0; try_count < 1; try_count++) {
      dist = 6.0 * ((double)rand() / RAND_MAX - 0.5);
      Eigen::Matrix4f shift_matrix = Eigen::Matrix4f::Identity(4, 4);
      shift_matrix(0, 3) = dist;
      const Eigen::Matrix4f mc_pose_matrix = result_pose_matrix * shift_matrix;

      ndt_ptr->setInputSource(sensor_points_baselinkTF_ptr);
      ndt_ptr->align(*output_cloud, mc_pose_matrix);
      double tp = ndt_ptr->getTransformationProbability();

      if (mc_max_tp < tp) {  // find max tp
        mc_max_tp = tp;
        mc_slip_pose_ =
          result_pose_matrix.inverse() * mc_pose_matrix;  //*mc_center_pose_matrix.inverse();
      }
    }
    // return to original
    ndt_ptr->setMaximumIterations(max_iter);
  }

  // replace initial position for align process.
  //  if tp_check update slip_pose, then initial pose is updated.
  //  otherwise, initial pose is not changed.
  Eigen::Matrix4f pose_update(Eigen::Matrix4f origin)
  {
    Eigen::Matrix4f new_pose;
    new_pose = origin * mc_slip_pose_;
    mc_slip_pose_ = Eigen::Matrix4f::Identity(4, 4);
    return new_pose;
  }

  // position offset for initial position
  Eigen::Matrix4f mc_slip_pose_;
};

#endif  // NDT_SCAN_MATCHER__SAMPLING_SEARCH_HPP_
