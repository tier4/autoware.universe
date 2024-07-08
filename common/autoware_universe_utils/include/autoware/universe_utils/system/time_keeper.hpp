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
#ifndef AUTOWARE__UNIVERSE_UTILS__SYSTEM__TIME_KEEPER_HPP_
#define AUTOWARE__UNIVERSE_UTILS__SYSTEM__TIME_KEEPER_HPP_

#include "autoware/universe_utils/system/stop_watch.hpp"

#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/string.hpp>
#include <tier4_debug_msgs/msg/processing_time_node.hpp>
#include <tier4_debug_msgs/msg/processing_time_tree.hpp>

#include <fmt/format.h>

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace autoware::universe_utils
{
/**
 * @brief Class representing a node in the time tracking tree
 */
class ProcessingTimeNode : public std::enable_shared_from_this<ProcessingTimeNode>
{
public:
  /**
   * @brief Construct a new ProcessingTimeNode object
   *
   * @param name Name of the node
   */
  explicit ProcessingTimeNode(const std::string & name);

  /**
   * @brief Add a child node
   *
   * @param name Name of the child node
   * @return std::shared_ptr<ProcessingTimeNode> Shared pointer to the newly added child node
   */
  std::shared_ptr<ProcessingTimeNode> add_child(const std::string & name);

  /**
   * @brief Get the result string representing the node and its children in a tree structure
   *
   * @return std::string Result string representing the node and its children
   */
  std::string to_string() const;

  /**
   * @brief Construct a ProcessingTimeTree message from the node and its children
   *
   * @return tier4_debug_msgs::msg::ProcessingTimeTree Constructed ProcessingTimeTree message
   */
  tier4_debug_msgs::msg::ProcessingTimeTree to_msg() const;

  /**
   * @brief Get the parent node
   *
   * @return std::shared_ptr<ProcessingTimeNode> Shared pointer to the parent node
   */
  std::shared_ptr<ProcessingTimeNode> get_parent_node() const;

  /**
   * @brief Get the child nodes
   *
   * @return std::vector<std::shared_ptr<ProcessingTimeNode>> Vector of shared pointers to the child
   * nodes
   */
  std::vector<std::shared_ptr<ProcessingTimeNode>> get_child_nodes() const;

  /**
   * @brief Set the processing time for the node
   *
   * @param processing_time Processing time to be set
   */
  void set_time(const double processing_time);

  /**
   * @brief Get the name of the node
   *
   * @return std::string Name of the node
   */
  std::string get_name() const;

private:
  const std::string name_;                                    //!< Name of the node
  double processing_time_{0.0};                               //!< Processing time of the node
  std::shared_ptr<ProcessingTimeNode> parent_node_{nullptr};  //!< Shared pointer to the parent node
  std::vector<std::shared_ptr<ProcessingTimeNode>>
    child_nodes_;  //!< Vector of shared pointers to the child nodes
};

using ProcessingTimeDetail =
  tier4_debug_msgs::msg::ProcessingTimeTree;  //!< Alias for the ProcessingTimeTree message

/**
 * @brief Class for tracking and reporting the processing time of various functions
 */
class TimeKeeper
{
public:
  template <typename... Reporters>
  explicit TimeKeeper(Reporters... reporters) : current_time_node_(nullptr), root_node_(nullptr)
  {
    reporters_.reserve(sizeof...(Reporters));
    (add_reporter(reporters), ...);
  }

  void add_reporter(std::ostream * os)
  {
    reporters_.emplace_back([os](const std::shared_ptr<ProcessingTimeNode> & node) {
      *os << "==========================" << std::endl;
      *os << node->to_string() << std::endl;
    });
  }

  void add_reporter(rclcpp::Publisher<ProcessingTimeDetail>::SharedPtr publisher)
  {
    reporters_.emplace_back([publisher](const std::shared_ptr<ProcessingTimeNode> & node) {
      publisher->publish(node->to_msg());
    });
  }

  void add_reporter(rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher)
  {
    reporters_.emplace_back([publisher](const std::shared_ptr<ProcessingTimeNode> & node) {
      std_msgs::msg::String msg;
      msg.data = node->to_string();
      publisher->publish(msg);
    });
  }

  /**
   * @brief Start tracking the processing time of a function
   *
   * @param func_name Name of the function to be tracked
   */
  void start_track(const std::string & func_name);

  /**
   * @brief End tracking the processing time of a function
   *
   * @param func_name Name of the function to end tracking
   */
  void end_track(const std::string & func_name);

private:
  void report()
  {
    if (current_time_node_ != nullptr) {
      throw std::runtime_error(fmt::format(
        "You must call end_track({}) first, but report() is called",
        current_time_node_->get_name()));
    }
    for (const auto & reporter : reporters_) {
      reporter(root_node_);
    }
    current_time_node_.reset();
    root_node_.reset();
  }

  std::shared_ptr<ProcessingTimeNode>
    current_time_node_;                            //!< Shared pointer to the current time node
  std::shared_ptr<ProcessingTimeNode> root_node_;  //!< Shared pointer to the root time node
  autoware::universe_utils::StopWatch<
    std::chrono::milliseconds, std::chrono::microseconds, std::chrono::steady_clock>
    stop_watch_;  //!< StopWatch object for tracking the processing time

  std::vector<std::function<void(const std::shared_ptr<ProcessingTimeNode> &)>>
    reporters_;  //!< Vector of functions for reporting the processing times
};

/**
 * @brief Class for automatically tracking the processing time of a function within a scope
 */
class ScopedTimeTrack
{
public:
  /**
   * @brief Construct a new ScopedTimeTrack object
   *
   * @param func_name Name of the function to be tracked
   * @param time_keeper Reference to the TimeKeeper object
   */
  ScopedTimeTrack(const std::string & func_name, TimeKeeper & time_keeper);

  /**
   * @brief Destroy the ScopedTimeTrack object, ending the tracking of the function
   */
  ~ScopedTimeTrack();

private:
  const std::string func_name_;  //!< Name of the function being tracked
  TimeKeeper & time_keeper_;     //!< Reference to the TimeKeeper object
};

// class TimeKeeperManager
// {
// public:
//   static std::shared_ptr<TimeKeeper> bring_time_keeper(const std::string & time_keeper_name)
//   {
//     if (time_keeper_map_.find(time_keeper_name) == time_keeper_map_.end()) {
//       time_keeper_map_[time_keeper_name] = std::make_shared<TimeKeeper>();
//     }
//     return time_keeper_map_[time_keeper_name];
//   }

// private:
//   static std::map<std::string, std::shared_ptr<TimeKeeper>> time_keeper_map_;
// };

}  // namespace autoware::universe_utils

#endif  // AUTOWARE__UNIVERSE_UTILS__SYSTEM__TIME_KEEPER_HPP_
