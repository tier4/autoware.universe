// Copyright 2023 The Autoware Foundation.
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

#ifndef SIMPLE_PLANNING_SIMULATOR__ROSBAG_REPLAYER_HPP_
#define SIMPLE_PLANNING_SIMULATOR__ROSBAG_REPLAYER_HPP_

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_system_msgs/msg/autoware_state.hpp>
#include <autoware_auto_system_msgs/msg/float32_multi_array_diagnostic.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tier4_debug_msgs/msg/float32_stamped.hpp>
#include <tier4_external_api_msgs/srv/engage.hpp>
#include <tier4_planning_msgs/msg/velocity_limit.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace std::chrono_literals;
using namespace std::chrono;

class TimeController
{
public:
  TimeController(int cycle_time_ms, rclcpp::Clock::SharedPtr clock) : clock_(clock)
  {
    cycle_time_ = getDuration(cycle_time_ms * 1e6);
  }

  ~TimeController() {}

  void on_timer()
  {
    auto current_time = clock_->now();
    /* average offset estimation between control frame and simulator frame */
    if (!is_started_ && last_command_time_.get_clock_type() == RCL_ROS_TIME) {
      start_time_ = last_command_time_;
      is_started_ = true;
    }

    if (is_started_) {
      double cmd_ns = last_command_time_.nanoseconds();
      const double diff_cmd_ns =
        cmd_ns - start_time_.nanoseconds() + cycle_time_.nanoseconds() / 2.0;
      int cmd_cycle_num = static_cast<int>(diff_cmd_ns / cycle_time_.nanoseconds());
      double estimated_cmd_offset_time_ns =
        diff_cmd_ns - (cmd_cycle_num * cycle_time_.nanoseconds());

      static std::vector<double> estimated_offset_time_buffer;

      constexpr int QUEUE_SIZE = 200;
      if (estimated_offset_time_buffer.size() < QUEUE_SIZE) {
        // store 200 estimated offset time
        estimated_offset_time_buffer.push_back(estimated_cmd_offset_time_ns);
        // show estimated offset time
        {
          std::string bar;
          for (int i = 0; i < estimated_cmd_offset_time_ns / 1000000; i++) {
            bar += "=";
          }
          std::cout << "estimated_offset_time_ns : " << bar << std::endl;
        }
      } else if (estimated_offset_time_buffer.size() == QUEUE_SIZE) {
        // calculate average of 200 estimated offset time
        // and set it to start_time_offset_
        auto average =
          std::accumulate(
            estimated_offset_time_buffer.begin(), estimated_offset_time_buffer.end(), 0.0) /
          estimated_offset_time_buffer.size();
        average -= cycle_time_.nanoseconds() / 2.0;
        start_time_offset_ = getDuration(average);
        std::cout << "average_offset_time_ns : " << average / 1000000 << std::endl;
        // add the extra one to avoid the offset calculation on every frame
        estimated_offset_time_buffer.push_back(estimated_cmd_offset_time_ns);
        // initialize cycle_num
        cycle_num_ = static_cast<int>(
          (current_time - (start_time_ + start_time_offset_) + (cycle_time_*0.5)).nanoseconds() /
          cycle_time_.nanoseconds());
        auto cmd_frame_time = start_time_ + start_time_offset_ + cycle_time_ * cycle_num_;
        // if offset time is larger than next sim frame time, use next frame
        if ((cmd_frame_time + sim_frame_offset) > (current_time + cycle_time_)) {
          cycle_num_++;
        }
        cmd_frame_time = start_time_ + start_time_offset_ + cycle_time_ * cycle_num_;
        std::cout << "frame_time_precision: "
                  << (cmd_frame_time - last_command_time_).nanoseconds() / 1000000 << std::endl;
      } else {
        has_measured_offset_ = true;
        std::stringstream ss;
        // sleep
        {
          cycle_num_++;
          auto cmd_frame_time = start_time_ + start_time_offset_ + cycle_time_ * cycle_num_;
          ss << "offset[ms]: " << start_time_offset_.nanoseconds() / 1000000. << ", ";
          ss << "frame_time_precision: "
             << (cmd_frame_time - last_command_time_).nanoseconds() / 1000000. << ", ";
          auto sleep_duration = cmd_frame_time + sim_frame_offset - current_time;
          ss << "sleep: " << sleep_duration.nanoseconds() / 1000000.;
          clock_->sleep_until(cmd_frame_time + sim_frame_offset);
          //          clock_->sleep_until(cmd_frame_time + sim_frame_offset);
        }

        if (on_sim) {
          on_sim(clock_->now());
        }

        if (on_calc_offset) {
          on_calc_offset((clock_->now() - last_command_time_));
        }

        frame_diff_time_buffer_.push_back(clock_->now() - last_command_time_);
        ss << ", frame_diff_time: " << (clock_->now() - last_command_time_).nanoseconds() / 1000000. << std::endl;
        std::cout << ss.str() << std::endl;
        // save log and exit
        if(frame_diff_time_buffer_.size() > 1000){
          // get current time string
          using namespace std::chrono;
          std::time_t current_time_t = system_clock::to_time_t(system_clock::now());
          std::tm * ptm = std::localtime(&current_time_t);
          std::stringstream ss_now;
          ss_now << std::put_time(ptm, "%Y%m%d%H%M");
  std::function<void(rclcpp::Duration)> on_calc_offset = nullptr;
  std::function<void(rclcpp::Time)> on_cmd = nullptr;
  std::function<void(rclcpp::Time)> on_sim = nullptr;

          // export frame_diff_time_buffer_ to csv file
          std::string file_name = "frame_diff_time_" + ss_now.str() + ".csv";
          std::ofstream ofs(file_name);
          for (const auto & diff_time : frame_diff_time_buffer_) {
            ofs << diff_time.nanoseconds() << std::endl;
          }
          ofs.close();
          std::exit(0);
        }
      }
    }
  }

  void on_command()
  {
    last_command_time_ = clock_->now();
    if (on_cmd) {
      on_cmd(last_command_time_);
    }
  }

  rclcpp::Duration getDuration(double ns)
  {
    return rclcpp::Duration(nanoseconds(static_cast<int64_t>(ns)));
  }

  bool has_measured_offset() const { return has_measured_offset_; }

private:
  rclcpp::Time start_time_;
  rclcpp::Duration start_time_offset_ = -100ms;
  rclcpp::Duration cycle_time_ = 30ms;
  rclcpp::Time last_command_time_;
  bool is_started_ = false;
  std::vector<rclcpp::Duration> diff_time_buffer_;
  std::vector<rclcpp::Duration> frame_diff_time_buffer_;
  int cycle_num_ = 0;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Duration sim_frame_offset = 15ms;
  bool has_measured_offset_ = false;
};

using Engage = tier4_external_api_msgs::srv::Engage;
class AutowareOperator
{
public:
  AutowareOperator(rclcpp::Node & node)
  {
    client_engage = node.create_client<Engage>("/api/external/set/engage");
    pub_pose_estimation =
      node.create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 1);
    pub_goal_pose =
      node.create_publisher<geometry_msgs::msg::PoseStamped>("/planning/mission_planning/goal", 1);
    pub_velocity_limit = node.create_publisher<tier4_planning_msgs::msg::VelocityLimit>(
      "/planning/scenario_planning/max_velocity_default",
      rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());

    sub_autoware_state = node.create_subscription<autoware_auto_system_msgs::msg::AutowareState>(
      "/autoware/state", 1,
      [this](const autoware_auto_system_msgs::msg::AutowareState & msg) { state = msg; });

    checkServiceConnection();
  }

  void checkServiceConnection(void)
  {
    while (!client_engage->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_INFO(
        rclcpp::get_logger("AutowareOperator"), "engage service not available, waiting again...");
    }
  }

  void engage(bool engage, bool wait = true)
  {
    auto req = std::make_shared<Engage::Request>();
    req->engage = engage;
    auto future = client_engage->async_send_request(req);
    if (!wait) {
      return;
    }
    auto status = future.wait_for(std::chrono::seconds(1));
    if (status == std::future_status::ready) {
      auto response = future.get();
      if (response->status.code == response->status.SUCCESS) {
        RCLCPP_INFO(rclcpp::get_logger("AutowareOperator"), "engage success");
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("AutowareOperator"), "engage failed");
      }
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("AutowareOperator"), "engage timeout");
    }
  }

  void setVelocityLimit(float velocity_limit)
  {
    auto msg = std::make_shared<tier4_planning_msgs::msg::VelocityLimit>();
    msg->max_velocity = velocity_limit;
    pub_velocity_limit->publish(*msg);
  }

  void setGoalPose()
  {
    auto msg = std::make_shared<geometry_msgs::msg::PoseStamped>();
    msg->header.frame_id = "map";
    msg->header.stamp = rclcpp::Clock().now();
    msg->pose.position.x = 16713.16796875;
    msg->pose.position.y = 93383.9296875;
    msg->pose.orientation.set__x(0).set__y(0).set__z(0.6811543441258587).set__w(0.7321398496725002);
    pub_goal_pose->publish(*msg);
  }

  void setPoseEstimation() { pub_pose_estimation->publish(getInitialPose()); }

  auto getInitialPose() -> geometry_msgs::msg::PoseWithCovarianceStamped
  {
    auto msg = geometry_msgs::msg::PoseWithCovarianceStamped();
    msg.header.frame_id = "map";
    msg.header.stamp = rclcpp::Clock().now();
    msg.pose.pose.position.x = 16673.787109375;
    msg.pose.pose.position.y = 92971.7265625;
    msg.pose.pose.orientation.set__x(0)
      .set__y(0)
      .set__z(0.6773713996525991)
      .set__w(0.7356412080169781);
    return msg;
  }

  auto getAutowareState() { return state; }

private:
  rclcpp::Client<Engage>::SharedPtr client_engage;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_estimation;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_pose;
  rclcpp::Publisher<tier4_planning_msgs::msg::VelocityLimit>::SharedPtr pub_velocity_limit;
  rclcpp::Subscription<autoware_auto_system_msgs::msg::AutowareState>::SharedPtr sub_autoware_state;
  autoware_auto_system_msgs::msg::AutowareState state;
};

class Config
{
public:
  Config(std::string name = "no27")
  {
    std::string directory_path =
      ament_index_cpp::get_package_share_directory("simple_planning_simulator");
    std::ifstream ifs(directory_path + "/param/config.json");
    if (!ifs.is_open()) {
      std::cerr << "cannot open config file" << std::endl;
      exit(1);
    }
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    using json = nlohmann::json;
    auto configs = json::parse(json_str);

    auto config = configs[name];

    velocity_limit_mps = config["velocity_limit"].get<double>() / 3.6;

    start_line_left_x = config["bag_start_line"]["left"][0].get<double>();
    start_line_left_y = config["bag_start_line"]["left"][1].get<double>();
    start_line_right_x = config["bag_start_line"]["right"][0].get<double>();
    start_line_right_y = config["bag_start_line"]["right"][1].get<double>();

    forward_vehicle_uuid.push_back(config["vehicle_uuid"][0].get<uint8_t>());
    forward_vehicle_uuid.push_back(config["vehicle_uuid"][1].get<uint8_t>());

    rosbag_start_time =
      config["bag_start_time"].get<int64_t>() + config["start_offset"].get<int64_t>();

    // wheelbase: 2.75m, front overhang: 0.8m
    baselink_to_front = 3.55;

    rosbag_directory = configs["rosbag_directory"].get<std::string>();
    rosbag_path = rosbag_directory + "/" + config["bag_name"].get<std::string>();
  }

  float velocity_limit_mps;
  float start_line_left_x;
  float start_line_left_y;
  float start_line_right_x;
  float start_line_right_y;
  std::vector<uint8_t> forward_vehicle_uuid;
  int64_t rosbag_start_time;
  float baselink_to_front;
  std::filesystem::path rosbag_path;
  std::string rosbag_directory;
};

struct SeriesData
{
  std::vector<double> data;
  std::vector<rclcpp::Time> stamp;
  rclcpp::Time stamp_offset;
  std::string name;

  SeriesData(std::string name) : name(name) {}
  void push_back(double data_, rclcpp::Time stamp_)
  {
    this->data.push_back(data_);
    this->stamp.push_back(stamp_);
  }
  void set_offset(rclcpp::Time offset) { stamp_offset = offset; }
  void save(std::string dir)
  {
    dir = dir + "/" + name;
    if (not std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    {
      std::ofstream ofs(dir + "/" + name + "_stamp.csv");
      for (size_t i = 0; i < stamp.size(); i++) {
        ofs << (stamp[i] - stamp_offset).nanoseconds() / 1000000. << std::endl;
      }
      ofs.close();
    }
    {
      std::ofstream ofs(dir + "/" + name + "_data.csv");
      for (size_t i = 0; i < data.size(); i++) {
        ofs << data[i] << std::endl;
      }
      ofs.close();
    }
  }
};

struct DataCollection
{
  std::map<std::string, SeriesData> series;
  DataCollection()
  {
    series.emplace("ttc_sim", SeriesData{"ttc_sim"});
    series.emplace("ttc_real", SeriesData{"ttc_real"});
    series.emplace("distance_sim", SeriesData{"distance_sim"});
    series.emplace("distance_real", SeriesData{"distance_real"});
    series.emplace("velocity_sim", SeriesData{"velocity_sim"});
    series.emplace("velocity_real", SeriesData{"velocity_real"});
    series.emplace("sim_frame_stamp", SeriesData{"sim_frame_stamp"});
    series.emplace("cmd_frame_stamp", SeriesData{"cmd_frame_stamp"});
    series.emplace("frame_stamp_diff", SeriesData{"frame_stamp_diff"});
  }

  bool is_started = false;
  rclcpp::Time start_time;
  rclcpp::Duration log_duration = 30s;

  void start()
  {
    start_time = rclcpp::Clock().now();
    for (auto & s : series) {
      s.second.set_offset(start_time);
    }
    is_started = true;
  }

  bool should_finish()
  {
    if (not is_started) {
      return false;
    }

    auto duration = rclcpp::Clock().now() - start_time;
    std::cout << "duration: " << duration.seconds() << std::endl;
    return (rclcpp::Clock().now() - start_time) > log_duration;
  }

  void save(std::string dir)
  {
    if (not std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    for (auto & s : series) {
      s.second.save(dir);
    }
  }
};

class RealRosbagReplayer
{
public:
  std::unique_ptr<TimeController> time_controller_ = nullptr;

  RealRosbagReplayer(rclcpp::Node & node, std::string config_name, int timer_sampling_time_ms)
  : config(config_name), rosbag_data(node), autoware(node)
  {
    time_controller_ = std::make_unique<TimeController>(timer_sampling_time_ms, node.get_clock());
    marker_publisher = node.create_publisher<visualization_msgs::msg::Marker>(
      "/planning/mission_planning/mission_planning/debug/trajectory_marker", 1);
    distance_publisher = node.create_publisher<tier4_debug_msgs::msg::Float32Stamped>(
      "/planning/mission_planning/mission_planning/debug/distance", 1);
    ttc_publisher = node.create_publisher<tier4_debug_msgs::msg::Float32Stamped>(
      "/planning/mission_planning/mission_planning/debug/ttc", 1);
  }

  auto getInitialPose() -> geometry_msgs::msg::PoseWithCovarianceStamped
  {
    return autoware.getInitialPose();
  }

  auto setPoseEstimation() { autoware.setPoseEstimation(); }

  auto getAutowareState() { return autoware.getAutowareState(); }

  void loadRosbag()
  {
    rosbag2_cpp::Reader reader;
    reader.open(config.rosbag_path.string());

    std::cout << "loading rosbag: " << config.rosbag_path.string() << std::endl;

    while (reader.has_next()) {
      auto serialized_message = reader.read_next();
      rclcpp::SerializedMessage extracted_serialized_msg(*serialized_message->serialized_data);
      auto topic = serialized_message->topic_name;
      auto stamp_ns = serialized_message->time_stamp - config.rosbag_start_time;
      if (stamp_ns < 0) {
        continue;
      }
      if (topic == "/localization/kinematic_state") {
        rosbag_data.ego_odom.store.push_back([&extracted_serialized_msg, stamp_ns]() {
          static rclcpp::Serialization<nav_msgs::msg::Odometry> serialization;
          nav_msgs::msg::Odometry msg;
          serialization.deserialize_message(&extracted_serialized_msg, &msg);
          return std::make_pair(stamp_ns, msg);
        }());
      } else if (topic == "/control/command/control_cmd") {
        rosbag_data.ego_control_cmd.store.push_back([&extracted_serialized_msg, stamp_ns]() {
          static rclcpp::Serialization<autoware_auto_control_msgs::msg::AckermannControlCommand>
            serialization;
          autoware_auto_control_msgs::msg::AckermannControlCommand msg;
          serialization.deserialize_message(&extracted_serialized_msg, &msg);
          return std::make_pair(stamp_ns, msg);
        }());
      } else if (topic == "/control/trajectory_follower/longitudinal/diagnostic") {
        rosbag_data.ego_control_debug.store.push_back([&extracted_serialized_msg, stamp_ns]() {
          static rclcpp::Serialization<autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic>
            serialization;
          autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic msg;
          serialization.deserialize_message(&extracted_serialized_msg, &msg);
          return std::make_pair(stamp_ns, msg);
        }());
      } else if (topic == "/perception/object_recognition/objects") {
        rosbag_data.perception.store.push_back([&extracted_serialized_msg, stamp_ns]() {
          static rclcpp::Serialization<autoware_auto_perception_msgs::msg::PredictedObjects>
            serialization;
          autoware_auto_perception_msgs::msg::PredictedObjects msg;
          serialization.deserialize_message(&extracted_serialized_msg, &msg);
          return std::make_pair(stamp_ns, msg);
        }());
      }
    }
  }

  bool setRoute()
  {
    // do if autoware is ready
    if (
      autoware.getAutowareState().state ==
      autoware_auto_system_msgs::msg::AutowareState::WAITING_FOR_ROUTE) {
      autoware.checkServiceConnection();
      autoware.setVelocityLimit(config.velocity_limit_mps);
      autoware.setGoalPose();
      return true;
    }
    return false;
  }

  void setRouteWithoutValidation()
  {
    autoware.checkServiceConnection();
    autoware.setVelocityLimit(config.velocity_limit_mps);
    autoware.setGoalPose();
  }

  void publishEmptyObjects()
  {
    auto msg = std::make_shared<autoware_auto_perception_msgs::msg::PredictedObjects>();
    msg->header.frame_id = "map";
    msg->header.stamp = rclcpp::Clock().now();
    rosbag_data.perception.publisher->publish(*msg);
  }

  bool engageAutoware()
  {
    if (
      autoware.getAutowareState().state ==
      autoware_auto_system_msgs::msg::AutowareState::WAITING_FOR_ENGAGE) {
      autoware.engage(true, false);
      return true;
    }
    return false;
  }

  void publishRosbagData(int64_t current_time_ns)
  {
    auto publish = [this, current_time_ns](auto & store, auto & publisher) {
      while (store.iterator != store.store.end() && store.iterator->first <= current_time_ns) {
        publisher->publish(store.iterator->second);
        store.iterator++;
      }
    };
    publish(rosbag_data.ego_odom, rosbag_data.ego_odom.publisher);
    publish(rosbag_data.ego_control_cmd, rosbag_data.ego_control_cmd.publisher);
    publish(rosbag_data.ego_control_debug, rosbag_data.ego_control_debug.publisher);
    publish(rosbag_data.perception, rosbag_data.perception.publisher);
  }

  void checkStartLineOnOdom(const nav_msgs::msg::Odometry & msg)
  {
    // check if ego vehicle is over start line
    auto dx_line = config.start_line_left_x - config.start_line_right_x;
    auto dy_line = config.start_line_left_y - config.start_line_right_y;
    auto dx_ego = config.start_line_left_x - msg.pose.pose.position.x;
    auto dy_ego = config.start_line_left_y - msg.pose.pose.position.y;
    bool is_over_line = (dx_line * dy_ego - dx_ego * dy_line > 0);
    if (is_over_line) {
      if (not rosbag_data.perception.publish_thread) {
        rosbag_data.startPublishThreads(std::chrono::system_clock::now());
      }
    }
    on_odom_sim(msg);

    if (not rosbag_data.perception.publish_thread) {
      publishEmptyObjects();
    }
  }

private:
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher;
  rclcpp::Publisher<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr distance_publisher;
  rclcpp::Publisher<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr ttc_publisher;

  Config config;

  template <typename MessageT>
  struct TopicStore
  {
    TopicStore(std::string topic_name) : topic_name(topic_name) { iterator = store.begin(); }
    std::string topic_name;
    using MessageWithStamp = std::pair<rcutils_time_point_value_t, MessageT>;
    std::vector<MessageWithStamp> store;
    typename std::vector<MessageWithStamp>::iterator iterator;
    typename rclcpp::Publisher<MessageT>::SharedPtr publisher = nullptr;
    std::unique_ptr<std::thread> publish_thread = nullptr;
    std::function<void(const MessageT &)> on_publish = nullptr;

    void createPublisher(rclcpp::Node & node)
    {
      publisher = node.create_publisher<MessageT>(topic_name, 1);
    }

    void createPublishThead(std::chrono::system_clock::time_point start_time)
    {
      // check start_time is nearly now;
      auto diff = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() - start_time);
      if (std::abs(diff.count()) > 100) {
        std::cerr << "start time is too old / " << std::endl;
        exit(1);
      }
      iterator = store.begin();
      publish_thread = std::make_unique<std::thread>([this, start_time]() {
        while (1) {
          // sleep until next message time stamp is reached
          if (iterator != store.end()) {
            std::this_thread::sleep_until(start_time + std::chrono::nanoseconds(iterator->first));
            publisher->publish(iterator->second);
            if (on_publish) {
              on_publish(iterator->second);
            }
            iterator++;
          } else {
            break;
          }
        }
      });
    }
  };

  struct RosbagData
  {
    TopicStore<nav_msgs::msg::Odometry> ego_odom;
    TopicStore<autoware_auto_perception_msgs::msg::PredictedObjects> perception;
    TopicStore<autoware_auto_control_msgs::msg::AckermannControlCommand> ego_control_cmd;
    TopicStore<autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic> ego_control_debug;
    RosbagData(rclcpp::Node & node)
    : ego_odom("/localization/kinematic_state_rosbag"),
      perception("/perception/object_recognition/objects"),
      ego_control_cmd("/control/command/control_cmd_rosbag"),
      ego_control_debug("/control/trajectory_follower/longitudinal/diagnostic_rosbag")
    {
      ego_odom.createPublisher(node);
      perception.createPublisher(node);
      ego_control_cmd.createPublisher(node);
      ego_control_debug.createPublisher(node);
    }

    void startPublishThreads(std::chrono::system_clock::time_point start_time)
    {
      ego_odom.createPublishThead(start_time);
      perception.createPublishThead(start_time);
      ego_control_cmd.createPublishThead(start_time);
      ego_control_debug.createPublishThead(start_time);
    }
  } rosbag_data;

  AutowareOperator autoware;
};

#endif  // SIMPLE_PLANNING_SIMULATOR__ROSBAG_REPLAYER_HPP_
