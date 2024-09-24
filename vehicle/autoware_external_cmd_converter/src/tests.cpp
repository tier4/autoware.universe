#include "autoware_external_cmd_converter/tests.hpp"

#include <autoware_test_utils/autoware_test_utils.hpp>
#include <rclcpp/publisher.hpp>

#include <tier4_control_msgs/msg/detail/gate_mode__struct.hpp>
#include <tier4_external_api_msgs/msg/detail/heartbeat__struct.hpp>

#include <memory>

namespace autoware::external_cmd_converter::test
{
using tier4_external_api_msgs::msg::Heartbeat;
std::shared_ptr<ExternalCmdConverterNode> generateExternalCmdConverterNode()
{
  auto node_options = rclcpp::NodeOptions{};

  const auto autoware_test_utils_dir =
    ament_index_cpp::get_package_share_directory("autoware_test_utils");
  const auto external_cmd_converter_dir =
    ament_index_cpp::get_package_share_directory("autoware_external_cmd_converter");

  node_options.append_parameter_override("enable_slow_down", false);

  node_options.arguments(
    {"--ros-args", "--params-file", autoware_test_utils_dir + "/config/test_common.param.yaml",
     "--params-file", autoware_test_utils_dir + "/config/test_nearest_search.param.yaml",
     "--params-file", autoware_test_utils_dir + "/config/test_vehicle_info.param.yaml",
     "--params-file", external_cmd_converter_dir + "/config/external_cmd_converter.param.yaml"});

  return std::make_shared<ExternalCmdConverterNode>(node_options);
}

TEST(ExternalCmdConverterTest, NodeTest)
{
  auto test_node = generateTestNode();
  auto external_cmd_converter_node = generateExternalCmdConverterNode();
  rclcpp::Publisher<tier4_external_api_msgs::msg::ControlCommandStamped>::SharedPtr control_cmd_pub;
  rclcpp::Publisher<tier4_external_api_msgs::msg::Heartbeat>::SharedPtr
    emergency_stop_heartbeat_pub;
  rclcpp::Publisher<Odometry>::SharedPtr velocity_pub;
  rclcpp::Publisher<GearCommand>::SharedPtr shift_cmd_pub;
  rclcpp::Publisher<GateMode>::SharedPtr gate_mode_pub;
  rclcpp::Publisher<ExternalControlCommand>::SharedPtr external_cmd_pub;
  GearCommand gear_cmd;
  gear_cmd.command = GearCommand::DRIVE;
  gear_cmd.stamp = external_cmd_converter_node->now();

  GateMode gate_mode;
  gate_mode.data = GateMode::AUTO;

  Heartbeat heartbeat;

  ExternalControlCommand external_cmd;

  autoware::test_utils::publishToTargetNode(
    test_node, external_cmd_converter_node, "in/current_gate_mode", gate_mode_pub, gate_mode);
  autoware::test_utils::publishToTargetNode(
    test_node, external_cmd_converter_node, "in/shift_cmd", shift_cmd_pub, gear_cmd);
  autoware::test_utils::publishToTargetNode(
    test_node, external_cmd_converter_node, "in/emergency_stop", emergency_stop_heartbeat_pub,
    heartbeat);
  autoware::test_utils::publishToTargetNode(
    test_node, external_cmd_converter_node, "in/external_control_cmd", external_cmd_pub,
    external_cmd);
}

}  // namespace autoware::external_cmd_converter::test
