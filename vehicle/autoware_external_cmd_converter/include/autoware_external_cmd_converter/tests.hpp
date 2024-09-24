// Copyright 2024 Tier IV, Inc.
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

#ifndef AUTOWARE_EXTERNAL_CMD_CONVERTER_AUTOWARE_EXTERNAL_CMD_CONVERTER_HPP_
#define AUTOWARE_EXTERNAL_CMD_CONVERTER_AUTOWARE_EXTERNAL_CMD_CONVERTER_HPP_

#include "autoware_external_cmd_converter/node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoware_test_utils/autoware_test_utils.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace autoware::external_cmd_converter::test
{
using autoware::external_cmd_converter::ExternalCmdConverterNode;
// since ASSERT_NO_THROW in gtest masks the exception message, redefine it.
#define ASSERT_NO_THROW_WITH_ERROR_MSG(statement)                                                \
  try {                                                                                          \
    statement;                                                                                   \
    SUCCEED();                                                                                   \
  } catch (const std::exception & e) {                                                           \
    FAIL() << "Expected: " << #statement                                                         \
           << " doesn't throw an exception.\nActual: it throws. Error message: " << e.what()     \
           << std::endl;                                                                         \
  } catch (...) {                                                                                \
    FAIL() << "Expected: " << #statement                                                         \
           << " doesn't throw an exception.\nActual: it throws. Error message is not available." \
           << std::endl;                                                                         \
  }

std::shared_ptr<rclcpp::Node> generateTestNode()
{
  return std::make_shared<rclcpp::Node>("external_cmd_converter_test_node");
}

std::shared_ptr<ExternalCmdConverterNode> generateExternalCmdConverterNode();
}  // namespace autoware::external_cmd_converter::test
#endif  // AUTOWARE_EXTERNAL_CMD_CONVERTER_AUTOWARE_EXTERNAL_CMD_CONVERTER_HPP_
