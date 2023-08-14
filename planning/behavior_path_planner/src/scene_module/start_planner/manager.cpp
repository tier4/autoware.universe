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

#include "behavior_path_planner/scene_module/start_planner/manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

namespace behavior_path_planner
{

StartPlannerModuleManager::StartPlannerModuleManager(
  rclcpp::Node * node, const std::string & name, const ModuleConfigParameters & config,
  const std::shared_ptr<StartPlannerParameters> & parameters)
: SceneModuleManagerInterface(node, name, config, {""}), parameters_{parameters}
{
}

void StartPlannerModuleManager::updateModuleParams(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  auto & p = parameters_;

  [[maybe_unused]] std::string ns = name_ + ".";

  std::for_each(registered_modules_.begin(), registered_modules_.end(), [&p](const auto & m) {
    m->updateModuleParams(p);
  });
}

bool StartPlannerModuleManager::isSimultaneousExecutableAsApprovedModule() const
{
  const auto checker = [this](const auto & module) {
    // Currently simultaneous execution with other modules is not supported while backward driving
    if (!module->isBackFinished()) {
      return false;
    }
    return enable_simultaneous_execution_as_candidate_module_;
  };

  return std::all_of(registered_modules_.begin(), registered_modules_.end(), checker);
}

bool StartPlannerModuleManager::isSimultaneousExecutableAsCandidateModule() const
{
  const auto checker = [this](const auto & module) {
    // Currently simultaneous execution with other modules is not supported while backward driving
    if (!module->isBackFinished()) {
      return false;
    }
    return enable_simultaneous_execution_as_candidate_module_;
  };

  return std::all_of(registered_modules_.begin(), registered_modules_.end(), checker);
}
}  // namespace behavior_path_planner
