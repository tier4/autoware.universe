#!/usr/bin/env python3

# Copyright 2023 Tier IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from tier4_debug_msgs.msg import Int32Stamped

MAX_COMMAND = 3  # [-]
MIN_COMMAND = 0  # [-]


class MrmTestCommander(Node):
    def __init__(self):
        super().__init__("mrm_test_commander")
        self.pub = self.create_publisher(Int32Stamped, "/mrm_test/command", 1)

    def run(self):
        while rclpy.ok():
            value = int(
                input(
                    "MRM test command [" + str(MIN_COMMAND) + " ~ " + str(MAX_COMMAND) + "] > "
                )
            )

            if value > MAX_COMMAND:
                print(
                    "input value is larger than max command!"
                    + f"input: {value} max: {MAX_COMMAND}"
                )
                value = MAX_COMMAND
            elif value < MIN_COMMAND:
                print(
                    "input value is smaller than min command!"
                    + f"input: {value} min: {MIN_COMMAND}"
                )
                value = MIN_COMMAND

            msg = Int32Stamped(stamp=self.get_clock().now().to_msg(), data=value)
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    commander = MrmTestCommander()
    commander.run()

    commander.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()