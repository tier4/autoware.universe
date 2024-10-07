#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Tier IV, Inc. All rights reserved.
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
from tier4_debug_msgs.msg import Float32Stamped

MAX_SUB_BRAKE = 100.0  # [-]
MIN_SUB_BRAKE = 0.0  # [-]


class SubBrakeTester(Node):
    def __init__(self):
        super().__init__("vehicle_sub_brake_tester")
        self.pub = self.create_publisher(Float32Stamped, "/vehicle/tester/sub_brake", 1)

    def run(self):
        while rclpy.ok():
            value = float(
                input("target sub brake [" + str(MIN_SUB_BRAKE) + " ~ " + str(MAX_SUB_BRAKE) + "] > ")
            )

            if value > MAX_SUB_BRAKE:
                print("input value is larger than max sub brake!" + f"input: {value} max: {MAX_SUB_BRAKE}")
                value = MAX_SUB_BRAKE
            elif value < MIN_SUB_BRAKE:
                print("input value is smaller than min sub brake!" + f"input: {value} min: {MIN_SUB_BRAKE}")
                value = MIN_SUB_BRAKE

            msg = Float32Stamped(stamp=self.get_clock().now().to_msg(), data=value)

            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    sub_brake_tester = SubBrakeTester()
    sub_brake_tester.run()

    sub_brake_tester.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
