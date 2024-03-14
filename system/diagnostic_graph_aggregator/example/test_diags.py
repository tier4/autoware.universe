#!/usr/bin/env python3

# Copyright 2023 The Autoware Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
import rclpy
import rclpy.node
import rclpy.qos


class DummyDiagnostics(rclpy.node.Node):
    def __init__(self, names):
        super().__init__("dummy_diagnostics")
        qos = rclpy.qos.qos_profile_system_default
        self.diags = self.create_publisher(DiagnosticArray, "/diagnostics", qos)
        # Timer set to trigger `on_timer` method every 0.5 seconds
        self.timer = self.create_timer(0.5, self.on_timer)
        self.count = 0
        # Create a status for each name provided
        self.array = [self.create_status(name) for name in names]

    def on_timer(self):
        # Calculate the current phase in the loop based on `self.count`
        # Each phase lasts 10 counts (5 seconds), cycling through the specified states
        transition_size = 7
        phase = (self.count // 5) % transition_size
        print(phase)

        if phase == 0:
            print("All OK")

        # Loop through each diagnostic status to set its level and message
        for status in self.array:
            # Default to OK status
            status.level = DiagnosticStatus.OK
            status.message = "OK"

            # Identify the specific condition based on the current phase
            if phase == 1 and "pandar_monitor: /sensing/lidar/front_lower: pandar_temperature" in status.name:
                print("pandar_temperature WARN")
                status.level = DiagnosticStatus.WARN
                status.message = "WARN"
            elif phase == 2 and "pandar_monitor: /sensing/lidar/front_lower: pandar_temperature" in status.name:
                print("pandar_temperature ERROR")
                status.level = DiagnosticStatus.ERROR
                status.message = "ERROR"
            elif phase == 3 and "pandar_monitor: /sensing/lidar/front_lower: pandar_connection" in status.name:
                print("pandar_connection WARN")
                status.level = DiagnosticStatus.WARN
                status.message = "WARN"
            elif phase == 4 and "pandar_monitor: /sensing/lidar/front_lower: pandar_connection" in status.name:
                print("pandar_connection ERROR")
                status.level = DiagnosticStatus.ERROR
                status.message = "ERROR"
            elif phase == 5 and "imu_monitor: yaw_rate_status" in status.name:
                print("yaw_rate WARN")
                status.level = DiagnosticStatus.WARN
                status.message = "WARN"
            elif phase == 6 and "imu_monitor: yaw_rate_status" in status.name:
                print("yaw_rate ERROR")
                status.level = DiagnosticStatus.ERROR
                status.message = "ERROR"

        # Publish the diagnostic array with updated statuses
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        diagnostics.status = self.array
        self.diags.publish(diagnostics)

        # Increment the count, effectively moving time forward by 0.5 seconds
        self.count = (self.count + 1) % 10000
        print(self.count)

    @staticmethod
    def create_status(name: str):
        # Create and return a DiagnosticStatus with a given name and a generic hardware_id
        return DiagnosticStatus(name=name, hardware_id="example")


if __name__ == "__main__":

    positions = [
        'front_lower', 'front_upper',
        'left_lower', 'left_upper',
        'right_lower', 'right_upper',
        'rear_lower', 'rear_upper'
    ]
    types = ['pandar_temperature', 'pandar_connection']
    
    print("### diags ###")
    diags = [
        f"pandar_monitor: /sensing/lidar/{position}: {type_}"
        for position in positions for type_ in types
    ]
    diags += ["imu_monitor: yaw_rate_status"]
    
    print(diags)

    rclpy.init()
    rclpy.spin(DummyDiagnostics(diags))
    rclpy.shutdown()
