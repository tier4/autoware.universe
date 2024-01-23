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
from tier4_debug_msgs.msg import Int32Stamped


class DummyDiagnostics(rclpy.node.Node):
    def __init__(self, names):
        super().__init__("dummy_diagnostics")
        qos = rclpy.qos.qos_profile_system_default
        self.diags = self.create_publisher(DiagnosticArray, "/diagnostics", qos)
        self.timer = self.create_timer(0.5, self.on_timer)
        self.sub = self.create_subscription(
            Int32Stamped, "/mrm_test/command", self.on_command, 1
        )
        self.array = [self.create_status(name) for name in names]

    def on_timer(self):
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        diagnostics.status = self.array
        self.diags.publish(diagnostics)
        
    def on_command(self, msg):
        if (msg.data == 0):
            print("autonomous mode ok")
            self.array[0].level = DiagnosticStatus.OK
            self.array[0].message = "OK"
            self.array[1].level = DiagnosticStatus.OK
            self.array[1].message = "OK"
            self.array[2].level = DiagnosticStatus.OK
            self.array[2].message = "OK"
            self.array[3].level = DiagnosticStatus.OK
            self.array[3].message = "OK"
        elif (msg.data == 1):
            print("autonomous mode ng")
            self.array[0].level = DiagnosticStatus.ERROR
            self.array[0].message = "ERROR"
        elif (msg.data == 2):
            print("autonomous and pull_over ng")
            self.array[0].level = DiagnosticStatus.ERROR
            self.array[0].message = "ERROR"
            self.array[3].level = DiagnosticStatus.ERROR
            self.array[3].message = "ERROR"
        elif (msg.data == 3):
            print("autonomous, pull_over and comfortable_stop ng")
            self.array[0].level = DiagnosticStatus.ERROR
            self.array[0].message = "ERROR"
            self.array[2].level = DiagnosticStatus.ERROR
            self.array[2].message = "ERROR"
            self.array[3].level = DiagnosticStatus.ERROR
            self.array[3].message = "ERROR"

    @staticmethod
    def create_status(name: str):
        return DiagnosticStatus(name=name, hardware_id="example", level=DiagnosticStatus.OK, message="OK")


if __name__ == "__main__":
    diags = [
        "test/autonomous: status",
        "test/emergency_stop: status",
        "test/comfortable_stop: status",
        "test/pull_over: status",
    ]
    rclpy.init()
    rclpy.spin(DummyDiagnostics(diags))
    rclpy.shutdown()
