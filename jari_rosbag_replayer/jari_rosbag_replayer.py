#!/usr/bin/env python3


import time
import rclpy
import rosbag2_py
from rosbag2_py import StorageFilter
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import Pose, Point
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from autoware_auto_perception_msgs.msg import DetectedObjects, PredictedObjects
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_system_msgs.msg import Float32MultiArrayDiagnostic
from visualization_msgs.msg import Marker

# set line: when the ego is over the line, the rosbag start is triggered
POS_X_L = 16690.2
POS_Y_L = 93171.8
POS_X_R = 16695.0
POS_Y_R = 93171.4

# start rosbag time: start time of the replayed rosbag
# Note: this time is available by checking the time when the ego in rosbag exceeds the line above.
# T0 = 1668048507772541470 # 1668048507, 772541470
# T0 = 1664876748366648640
T0 = 1664876748066648640

def get_rosbag_options(path, serialization_format="cdr"):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    return storage_options, converter_options

def open_reader(path: str):
    storage_options, converter_options = get_rosbag_options(path)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader

class JariRosbagReplayer(Node):
    def __init__(self):
        super().__init__("jari_rosbag_replayer")

        self.triggered_time = None
        self.next_pub_index_ego_odom = 0
        self.next_pub_index_perception = 0
        self.next_pub_index_ego_control_cmd = 0
        self.next_pub_index_ego_control_debug = 0

        self.sub_odom = self.create_subscription(
            Odometry, "/localization/kinematic_state", self.onOdom, 1
        )
        self.pub_ego_odom = self.create_publisher(
            Odometry, "/localization/kinematic_state_rosbag", 1
        )
        self.pub_ego_control_cmd = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd_rosbag", 1
        )
        self.pub_ego_control_debug = self.create_publisher(
            Float32MultiArrayDiagnostic, "/control/trajectory_follower/longitudinal/diagnostic_rosbag", 1
        )
        self.pub_perception = self.create_publisher(
            PredictedObjects, "/perception/object_recognition/objects", 1
        )
        self.pub_marker = self.create_publisher(
            Marker, "/jari/debug_marker", 1
        )
        self.rosbag_objects_data = []
        self.rosbag_ego_data = []
        self.rosbag_ego_control_cmd = []
        self.rosbag_ego_control_debug = []
        self.load_rosbag("/mnt/data/rosbags/230118_jari_planning_sim/16_1")

        self.publish_empty_object()
        self.publish_line_marker()

        time.sleep(1.0)  # wait for ready to publish/subscribe

        self.timer = self.create_timer(0.005, self.on_timer)

    def publish_line_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = 1
        marker.ns = "trigger_line"
        p1 = Point()
        p1.x = POS_X_L
        p1.y = POS_Y_L
        marker.points.append(p1)
        p2 = Point()
        p2.x = POS_X_R
        p2.y = POS_Y_R
        marker.points.append(p2)
        marker.color.a = 0.99
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        self.pub_marker.publish(marker)

    def on_timer(self):
        if self.triggered_time is None:  # not triggered yet
            msg = PredictedObjects()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_perception.publish(msg)
            return

        (sec, nanosec) = self.get_clock().now().seconds_nanoseconds()
        t_now = int(sec * 1e9) + nanosec
        t_spent_real = t_now - self.triggered_time
        # print("t_now = ", t_now, ", triggered_time = ", self.triggered_time, "t_spent_from_triggered = ", t_spent_real*1.0e-9)

        # publish_until_spent_time (todo: make function)
        while rclpy.ok() and self.next_pub_index_perception < len(self.rosbag_objects_data):
            object = self.rosbag_objects_data[self.next_pub_index_perception]
            t_spent_rosbag = object[0] - T0
            if t_spent_rosbag < 0: # do not use data before T0
                self.next_pub_index_perception += 1
                continue
            if t_spent_rosbag < t_spent_real:
                msg = object[1]
                msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_perception.publish(msg)
                self.next_pub_index_perception += 1
                # print("perception published: ", self.next_pub_index_perception, "t_spent_rosbag: ", t_spent_rosbag)
            else:
                break

        # publish_until_spent_time (todo: make function)
        while rclpy.ok() and self.next_pub_index_ego_odom < len(self.rosbag_ego_data):
            object = self.rosbag_ego_data[self.next_pub_index_ego_odom]
            t_spent_rosbag = object[0] - T0
            if t_spent_rosbag < 0: # do not use data before T0
                self.next_pub_index_ego_odom += + 1
                continue
            if t_spent_rosbag < t_spent_real:
                msg = object[1]
                msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_ego_odom.publish(msg)
                self.next_pub_index_ego_odom += + 1
                # print("odom published: ", self.next_pub_index_ego_odom, "t_spent_rosbag: ", t_spent_rosbag)
            else:
                break

        # publish_until_spent_time (todo: make function)
        while rclpy.ok() and self.next_pub_index_ego_control_cmd < len(self.rosbag_ego_control_cmd):
            object = self.rosbag_ego_control_cmd[self.next_pub_index_ego_control_cmd]
            t_spent_rosbag = object[0] - T0
            if t_spent_rosbag < 0: # do not use data before T0
                self.next_pub_index_ego_control_cmd += + 1
                continue
            if t_spent_rosbag < t_spent_real:
                msg = object[1]
                msg.stamp = self.get_clock().now().to_msg()
                self.pub_ego_control_cmd.publish(msg)
                self.next_pub_index_ego_control_cmd += + 1
                # print("odom published: ", self.next_pub_index_ego_control_cmd, "t_spent_rosbag: ", t_spent_rosbag)
            else:
                break

        # publish_until_spent_time (todo: make function)
        while rclpy.ok() and self.next_pub_index_ego_control_debug < len(self.rosbag_ego_control_debug):
            object = self.rosbag_ego_control_debug[self.next_pub_index_ego_control_debug]
            t_spent_rosbag = object[0] - T0
            if t_spent_rosbag < 0: # do not use data before T0
                self.next_pub_index_ego_control_debug += + 1
                continue
            if t_spent_rosbag < t_spent_real:
                msg = object[1]
                msg.diag_header.data_stamp = self.get_clock().now().to_msg()
                msg.diag_header.computation_start = self.get_clock().now().to_msg()
                self.pub_ego_control_debug.publish(msg)
                self.next_pub_index_ego_control_debug += + 1
                # print("odom published: ", self.next_pub_index_ego_control_debug, "t_spent_rosbag: ", t_spent_rosbag)
            else:
                break



    def onOdom(self, odom):
        pos = odom.pose.pose.position
        if self.isOverLine(pos) == True and self.triggered_time is None:
            (sec, nanosec) = self.get_clock().now().seconds_nanoseconds()
            self.triggered_time = int(sec * 1e9) + nanosec



    def load_rosbag(self, rosbag2_path: str):
        reader = open_reader(str(rosbag2_path))

        topic_types = reader.get_all_topics_and_types()
        # Create a map for quicker lookup
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        perception_topic = '/perception/object_recognition/objects'
        ego_odom_topic = '/localization/kinematic_state'
        ego_control_cmd_topic = '/control/command/control_cmd'
        ego_control_debug_topic = '/control/trajectory_follower/longitudinal/diagnostic'
        topic_filter = StorageFilter(
            topics=[perception_topic, ego_odom_topic, ego_control_cmd_topic, ego_control_debug_topic])
        reader.set_filter(topic_filter)

        while reader.has_next():
            (topic, data, stamp) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            if (topic == perception_topic):
                self.rosbag_objects_data.append((stamp, msg))
            if (topic == ego_odom_topic):
                self.rosbag_ego_data.append((stamp, msg))
            if (topic == ego_control_cmd_topic):
                self.rosbag_ego_control_cmd.append((stamp, msg))
            if (topic == ego_control_debug_topic):
                self.rosbag_ego_control_debug.append((stamp, msg))

        # print(self.rosbag_objects_data[0])
        # print(self.rosbag_ego_data[0])
        # print("rosbag_objects_data size: ", len(self.rosbag_objects_data))
        # print("rosbag_ego_data size: ", len(self.rosbag_ego_data))
        # print(self.rosbag_objects_data[0][1])


    def isOverLine(self, p):
        dx0 = POS_X_L - POS_X_R
        dy0 = POS_Y_L - POS_Y_R
        dx1 = POS_X_L - p.x
        dy1 = POS_Y_L - p.y
        outer = dx0 * dy1 - dy0 * dx1
        result = 'before line' if outer < 0 else 'OVER LINE!!!'
        print(self.get_clock().now().seconds_nanoseconds(), result)
        # self.get_logger().info(result)
        return bool(outer > 0)

    def publish_empty_object(self):
        msg = PredictedObjects()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_perception.publish(msg)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = JariRosbagReplayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
