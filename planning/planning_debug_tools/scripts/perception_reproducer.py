#!/usr/bin/env python3

import argparse
import copy
import math
import os
import subprocess
import time

from autoware_auto_perception_msgs.msg import DetectedObjects
from autoware_auto_perception_msgs.msg import PredictedObjects
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosbag2_py import StorageFilter
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import tf2_geometry_msgs
from tf2_ros import LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler


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


def calc_squared_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def create_empty_pointcloud(timestamp):
    pointcloud_msg = PointCloud2()
    pointcloud_msg.header.stamp = timestamp
    pointcloud_msg.header.frame_id = "map"
    pointcloud_msg.height = 1
    pointcloud_msg.is_dense = True
    pointcloud_msg.point_step = 16
    field_name_vec = ["x", "y", "z"]
    offset_vec = [0, 4, 8]
    for field_name, offset in zip(field_name_vec, offset_vec):
        field = PointField()
        field.name = field_name
        field.offset = offset
        field.datatype = 7
        field.count = 1
        pointcloud_msg.fields.append(field)
    return pointcloud_msg


class PerceptionReproducer(Node):
    def __init__(self, args):
        super().__init__("perception_reproducer")
        self.args = args

        # kill empty_objects_publisher
        if self.args.predicted_object:
            cmd = "ps aux | grep empty_objects_publisher | awk '{ print \"kill \", $2 }' | sh"
        else:
            cmd = "ps aux | grep dummy_perception_publisher | awk '{ print \"kill \", $2 }' | sh"
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ps.communicate()[0]

        # subscriber
        self.sub_odom = self.create_subscription(
            Odometry, "/localization/kinematic_state", self.on_odom, 1
        )

        # publisher
        if self.args.predicted_object:
            self.objects_pub = self.create_publisher(
                PredictedObjects, "/perception/object_recognition/objects", 1
            )
        else:
            self.objects_pub = self.create_publisher(
                DetectedObjects, "/perception/object_recognition/detection/objects", 1
            )
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "/perception/obstacle_segmentation/pointcloud", 1
        )

        # tf
        if not self.args.predicted_object:
            self.tf_buffer = Buffer()
            self._tf_listener = TransformListener(self.tf_buffer, self)

        self.ego_pose_idx = None
        self.ego_pose = None

        self.rosbag_objects_data = []
        self.rosbag_ego_odom_data = []

        # load rosbag
        print("Stared loading rosbag")
        if args.bag:
            self.load_rosbag(args.bag)
        elif args.directory:
            for bag_file in sorted(os.listdir(args.directory)):
                self.load_rosbag(args.directory + "/" + bag_file)
        print("Ended loading rosbag")

        # wait for ready to publish/subscribe
        time.sleep(1.0)

        self.timer = self.create_timer(0.01, self.on_timer)
        print("Start timer callback")

    def on_odom(self, odom):
        self.ego_pose = odom.pose.pose

    def on_timer(self):
        timestamp = self.get_clock().now().to_msg()

        if not self.args.predicted_object:
            pointcloud_msg = create_empty_pointcloud(timestamp)
            self.pointcloud_pub.publish(pointcloud_msg)

        if self.ego_pose:
            pose_timestamp = self.find_time_by_ego_pose(self.ego_pose)
            objects_msg = copy.deepcopy(self.find_object_by_timestamp(pose_timestamp))
            if objects_msg:
                objects_msg.header.stamp = timestamp
                if not self.args.predicted_object:
                    objects_msg.header.frame_id = "map"
                    for o in objects_msg.objects:
                        object_pose = o.kinematics.pose_with_covariance.pose
                        ego_orientation_list = [
                            self.ego_pose.orientation.x,
                            self.ego_pose.orientation.y,
                            self.ego_pose.orientation.z,
                            self.ego_pose.orientation.w,
                        ]
                        ego_yaw = euler_from_quaternion(ego_orientation_list)[2]
                        print(ego_yaw)
                        theta = math.atan2(object_pose.position.x, object_pose.position.y)
                        length = math.hypot(object_pose.position.x, object_pose.position.y)

                        object_pose.position.x = self.ego_pose.position.x + length * math.cos(
                            ego_yaw + theta
                        )
                        object_pose.position.y = self.ego_pose.position.y + length * math.sin(
                            ego_yaw + theta
                        )

                        obj_orientation_list = [
                            object_pose.orientation.x,
                            object_pose.orientation.y,
                            object_pose.orientation.z,
                            object_pose.orientation.w,
                        ]
                        obj_yaw = euler_from_quaternion(obj_orientation_list)[2]
                        obj_q = quaternion_from_euler(0, 0, ego_yaw + obj_yaw)
                        object_pose.orientation.x = obj_q[0]
                        object_pose.orientation.y = obj_q[1]
                        object_pose.orientation.z = obj_q[2]
                        object_pose.orientation.w = obj_q[3]

                self.objects_pub.publish(objects_msg)
        else:
            print("No ego pose found.")

    def find_time_by_ego_pose(self, ego_pose):
        if self.ego_pose_idx:
            start_idx = self.ego_pose_idx - 10
            end_idx = self.ego_pose_idx + 10
        else:
            start_idx = 0
            end_idx = len(self.rosbag_ego_odom_data) - 1

        nearest_idx = 0
        nearest_dist = float("inf")
        for idx in range(start_idx, end_idx + 1):
            data = self.rosbag_ego_odom_data[idx]
            dist = calc_squared_distance(data[1].pose.pose.position, ego_pose.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        return self.rosbag_ego_odom_data[nearest_idx][0]

    def find_object_by_timestamp(self, timestamp):
        for data in self.rosbag_objects_data:
            if timestamp < data[0]:
                return data[1]
        return None

    def load_rosbag(self, rosbag2_path: str):
        reader = open_reader(str(rosbag2_path))

        topic_types = reader.get_all_topics_and_types()
        # Create a map for quicker lookup
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        objects_topic = (
            "/perception/object_recognition/objects"
            if self.args.predicted_object
            else "/perception/object_recognition/detection/objects"
        )
        ego_odom_topic = "/localization/kinematic_state"
        topic_filter = StorageFilter(topics=[objects_topic, ego_odom_topic])
        reader.set_filter(topic_filter)

        while reader.has_next():
            (topic, data, stamp) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            if topic == objects_topic:
                self.rosbag_objects_data.append((stamp, msg))
                # print(msg.objects[0].kinematics.pose_with_covariance.pose)
            if topic == ego_odom_topic:
                self.rosbag_ego_odom_data.append((stamp, msg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag", help="rosbag", default=None)
    parser.add_argument("-d", "--directory", help="directory of rosbags", default=None)
    parser.add_argument(
        "-p", "--predicted-object", help="publish predicted object", action="store_true"
    )
    args = parser.parse_args()

    rclpy.init()
    node = PerceptionReproducer(args)
    rclpy.spin(node)

    try:
        rclpy.init()
        node = PerceptionReproducer(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
