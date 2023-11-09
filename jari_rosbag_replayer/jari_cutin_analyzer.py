#!/usr/bin/env python3

import math
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
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from autoware_auto_perception_msgs.msg import DetectedObjects, PredictedObjects
from tier4_debug_msgs.msg import Float32Stamped, Float32MultiArrayStamped
from visualization_msgs.msg import Marker
import tf_transformations


directory_path = Path(__file__).resolve().parent

configs = json.load(open(directory_path / "config.json", "r"))

config = configs["no27"]

FORWARD_VEHICLE_UUID = config["vehicle_uuid"]

BASELINK_TO_FRONT_EGO_VEHICLE = 3.55  # wheel base: 2.75m, front overhang: 0.8m

TRANSLATION_IDENTITY = [0.0, 0.0, 0.0]
ZOOM_IDENTITY = [1.0, 1.0, 1.0]

def quaternion_to_yaw(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    qlist = [x, y, z, w]
    euler = tf_transformations.euler_from_quaternion(qlist)
    # print(f"yaw: {euler[2]}")
    return euler[2]

def euclidean_dist_from_poses(pose0, pose1):
    x0 = pose0.position.x
    y0 = pose0.position.y
    x1 = pose1.position.x
    y1 = pose1.position.y
    return math.sqrt(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2))

class JariCutinAnalyzer(Node):
    def __init__(self):
        super().__init__("jari_cutin_analyzer")

        self.forward_vehicle_object = None

        self.sub_odom_sim = self.create_subscription(
            Odometry, "/localization/kinematic_state", self.onOdomSim, 1)
        self.sub_odom_real = self.create_subscription(
            Odometry, "/localization/kinematic_state_rosbag", self.onOdomReal, 1)
        self.sub_objects = self.create_subscription(
            PredictedObjects, "/perception/object_recognition/objects", self.onObjects, 1)

        self.pub_distance_sim = self.create_publisher(
            Float32Stamped, "/jari/distance_betwween_vehicles/sim", 1)
        self.pub_distance_real = self.create_publisher(
            Float32Stamped, "/jari/distance_betwween_vehicles/real", 1)
        self.pub_ttc_sim = self.create_publisher(
            Float32Stamped, "/jari/time_to_collision/sim", 1)
        self.pub_ttc_real = self.create_publisher(
            Float32Stamped, "/jari/time_to_collision/real", 1)
        self.pub_debug_sim = self.create_publisher(
            Float32MultiArrayStamped, "/jari/debug/sim", 1)
        self.pub_debug_real = self.create_publisher(
            Float32MultiArrayStamped, "/jari/debug/real", 1)

        self.msg_debug_sim = Float32MultiArrayStamped()
        self.msg_debug_real = Float32MultiArrayStamped()
        # data[0]: yaw ego [rad]
        # data[1]: yaw forward vehicle [rad]
        # data[2]: yaw diff [rad]
        # data[3]: euclidean distance [m]
        # data[4]: euclidean distance * cos(yaw_diff) [m]
        # data[5]: length / 2 of the forward vehicle [m]
        # data[6]: width / 2 of the forward vehicle [m]
        # data[7]: length / 2 * cos(yaw_diff) [m]
        # data[8]: width / 2 * sin(yaw_diff) [m]
        # data[9]: baselink to front [m]
        # data[10]: sum of length deduction [m]
        # data[11]: distance between the vehicles [m]
        # data[12]: pose.position.x [m]
        # data[13]: pose.position.y [m]

        time.sleep(1.0)  # wait for ready to publish/subscribe

    def onOdomSim(self, msg):
        if self.forward_vehicle_object == None:
            return

        dist_between_vehicles = self.calc_dist_between_vehicles(
            msg.pose.pose,
            self.forward_vehicle_object.kinematics.initial_pose_with_covariance.pose,
            True)

        velocity_ego_vehicle = msg.twist.twist.linear.x
        time_to_collision = dist_between_vehicles / velocity_ego_vehicle
        # print(f"time_to_collision: {time_to_collision}")

        msg_dist = Float32Stamped()
        msg_dist.stamp = self.get_clock().now().to_msg()
        msg_dist.data = dist_between_vehicles
        self.pub_distance_sim.publish(msg_dist)

        msg_ttc = Float32Stamped()
        msg_ttc.stamp = self.get_clock().now().to_msg()
        msg_ttc.data = time_to_collision
        self.pub_ttc_sim.publish(msg_ttc)

        self.msg_debug_sim.stamp = self.get_clock().now().to_msg()
        self.pub_debug_sim.publish(self.msg_debug_sim)

    def onOdomReal(self, msg):
        if self.forward_vehicle_object == None:
            return

        dist_between_vehicles = self.calc_dist_between_vehicles(
            msg.pose.pose,
            self.forward_vehicle_object.kinematics.initial_pose_with_covariance.pose,
            False)

        velocity_ego_vehicle = msg.twist.twist.linear.x
        time_to_collision = dist_between_vehicles / velocity_ego_vehicle
        # print(f"time_to_collision: {time_to_collision}")

        msg_dist = Float32Stamped()
        msg_dist.stamp = self.get_clock().now().to_msg()
        msg_dist.data = dist_between_vehicles
        self.pub_distance_real.publish(msg_dist)

        msg_ttc = Float32Stamped()
        msg_ttc.stamp = self.get_clock().now().to_msg()
        msg_ttc.data = time_to_collision
        self.pub_ttc_real.publish(msg_ttc)

        self.msg_debug_real.stamp = self.get_clock().now().to_msg()
        self.pub_debug_real.publish(self.msg_debug_real)

    def calc_dist_between_vehicles(self, ego_vehicle_pose, forward_vehicle_pose, is_sim):
        yaw_ego_vehicle = quaternion_to_yaw(ego_vehicle_pose.orientation)
        yaw_forward_vehicle = quaternion_to_yaw(forward_vehicle_pose.orientation)
        yaw_diff = math.fabs(yaw_ego_vehicle - yaw_forward_vehicle)
        # print(f"yaw ego: {yaw_ego_vehicle}, yaw forward: {yaw_forward_vehicle}, diff: {yaw_diff}")

        euclidean_dist = euclidean_dist_from_poses(ego_vehicle_pose, forward_vehicle_pose)

        length_forward_vehicle = self.forward_vehicle_object.shape.dimensions.x
        width_forward_vehicle = self.forward_vehicle_object.shape.dimensions.y
        dist_between_vehicles = \
            euclidean_dist * math.cos(yaw_diff) - BASELINK_TO_FRONT_EGO_VEHICLE - \
            (length_forward_vehicle / 2.0 * math.cos(yaw_diff) + \
             width_forward_vehicle / 2.0 * math.sin(yaw_diff))
        # print(f"dist: {dist_between_vehicles}")
        # print(f"euclidean_dist: {euclidean_dist}")
        # print(f"forward vehicle length/2: {length_forward_vehicle/2.0}, width/2: {width_forward_vehicle/2.0}")
        # print(f"euclidean_dist * cos: {euclidean_dist * math.cos(yaw_diff)}")
        # print(f"length/2 * cos: {length_forward_vehicle / 2.0 * math.cos(yaw_diff)}")
        # print(f"width/2 * sin: {width_forward_vehicle / 2.0 * math.sin(yaw_diff)}")

        debug_array = []
        debug_array.append(yaw_ego_vehicle)
        debug_array.append(yaw_forward_vehicle)
        debug_array.append(yaw_diff)
        debug_array.append(euclidean_dist)
        debug_array.append(euclidean_dist * math.cos(yaw_diff))
        debug_array.append(length_forward_vehicle / 2.0)
        debug_array.append(width_forward_vehicle / 2.0)
        debug_array.append((length_forward_vehicle / 2.0) * math.cos(yaw_diff))
        debug_array.append((width_forward_vehicle / 2.0) * math.sin(yaw_diff))
        debug_array.append(BASELINK_TO_FRONT_EGO_VEHICLE)
        debug_array.append(BASELINK_TO_FRONT_EGO_VEHICLE + length_forward_vehicle / 2.0 * math.cos(yaw_diff) + width_forward_vehicle / 2.0 * math.sin(yaw_diff))
        debug_array.append(debug_array[4] - debug_array[10])
        debug_array.append(ego_vehicle_pose.position.x)
        debug_array.append(ego_vehicle_pose.position.y)

        if is_sim:
            self.msg_debug_sim.data = debug_array
        else:
            self.msg_debug_real.data = debug_array

        return dist_between_vehicles

    def onObjects(self, msg):
        for obj in msg.objects:
            uuid0 = obj.object_id.uuid[0]
            uuid1 = obj.object_id.uuid[1]
            if uuid0 == FORWARD_VEHICLE_UUID[0] and uuid1 == FORWARD_VEHICLE_UUID[1]:
                self.forward_vehicle_object = obj


def main(args=None):
    try:
        rclpy.init(args=args)
        node = JariCutinAnalyzer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
