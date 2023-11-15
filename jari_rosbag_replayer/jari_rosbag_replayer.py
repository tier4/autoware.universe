#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import json
import math
import threading
from pathlib import Path

import rclpy
import rosbag2_py
from rosbag2_py import StorageFilter
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Point, Quaternion, Vector3
from nav_msgs.msg import Odometry
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_system_msgs.msg import Float32MultiArrayDiagnostic
from std_msgs.msg import ColorRGBA
from tier4_debug_msgs.msg import Float32Stamped, Float32MultiArrayStamped
from tier4_planning_msgs.msg import VelocityLimit
from autoware_auto_system_msgs.msg import AutowareState
from tier4_external_api_msgs.srv import Engage
import tf_transformations
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import numpy as np

TRANSLATION_IDENTITY = [0.0, 0.0, 0.0]
ZOOM_IDENTITY = [1.0, 1.0, 1.0]


def quaternion_to_yaw(quaternion):
    return tf_transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]


def euclidean_dist_from_poses(pose0, pose1):
    return math.hypot(pose1.position.x - pose0.position.x, pose1.position.y - pose0.position.y)


class Config:
    def __init__(self, name="no27"):
        directory_path = Path(__file__).resolve().parent
        configs = json.load(open(directory_path / "config.json", "r"))

        config = configs[name]

        self.velocity_limit_mps = config["velocity_limit"] / 3.6

        self.start_line_left_x = config["bag_start_line"]["left"][0]
        self.start_line_left_y = config["bag_start_line"]["left"][1]
        self.start_line_right_x = config["bag_start_line"]["right"][0]
        self.start_line_right_y = config["bag_start_line"]["right"][1]

        self.forward_vehicle_uuid = config["vehicle_uuid"]

        self.rosbag_start_time = config["bag_start_time"] + config["start_offset"]

        # wheelbase: 2.75m, front overhang: 0.8m
        self.baselink_to_front = 3.55

        self.rosbag_path = configs["rosbag_directory"] + "/" + config["bag_name"]
        self.rosbag_directory = configs["rosbag_directory"]


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


class BackgroundRosBagRecorder:
    def __init__(self):
        self.finish_flag = False
        self.worker_thread = None

    def __record_rosbag(self, path):
        os.system('notify-send "start rosbag record!"')
        command = ["ros2", "bag", "record", "-a", "-o", path, "-s", "mcap"]
        log_path = path + "-log.txt"
        log_file = open(log_path, 'w')
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                   preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        while not self.finish_flag:
            time.sleep(1.0)

        os.system('notify-send "finish rosbag record!"')

        process.send_signal(signal.SIGINT)
        process.wait()
        log_file.close()

    def start(self, path):
        import threading
        if self.worker_thread is None:
            self.worker_thread = threading.Thread(target=self.__record_rosbag, args=(path,))
            self.worker_thread.start()

    def stop(self):
        self.finish_flag = True
        if self.worker_thread is not None:
            self.worker_thread.join()
            self.worker_thread = None


class SeriesData:
    def __init__(self, name, stamp_offset=time.time()):
        self.name = name
        self.stamp_offset = None
        self.stamp = []
        self.data = []

    def set_start_offset(self, offset):
        self.stamp_offset = offset

    def clear(self):
        self.stamp = []
        self.data = []

    def append(self, data):
        self.stamp.append(time.time() - self.stamp_offset)
        self.data.append(data)

    def save(self, dir):
        self.dir = dir + self.name + '/'
        os.makedirs(self.dir, exist_ok=True)
        np.save(self.dir + 'stamp.npy', np.array(self.stamp))
        np.save(self.dir + 'data.npy', np.array(self.data))
        #         save plot of series
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(self.stamp, self.data)
        ax.set_title(self.name)
        ax.set_ylabel(self.name)
        ax.set_xlabel('time (s)')
        fig.savefig(self.dir + 'plot.svg')


class LogAnalyzer:
    def __init__(self):
        self.distance_real = SeriesData('distance_real')
        self.distance_sim = SeriesData('distance_sim')
        self.ttc_real = SeriesData('ttc_real')
        self.ttc_sim = SeriesData('ttc_sim')
        self.speed_real = SeriesData('speed_real')
        self.speed_sim = SeriesData('speed_sim')
        self.start_time = None
        self.log_duration_sec = 20

    def start(self):
        self.start_time = time.time()
        self.distance_real.set_start_offset(self.start_time)
        self.distance_sim.set_start_offset(self.start_time)
        self.ttc_real.set_start_offset(self.start_time)
        self.ttc_sim.set_start_offset(self.start_time)
        self.speed_real.set_start_offset(self.start_time)
        self.speed_sim.set_start_offset(self.start_time)

    def on_finish(self):
        dir = "analysis/" + str(time.time()) + "/"

        # save log
        os.makedirs(dir, exist_ok=True)
        # save data and stamp to npy file
        self.distance_sim.save(dir)
        self.distance_real.save(dir)
        self.ttc_sim.save(dir)
        self.ttc_real.save(dir)
        self.speed_sim.save(dir)
        self.speed_real.save(dir)

        # plot distance of real and sim
        plt.figure()
        plt.title("distance")
        plt.xlabel("time [s]")
        plt.ylabel("distance [m]")
        plt.plot(self.distance_real.stamp, self.distance_real.data, label="real")
        plt.plot(self.distance_sim.stamp, self.distance_sim.data, label="sim")
        plt.legend()
        # savefig as svg
        plt.savefig(dir + "distance.svg")
        plt.close()

        # plot ttc of real and sim
        plt.figure()
        plt.title("ttc")
        plt.xlabel("time [s]")
        plt.ylabel("ttc [s]")
        plt.plot(self.ttc_real.stamp, self.ttc_real.data, label="real")
        plt.plot(self.ttc_sim.stamp, self.ttc_sim.data, label="sim")
        plt.legend()
        plt.savefig(dir + "ttc.svg")
        plt.close()

        # plot speed of real and sim
        plt.figure()
        plt.title("speed")
        plt.xlabel("time [s]")
        plt.ylabel("speed [m/s]")
        plt.plot(self.speed_real.stamp, self.speed_real.data, label="real")
        plt.plot(self.speed_sim.stamp, self.speed_sim.data, label="sim")
        plt.legend()
        plt.savefig(dir + "speed.svg")
        plt.close()

    def should_finish(self):
        if self.start_time is None:
            return False
        flag = (time.time() - self.start_time) > self.log_duration_sec
        if flag:
            self.on_finish()
            return True
        return False


class AutowareOperator:
    def __init__(self, node):
        self.node = node
        self.client_engage = node.create_client(
            Engage, "/api/external/set/engage")

        self.state = None
        self.sub_autoware_state = node.create_subscription(
            AutowareState, "/autoware/state", self.on_autoware_state, 1)

        self.pub_pose_estimation = node.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 1)
        self.pub_goal_pose = node.create_publisher(
            PoseStamped, "/planning/mission_planning/goal", 1)

        self.pub_velocity_limit = node.create_publisher(
            VelocityLimit, '/planning/scenario_planning/max_velocity_default',
            rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))

        # check client is ready
        while not self.client_engage.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('engage service not available, waiting again...')

    def on_autoware_state(self, msg):
        self.state = msg.state

    def engage(self, engage=True):
        req = Engage.Request()
        req.engage = engage
        self.client_engage.call_async(req)
        print("engage")

    def set_velocity_limit(self, velocity_limit):
        velocity_limit_msg = VelocityLimit()
        velocity_limit_msg.max_velocity = velocity_limit
        self.pub_velocity_limit.publish(velocity_limit_msg)
        print("set velocity limit")


class JariRosbagReplayer(Node):
    def __init__(self):
        super().__init__("jari_rosbag_replayer")

        self.config = Config()

        self.log_analyzer = LogAnalyzer()

        self.triggered_time = None
        self.next_pub_index_ego_odom = 0
        self.next_pub_index_perception = 0
        self.next_pub_index_ego_control_cmd = 0
        self.next_pub_index_ego_control_debug = 0

        self.rosbag_objects_data = []
        self.rosbag_ego_data = []
        self.rosbag_ego_control_cmd = []
        self.rosbag_ego_control_debug = []

        self.rosbag_recorder = BackgroundRosBagRecorder()

        self.autoware = AutowareOperator(self)

        self.sub_odom_sim = self.create_subscription(
            Odometry, "/localization/kinematic_state", self.on_odom_sim, 1)
        self.pub_odom_real = self.create_publisher(
            Odometry, "/localization/kinematic_state_rosbag", 1)
        self.pub_control_cmd_real = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd_rosbag", 1)
        self.pub_ego_control_debug = self.create_publisher(
            Float32MultiArrayDiagnostic, "/control/trajectory_follower/longitudinal/diagnostic_rosbag", 1)
        self.pub_perception_real = self.create_publisher(
            PredictedObjects, "/perception/object_recognition/objects", 1)
        self.pub_marker = self.create_publisher(
            Marker, "/jari/debug_marker", 1)

        # analyzer
        self.forward_vehicle_object = None
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

        # log path with timestamp
        log_path = self.config.rosbag_directory + "/rosbag/" + str(time.time())
        self.rosbag_recorder.start(log_path)

        time.sleep(1.0)  # wait for ready to publish/subscribe

        self.publish_pose_estimation()

        time.sleep(5.0)

        self.load_rosbag(self.config.rosbag_path)
        print("rosbag is loaded")

        self.publish_goal_pose()
        self.publish_empty_object()
        self.publish_line_marker()

        self.timer = self.create_timer(0.005, self.on_timer)

    def publish_pose_estimation(self):
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.header.frame_id = "map"
        initial_pose.pose.pose.position = Point(x=16673.787109375, y=92971.7265625, z=0.0)
        initial_pose.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.6773713996525991, w=0.7356412080169781)
        self.autoware.pub_pose_estimation.publish(initial_pose)
        print("send pose estimation")

    def publish_goal_pose(self):
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position = Point(x=16713.16796875, y=93383.9296875, z=0.0)
        goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.6811543441258587, w=0.7321398496725002)
        self.autoware.pub_goal_pose.publish(goal_pose)
        print("send goal_pose")

    def analyze_on_odom_sim(self, msg):
        if self.forward_vehicle_object is None:
            return

        dist_between_vehicles = self.calc_dist_between_vehicles(
            msg.pose.pose,
            self.forward_vehicle_object.kinematics.initial_pose_with_covariance.pose,
            True)

        velocity_ego_vehicle = msg.twist.twist.linear.x
        self.log_analyzer.speed_sim.append(velocity_ego_vehicle)

        time_to_collision = 0.0
        if velocity_ego_vehicle != 0.0:
            time_to_collision = dist_between_vehicles / velocity_ego_vehicle

        stamp = self._clock.now().to_msg()

        self.pub_distance_sim.publish(Float32Stamped(stamp=stamp, data=dist_between_vehicles))
        self.pub_ttc_sim.publish(Float32Stamped(stamp=stamp, data=time_to_collision))

        self.log_analyzer.ttc_sim.append(time_to_collision)
        self.log_analyzer.distance_sim.append(dist_between_vehicles)

        self.msg_debug_sim.stamp = stamp
        self.pub_debug_sim.publish(self.msg_debug_sim)

    def analyze_on_odom_real(self, msg):
        if self.forward_vehicle_object is None:
            return

        dist_between_vehicles = self.calc_dist_between_vehicles(
            msg.pose.pose,
            self.forward_vehicle_object.kinematics.initial_pose_with_covariance.pose,
            False)

        velocity_ego_vehicle = msg.twist.twist.linear.x
        self.log_analyzer.speed_real.append(velocity_ego_vehicle)

        time_to_collision = 0.0
        if velocity_ego_vehicle != 0.0:
            time_to_collision = dist_between_vehicles / velocity_ego_vehicle

        stamp = self._clock.now().to_msg()

        self.pub_distance_real.publish(
            Float32Stamped(stamp=stamp, data=dist_between_vehicles))
        self.pub_ttc_real.publish(Float32Stamped(stamp=stamp, data=time_to_collision))

        self.log_analyzer.ttc_real.append(time_to_collision)
        self.log_analyzer.distance_real.append(dist_between_vehicles)

        self.msg_debug_real.stamp = stamp
        self.pub_debug_real.publish(self.msg_debug_real)

    def analyze_on_objects(self, msg):
        for obj in msg.objects:
            uuid0 = obj.object_id.uuid[0]
            uuid1 = obj.object_id.uuid[1]
            if uuid0 == self.config.forward_vehicle_uuid[0] and uuid1 == self.config.forward_vehicle_uuid[1]:
                self.forward_vehicle_object = obj

    def publish_line_marker(self):
        marker = Marker(id=1, type=Marker.LINE_STRIP, action=Marker.ADD, ns="trigger_line")
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.points.append(Point(x=self.config.start_line_left_x, y=self.config.start_line_left_y))
        marker.points.append(Point(x=self.config.start_line_right_x, y=self.config.start_line_right_y))

        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.99)
        marker.scale = Vector3(x=0.3, y=0.3, z=0.3)
        self.pub_marker.publish(marker)

    def publish_until_spent_time(self, target, t_spent_real):
        def get_current_object():
            if target == 'perception':
                return self.rosbag_objects_data[self.next_pub_index_perception]
            elif target == 'ego_odom':
                return self.rosbag_ego_data[self.next_pub_index_ego_odom]
            elif target == 'ego_control_cmd':
                return self.rosbag_ego_control_cmd[self.next_pub_index_ego_control_cmd]
            elif target == 'ego_control_debug':
                return self.rosbag_ego_control_debug[self.next_pub_index_ego_control_debug]
            else:
                raise NotImplementedError()

        def increment_index():
            if target == 'perception':
                self.next_pub_index_perception += 1
            elif target == 'ego_odom':
                self.next_pub_index_ego_odom += 1
            elif target == 'ego_control_cmd':
                self.next_pub_index_ego_control_cmd += 1
            elif target == 'ego_control_debug':
                self.next_pub_index_ego_control_debug += 1
            else:
                raise NotImplementedError()

        def publish_msg(msg):
            if target == 'perception':
                self.pub_perception_real.publish(msg)
                self.analyze_on_objects(msg)
            elif target == 'ego_odom':
                self.pub_odom_real.publish(msg)
                self.analyze_on_odom_real(msg)
            elif target == 'ego_control_cmd':
                self.pub_control_cmd_real.publish(msg)
            elif target == 'ego_control_debug':
                self.pub_ego_control_debug.publish(msg)
            else:
                raise NotImplementedError()

        def has_unpublished_data():
            if target == 'perception' and self.next_pub_index_perception < len(self.rosbag_objects_data):
                return True
            if target == 'ego_odom' and self.next_pub_index_ego_odom < len(self.rosbag_ego_data):
                return True
            if target == 'ego_control_cmd' and self.next_pub_index_ego_control_cmd < len(self.rosbag_ego_control_cmd):
                return True
            if target == 'ego_control_debug' and self.next_pub_index_ego_control_debug < len(
                    self.rosbag_ego_control_debug):
                return True
            return False

        while rclpy.ok() and has_unpublished_data():
            current_object = get_current_object()
            t_spent_rosbag = current_object[0] - self.config.rosbag_start_time
            if t_spent_rosbag < 0:
                increment_index()
                continue
            if t_spent_rosbag < t_spent_real:
                msg = current_object[1]
                if target == 'ego_control_debug':
                    msg.diag_header.data_stamp = self.get_clock().now().to_msg()
                    msg.diag_header.computation_start = self.get_clock().now().to_msg()
                elif target == 'ego_control_cmd':
                    msg.stamp = self.get_clock().now().to_msg()
                else:
                    msg.header.stamp = self.get_clock().now().to_msg()
                publish_msg(msg)
                increment_index()
            else:
                break

    def on_timer(self):
        if self.triggered_time is None:  # not triggered yet
            msg = PredictedObjects()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_perception_real.publish(msg)
            self.analyze_on_objects(msg)
            return

        (sec, nano_sec) = self.get_clock().now().seconds_nanoseconds()
        t_now = int(sec * 1e9) + nano_sec
        t_spent_real = t_now - self.triggered_time

        self.publish_until_spent_time('perception', t_spent_real)

        self.publish_until_spent_time('ego_odom', t_spent_real)

        self.publish_until_spent_time('ego_control_cmd', t_spent_real)

        self.publish_until_spent_time('ego_control_debug', t_spent_real)

    def on_odom_sim(self, odom):
        pos = odom.pose.pose.position
        if self.is_over_line(pos) and self.triggered_time is None:
            (sec, nano_sec) = self.get_clock().now().seconds_nanoseconds()
            self.triggered_time = int(sec * 1e9) + nano_sec
            self.log_analyzer.start()
        self.analyze_on_odom_sim(odom)

    def load_rosbag(self, rosbag2_path: str):
        reader = open_reader(str(rosbag2_path))

        topic_types = reader.get_all_topics_and_types()
        # Create a map for quicker lookup
        type_map = {
            topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        odom_real_topic = '/localization/kinematic_state'
        conrol_cmd_real_topic = '/control/command/control_cmd'
        ego_control_debug_topic = '/control/trajectory_follower/longitudinal/diagnostic'

        topic_filter = StorageFilter(
            topics=[self.pub_perception_real.topic_name, odom_real_topic,
                    conrol_cmd_real_topic, ego_control_debug_topic])
        reader.set_filter(topic_filter)

        while reader.has_next():
            (topic, data, stamp) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            if topic == self.pub_perception_real.topic_name:
                self.rosbag_objects_data.append((stamp, msg))
            if topic == odom_real_topic:
                self.rosbag_ego_data.append((stamp, msg))
            if topic == conrol_cmd_real_topic:
                self.rosbag_ego_control_cmd.append((stamp, msg))
            if topic == ego_control_debug_topic:
                self.rosbag_ego_control_debug.append((stamp, msg))

    def calc_dist_between_vehicles(self, ego_vehicle_pose, forward_vehicle_pose, is_sim):
        yaw_ego_vehicle = quaternion_to_yaw(ego_vehicle_pose.orientation)
        yaw_forward_vehicle = quaternion_to_yaw(forward_vehicle_pose.orientation)
        yaw_diff = math.fabs(yaw_ego_vehicle - yaw_forward_vehicle)
        # print(f"yaw ego: {yaw_ego_vehicle}, yaw forward: {yaw_forward_vehicle}, diff: {yaw_diff}")

        euclidean_dist = euclidean_dist_from_poses(ego_vehicle_pose, forward_vehicle_pose)

        length_forward_vehicle = self.forward_vehicle_object.shape.dimensions.x
        width_forward_vehicle = self.forward_vehicle_object.shape.dimensions.y
        dist_between_vehicles = \
            euclidean_dist * math.cos(yaw_diff) - self.config.baselink_to_front - \
            (length_forward_vehicle / 2.0 * math.cos(yaw_diff) + width_forward_vehicle / 2.0 * math.sin(yaw_diff))

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
        debug_array.append(self.config.baselink_to_front)
        debug_array.append(self.config.baselink_to_front + length_forward_vehicle / 2.0 * math.cos(
            yaw_diff) + width_forward_vehicle / 2.0 * math.sin(yaw_diff))
        debug_array.append(debug_array[4] - debug_array[10])
        debug_array.append(ego_vehicle_pose.position.x)
        debug_array.append(ego_vehicle_pose.position.y)

        if is_sim:
            self.msg_debug_sim.data = debug_array
        else:
            self.msg_debug_real.data = debug_array

        return dist_between_vehicles

    def is_over_line(self, p):
        dx0 = self.config.start_line_left_x - self.config.start_line_right_x
        dy0 = self.config.start_line_left_y - self.config.start_line_right_y
        dx1 = self.config.start_line_left_x - p.x
        dy1 = self.config.start_line_left_y - p.y
        outer = dx0 * dy1 - dy0 * dx1
        return bool(outer > 0)

    def publish_empty_object(self):
        msg = PredictedObjects()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_perception_real.publish(msg)


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = JariRosbagReplayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.rosbag_recorder.stop()
            print("finish recording")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
