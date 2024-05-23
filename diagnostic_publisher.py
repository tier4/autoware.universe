import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import threading
import math
import copy


class DiagnosticPublisher(Node):
    def __init__(self):
        super().__init__('diagnostic_publisher')
        self.publisher_ = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self.status_level = DiagnosticStatus.OK  # Default status is OK
        self.status_lock = threading.Lock()

        self.initial_pose = None
        self.final_pose = None

        self.initial_pose_set = False
        self.final_pose_set = False

        # Subscribe to the odometry topic
        self.subscription = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.odometry_callback,
            10
        )

        # Thread to listen for key press
        self.key_thread = threading.Thread(target=self.key_listener)
        self.key_thread.daemon = True
        self.key_thread.start()

        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_diagnostics)

    def key_listener(self):
        while True:
            key = input()
            if key == 'w':
                with self.status_lock:
                    self.status_level = DiagnosticStatus.ERROR
                self.get_logger().info('Changed status to ERROR')
            elif key == 'r':
                with self.status_lock:
                    self.status_level = DiagnosticStatus.OK
                    self.initial_pose_set = False
                    self.final_pose_set = False
                self.get_logger().info('Changed status to OK')

    def odometry_callback(self, msg):
        current_vel = msg.twist.twist.linear.x
        current_pose = msg.pose.pose.position

        with self.status_lock:
            if self.status_level == DiagnosticStatus.ERROR and not self.initial_pose_set:
                self.initial_pose = copy.deepcopy(current_pose)
                self.get_logger().info(f'Registered initial position at ERROR: {self.initial_pose}')
                self.initial_pose_set = True

            if self.status_level == DiagnosticStatus.ERROR and not self.final_pose_set and self.initial_pose_set and abs(current_vel) < 0.01:
                self.final_pose = copy.deepcopy(current_pose)
                self.final_pose_set = True
                self.get_logger().info(f'Registered final position at stop: {self.final_pose}')
                self.calculate_distance()

    def calculate_distance(self):
        if self.initial_pose and self.final_pose:
            dx = self.final_pose.x - self.initial_pose.x
            dy = self.final_pose.y - self.initial_pose.y
            distance = math.sqrt(dx**2 + dy**2)
            self.get_logger().info(f'Distance traveled: {distance} meters')
            # Reset poses after calculation
            self.initial_pose = None
            self.final_pose = None

    def publish_diagnostics(self):
        msg = DiagnosticArray()

        # Setting the header with the current time
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = ''
        msg.header = header

        # Creating the DiagnosticStatus
        status = DiagnosticStatus()
        with self.status_lock:
            status.level = self.status_level
        status.name = 'autonomous_emergency_braking: aeb_emergency_stop'
        status.message = '[AEB]: Emergency Brake' if status.level == DiagnosticStatus.ERROR else '[AEB]: System OK'
        status.hardware_id = 'autonomous_emergency_braking'

        # Adding key-value pairs to the status
        key_values = [
            KeyValue(key='RSS', value='15.81'),
            KeyValue(key='Distance', value='6.46'),
            KeyValue(key='Object Speed', value='0.40')
        ]
        status.values = key_values

        # Adding the status to the DiagnosticArray
        msg.status.append(status)

        # Publishing the message
        self.publisher_.publish(msg)
        # self.get_logger().info('Published diagnostic message')


def main(args=None):
    rclpy.init(args=args)
    diagnostic_publisher = DiagnosticPublisher()

    try:
        rclpy.spin(diagnostic_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        diagnostic_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
