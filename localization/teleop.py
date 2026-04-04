import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

SPEED = 1.0
STEERING = 0.34


class Teleop(Node):
    def __init__(self):
        super().__init__('teleop')
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)

        self.speed = 0.0
        self.steering = 0.0

        self.timer = self.create_timer(0.05, self.publish_cmd)

        self.get_logger().info(
            "Teleop ready: W=fwd, S=back, A=left, D=right, SPACE=stop, Q=quit"
        )

    def publish_cmd(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering
        self.pub.publish(msg)


def get_key(timeout=0.05):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = Teleop()

    try:
        while rclpy.ok():
            key = get_key()
            if key == 'q' or key == '\x03':
                break
            elif key == 'w':
                node.speed = SPEED if node.speed <= 0.0 else 0.0
            elif key == 's':
                node.speed = -SPEED if node.speed >= 0.0 else 0.0
            elif key == 'a':
                node.steering = STEERING if node.steering <= 0.0 else 0.0
            elif key == 'd':
                node.steering = -STEERING if node.steering >= 0.0 else 0.0
            elif key == ' ':
                node.speed = 0.0
                node.steering = 0.0

            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        node.speed = 0.0
        node.steering = 0.0
        node.publish_cmd()
        node.destroy_node()
        rclpy.shutdown()
