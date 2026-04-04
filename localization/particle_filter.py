from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose

from rclpy.node import Node
import rclpy

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        pf_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.particle_filter_frame = pf_frame.lstrip('/')

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.declare_parameter('num_particles', 200)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        # Already declared by SensorModel, just read it
        self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value

        # initialies particle array with zeros, weights all particles equally
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particle_init = False
        self.last_odom_time = None

        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_array_pub = self.create_publisher(PoseArray, "/particles", 1)
        self.scan_pub = self.create_publisher(LaserScan, "/pf/scan", 1)
        self.latest_scan = None

        self.get_logger().info("=============+READY+=============")

    def pose_callback(self, msg):
        """
        This is called when the user clicks 2d pose estimate in rviz, 
        which results in a pose publisehd to /initialpose topic. It then 
        spreads num_particles around the clicked pose with noise, 
        reset the weights so they are all equal, and enable the particle filter.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # yaw
        quat = msg.pose.pose.orientation
        theta = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')[2]

        # noiseily scatter partildes aroudn clicked pose
        center = np.array([x, y, theta])
        noise = np.random.normal(0, 0.1, size=(self.num_particles, 3))
        # noise = np.zeros((self.num_particles, 3))
        self.particles = center + noise
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particle_init = True

        self.get_logger().info("Particles initialized around (%.2f, %.2f, %.2f)" % (x, y, theta))
        
        
    def odom_callback(self, msg):
        """
        This is called on every new odometry message. It updates the particles 
        based on the motion model by gathering information about how the robot 
        moves and then sending that information to the motion model along 
        with the particles.
        """
        if not self.particle_init:
            return

        now = self.get_clock().now()

        # first odometry message so return without updating particles
        if self.last_odom_time is None:
            self.last_odom_time = now
            return

        # ros time to nano seconds to seconds covnerisobn
        dt = (now - self.last_odom_time).nanoseconds / 1e9
        self.last_odom_time = now

        # gather twist data to feed into motion model
        dx = msg.twist.twist.linear.x * dt
        dy = msg.twist.twist.linear.y * dt
        dtheta = msg.twist.twist.angular.z * dt
        odometry = np.array([dx, dy, dtheta])

        self.particles = self.motion_model.evaluate(self.particles, odometry)
        self.publish_pose_estimate()
            
    def laser_callback(self, msg):
        """
        This function is called on every new lidar scan. It first downsamples the scan,
        then scores the particles based on the sensor model which tells how likely each
        particle is to observe the scan. Then it resamples the particles based on weights.
        """
        self.latest_scan = msg

        if not self.particle_init:
            return

        # downsample lidar scan to num_beams_per_particle evenly spaced beams
        ranges = np.array(msg.ranges)
        indices = np.linspace(0, len(ranges) - 1, self.num_beams_per_particle, dtype=int)
        downsampled = ranges[indices]

        weights = self.sensor_model.evaluate(self.particles, downsampled)
        if weights is None:
            return
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return
        self.weights = weights / weight_sum

        # resample particles based on their weights (higher weighted particles are more likely to be selected)
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

        # add small noise after resampling to maintain particle diversity
        self.particles += np.random.normal(0, 0.05, size=self.particles.shape)

        self.publish_pose_estimate()

    def publish_pose_estimate(self):
        mean_x = np.mean(self.particles[:, 0])
        mean_y = np.mean(self.particles[:, 1])
        mean_theta = np.arctan2(
            np.mean(np.sin(self.particles[:, 2])),
            np.mean(np.cos(self.particles[:, 2]))
        )

        half = float(mean_theta) * 0.5
        qz = float(np.sin(half))
        qw = float(np.cos(half))
        now = self.get_clock().now().to_msg()

        # 1. Publish odom message with PF estimate (required by autograder)
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "map"
        odom_msg.pose.pose.position.x = float(mean_x)
        odom_msg.pose.pose.position.y = float(mean_y)
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom_msg)

        # 2. Publish map -> particle_filter_frame TF (required by README)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = self.particle_filter_frame
        t.transform.translation.x = float(mean_x)
        t.transform.translation.y = float(mean_y)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

        # 3. Publish particle cloud in map frame
        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = "map"
        half_thetas = self.particles[:, 2] * 0.5
        sin_h = np.sin(half_thetas)
        cos_h = np.cos(half_thetas)
        for i in range(self.num_particles):
            pose = Pose()
            pose.position.x = float(self.particles[i, 0])
            pose.position.y = float(self.particles[i, 1])
            pose.orientation.z = float(sin_h[i])
            pose.orientation.w = float(cos_h[i])
            pose_array.poses.append(pose)
        self.pose_array_pub.publish(pose_array)

        # 4. Republish laser scan in the PF frame so it aligns with the map in RViz
        if self.latest_scan is not None:
            scan = LaserScan()
            scan.header.stamp = now
            scan.header.frame_id = self.particle_filter_frame
            scan.angle_min = self.latest_scan.angle_min
            scan.angle_max = self.latest_scan.angle_max
            scan.angle_increment = self.latest_scan.angle_increment
            scan.time_increment = self.latest_scan.time_increment
            scan.scan_time = self.latest_scan.scan_time
            scan.range_min = self.latest_scan.range_min
            scan.range_max = self.latest_scan.range_max
            scan.ranges = self.latest_scan.ranges
            scan.intensities = self.latest_scan.intensities
            self.scan_pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
