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
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

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

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

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
        noise = np.random.normal(0, 0.5, size=(self.num_particles, 3))
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
        # return if particles have not been initialized
        if not self.particle_init:
            return

        # downsample lidar scan to num_beams_per_particle evenly spaced beams
        ranges = np.array(msg.ranges)
        indices = np.linspace(0, len(ranges) - 1, self.num_beams_per_particle, dtype=int)
        downsampled = ranges[indices]

        # evalute each particle with senosr model and normallize
        self.weights = self.sensor_model.evaluate(self.particles, downsampled)
        self.weights /= np.sum(self.weights)

        # resample particles based on their weights (higher weighted particles are more likely to be selected)
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

        self.publish_pose_estimate()

    def publish_pose_estimate(self):
        """
        We want to publish the estimated pose to:
        1. Odometry msg on /pf/pose/odom (required by autograder)
        2. TF broadcast from /map -> particle_filter_frame (for RViz car display)
        3. PoseArray on /particles for particle cloud visualziton
        """
        # average x and y, use circular mean for theta
        # CAN DO THIS A BETTER WAY PROBABLY BUT THIS IS SIMPLE SOLUTION FOR NOW
        mean_x = np.mean(self.particles[:, 0])
        mean_y = np.mean(self.particles[:, 1])
        mean_theta = np.arctan2(
            np.mean(np.sin(self.particles[:, 2])),
            np.mean(np.cos(self.particles[:, 2]))
        )

        quat = Rotation.from_euler('xyz', [0, 0, mean_theta]).as_quat()
        now = self.get_clock().now().to_msg()

        # publish odometry message with estimated pose  in /map frame
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "/map"
        odom_msg.pose.pose.position.x = mean_x
        odom_msg.pose.pose.position.y = mean_y
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        self.odom_pub.publish(odom_msg)

        # publish transform from /map to particle_filter_frame
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "/map"
        t.child_frame_id = self.particle_filter_frame
        t.transform.translation.x = mean_x
        t.transform.translation.y = mean_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

        # publish pose array of all particles for rviz visualization
        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = "/map"
        for p in self.particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            q = Rotation.from_euler('xyz', [0, 0, p[2]]).as_quat()
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)
        self.pose_array_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
