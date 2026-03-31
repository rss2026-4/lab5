from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan

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
        # self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.num_particles = 3
        self.init_mean = 0
        self.init_std_dev = .5
        self.particles = 0

    def pose_callback(self, msg):
        
        # initialize particles with certain spread around given point from rviz
        noise = np.random.normal(self.init_mean, self.init_std_dev, size = (self.num_particles, 3))
        
        # self.get_logger().info("Pose %s " % msg.pose)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        thx = msg.pose.pose.orientation.x
        thy = msg.pose.pose.orientation.y
        thz = msg.pose.pose.orientation.z
        thw = msg.pose.pose.orientation.w

        theta = Rotation.from_quat([thx, thy, thz, thw]).as_euler('xyz')[2] # converting quat -> euler -> taking z rotation; x and y should be 0
        # self.get_logger().info("Rotation %s " % theta)
        particles = np.array([x, y, theta])
        noisy_particles = particles + noise
        self.get_logger().info("og %s " % particles)
        self.get_logger().info("no %s " % noisy_particles)
        
        self.particles = noisy_particles = particles + noise
        
        
    # upon new odom msg, apply motion model to get new particle pos
    def odom_callback(self, msg):
        # extract data from msg
        odometry = 0

        # apply motion model
        updated_particles = self.motion_model.evaluate(self.particles, odometry)


        # determine new avg 
        
            
    # upon new sensor msg, apply sensor model to get partgicle probabilies
    def laser_callback(self, msg):
        # extract & filter lidar data
        laser_observations = 0 


        # apply sensor model
        updated_probabilties = self.sensor_model.evaluate(self.particles, laser_observations)

        # return new avg
        pass
    
    # some sort of visualization function; maybe want to create whole other node for it

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
