import numpy as np
import matplotlib.pyplot as plt

class MotionModel:

    def __init__(self, node):
        ####################################
        # Do any precomputation for the motion
        # model here.

        self.deterministic = False 
        self.node = node
        self.mean = 0
        self.std_dev = 0.1
            
        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        x = particles[:, 0:1]
        y = particles[:, 1:2]
        theta = particles[:, 2:]

        N = len(x)

        dx = odometry[0]
        dy = odometry[1]
        dtheta = odometry[2] 
        
        # apply motion 
        new_x = x + dx*np.cos(theta) - dy*np.sin(theta)
        new_y = y + dx*np.sin(theta) + dy*np.cos(theta)
        new_theta = theta + dtheta
        updated_particles = np.hstack([new_x, new_y, new_theta])


        # no noise
        if self.deterministic:
            # plt.plot(x, y, 'bo', label="original pos")
            # plt.plot(new_x, new_y, 'ro', label="new pos")
            # plt.title("Motion Model (noiseless)")
            # for i in range(N):
            #     plt.plot([new_x[i],x[i]], [new_y[i], y[i]], 'g')
            # plt.ylabel("y_pos")
            # plt.xlabel("x_pos")
            # plt.legend()
            # plt.show()
            self.node.get_logger().info("No Noise")
            return updated_particles 
        
        # noise added
        else: 
            self.node.get_logger().info("Yes Noise")
            noise = np.random.normal(self.mean, self.std_dev, size = (N, 3)) # mean = 0, scale = 1
            noisy_particles = updated_particles + noise
            return noisy_particles
        
        ####################################
