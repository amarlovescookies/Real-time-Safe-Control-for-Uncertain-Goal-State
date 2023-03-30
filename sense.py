import numpy as np
import scipy as scipy
import scipy.stats

class Sense:
    def __init__(self, target, arena, maxNoise):
        self.target = target  # target as np.array([[x,y]])
        self.prev_x = -1
        self.prev_y = -1
        self.N = 200
        self.std = np.array([0.1, 0.1])
        self.arena = arena  # arena as np.array([[x_min,x_max], [y_min, y_max]])
        self.x_range = self.arena[0]  # np.array([0,800])
        self.y_range = self.arena[1]  # np.array([0,600])
        self.sensor_error_std = maxNoise
        self.landmarks = np.array([[144, 73], [410, 13], [336, 175], [718, 159], [178, 484], [665, 464]])
        self.NL = len(self.landmarks)
        self.particles = np.empty((self.N, 2))
        self.create_uniform_particles()
        # self.particles = create_uniform_particles(x_range, y_range, self.N)
        self.weights = np.array([1.0] * self.N)

    def create_uniform_particles(self):
        """
        function to sample N initial guesses for particles in the defined space
        Inputs:
        N = number of particles
        x_range =  2x1 array of range of x
        y_range = 2x1 array of range of y
        Output:
        particles = Nx2 array with coordinates of N particles sampled from a uniform distribution
        """
        # self.particles = np.empty((self.N, 2)) # create an empty array of shape Nx2
        self.particles[:, 0] = np.random.uniform(self.x_range[0], self.x_range[1], size=self.N)  # Sample N points from uniform dist in the interval [x_range[0], x_range[1])
        self.particles[:, 1] = np.random.uniform(self.y_range[0], self.y_range[1], size=self.N)  # Sample N points from uniform dist in the interval [y_range[0], y_range[1])
        return self.particles

    def particle_filter(self, targetPos):
        """
        Input set of particles each having a state and a weight from previous time step, some new action and new measurement at current time step.
        Start with empty set of particles and zero normalization factor
        for number of particles do the following steps recursively:
          1. sample a particle from old set with its state and weight
          2. sample a particle from the new distribution using old state and action
          3. Compute weight of this new particle with respect to the measurement
          4. Update normalization factor
          5. Insert particle in the new set
        For all particles in new set normalize weights
        return mean and covariance matrix of new distribution of particles
        """
        # global target
        # global prev_x
        # global prev_y
        # global trajectory
        self.target = targetPos
        x = self.target[0]
        y = self.target[1]
        # target = np.array([[x,y]])
        # trajectory=np.vstack((trajectory,np.array([x,y])))

        if self.prev_x > 0:
            heading = np.arctan2(np.array([y - self.prev_y]), np.array([self.prev_x - x]))  # cal heading of target relative to prev position

            if heading > 0:
                heading = -(heading - np.pi)
            else:
                heading = -(heading + np.pi)

            distance = np.linalg.norm(np.array([[self.prev_x, self.prev_y]]) - np.array([[x, y]]), axis=1)  # calc diistance of target from prev position

            u = np.array([heading, distance])
            self.predict(u, dt=1.)

            # Lets add random noise to the target position directly
            # newTarget = np.array([x+np.random.randn(1) * self.sensor_error_std, y+np.random.randn(1) * self.sensor_error_std]).reshape((1,-1))
            newTarget = self.target


            # zs = (np.linalg.norm(self.landmarks - self.target, axis=1) + (np.random.randn(self.NL) * self.sensor_error_std))
            zs = np.linalg.norm(self.landmarks - newTarget, axis=1)# + (np.random.randn(self.NL) * self.sensor_error_std))
            self.update(z=zs, R=50)

            indices = self.systematic_resample()
            self.resample_from_index(indices)

        self.prev_x = x
        self.prev_y = y

        mean, covariance_matrix = self.estimate()
        w1, w2, w3, w4, theta = self.get_points(mean, covariance_matrix)

        return w1, w2, w3, w4, theta, self.particles, self.weights, mean, covariance_matrix 

    def predict(self, u, dt=0.4):
        """
        function to update coordinates of particles based on predictions claculated based on distance moved by the target
        Inputs:
        particles = Nx2 array with coordinates of N particles sampled from a distribution
        u = 2x1 array with heading of target and distance of target from its previous position
        std = 2x1 array initialized standard deviation of the distribution
        dt =  scaling factor on distance moved by target
        Output:
        updates the coordinates of the particles based on prediction
        """
        N = len(self.particles)  # number of particles
        dist = (u[1] * dt) + (np.random.randn(N) * self.std[1])  # create a random (N,) array multiply it with some standard deviation and add the distance moved by the target
        self.particles[:, 0] += np.cos(u[0]) * dist  # update x position of all particles
        self.particles[:, 1] += np.sin(u[0]) * dist  # update y position of all particles

    def update(self, z, R):
        """
        function to update weights of particles using the pdf of distance between landmarks and particles
        Inputs:
        particles = Nx2 array with coordinates of N particles sampled from a distribution
        weights = current weights corresponding to every particle
        z = distances of landmarks from the target
        R = upper tail probability
        landmarks = list of landmarks
        Output:
        update weights of all particles based on
        """
        self.weights.fill(1.)  # fill weights array with ones
        for i, landmark in enumerate(self.landmarks):  # loop over number of landmarks

            distance = np.power((self.particles[:, 0] - landmark[0]) ** 2 + (self.particles[:, 1] - landmark[1]) ** 2,0.5)  # calc distance between all particles and the landmark
            self.weights *= scipy.stats.norm(distance, R).pdf(z[i])  # update weights based on pdf of distances of target from landmarks

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize weights

    def systematic_resample(self):
        """
        function to resample the particles with updated weights
        Inputs:
        weights = Nx1 array with current weights corresponding to every particle
        Output:
        indices = indices of particles with higher probability
        """
        N = len(self.weights)
        positions = (np.arange(N) + np.random.random()) / N

        indices = np.zeros(N, 'i')
        cum_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N and j < N:
            if positions[i] < cum_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def resample_from_index(self, indices):
        """
        function to resample all particles and weights with indices
        Inputs:
        indices = indices of particles that need to be picked
        particles = Nx2 array with coordinates of particles
        weights = Nx1 array with current weights corresponding to every particle
        Output:
        updates all particles and weights to only the indices
        """
        self.particles[:] = self.particles[indices]
        self.weights[:] = self.weights[indices]
        self.weights /= np.sum(self.weights)  # normalize weights

    def estimate(self):
        """
        function to calculate mean and variance of distribution of particles based on weights
        Inputs:
        particles = Nx2 array with coordinates of particles
        weights = Nx1 array with current weights corresponding to every particle
        Output:
        mean = mean of the distribution of particles based on weights
        covar = covariance of the distribution of particles based on weights
        """
        mean = np.average(self.particles, axis=0, weights=self.weights) # calc mean of distribution 

        covar = self.cov_mat(self.particles.T) # calc covariance of distribution
        return mean, covar

    def var(self, x, y):
        """
        function to calculate mean variance between x and y
        Inputs:
        x, y = np.arrays of same length
        Output:
        var = variance between x and y
        """
        xbar, ybar = x.mean(), y.mean()
        return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)


    def cov_mat(self, X):
        """
        function to calculate covariance matrix from variance of variables of distribution
        Input:
        particles = 2XN array of coordinates of the particles
        Output:
        covariance matrix of distribution of particles
        """
        return np.array([[self.var(X[0], X[0]), self.var(X[0], X[1])], 
                        [self.var(X[1], X[0]), self.var(X[1], X[1])]])

    def get_points(self, mean, covariance_matrix):
        """
        function to calculate mean and variance of distribution of particles based on weights
        Inputs:
        mean = mean of the distribution of particles based on weights
        var = variance of the distribution of particles based on weights
        Output:
        w1, w2, w3, w4 = max bounds of the distribtuion
        theta = angle of the w1 wrt x-axis
        """
        eVe, eVa = np.linalg.eig(covariance_matrix)
        vec1 = np.array([3 * np.sqrt(eVe[0]) * eVa.T[0][0], 3 * np.sqrt(eVe[0]) * eVa.T[0][1]])
        vec2 = np.array([3 * np.sqrt(eVe[1]) * eVa.T[1][0], 3 * np.sqrt(eVe[1]) * eVa.T[1][1]])
        w1 = mean + vec1
        w2 = mean - vec1
        w3 = mean + vec2
        w4 = mean - vec2
        theta = np.arctan(w1[1] / w1[0])
        return w1, w2, w3, w4, theta