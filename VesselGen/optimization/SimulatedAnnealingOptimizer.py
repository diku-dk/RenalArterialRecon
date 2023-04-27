"""
Author: Junhong Shen (jhshen@g.ucla.edu) and Peidi Xu (peidi@di.ku.dk)

Description: Simulated Annealing for the specific case of vascular network optimization.

This one is not used in our paper

"""

import numpy as np


class SA_Optimizer:
    """ 
    Given a set of nodes, find the optimal location of the branching point and the radius for each vessel 
    connecting the branching point with one of the nodes.
    """

    def __init__(self, locs, radii, init_loc, mode, T=1, a=0.99, w1=642, w2=1, w3=5e3, c=3,
                 max_try=5, min_r=0.5, max_r=2, radius_range=2, loc_range=20,
                 optimize_r=True, r_min_thresh=0.01,
                 include_surface_area=False, vsize=1):
        self.dataPoints = locs
        self.num_points = len(self.dataPoints)
        self.testRadii = [radii]
        self.testMedians = [init_loc]
        self.cost_mode = mode
        self.count = 0

        self.vsize = vsize

        
        self.T = T
        self.a = a
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.c = c
        self.max_try = max_try
        self.min_r = min_r
        self.max_r = max_r
        self.include_surface_area = include_surface_area
        self.radius_range = radius_range
        self.loc_range = loc_range
        self.optimize_r = optimize_r
        self.r_min_thresh = r_min_thresh

        self.dim = len(self.dataPoints[0])
        self.costs = [self.cost(self.testMedians[self.count], self.testRadii[self.count])]


    def optimize(self):
        """ 
        Take random moves and find the minimum cost in all the moves.
        """

        while self.T > 1e-5:
            i = 0
            while i < self.max_try:
                self.move()
                cost_old = self.costs[self.count - 1]
                cost_new = self.costs[self.count]
                if cost_new < cost_old:
                    break
                else:
                    prob = np.exp((cost_old - cost_new) / self.T)
                    pa = np.random.random()
                    if prob <= pa or np.any(self.testRadii[-1]) <= self.r_min_thresh:
                        del self.testMedians[-1]
                        del self.testRadii[-1]
                        del self.costs[-1]
                        self.count -= 1
                    i += 1
            self.T = self.T * self.a
        min_idx = np.argmin(self.costs)
        return self.testMedians[min_idx], self.testRadii[min_idx], self.costs[min_idx]

    def move(self):
        """ 
        Take a random move.
        """
        t = self.T
        t = 1
        loc_new = self.testMedians[self.count] + (2 * np.random.rand(1, self.dim)[0] - 1
                                                  ) * 0.005 * t * self.get_loc_range()

        if self.optimize_r:
            t = self.T
            t = 1

            rand = np.concatenate((np.array([0.5]), np.random.rand(1, self.num_points - 1)[0]))
            radii_new = self.testRadii[self.count] + (2 * rand - 1) * 0.005 * self.get_radius_range()

            radii_new = np.maximum(self.r_min_thresh, radii_new)

            radii_new[0] = self.get_first_r(radii_new)

        else:
            radii_new = self.testRadii[self.count]

        self.count += 1
        self.testMedians.append(loc_new)
        self.testRadii.append(radii_new)
        self.costs.append(self.cost(self.testMedians[self.count], self.testRadii[self.count]))

    def cost(self, testMedian, testRadius):
        """ 
        Calculate cost.
        """

        
        if self.cost_mode == 'MC':
            temp = 0.0
            for i in range(self.num_points):
                surface_term = self.vsize * 2 * np.linalg.norm(testMedian - self.dataPoints[i]) * testRadius[i] if \
                    self.include_surface_area else 0
                temp += self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]) * (testRadius[i] ** 2) + \
                        surface_term + max(0, testRadius[i] - self.max_r) ** 2 * self.w1 + \
                        max(0, self.min_r - testRadius[i]) ** 2 * self.w2
            return temp

        
        else:
            if self.include_surface_area:
                temp1 = np.array([self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]) * testRadius[i] ** 2 +
                                  2 * self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]) * testRadius[i]
                                  for i in range(self.num_points)])
            else:
                temp1 = np.array([self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]) * testRadius[i] ** 2
                                  for i in range(self.num_points)])
            temp1 = self.w1 * np.sum(temp1)
            temp2 = np.array([testRadius[i] ** 4 / (self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]))
                              for i in range(1, len(self.dataPoints))])
            temp2 = 1 / np.sum(temp2) + self.vsize * np.linalg.norm(testMedian - self.dataPoints[0]) / testRadius[0] ** 4
            temp2 *= self.w2
            temp3 = np.array([self.penalty(testRadius, i) for i in range(self.num_points)])
            temp3 = self.w3 * np.sum(temp3)
            return temp1 + temp2 + temp3

    def penalty(self, testRadius, k):
        """ 
        Calculate penalty which limits the radius within a resonable range.
        """

        return max(0, testRadius[k] - self.max_r) ** 2 + max(0, self.min_r - testRadius[k]) ** 2


    def get_radius_range(self):
        return np.max(self.testRadii[self.count]) - np.min(self.testRadii[self.count])
        return self.radius_range

    def get_loc_range(self):

        return np.max(self.testMedians[self.count]) - np.min(self.testMedians[self.count])
        return self.loc_range

    def get_first_r(self, testRadius):
        return ((np.sum(testRadius ** self.c) - testRadius[0] ** self.c)) ** (1 / self.c)
