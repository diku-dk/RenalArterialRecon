"""
Author: Yifan Jiang (yifanyifan.jiang@mail.utoronto.ca) and Peidi Xu (peidi@di.ku.dk)

Description: Given a bifurcation, finds the optimal position of the branching point
    and the optimal radii.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
from scipy.optimize import minimize

class GD_OptimizerUnit:

    def __init__(self, locs, radii, init_loc, mode, T=1, a=0.99, w1=642, w2=1, w3=5e3, c=3,
                 max_try=5, min_r=0.5, max_r=2, radius_range=2, loc_range=20,
                 optimize_r=True, r_min_thresh=0.01,
                 include_surface_area=False, vsize=1, flow=np.array(1), viscosity=1):

        self.dataPoints = locs
        self.num_points = len(self.dataPoints)
        self.testRadius = radii
        self.testMedians = init_loc
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

        self.flow = flow
        self.viscosity = viscosity

    def cost(self, testMedian):
        """
        Calculate cost.
        """

        
        temp3 = 0

        
        if self.cost_mode == 'MC':
            temp = 0.0
            for i in range(self.num_points):
                surface_term = self.vsize * 2 * np.linalg.norm(testMedian - self.dataPoints[i]) * self.testRadius[i] * np.pi if \
                    self.include_surface_area else 0
                temp += self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]) * (self.testRadius[i] ** 2) * np.pi + \
                        surface_term

            temp *= self.w1

            return temp

        
        else:
            if self.include_surface_area:
                temp1 = np.array([self.vsize * np.pi * np.linalg.norm(testMedian - self.dataPoints[i]) * self.testRadius[i] ** 2 +
                                  2 * self.vsize * np.linalg.norm(testMedian - self.dataPoints[i]) * self.testRadius[i]
                                  for i in range(self.num_points)])
            else:
                temp1 = np.array([self.vsize * np.pi * np.linalg.norm(testMedian - self.dataPoints[i]) * self.testRadius[i] ** 2
                                  for i in range(self.num_points)])

            temp1 = self.w1 * np.sum(temp1)


            temp2 = np.array([self.flow[i]**2 * 8 * self.viscosity * self.vsize * np.linalg.norm(
                testMedian - self.dataPoints[i])/(np.pi * self.testRadius[i] ** 4)
                              for i in range(1, len(self.dataPoints))])

            temp2 = self.w2 * np.sum(temp2)
            
            

            return temp1 + temp2 + temp3


    def optimize(self):
        maxs = np.max(self.dataPoints, axis=0) - np.min(self.testRadius)/(2 * self.vsize)
        mins = np.min(self.dataPoints, axis=0) + np.min(self.testRadius)/(2 * self.vsize)
        

        if not np.all(maxs >= mins):
            return self.testMedians, self.testRadius, self.cost(self.testMedians)

        bnds = [(i, j) for i, j in zip(mins, maxs)]
        x0 = self.testMedians
        res = minimize(self.cost, x0, bounds=bnds)
        return res.x, self.testRadius, self.cost(res.x)

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

