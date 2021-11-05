from collections import deque
import numpy as np
from KalmanFilter import KalmanFilter
import math
from parameters import *


class speedEstimator_vz():
    def __init__(self):
        self.validSpeedUpdated = False
        self.lastPoint = 0.
        self.distanceThreshold = 0.05
        self.speedUpdateTime = []
        self.speedWindowSize = 10
        self.keyMeasPairs = deque(maxlen=80)
        self.speedWindow = []
        self.filtedRange = []
        self.curSpeed = 0.0
        self.speedRecord = []
        self.rangeKF = KalmanFilter(dim_x=1, dim_z=1)
        self.rangeKF.x = np.array([0.])
        self.rangeKF.F = np.array([[1.]])
        self.rangeKF.H = np.array([[1.]])
        self.rangeKF.P *= 100.
        self.rangeKF.R = 0.1
        self.rangeKF.Q = 0.001
        self.rangeKF.initialized = False
        self.L = 10
        self.last_z  = 0
        self.zMeasPairs = deque(maxlen=80)
        self.initialized()
    

    def vel_from_dis(self, l_0, l_1, l_2, t0, t1, t2):
        t_1 = t1 - t0
        t_2 = t2 - t1
        if (l_2 - l_1) * (l_1 - l_0) > 0:
            d = abs(l_2 * l_2 - l_1 * l_1 - (l_1 * l_1 - l_0 * l_0) * t_2 / t_1)
            tl = t_1 * t_2 + t_2 * t_2
            return math.sqrt(d / tl)
        else:
            return False


    def range_key_pairs_maintaince(self, range, time):
        if abs(range-self.lastPoint) > self.distanceThreshold:
            self.lastPoint = range
            self.keyMeasPairs.append([range, time])
            return True
        else:
            return False

    def swfilter_timedata(self, sequence, radius, iter):
        sequence = np.array(sequence)
        box_filter = np.ones((radius))
        ones_1 = np.ones((radius-1,radius))
        triu_filter = np.triu(ones_1)
        filter = triu_filter
        d = np.zeros(radius-1)
        for i in range(radius-1):
            d[i] = np.sqrt((np.dot(filter[i], sequence) / np.sum(filter[i]) - sequence[-1])**2)
        index = np.argmin(d)
        box_filter = box_filter * triu_filter[index]
        output = np.dot(box_filter, sequence) / np.sum(box_filter)
        return output
    
    def cal_z(self, range,time):
        z = (self.L**2+range[0]**2-range[1]**2)/(2*self.L)
        # print(range,z)
        if abs(z-self.last_z) > self.distanceThreshold:
            self.last_z = z
            self.zMeasPairs.append([z, time])
            return True
        else:
            return False
    def vz_from_z(self, z0,z1,t0,t1):
        vz = (z1-z0)/(t1-t0)
        return vz
        

    def estimate_speed(self, range0, range1, time, interval):
        fdragne = self.filter_range(range0, range1)
        self.speedWindowSize = 5+ 0.1*fdragne[0]        
        if self.cal_z(fdragne, time) and len(self.zMeasPairs) >= 2*interval:
            tempresult = self.vz_from_z(self.zMeasPairs[-2*interval][0], 
                                           self.zMeasPairs[-1][0], self.zMeasPairs[-2*interval][1],
                                            self.zMeasPairs[-1][1])

            self.speedWindow.append(tempresult)
            # print(len(self.speedWindow))
            if len(self.speedWindow)>(self.speedWindowSize-1):
                self.curSpeed = np.median(self.speedWindow)  # Estimation of this linear motion speed
                #self.curSpeed = self.swfilter_timedata(self.speedWindow, len(self.speedWindow), 1)
                #self.speedWindow[-1] = self.curSpeed
                self.speedRecord.append(self.curSpeed)
                self.speedUpdateTime.append(time)
                self.validSpeedUpdated = True
                while(len(self.speedWindow)>(self.speedWindowSize-1)):
                    self.speedWindow.pop(0)

        else:
            self.validSpeedUpdated = False


    def filter_range(self, range0, range1):
        if self.rangeKF.initialized == False:
            self.rangeKF.x = np.array([range0, range1])
            self.filtedRange.append(np.array([range0, range1]))
            self.rangeKF.initialized = True
            # print("kf initialized")
            # print(self.rangeKF.F.shape)
        else:
            self.rangeKF.predict()
            self.rangeKF.update(np.array([range0, range1]))
            self.filtedRange.append(self.rangeKF.x)
        return self.filtedRange[-1]


    def get_vel(self):
        return self.curSpeed

    def initialized(self, ):
        self.rangeKF = KalmanFilter(dim_x=2, dim_z=2)
        self.rangeKF.x = np.array([0., 0.])
        self.rangeKF.F = np.array([[1., 0.],[0., 1.]])
        self.rangeKF.H = np.array([[1., 0.],[0., 1.]])
        self.rangeKF.P *= 100.
        self.rangeKF.R = 0.1
        self.rangeKF.Q = 0.001
        self.rangeKF.initialized = False
