from KalmanFilter import KalmanFilter
from ExtendedKalmanFilter import ExtendedKalmanFilter
import numpy as np
import math
from math import sin, cos, pi
from collections import deque
from utility import uwbPassOutlierDetector, normalizeAngle
from speedEstimator import speedEstimator
from speedEstimator_vz import speedEstimator_vz



class tracker():
    def __init__(self,lineMovingThreshold=0.1,ElevMovingThreshold=0.1):
        self.speedEstimator0 = speedEstimator()
        self.speedEstimator1 = speedEstimator()
        self.speedEstimator_vz = speedEstimator_vz()
        self.ekf = customizedEKF()
        self.accData = deque(maxlen=100)
        self.staticThreshold = 0.01
        self.isSimulation = False
        self.lineMovingThreshold = lineMovingThreshold
        self.ElevMovingThreshold = ElevMovingThreshold
        self.curHeading = 0.
        self.curPitch = 0.
        self.measUpdateTime = 0.
        self.newMeasHeading = 0.
        self.newMeasRange0 = 0.
        self.newMeasRange1 = 0.
        self.newMeasPitch = 0.
        self.histRange = deque(maxlen=20)
        self.withSpeedEstimator = True
        self.anchor = (0,0)


    def setup_mode(self, simulation, speedEstimatorSwitch = True):
        self.isSimulation = simulation
        self.withSpeedEstimator = speedEstimatorSwitch
        if self.isSimulation:
            self.speedIinterval = 10
        else:
            self.speedIinterval = 20


    def check_static(self, acc):
        self.accData.append(acc)
        if np.std(self.accData) > self.staticThreshold:
            return False
        else:
            return True



    def get_valid_measurement_range(self, rangeMeas, time):
        if uwbPassOutlierDetector(self.histRange, rangeMeas):
            calibUWB = 1.11218892 * rangeMeas - 0.03747436  # TUM basketball calibration result
            self.newMeasRange = calibUWB
            self.rangeMeasUpdated = True
            self.measUpdateTime = time
            return True
        else:
            return False

    def update_sim_measurement_range0(self, rangeMeas, time):
        self.newMeasRange0 = rangeMeas
        self.rangeMeasUpdated = True
        self.measUpdateTime = time
    def update_sim_measurement_range1(self, rangeMeas, time):
        self.newMeasRange1 = rangeMeas
        self.rangeMeasUpdated = True
        self.measUpdateTime = time

    def update_heading_measurement(self, headingMeas, time):
        self.newMeasHeading = headingMeas
        self.measUpdateTime = time
        self.headingMeasUpdated = True

    def update_pitch_measurement(self, pitchMeas, time):
        self.newMeasPitch = pitchMeas
        self.measUpdateTime = time
        self.pitchMeasUpdated = True

    def linear_motion_check(self):
        if abs(self.newMeasHeading-self.curHeading) < self.lineMovingThreshold and abs(self.newMeasPitch-self.curPitch) < self.ElevMovingThreshold:
            return True
        else:
            self.curHeading = self.newMeasHeading
            self.curPitch = self.newMeasPitch
            self.speedEstimator0.keyMeasPairs.clear()
            self.speedEstimator1.keyMeasPairs.clear()
            return False

    def real_step(self,measurement):
        rangemeas = measurement[0]
        headmeas = measurement[1]
        timeStamp = measurement[2]
        acc = measurement[3]
        if self.get_valid_measurement_range(rangemeas, timeStamp):
            self.speedEstimator0.estimate_speed(measurement[0], timeStamp, self.speedIinterval)
            self.speedEstimator1.estimate_speed(measurement[0], timeStamp, self.speedIinterval)
        self.update_heading_measurement(headmeas,timeStamp)
        if self.check_static(acc):
            self.ekf.x[3] = 0
            self.speedEstimator0.keyMeasPairs = []
            self.speedEstimator1.keyMeasPairs = []
        else:
            self.ekf.ekfStep([self.newMeasRange, self.newMeasHeading])
            if self.withSpeedEstimator:
                if self.newMeasRange0 <= self.newMeasRange1:
                    if self.speedEstimator0.validSpeedUpdated:
                        estimatedVel = self.speedEstimator0.get_vel()
                        self.ekf.x[3] = 0.5*self.ekf.x[3] + 0.5*estimatedVel
                else:
                    if self.speedEstimator1.validSpeedUpdated:
                        estimatedVel = self.speedEstimator1.get_vel()
                        self.ekf.x[3] = 0.5*self.ekf.x[3] + 0.5*estimatedVel

            self.ekf.records()

    def sim_step(self, measurement):
        rangemeas0 = measurement[0]
        rangemeas1 = measurement[1]
        headmeas = measurement[2]
        timeStamp = measurement[3]
        self.update_sim_measurement_range0(rangemeas0,timeStamp)
        self.update_sim_measurement_range1(rangemeas1,timeStamp)
        self.speedEstimator0.estimate_speed(rangemeas0, timeStamp, self.speedIinterval)
        self.speedEstimator1.estimate_speed(rangemeas1, timeStamp, self.speedIinterval)
        self.speedEstimator_vz.estimate_speed(rangemeas0 ,rangemeas1, timeStamp, self.speedIinterval)
        self.update_heading_measurement(headmeas,timeStamp)
       
        # if self.newMeasRange0<=self.newMeasRange1:
        #     vz = self.speedEstimator_vz.get_vel()
        #     v = self.speedEstimator0.get_vel()
        #     print("vz:",vz)
        #     print("v:",v)
        #     self.newMeasPitch =  normalizeAngle(np.degrees(np.arcsin(vz/(v+1e-12))))
        # else:
        #     vz = self.speedEstimator_vz.get_vel()
        #     v = self.speedEstimator1.get_vel()
        #     print("vz:",vz)
        #     print("v:",v)
        #     self.newMeasPitch =  normalizeAngle(np.degrees(np.arcsin(vz/(v+1e-12))))
        
        # print(self.speedEstimator_vz.last_z)
        self.ekf.ekfStep([self.newMeasRange0,self.newMeasRange1, self.newMeasHeading,  self.speedEstimator_vz.last_z])
        if self.newMeasRange0<=self.newMeasRange1:
            if self.withSpeedEstimator and self.linear_motion_check() and self.speedEstimator0.validSpeedUpdated:
                estimatedVel = self.speedEstimator0.get_vel()
                self.ekf.x[5] = 0.5*self.ekf.x[5] + 0.5*estimatedVel
        else:
            if self.withSpeedEstimator and self.linear_motion_check() and self.speedEstimator1.validSpeedUpdated:
                estimatedVel = self.speedEstimator1.get_vel()
                self.ekf.x[5] = 0.5*self.ekf.x[5] + 0.5*estimatedVel
        self.ekf.records()


    def step(self, measurement):
        if self.isSimulation:
            self.sim_step(measurement)
        else:
            self.real_step(measurement)


class customizedEKF(ExtendedKalmanFilter):
    
    def __init__(self, dim_x=6, dim_z=4):
        super(customizedEKF, self).__init__(dim_x, dim_z)
        self.dt = 0.005
        self.recordState = []
        self.recordResidual = []
        self.recordP = []
        self.anchor0 = (0,0,0)
        self.anchor1 = (0,0,10)
 
    def set_covs(self, covS_X, covS_Y, covS_Z, covS_Ori, covS_Pitch, covS_LVel, covM_Range, covM_Ori, covM_Pitch):
        self.Q = np.array([ [covS_X, 0., 0., 0., 0., 0.],
                            [0., covS_Y, 0., 0., 0., 0.],
                            [0., 0., covS_Z, 0., 0., 0.],
                            [0., 0., 0, covS_Ori, 0., 0.],
                            [0., 0., 0, 0., covS_Pitch, 0.],
                            [0., 0., 0., 0., 0.,covS_LVel,]])

        self.R = np.array([[covM_Range, 0., 0., 0.,],
                                [0, covM_Range, 0., 0., ],
                                [0, 0., covM_Ori, 0.,],                               
                                [0, 0., 0., 0.01]])

    def set_initial_state(self, initialState):
        self.x = initialState

    def predict_x(self, state):
        x = self.x[0]
        y = self.x[1]
        z = self.x[2]
        o = self.x[3]
        theta = self.x[4]
        v = self.x[5]

        self.x[0] = x + v * cos(theta) * cos(o) * self.dt
        self.x[1] = y + v * cos(theta) * sin(o) * self.dt
        self.x[2] = z + v * sin(theta) * self.dt
        self.x[3] = o
        self.x[4] = theta
        self.x[5] = v

    def calF(self):
        x = self.x[0]
        y = self.x[1]
        z = self.x[2]
        o = self.x[3]
        theta = self.x[4]
        v = self.x[5]

        dx_dx = 1.
        dx_do = -sin(o) * cos(theta) * v * self.dt
        dx_dtheta = -sin(theta) * cos(o) * v * self.dt
        dx_dv = cos(o) * cos(theta) * self.dt
        

        dy_dy = 1.
        dy_do = cos(o) * cos(theta) * v * self.dt
        dy_dtheta =  -sin(theta) * sin(o) * v * self.dt
        dy_dv = sin(o) * cos(theta) * self.dt
       

        dz_dz = 1.
        dz_dtheta = cos(theta) * v * self.dt
        dz_dv = sin(theta) * self.dt


        self.F = np.array([[dx_dx, 0., 0., dx_do, dx_dtheta, dx_dv],
                           [0., dy_dy, 0., dy_do, dy_dtheta, dy_dv],
                           [0., 0., dz_dz, 0., dz_dtheta, dz_dv],
                           [0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 0., 1.],])

    def residualWithAng(self, zmeas, zpre):
        pi = math.pi
        resVal = np.subtract(zmeas, zpre)
        resVal[1] = normalizeAngle(resVal[1])
        resVal[2] = normalizeAngle(resVal[2])
        return resVal

    def H_Jac(self, s):
        xnorm = np.linalg.norm([self.x[0], self.x[1], self.x[2]])
        dr_dx = self.x[0] / xnorm
        dr_dy = self.x[1] / xnorm
        dr_dz = self.x[2] / xnorm
        xnorm1 =  np.linalg.norm([self.x[0], self.x[1], self.x[2]-10])
        dr1_dx = self.x[0]/xnorm1
        dr1_dy = self.x[1]/xnorm1
        dr1_dz = (self.x[2]-self.anchor1[2])/xnorm1
        Hjac = np.array([[dr_dx, dr_dy, dr_dz, 0,0,  0],
                         [dr1_dx, dr1_dy, dr1_dz, 0,0,  0],
                         [0., 0., 0., 1.,0, 0],
                         [0., 0., 1., 0.,0, 0]])
        return Hjac

    def H_state(self, s):
        xnorm0 = np.linalg.norm([self.x[0], self.x[1], self.x[2]])
        xnorm1 = np.linalg.norm([self.x[0], self.x[1], self.x[2]-self.anchor1[2]])
        h_x = np.array([xnorm0,xnorm1, self.x[3], self.x[2]])
        return h_x

    def ekfStep(self, measurement):
        self.calF()
        self.predict()
        self.x[3] = normalizeAngle(self.x[3])
        self.x[4] = normalizeAngle(self.x[4])
        self.update(measurement, self.H_Jac, self.H_state, residual=self.residualWithAng)

    def records(self):
        state = [self.x[0] + self.anchor0[0], self.x[1] + self.anchor0[1], self.x[2]+ self.anchor0[2], self.x[3], self.x[4],self.x[5]]
        self.recordState.append(state)
        self.recordP.append(self.P)
        self.recordResidual.append(self.y)
