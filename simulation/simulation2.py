import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'trackerCore'))

from tracker import tracker
from simulationData import sim, readFromFile, saveToFile
from plot import plot_sim, plot_sim_error, plot_trajectory, plot_anchor_info,plot_trajectory_useplotly
from parameters import *
import numpy as np
import random
if __name__ == '__main__':
    np.random.seed(seed)
    random.seed(seed)
    #Using last time data or generate new simulation data
    if UsingLastTimeData:
        simData = readFromFile()
    else:
        simData = sim()
        simData.generate_sim_root()
        simData.generate_sim()
        saveToFile(simData)

    # Create two tracker for compare
    myTrackerExp = tracker()
    refMyTracker = tracker()
    AnchoTracker = tracker()

    # Set tracker mode to simulation, disable the speedEstimator of the reference tracker
    AnchoTracker.setup_mode(IsSimulation) #red
    myTrackerExp.setup_mode(IsSimulation) #green
    speedEstimatorSwitch = False
    refMyTracker.setup_mode(IsSimulation,speedEstimatorSwitch) #yello


    # Set covariance for the EKF
    refMyTracker.ekf.set_covs(covS_X, covS_Y, covS_Z, covS_Ori, covS_Pitch, covS_LVel, covM_Range, covM_Ori, covM_Pitch)
    myTrackerExp.ekf.set_covs(covS_X, covS_Y, covS_Z, covS_Ori, covS_Pitch, covS_LVel, covM_Range, covM_Ori, covM_Pitch)
    AnchoTracker.ekf.set_covs(covS_X, covS_Y, covS_Z, covS_Ori, covS_Pitch, covS_LVel, covM_Range, covM_Ori, covM_Pitch)

    # Set initial state for the EKF
    refMyTracker.ekf.set_initial_state(initialState)
    myTrackerExp.ekf.set_initial_state(initialState)
    AnchoTracker.ekf.set_initial_state(initialState)

    # Choose measurement input
    uwbInput = simData.uwbNoisy
    yawInput = simData.yawNoisy
    pitchInput = simData.pitchNoisy
    timeInput = simData.timestamp

    print("Start the Tracker")
    for step in range(len(uwbInput)):
        measurement = [uwbInput[step], yawInput[step], pitchInput[step], timeInput[step]]
        refMyTracker.step(measurement)
        myTrackerExp.step(measurement)

    # Plot the result to result_cache folder
    #plot_sim(simData,refMyTracker,myTrackerExp)
    #plot_sim_error(simData,refMyTracker,myTrackerExp)

    #new anchor tracking
    anchor_x = 0
    anchor_y = 0
    anchor_z = 0
    record_anchor_x = []
    record_anchor_y = []
    record_anchor_z = []
    record_switch_hand_step = []
    for step in range(len(uwbInput)):
        uwbInput_ = np.linalg.norm([simData.x[step]-anchor_x, simData.y[step]-anchor_y, simData.z[step]-anchor_z]) + np.random.normal(0, 0.2) #new range and noise\
        if uwbInput_ > 15 :#and len(AnchoTracker.ekf.recordState) > 0:
            anchor_x = AnchoTracker.ekf.recordState[-1][0]
            anchor_y = AnchoTracker.ekf.recordState[-1][1]
            anchor_z = AnchoTracker.ekf.recordState[-1][2]
            record_anchor_x.append(anchor_x)
            record_anchor_y.append(anchor_y)
            record_anchor_z.append(anchor_z)
            record_switch_hand_step.append(step)


            uwbInput_ = np.linalg.norm([simData.x[step]-anchor_x, simData.y[step]-anchor_y, simData.z[step]-anchor_z]) + np.random.normal(0, 0.2)
            initialState = [0,0,0, AnchoTracker.ekf.x[3], AnchoTracker.ekf.x[4], AnchoTracker.ekf.x[5]] #AnchoTracker.ekf.x[0], AnchoTracker.ekf.x[1]
            
            recordState = AnchoTracker.ekf.recordState
            recordP = AnchoTracker.ekf.recordP
            recordResidual = AnchoTracker.ekf.recordResidual
            recordspeedEstimator = AnchoTracker.speedEstimator

            AnchoTracker = tracker()
            AnchoTracker.setup_mode(IsSimulation)
            AnchoTracker.ekf.set_covs(covS_X, covS_Y, covS_Z, covS_Ori, covS_Pitch, covS_LVel, covM_Range, covM_Ori, covM_Pitch)
            AnchoTracker.ekf.anchor = (anchor_x, anchor_y, anchor_z)
            AnchoTracker.ekf.set_initial_state(initialState)
            AnchoTracker.ekf.P = recordP[-1]

            AnchoTracker.speedEstimator = recordspeedEstimator
            AnchoTracker.speedEstimator.initialized()

            AnchoTracker.ekf.recordState.extend(recordState)
            AnchoTracker.ekf.recordP.extend(recordP)
            AnchoTracker.ekf.recordResidual.extend(recordResidual)


        measurement = [uwbInput_, yawInput[step], pitchInput[step], timeInput[step]]
        AnchoTracker.step(measurement)
    print(len(record_anchor_x))
    # Plot the result to result_cache folder

    plot_trajectory(simData, refMyTracker,myTrackerExp, AnchoTracker,name='new_method_',
                                anchor_x=record_anchor_x, anchor_y=record_anchor_y, anchor_z=record_anchor_z)
    #plot_trajectory_useplotly(simData, refMyTracker,myTrackerExp, AnchoTracker,name='new_method_',
    #                            anchor_x=record_anchor_x, anchor_y=record_anchor_y, anchor_z=record_anchor_z)
    plot_anchor_info(simData, refMyTracker,myTrackerExp, AnchoTracker,name='new_method_', record_switch_hand_step=record_switch_hand_step)
    plot_sim_error(simData,refMyTracker,myTrackerExp, AnchoTracker,'new_method_')
