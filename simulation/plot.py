import os, sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'trackerCore'))

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

savePath = sys.path[0] + "/result_cache/"

from simulationData import readFromFile
from simulationData import sim
import plotly.graph_objs as go
import plotly


def plot_trajectory(sim, ref_ekf, my_ekf, anchor_ekf,name='',anchor_x=[], anchor_y=[], anchor_z=[]):
    posXRef, posYRef, posZRef, orientationRef, pitchRef, linVelRef, posX, posY, posZ, orientation, pitch, linVel = ([] for _ in range(12))
    anchor_posX, anchor_posY, anchor_posZ, anchor_orientation, anchor_pitch, anchor_linVel = ([] for _ in range(6))
    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        posZRef.append(ref_ekf.ekf.recordState[idx][2])
        orientationRef.append(ref_ekf.ekf.recordState[idx][3])
        pitchRef.append(ref_ekf.ekf.recordState[idx][4])
        linVelRef.append(ref_ekf.ekf.recordState[idx][5])

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        posZ.append(my_ekf.ekf.recordState[idx][2])
        orientation.append(my_ekf.ekf.recordState[idx][3])
        pitch.append(my_ekf.ekf.recordState[idx][4])
        linVel.append(my_ekf.ekf.recordState[idx][5])

    for idx in range(len(anchor_ekf.ekf.recordState)):
        anchor_posX.append(anchor_ekf.ekf.recordState[idx][0])
        anchor_posY.append(anchor_ekf.ekf.recordState[idx][1])
        anchor_posZ.append(anchor_ekf.ekf.recordState[idx][2])
        anchor_orientation.append(anchor_ekf.ekf.recordState[idx][3])
        anchor_pitch.append(anchor_ekf.ekf.recordState[idx][4])
        anchor_linVel.append(anchor_ekf.ekf.recordState[idx][5])

    fig = plt.figure(figsize=(7, 7))
    #colors = plt.cm.rainbow(np.linspace(0, 1, len(posX)))
    ax = fig.gca(projection='3d')
    ax.force_zorder = True
    ax.scatter(sim.x, sim.y, sim.z, s=5, c='black', label="Ground Truth",zorder=1)
    ax.scatter(posXRef, posYRef, posZRef, s=2, c='red', label="EKF without RVE",zorder=2)
    ax.scatter(posX, posY, posZ, s=2, c='green', label="Proposed Method",zorder=3)
    ax.scatter(anchor_posX, anchor_posY, anchor_posZ, s=2, c='yellow', label="Anchor Method",zorder=4)
    ax.scatter(0.,0.,0., marker = '^', s=100, c='black', label="Anchor" ,zorder=5)
    blue_patch = ax.scatter(anchor_x,anchor_y,anchor_z, marker = '^', s=100, c='b', label="new Anchor", zorder=6)



    black_patch = mpatches.Patch(color='black', label="Ground truth")
    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    yellow_patch = mpatches.Patch(color='yellow', label="With speed estimator and hand over")
    ax.legend(handles=[black_patch, red_patch, green_patch, yellow_patch, blue_patch])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    #plt.axis('equal')
    #plt.show()
    plt.savefig(savePath+"%ssim_result_trajectory_anchor.svg" % name)
    plt.savefig(savePath+"%ssim_result_trajectory_anchor.png" % name, bbox_inches='tight', dpi=600)


    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Orientation/ aVel/ lVel')
    #ax1.plot(orientationRef, label="Ref Yaw")
    ax1.plot(orientation, label=" Yaw ")
    #plt.show()

    #print('ax.azim {}'.format(ax.azim))
    #print('ax.elev {}'.format(ax.elev))
    fig.savefig(savePath+"%ssim_result_info_anchor.svg" % name)
    plt.savefig(savePath+"%ssim_result_info_anchor.png" % name, bbox_inches='tight', dpi=600)



def plot_anchor_info(sim, ref_ekf, my_ekf, anchor_ekf,name='',anchor_x=[], anchor_y=[], record_switch_hand_step=[]):

    posXRef, posYRef, posZRef, orientationRef, pitchRef, linVelRef, posX, posY, posZ, orientation, pitch, linVel = ([] for _ in range(12))
    anchor_posX, anchor_posY, anchor_posZ, anchor_orientation, anchor_pitch, anchor_linVel = ([] for _ in range(6))
    filtedRange = []
    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        posZRef.append(ref_ekf.ekf.recordState[idx][2])
        orientationRef.append(ref_ekf.ekf.recordState[idx][3])
        pitchRef.append(ref_ekf.ekf.recordState[idx][4])
        linVelRef.append(ref_ekf.ekf.recordState[idx][5])

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        posZ.append(my_ekf.ekf.recordState[idx][2])
        orientation.append(my_ekf.ekf.recordState[idx][3])
        pitch.append(my_ekf.ekf.recordState[idx][4])
        linVel.append(my_ekf.ekf.recordState[idx][5])

    for idx in range(len(anchor_ekf.ekf.recordState)):
        anchor_posX.append(anchor_ekf.ekf.recordState[idx][0])
        anchor_posY.append(anchor_ekf.ekf.recordState[idx][1])
        anchor_posZ.append(anchor_ekf.ekf.recordState[idx][2])
        anchor_orientation.append(anchor_ekf.ekf.recordState[idx][3])
        anchor_pitch.append(anchor_ekf.ekf.recordState[idx][4])
        anchor_linVel.append(anchor_ekf.ekf.recordState[idx][5])
    
    

    for idx in range(len(anchor_ekf.ekf.recordState)):
        filtedRange.append(np.linalg.norm([sim.x[idx],sim.y[idx],sim.z[idx]]))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Linear Velocity(m/s)')
    #ax1.plot(orientation, label="orientation")
    ax1.plot(sim.lVel,c='black', label=" GT Linear Vel ")
    ax1.plot(linVelRef,c='red', label=" Vanilla EKF Linear Vel. ")
    ax1.plot(linVel, c='green',label=" Proposed Method Linear Vel. ")
    ax1.plot(anchor_linVel, c='Yellow',label=" Anchor Method Linear Vel. ")

    #for idx in range(len(record_switch_hand_step)):
    #    plt.vlines(record_switch_hand_step[idx],0,10,color="green")
    black_patch = mpatches.Patch(color='black', label="Ground truth")
    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    yellow_patch = mpatches.Patch(color='yellow', label="Anchor with speed estimator")
    ax1.legend(loc='upper left', handles=[black_patch, red_patch, green_patch, yellow_patch])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #ax2.plot(uwbInput, color='c', label=" Range Measurement ")
    ax2.plot(filtedRange, color='goldenrod', label=" Filtered range ")
    ax2.set_ylabel('Simulated UWB range(m)')
    glodenrod_patch = mpatches.Patch(color='goldenrod', label="Range")
    ax2.legend(loc='upper right', handles=[glodenrod_patch])

    fig.savefig(savePath+"%ssim_result_info.svg" % name)
    plt.savefig(savePath+"%ssim_result_info.png" % name, bbox_inches='tight', dpi=600)


def plot_sim(sim, ref_ekf, my_ekf,name=''):

    posXRef, posYRef, orientationRef, linVelRef, posX, posY, orientation, linVel = ([] for _ in range(8))

    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        orientationRef.append(ref_ekf.ekf.recordState[idx][2])
        linVelRef.append(ref_ekf.ekf.recordState[idx][3])

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        orientation.append(my_ekf.ekf.recordState[idx][2])
        linVel.append(my_ekf.ekf.recordState[idx][3])

    plt.figure(figsize=(7, 7))
    #colors = plt.cm.rainbow(np.linspace(0, 1, len(posX)))
    plt.scatter(sim.x, sim.y, s=2, c='gray', label="Ground Truth")
    plt.scatter(posXRef, posYRef, s=2, c='red', label="EKF without RVE")
    plt.scatter(posX, posY, s=2, c='green', label="Proposed Method")
    plt.scatter(0.,0., marker = '^', s=100, c='b', label="Anchor" )


    gray_patch = mpatches.Patch(color='gray', label="Ground truth")
    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    plt.legend(handles=[gray_patch, red_patch, green_patch])
    plt.xlabel('(m)')
    plt.ylabel('(m)')

    plt.axis('equal')
    plt.savefig(savePath+"%ssim_result_trajectory.svg" % name)


    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Linear Velocity(m/s)')
    #ax1.plot(orientation, label="orientation")
    ax1.plot(sim.lVel,c='gray', label=" GT Linear Vel ")
    ax1.plot(linVelRef,c='red', label=" Vanilla EKF Linear Vel. ")
    ax1.plot(linVel, c='green',label=" Proposed Method Linear Vel. ")

    gray_patch = mpatches.Patch(color='gray', label="Ground truth")
    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    ax1.legend(loc='upper left', handles=[gray_patch, red_patch, green_patch])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #ax2.plot(uwbInput, color='c', label=" Range Measurement ")
    ax2.plot(my_ekf.speedEstimator.filtedRange, color='goldenrod', label=" Filtered range ")
    ax2.set_ylabel('Simulated UWB range(m)')
    glodenrod_patch = mpatches.Patch(color='goldenrod', label="Range")
    ax2.legend(loc='upper right', handles=[glodenrod_patch])

    fig.savefig(savePath+"%ssim_result_info.svg" % name)



def plot_sim_error(sim, ref_ekf, my_ekf, anchor_ekf,name='',):

    posXRef, posYRef, posZRef, orientationRef, pitchRef, linVelRef, posX, posY, posZ, orientation, pitch, linVel = ([] for _ in range(12))
    anchor_posX, anchor_posY, anchor_posZ, anchor_orientation, anchor_pitch, anchor_linVel = ([] for _ in range(6))
    error  = []
    errorRef = []
    errorAnchor = []

    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        posZRef.append(ref_ekf.ekf.recordState[idx][2])
        orientationRef.append(ref_ekf.ekf.recordState[idx][3])
        pitchRef.append(ref_ekf.ekf.recordState[idx][4])
        linVelRef.append(ref_ekf.ekf.recordState[idx][5])
        errorRef.append(np.linalg.norm([posXRef[idx] - sim.x[idx],posYRef[idx]-sim.y[idx], posZRef[idx]-sim.z[idx]]))

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        posZ.append(my_ekf.ekf.recordState[idx][2])
        orientation.append(my_ekf.ekf.recordState[idx][3])
        pitch.append(my_ekf.ekf.recordState[idx][4])
        linVel.append(my_ekf.ekf.recordState[idx][5])
        error.append(np.linalg.norm([posX[idx] - sim.x[idx],posY[idx]-sim.y[idx],posZ[idx]-sim.z[idx]]))

    for idx in range(len(anchor_ekf.ekf.recordState)):
        anchor_posX.append(anchor_ekf.ekf.recordState[idx][0])
        anchor_posY.append(anchor_ekf.ekf.recordState[idx][1])
        anchor_posZ.append(anchor_ekf.ekf.recordState[idx][2])
        anchor_orientation.append(anchor_ekf.ekf.recordState[idx][3])
        anchor_pitch.append(anchor_ekf.ekf.recordState[idx][4])
        anchor_linVel.append(anchor_ekf.ekf.recordState[idx][5])
        errorAnchor.append(np.linalg.norm([anchor_posX[idx] - sim.x[idx], anchor_posY[idx]-sim.y[idx],anchor_posZ[idx]-sim.z[idx]]))

    plt.figure(figsize=(10 , 5))
    plt.plot(errorRef,color='red',  label = ' Vanilla EKF')
    plt.plot(error, color='green', label = 'Paper Method')
    plt.plot(errorAnchor, color='yellow', label = 'Anchor Method')

    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    yellow_patch = mpatches.Patch(color='yellow', label="Handover with speed estimator")
    plt.legend(loc='upper left', handles=[red_patch, green_patch, yellow_patch])
    plt.xlabel('Time (steps)')
    plt.ylabel('RMSE(m)')

    print("RMSE With Speed Estimator", np.mean(error), "; Without", np.mean(errorRef), 'Anchor With Speed Estimator', np.mean(errorAnchor))
    plt.savefig(savePath+"%ssim_RMS.svg" % name)
    plt.savefig(savePath+"%ssim_RMS.png" % name, bbox_inches='tight', dpi=600)


def vel_from_dis( l_0, l_1, l_2, t0, t1, t2):
    t_1 = t1 - t0
    t_2 = t2 - t1
    d = abs(l_2 * l_2 - l_1 * l_1 - (l_1 * l_1 - l_0 * l_0) * t_2 / t_1)
    tl = t_1 * t_2 + t_2 * t_2
    return sqrt(d / tl)


def brute_vel_estimate(range_measurement, bVel, interval=500):
    interval = 50
    dt = 0.005
    for i in range( 2*interval, len(range_measurement)):
        t0 = i * dt
        t1 = (i + interval) * dt
        t2 = (i + interval * 2) * dt
        bVel.append(
            vel_from_dis(range_measurement[i-2*interval], range_measurement[i - interval], range_measurement[i], t0, t1, t2))

def plot_trajectory_useplotly(sim, ref_ekf, my_ekf, anchor_ekf,name='',anchor_x=[], anchor_y=[], anchor_z=[]):
    posXRef, posYRef, posZRef, orientationRef, pitchRef, linVelRef, posX, posY, posZ, orientation, pitch, linVel = ([] for _ in range(12))
    anchor_posX, anchor_posY, anchor_posZ, anchor_orientation, anchor_pitch, anchor_linVel = ([] for _ in range(6))
    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        posZRef.append(ref_ekf.ekf.recordState[idx][2])
        orientationRef.append(ref_ekf.ekf.recordState[idx][3])
        pitchRef.append(ref_ekf.ekf.recordState[idx][4])
        linVelRef.append(ref_ekf.ekf.recordState[idx][5])

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        posZ.append(my_ekf.ekf.recordState[idx][2])
        orientation.append(my_ekf.ekf.recordState[idx][3])
        pitch.append(my_ekf.ekf.recordState[idx][4])
        linVel.append(my_ekf.ekf.recordState[idx][5])

    for idx in range(len(anchor_ekf.ekf.recordState)):
        anchor_posX.append(anchor_ekf.ekf.recordState[idx][0])
        anchor_posY.append(anchor_ekf.ekf.recordState[idx][1])
        anchor_posZ.append(anchor_ekf.ekf.recordState[idx][2])
        anchor_orientation.append(anchor_ekf.ekf.recordState[idx][3])
        anchor_pitch.append(anchor_ekf.ekf.recordState[idx][4])
        anchor_linVel.append(anchor_ekf.ekf.recordState[idx][5])

    
    Ground_Truth = go.Scatter3d(x=sim.x, y=sim.y, z=sim.z, mode='markers', marker=dict(
                                                                size=12,
                                                                line=dict(color='gray',width=0.5),
                                                                opacity=0))
    without_v = go.Scatter3d(x=posXRef, y=posYRef, z=posZRef, mode='markers', marker=dict(
                                                                size=12,
                                                                line=dict(color='red',width=0.2),
                                                                opacity=0))
    with_v = go.Scatter3d(x=posX, y=posY, z=posZ, mode='markers', marker=dict(
                                                                size=12,
                                                                line=dict(color='green',width=0.2),
                                                                opacity=0))
    anchor_with_v = go.Scatter3d(x=anchor_posX, y=anchor_posY, z=anchor_posZ, mode='markers', marker=dict(
                                                                size=12,
                                                                line=dict(color='yellow',width=0.2),
                                                                opacity=0))
    anchor = go.Scatter3d(x=anchor_x,y=anchor_y,z=anchor_z, mode='markers', marker=dict(
                                                                size=12,
                                                                line=dict(color='blue',width=0.2),
                                                                opacity=0))
    

    data = [Ground_Truth, without_v, with_v, anchor_with_v, anchor, ]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=savePath+"%ssim_result_info.html" % name, auto_open=True)


if __name__ == '__main__':
    simData = readFromFile()