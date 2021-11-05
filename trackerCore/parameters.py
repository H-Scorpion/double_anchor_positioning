
# Parameters for simulation
IsSimulation = True
UsingLastTimeData = True

# Set covariance of the EKF
covS_X = 0.01  # covariance of state
covS_Y = 0.01
covS_Z = 0.01
covS_Ori = 0.01
covS_Pitch = 0.01
covS_LVel = 0.01

covM_Range = 0.2  # Covariance of range measurement
covM_Ori = 0.01
covM_Pitch = 0.01

# Set initial state of the EKF
initialState = [10., 0., 0., 2., 0., 0.0] 

seed = 886
