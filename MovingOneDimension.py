# Setting vectors and matrices for this problem
dt = 1.0
F = np.array([[1, dt], [0, 1]])
G = np.array([[0, 0]]).reshape(2,1) # considering constant velocity, then acceleration is 0
H = np.array([1, 0]).reshape(1, 2)
Q = np.array([[0.0, 0.0], [0.0, 0.0]]).reshape(2,2) # No process noise
R = np.array([25]).reshape(1, 1) # radar noise = 5 m, measurement covariance noise R = 25 m2
P0 = np.array([[30, 0], [0, 10]]) # initial covariance error matrix P0
x0 = np.array([10, 90]).reshape(2, 1) # initial estimated state vector x0, position = 10 m and velocity = 90 m/s

# Generating hypothetical values
x = np.linspace(0, 720, 10)

# Generating measurements with noise
measurements = x + np.random.normal(0, 5, 10) # mean = 0, sigma = 5, samples = 10

# Creating a Kalman Filter instance
kf = KalmanFilter(F = F, G = G, H = H, Q = Q, R = R, P = P0, x0 = x0)

# Creating lists to store generated values during Kalman Filter iterations
estimates = []
position_uncertainty = []
velocity_uncertainty = []

# For each measurement with noise (z), get the uncertainties (position and velocity) and estimate state (position)
for z in measurements:
    # Position estimate uncertainty
    position_uncertainty.append(kf.P[0,0])
    # Velocity estimate uncertainty
    velocity_uncertainty.append(kf.P[1,1])
    kf.predict()
    estimates.append(kf.update(z)[0])

# Plotting measurements and estimates
import matplotlib.pyplot as plt
plt.figure()
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 22})
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.plot(range(len(x)), x, label = 'True Value')
plt.plot(range(len(measurements)), measurements, label = 'Measurements')
plt.plot(range(len(estimates)), np.array(estimates), label = 'Kalman Filter Estimate')
plt.legend()
plt.show()

# Plotting position uncertainty
plt.figure()
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 22})
plt.xlabel('Time (s)')
plt.ylabel('Position Estimate Uncertainty (m)')
plt.plot(range(len(position_uncertainty)), position_uncertainty, label = 'Position Estimate Uncertainty')
plt.legend()
plt.show()

# Plotting velocity uncertainty
plt.figure()
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 22})
plt.xlabel('Time (s)')
plt.ylabel('Velocity Estimate Uncertainty (m/s)')
plt.plot(range(len(velocity_uncertainty)), velocity_uncertainty, label = 'Velocity Estimate Uncertainty')
plt.legend()
plt.show()
