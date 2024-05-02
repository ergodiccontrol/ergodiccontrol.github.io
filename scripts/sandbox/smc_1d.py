import numpy as np
from js import Path2D


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.x0 = .1        # Initial position
param.nbData = 2000  # Number of datapoints
param.nbFct = 10     # Number of basis functions
param.nbGaussian = 1 # Number of Gaussians to represent the spatial distribution
param.dt = 1e-2      # Time step
param.u_max = 1e-0   # Maximum speed allowed
param.xlim = [0, 1]  # Domain limit

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
x = 0
t = 0
r_x = []
wt = np.zeros((param.nbFct, 1))

path_1d = None
path_2d = None
hist = None
bins = None


# Reset function
# ===============================
def reset():
    global x, r_x, t, wt, param, path_1d, path_2d

    # Retrieve the initial state defined by the user
    (param.x0, Mu, Sigma) = initialState()

    param.Mu = np.array(Mu)
    param.Sigma = np.array(Sigma)

    if len(param.Mu.shape) != 1:
        print("Error: 'Mu' must be a vector of 'N' values, with 'N' the number of gaussians")
        return

    param.nbGaussian = param.Mu.shape[0]

    if (len(param.Sigma.shape) != 1) or (param.Sigma.shape[0] != param.nbGaussian):
        print(f"Error: 'Sigma' must be a vector of {param.nbGaussian} values")
        return

    # Compute Fourier series coefficients w_hat of desired spatial distribution
    param.Priors = np.ones(param.nbGaussian) / param.nbGaussian # Mixing coefficients

    rg = np.arange(param.nbFct, dtype=float).reshape((param.nbFct, 1))
    param.kk = rg * param.omega
    param.Lambda = (rg**2 + 1) ** -1 # Weighting vector (Eq.(15)

    # Explicit description of w_hat by exploiting the Fourier transform
    # properties of Gaussians (optimized version by exploiting symmetries)
    param.w_hat = np.zeros((param.nbFct, 1))
    for j in range(param.nbGaussian):
        param.w_hat = param.w_hat + param.Priors[j] * np.cos(param.kk * Mu[j]) * np.exp(-.5 * param.kk**2 * param.Sigma[j]) # Eq.(22)

    param.w_hat = param.w_hat / param.L

    # Reset the variables
    x = param.x0
    t = 0
    r_x = []
    wt = np.zeros((param.nbFct, 1))

    path_1d = Path2D.new()
    path_2d = Path2D.new()


# Update function
# ===============================
def update():
    global x, r_x, t, wt, param, path_1d, path_2d, hist, bins

    # We only compute 'nbData' values
    if t >= param.nbData:
        return

    t += 1
    x_prev = x

    # Retrieve the command
    u, wt = controlCommand(x, t, wt, param)
    if isinstance(u, np.ndarray):
        u = u.flatten()[0]

    # Ensure that we don't go out of limits
    next_x = x + u * param.dt
    if (next_x < param.xlim[0]) or (next_x > param.xlim[1]):
        u = -u

    # Update of the position
    x += u * param.dt

    # Update the paths (for rendering)
    path_1d.moveTo(x_prev, 0.0)
    path_1d.lineTo(x, 0.0)

    path_2d.moveTo(x_prev, (t-1) / param.nbData * PATH_2D_HEIGHT)
    path_2d.lineTo(x, t / param.nbData * PATH_2D_HEIGHT)

    # Recompute the histogram (for rendering)
    r_x.append(x)
    hist, bins = np.histogram(r_x, bins=20, range=param.xlim)
