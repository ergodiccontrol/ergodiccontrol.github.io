import numpy as np
from js import Path2D


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.x0 = [.2, .3]  # Initial position
param.nbData = 5000  # Number of datapoints
param.nbFct = 10     # Number of basis functions along x and y
param.nbVar = 2      # Dimension of the space
param.nbGaussian = 2 # Number of Gaussians to represent the spatial distribution
param.dt = 1e-2      # Time step
param.u_max = 1e1    # Maximum speed allowed
param.xlim = [0, 1]  # Domain limit

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
x = None
t = None
wt = None
r_x = None

hist = None
xbins = None
ybins = None


# Reset function
# ===============================
def reset(reset_state=True):
    global x, t, wt, r_x, param, controls, paths

    # Retrieve the initial state defined by the user
    if reset_state:
        (x0, Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations) = initialState()

        param.x0 = np.array(x0)
        if (len(param.x0.shape) != 1) or (param.x0.shape[0] != 2):
            print("Error: 'x0' must be a vector of size 2")
            return

        param.x0 = np.clip(param.x0, 0.01, 0.99) # x0 should be within [0,1]

        if not create_gaussians(Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations):
            return

    # Compute the desired spatial distribution
    param.rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVar, param.nbFct, param.nbFct))
    KX[0,:,:], KX[1,:,:] = np.meshgrid(param.rg, param.rg)

    # Weighting vector (Eq.(16))
    sp = (param.nbVar + 1) / 2 # Sobolev norm parameter
    param.Lambda = np.array(KX[0,:].flatten()**2 + KX[1,:].flatten()**2 + 1).T**(-sp)
    param.op = hadamard_matrix(2**(param.nbVar-1))
    param.op = np.array(param.op)
    param.kk = KX.reshape(param.nbVar, param.nbFct**2) * param.omega

    alpha = np.ones(param.nbGaussian) / param.nbGaussian # mixing coeffs. Priors
    param.w_hat = fourier(alpha)

    param.xx, param.yy = np.meshgrid(np.arange(1, param.nbFct+1), np.arange(1, param.nbFct+1))

    # Reset the variables
    x = param.x0.copy()
    t = 0
    wt = np.zeros(param.nbFct**param.nbVar)
    r_x = np.array((0, 2))

    controls = create_gaussian_controls(param)
    paths = [ Path2D.new() ]
    hist = None


# Update function
# ===============================
def update():
    global x, t, wt, r_x, param, paths, hist, xbins, ybins

    # We only compute 'nbData' values
    if t >= param.nbData:
        return

    t += 1
    x_prev = x.copy()

    # Retrieve the command
    u, wt = controlCommand(x, t, wt, param)

    # Update of the position
    x += u * param.dt

    # Update the path (for rendering)
    path = paths[0]
    path.moveTo(x_prev[0], x_prev[1])
    path.lineTo(x[0], x[1])

    # Recompute the histogram (for rendering)
    r_x = np.vstack((r_x, x))
    hist, xbins, ybins = np.histogram2d(r_x[:, 1], r_x[:, 0], bins=20, range=np.array([param.xlim, param.xlim]))
