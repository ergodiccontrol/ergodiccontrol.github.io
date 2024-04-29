import numpy as np
from js import Path2D


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.x0 = [.2, .3]  # Initial position
param.nbFct = 10     # Number of basis functions along x and y
param.nbVar = 2      # Dimension of the space
param.nbGaussian = 2 # Number of Gaussians to represent the spatial distribution
param.dt = 1e-2      # Time step
param.u_max = 1e1    # Maximum speed allowed
param.xlim = [0, 1]  # Domain limit

param.target = np.array([.5, .5])
param.target_radius = .03

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
x = None
t = None
wt = None
target_reached = False


# Reset function
# ===============================
def reset(reset_state=True):
	global x, t, wt, param, controls, paths, target_reached

	# Retrieve the initial state defined by the user
	if reset_state:
		(x0, Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations) = initial_state()

		param.x0 = np.array(x0)
		if (len(param.x0.shape) != 1) or (param.x0.shape[0] != 2):
			print("Error: 'x0' must be a vector of size 2")
			return

		param.x0 = np.clip(param.x0, 0.01, 0.99) # x0 should be within [0,1]

		if not create_gaussians(Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations):
			return

	# Sampling from GMM to define the target
	gaussian_id = np.random.choice(np.arange(0, param.nbGaussian))
	param.target = np.random.multivariate_normal(param.Mu[:,gaussian_id], 0.5*param.Sigma[:,:,gaussian_id])
	param.target = np.clip(param.target, 0.01, 0.99) # Target within [0,1]

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

	target_reached = False

	controls = create_gaussian_controls(param)
	paths = [ Path2D.new() ]


# Update function
# ===============================
def update():
	global x, t, wt, param, paths, target_reached

	# We stop computing once the target is reached
	if target_reached:
		return

	t += 1
	x_prev = x.copy()

	u, wt = command(x, t, wt, param)
	x += u * param.dt # Update of position

	path = paths[0]

	path.moveTo(x_prev[0], x_prev[1])
	path.lineTo(x[0], x[1])

	if (not target_reached) and \
		line_segment_and_circle_intersect(param.target[0], param.target[1], \
										  param.target_radius, x_prev[0], x_prev[1], x[0], x[1]):
		target_reached = True
