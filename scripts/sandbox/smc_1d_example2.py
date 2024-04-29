def initial_state():
	"""The initial state, customize it to your liking"""

	# Initial position
	x0 = 0.1

	# Gaussian centers (as many as you want, one per row)
	Mu = np.array([
		0.3,
	])

	# Gaussian variances (one per row)
	Sigma = np.array([
		0.01,
	])

	return (x0, Mu, Sigma)


def command(x, t, wt, param):
	# Fourier basis functions and derivatives for each dimension
	# (only cosine part on [0,L/2] is computed since the signal
	# is even and real by construction)
	phi = np.cos(x * param.kk)	# Eq.(18)

	# Gradient of basis functions
	dphi = -np.sin(x * param.kk) * param.kk

	# wt/t are the Fourier series coefficients along trajectory (Eq.(17))
	wt = wt + phi / param.L

	# Controller with constrained velocity norm
	u = -dphi.T @ (param.Lambda * (wt / (t + 1) - param.w_hat))	 # Eq.(24)
	u = u * param.u_max / (np.linalg.norm(u) + 1e-2)  # Velocity command

	return u, wt


reset()
