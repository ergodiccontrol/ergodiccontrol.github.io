def initialState():
    """The initial state, customize it to your liking"""

    # Initial position
    x0 = np.array([.2, .3])

    # Gaussian centers (as many as you want, one per row)
    Mu = np.array([
        [.3, .7],
        [.7, .3],
    ])

    # Gaussian covariances, defined by a direction vector, a scale and a regularization factor
    # direction vectors (one per row)
    Sigma_vectors = np.array([
        [.3, .1],
        [.1, .2],
    ])
    # scales
    Sigma_scales = np.array([
        1E-1,
        5E-1,
    ])
    # regularization factors
    Sigma_regularizations = np.array([
        1E-3,
        3E-3,
    ])

    return (x0, Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations)


def controlCommand(x, t, wt, param):
    # Depends on the current position only here, outputs: dphi, phix, phiy
    ang = x[:, np.newaxis] * param.rg * param.omega
    phi1 = np.cos(ang) #Eq.(18)
    dphi1 = -np.sin(ang) * np.tile(param.rg, (param.nbVar, 1)) * param.omega
    phix = phi1[0, param.xx-1].flatten()
    phiy = phi1[1, param.yy-1].flatten()
    dphix = dphi1[0, param.xx-1].flatten()
    dphiy = dphi1[1, param.yy-1].flatten()
    dphi = np.vstack([[dphix * phiy], [phix * dphiy]])

    # Depends on wt, wt starts with zeros, then updates
    wt = wt + (phix * phiy).T / (param.L**param.nbVar)

    # Controller with constrained velocity norm
    u = -dphi @ (param.Lambda * (wt/(t+1) - param.w_hat)) # Eq.(24)
    u = u * param.u_max / (np.linalg.norm(u) + 1e-1) # Velocity command

    return u, wt


reset()
