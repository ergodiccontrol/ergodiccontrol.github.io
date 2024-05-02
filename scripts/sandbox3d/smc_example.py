def initialState():
    """The initial state, customize it to your liking"""

    # Initial robot state
    x = [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 1.0]

    # Gaussian centers (as many as you want, one per row)
    Mu = np.array([
        [.8, .6, .5],
        [.8, .5, .6],
    ])

    # Gaussian covariances, defined by a direction vector, a scale and a regularization factor
    # direction vectors (one per row)
    Sigma_vectors = np.array([
        [.1, .1, 1.],
        [.1, 1., .1],
    ])
    # scales
    Sigma_scales = np.array([
        3E-1,
        2E-1,
    ])
    # regularization factors
    Sigma_regularizations = np.array([
        1E-3,
        1E-3,
    ])

    return (x, Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations)


def ergodicControl(x, t, wt, param):
    # Depends on the current position only here, outputs: dphi, phix, phiy, phiz
    ang = x[:, np.newaxis] * param.rg * param.omega
    phi1 = np.cos(ang) #Eq.(18)
    dphi1 = -np.sin(ang) * np.tile(param.rg, (param.nbVar, 1)) * param.omega
    phix = phi1[0, param.xx-1].flatten()
    phiy = phi1[1, param.yy-1].flatten()
    phiz = phi1[2, param.zz-1].flatten()
    dphix = dphi1[0, param.xx-1].flatten()
    dphiy = dphi1[1, param.yy-1].flatten()
    dphiz = dphi1[2, param.zz-1].flatten()
    dphi = np.vstack([[dphix * phiy * phiz], [phix * dphiy * phiz], [phix * phiy * dphiz]])

    # Depends on wt, wt starts with zeros, then updates
    wt = wt + (phix * phiy * phiz).T / (param.L**param.nbVar)

    # Controller with constrained velocity norm
    u = -dphi @ (param.Lambda * (wt/(t+1) - param.w_hat)) # Eq.(24)
    u = u * param.u_max / (np.linalg.norm(u) + 0.1) # Velocity command

    # Update of the position
    x = x + u * param.dt

    return x, wt


def controlCommand(x, t, wt, param):
    J = Jkin(x)
    f = fkin(x)
    e, wt = ergodicControl(f[:3], t, wt, param)
    u = np.linalg.pinv(J[:3,:]) @ (e - f[:3])
    return u, wt


reset()
