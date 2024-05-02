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


def ergodicControl(x, agent, goal_density, coverage_density, heat, coverage_block, param):
    # find agent pos on the grid as integer indices
    p = x * param.nbRes
    adjusted_position = p / param.dx
    col, row, depth = adjusted_position.astype(int)

    # each agent has a kernel around it,
    # clamp the kernel by the grid boundaries
    row_indices, row_start_kernel, num_kernel_rows = clamp_kernel_1d(
        row, 0, param.height, param.kernel_size
    )
    col_indices, col_start_kernel, num_kernel_cols = clamp_kernel_1d(
        col, 0, param.width, param.kernel_size
    )
    depth_indices, depth_start_kernel, num_kernel_depths = clamp_kernel_1d(
        depth, 0, param.depth, param.kernel_size
    )

    # add the kernel to the coverage density
    # effect of the agent on the coverage density
    coverage_density[depth_indices, row_indices, col_indices] += coverage_block[
        depth_start_kernel : depth_start_kernel + num_kernel_depths,
        row_start_kernel : row_start_kernel + num_kernel_rows,
        col_start_kernel : col_start_kernel + num_kernel_cols,
    ]

    coverage = normalize_mat(coverage_density)

    # this is the part we introduce exploration problem to the Heat Equation
    diff = goal_density - coverage
    source = np.maximum(diff, 0) ** 3
    source = normalize_mat(source) * param.area

    current_heat = np.zeros((param.depth, param.height, param.width))

    # 2-D heat equation (Partial Differential Equation)
    # In 2-D we perform this second-order central for x and y.
    # Note that, delta_x = delta_y = h since we have a uniform grid.
    # Accordingly we have -4.0 of the center element.

    # At boundary we have Neumann boundary conditions which assumes
    # that the derivative is zero at the boundary. This is equivalent
    # to having a zero flux boundary condition or perfect insulation.
    current_heat[1:-1, 1:-1, 1:-1] = param.dt * (
        (
            + param.alpha[2] * offset(heat, 1, 0, 0)
            + param.alpha[2] * offset(heat, -1, 0, 0)
            + param.alpha[1] * offset(heat, 0, 1, 0)
            + param.alpha[1] * offset(heat, 0, -1, 0)
            + param.alpha[0] * offset(heat, 0, 0, 1)
            + param.alpha[0] * offset(heat, 0, 0, -1)
            - 6.0 * offset(heat, 0, 0, 0)
        )
        / (param.dx * param.dx * param.dx)
        + param.source_strength * offset(source, 0, 0, 0)
    ) + offset(heat, 0, 0, 0)

    heat = current_heat.astype(np.float32)

    # Calculate the first derivatives (mind the order x, y and z)
    gradient_z, gradient_y, gradient_x = np.gradient(heat, 1, 1, 1)

    grad = calculate_gradient(
        p,
        agent,
        gradient_x,
        gradient_y,
        gradient_z,
    )
    p = agent.update(p, grad)

    return p / param.nbRes, coverage_density, heat


def controlCommand(x, agent, goal_density, coverage_density, heat, coverage_block, param):
    J = Jkin(x)
    f = fkin(x)
    e, coverage_density, heat = ergodicControl(f[:3], agent, goal_density, coverage_density, heat, coverage_block, param)
    u = np.linalg.pinv(J[:3,:]) @ (e - f[:3])
    return u, coverage_density, heat


reset()
