import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    h = np.empty((n, n), dtype=int)
    h[:half_size,:half_size] = h_half
    h[half_size:,:half_size] = h_half
    h[:half_size:,half_size:] = h_half
    h[half_size:,half_size:] = -h_half
    return h


def fourier(alpha):
    ## Compute Fourier series coefficients w_hat of desired spatial distribution
    w_hat = np.zeros(param.nbFct**param.nbVar)
    for j in range(param.nbGaussian):
        for n in range(param.op.shape[1]):
            MuTmp = np.diag(param.op[:,n]) @ param.Mu[:,j]
            SigmaTmp = np.diag(param.op[:,n]) @ param.Sigma[:,:,j] @ np.diag(param.op[:,n]).T
            cos_term = np.cos(param.kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-.5 * param.kk.T @ SigmaTmp @ param.kk))
            w_hat = w_hat + alpha[j] * cos_term * exp_term
    return w_hat / (param.L**param.nbVar) / (param.op.shape[1])


def create_gaussians(Mu, Sigma_vectors, Sigma_scales, Sigma_regularizations):
    param.Mu = np.array(Mu).T
    Sigma_vectors = np.array(Sigma_vectors)
    Sigma_scales = np.array(Sigma_scales)
    Sigma_regularizations = np.array(Sigma_regularizations)

    if (len(param.Mu.shape) != 2) or (param.Mu.shape[0] != 2):
        print("Error: 'Mu' must be a Nx2 matrix, with 'N' the number of gaussians")
        return False

    param.nbGaussian = param.Mu.shape[1]

    if (len(Sigma_vectors.shape) != 2) or (Sigma_vectors.shape[0] != param.nbGaussian) or (Sigma_vectors.shape[1] != 2):
        print(f"Error: 'Sigma_vectors' must be a {param.nbGaussian}x2 matrix")
        return False

    if (len(Sigma_scales.shape) != 1) or (Sigma_scales.shape[0] != param.nbGaussian):
        print(f"Error: 'Sigma_scales' must be a vector of {param.nbGaussian} values")
        return False

    if (len(Sigma_regularizations.shape) != 1) or (Sigma_regularizations.shape[0] != param.nbGaussian):
        print(f"Error: 'Sigma_regularizations' must be a vector of {param.nbGaussian} values")
        return False

    param.Sigma = np.zeros((param.nbVar, param.nbVar, param.nbGaussian))
    for i in range(param.nbGaussian):
        param.Sigma[:,:,i] = np.outer(Sigma_vectors[i,:], Sigma_vectors[i,:]) * Sigma_scales[i] + np.eye(param.nbVar) * Sigma_regularizations[i]

    return True
