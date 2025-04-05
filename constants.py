import numpy as np



# TODO check if g is needed 

def N(F: np.ndarray) -> float:
    return np.min(np.linalg.eigvalsh(F))

def rho_1(F_g: np.ndarray) -> float:
    result = None
    D_2 = np.diag(F_g)
    for j in range(F_g.shape[0]):
        current_sum = 0
        for m in range(F_g.shape[1]):
            if j != m:
                current_sum += F_g[j, m] ** 2 / D_2[m]
        current_sum /= D_2[j]
        if result is None or current_sum > result:
            result = current_sum
    return result

def r_inf(x: float, p: int, F_g: np.ndarray) -> float:
    return 2 / (1 - rho_1(F_g)) * np.sqrt(x + np.log(p))

def omega(x: float, p: int, F_g: np.ndarray) -> float:
    return 3 * r_inf(x, p, F_g) / np.sqrt(N(F_g))

def c(x: float, p: int, F_g: np.ndarray) -> float:
    omega_ = omega(x, p, F_g)
    rho_1_ = rho_1(F_g)
    return 1 / (1 - omega_) * (rho_1_ + 1/2 + 3 * (rho_1_ + 1/2) ** 2 / (4 * (1 - omega_) ** 2))

def tau_inf(x: float, p: int, F_g: np.ndarray) -> float:
    N_ = N(F_g)
    c_ = c(x, p, F_g)
    rho_1_ = rho_1(F_g)
    return  3 / (np.sqrt(N_)) * (5 / 2 + 2* (c + 1) / (1 - rho_1_)**2)
