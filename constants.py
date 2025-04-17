import numpy as np


# TODO check if g is needed


def N(F_g: np.ndarray) -> float:
    return np.min(np.linalg.eigvalsh(F_g))


def rho_1(F: np.ndarray) -> float:
    result = None
    D_squared = np.diag(F)
    p = F.shape[0]
    for j in range(p):
        current_sum = 0
        for m in range(p):
            if j != m:
                current_sum += F[j, m] ** 2 / D_squared[m]
        current_sum /= D_squared[j]
        if result is None or current_sum > result:
            result = current_sum
    return np.sqrt(result)


def r_inf(x: float, F: np.ndarray) -> float:
    p = F.shape[0]
    return 2 / (1 - rho_1(F)) * np.sqrt(x + np.log(p))


def omega(x: float, F: np.ndarray, F_g: np.ndarray) -> float:
    return 3 * r_inf(x, F) / np.sqrt(N(F_g))


def c(x: float, F: np.ndarray, F_g: np.ndarray) -> float:
    omega_ = omega(x, F, F_g)
    rho_1_ = rho_1(F)
    return (
        1
        / (1 - omega_)
        * (rho_1_ + 1 / 2 + 3 * (rho_1_ + 1 / 2) ** 2 / (4 * (1 - omega_) ** 2))
    )


def tau_inf(x: float, F: np.ndarray, F_g: np.ndarray) -> float:
    N_ = N(F_g)
    c_ = c(x, F, F_g)
    rho_1_ = rho_1(F)
    return 3 / (np.sqrt(N_)) * (5 / 2 + 2 * (c_ + 1) / (1 - rho_1_) ** 2)
