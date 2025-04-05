import numpy as np
from data import GaoData


def phi(v: float) -> float:
    return np.log(1 + np.exp(v))

 
def phi_1(v: float) -> float:  # == sigmoid(v)
    return 1 / (1 + np.exp(-v))


def phi_2(v: float) -> float:  # == sigmoid'(v) = sigmoid(v) * (1 - sigmoid(v))
    return np.exp(v) / (1 + np.exp(v)) ** 2


def p_win(v_i: float, v_j: float) -> float:  # == sigmoid(v_i - v_j)
    return 1 / (1 + np.exp(v_j - v_i))


def log_p_win(v_i: float, v_j: float) -> float:
    return -phi(v_j - v_i)


def F(v: np.ndarray, N: np.ndarray) -> np.ndarray:
    dif = v.reshape((-1, 1)) - v.reshape((1, -1))
    up = np.exp(dif)
    down = (1 + np.exp(dif)) ** 2
    res = -N * up / down
    res[np.arange(len(v)), np.arange(len(v))] = 0
    res[np.arange(len(v)), np.arange(len(v))] = -np.sum(res, axis=1)
    return res


def F_slow(v: np.ndarray, N: np.ndarray) -> np.ndarray:
    res = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            if i == j:
                continue
            else:
                res[i, j] = -N[i, j] * phi_2(v[i] - v[j])
        res[i, i] = -np.sum(res[i, :])
    return res


# for estimation of v^*_g
def calc_E_L(data: GaoData, v: np.ndarray) -> float:
    result = 0
    for m in range(data.p):
        for j in range(m):
            result += (v[j] - v[m]) * data.N[j, m] * p_win(
                data.v[j], data.v[m]
            ) - data.N[j, m] * phi(v[j] - v[m])
    return result


def calc_E_L_grad(data: GaoData, v: np.ndarray) -> np.ndarray:
    result = np.zeros(data.p)
    for m in range(data.p):
        for j in range(data.p):
            if j != m:
                # (vj - vm) * N * p_win(vj, vm)
                p = phi_1(data.v[j] - data.v[m])
                result[m] += -data.N[j, m] * p
                result[m] += data.N[j, m] * phi_1(v[j] - v[m])

    return result


def calc_L(data: GaoData, v: np.ndarray) -> float:
    result = 0
    for m in range(data.p):
        for j in range(m):
            result += (v[j] - v[m]) * data.S[j, m] - data.N[j, m] * phi(
                v[j] - v[m]
            )
    return result


def calc_L_grad(data: GaoData, v: np.ndarray) -> np.ndarray:
    result = np.zeros(data.p)
    for m in range(data.p):
        for j in range(data.p):
            if j != m:
                result[m] += -data.S[j, m] + data.N[j, m] * phi_1(v[j] - v[m])
    return result


def calc_L_hess(data: GaoData, v: np.ndarray) -> np.ndarray:
    return -F(v, data.N)
