import numpy as np
from btl_func import (
    F,
    calc_E_L,
    calc_E_L_grad,
    calc_L,
    calc_L_grad,
    calc_L_hess,
)
from data import GaoData


def F_g(v: np.ndarray, N: np.ndarray, g_l: float) -> np.ndarray:
    return F(v, N) + (g_l**2) * np.ones((len(v), len(v))) / len(v)


def D_g_squared(v: np.ndarray, N: np.ndarray, g_l: float) -> np.ndarray:
    raise NotImplementedError
    return np.diag(F_g(v, N, g_l))


def calc_L_g(data: GaoData, v: np.ndarray, g_l: float) -> float:
    return calc_L(data, v) - (g_l**2) * np.sum(v) ** 2 / (2 * len(v))

def calc_E_L_g(data: GaoData, v: np.ndarray, g_l: float) -> float:
    return calc_E_L(data, v) - (g_l**2) * np.sum(v) ** 2 / (2 * len(v))

def calc_L_g_grad(data: GaoData, v: np.ndarray, g_l: float) -> np.ndarray:
    return calc_L_grad(data, v) - (g_l**2) * np.sum(v) * np.ones((len(v),)) / len(v)

def calc_E_L_g_grad(data: GaoData, v: np.ndarray, g_l: float) -> np.ndarray:
    return calc_E_L_grad(data, v) - (g_l**2) * np.sum(v) * np.ones((len(v),)) / len(v)

def calc_L_g_hess(data: GaoData, v: np.ndarray, g_l: float) -> np.ndarray:
    return calc_L_hess(data, v) - (g_l**2) * np.ones((len(v), len(v))) / len(v)
