import numpy as np
from btl_func import F, calc_E_L, calc_E_L_grad, calc_L, calc_L_grad, calc_L_hess
from data import GaoData


def F_g(v_g: np.ndarray, N: np.ndarray, g_l: float) -> np.ndarray:
    return F(v_g, N) + g_l * np.eye(len(v_g))


def D_g_squared(v_g: np.ndarray, N: np.ndarray, g_l: float) -> np.ndarray:
    return np.diag(F_g(v_g, N, g_l))


def calc_E_L_g(data: GaoData, v: np.ndarray, g_l: float) -> float:
    return calc_E_L(data, v) - np.linalg.norm(g_l * v) ** 2 / 2


def calc_E_L_g_grad(data: GaoData, v: np.ndarray, g_l: float) -> np.ndarray:
    return calc_E_L_grad(data, v) - g_l * v


def calc_L_g(data: GaoData, v: np.ndarray, g_l: float) -> float:
    return calc_L(data, v) - np.linalg.norm(g_l * v) ** 2 / 2


def calc_L_g_grad(data: GaoData, v: np.ndarray, g_l: float) -> np.ndarray:
    return calc_L_grad(data, v) - g_l * v


def calc_L_g_hess(data: GaoData, v: np.ndarray, g_l: float) -> np.ndarray:
    return calc_L_hess(data, v) - g_l * np.eye(len(v))