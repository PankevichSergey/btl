import numpy as np
from btl_func_g import calc_E_L_g_grad, calc_E_L_g, calc_L_g_grad, calc_L_g
from btl_func import calc_L, calc_L_grad, calc_E_L, calc_E_L_grad
from data import GaoData
from optim import GradientDescent


def calc_v_star(data: GaoData) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_E_L_grad(data, v),
        np.random.rand(data.p),
        alpha=1 / (data.p * data.L),
        max_iter=1000,
        f=lambda v: -calc_E_L(data, v),
    )[0]


def calc_v_g_star(data: GaoData, g_l: float) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_E_L_g_grad(data, v, g_l),
        np.random.rand(data.p),
        alpha=1 / (g_l + data.p * data.L),
        max_iter=1000,
        f=lambda v: -calc_E_L_g(data, v, g_l),
    )[0]


def calc_v_tilda(data: GaoData) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_L_grad(data, v),
        np.random.rand(data.p),
        alpha=1 / (data.p * data.L),
        max_iter=100,
        f=lambda v: -calc_L(data, v),
    )[0]


def calc_v_g_tilda(data: GaoData, g_l: float) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_L_g_grad(data, v, g_l),
        np.random.rand(data.p),
        alpha=1 / (g_l + data.p * data.L),
        max_iter=100,
        f=lambda v: -calc_L_g(data, v, g_l),
    )[0]
