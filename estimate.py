import numpy as np
from btl_func_g import calc_E_L_g_grad, calc_E_L_g, calc_L_g_grad, calc_L_g
from btl_func import calc_L, calc_L_grad, calc_E_L, calc_E_L_grad
from data import GaoData
from optim import GradientDescent
from utils import normalize


# needed only for sanity check
def calc_v_star(data: GaoData) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_E_L_grad(data, v),
        np.random.rand(data.p),
        alpha=1 / (data.N.sum(axis=0).max() / 2),
        max_iter=1000,
        f=lambda v: -calc_E_L(data, v),
    )[0]


# needed only for sanity check
def calc_v_g_star(data: GaoData, g_l: float) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_E_L_g_grad(data, v, g_l),
        np.random.rand(data.p),
        alpha=2 / (g_l + data.N.sum(axis=0).max() / 2),
        max_iter=1000,
        f=lambda v: -calc_E_L_g(data, v, g_l),
    )[0]


def calc_v_tilda(data: GaoData) -> np.ndarray:
    return GradientDescent(
        lambda v: -calc_L_grad(data, v),
        data.v,
        alpha=0.1,
        # alpha= 1 / (data.N.sum(axis=0).max() / 2),
        max_iter=1000,
        f=lambda v: -calc_L(data, v),
    )[0]


# only when regularization is not (sum)**2
def calc_v_g_tilda(data: GaoData, g_l: float) -> np.ndarray:
    print("use this function only when regularization is not (sum)**2")
    return GradientDescent(
        lambda v: -calc_L_g_grad(data, v, g_l),
        data.v,
        alpha=1 / (g_l + data.N.sum(axis=0).max()),
        max_iter=1000,
        f=lambda v: -calc_L_g(data, v, g_l),
    )[0]


def calc_v_g_tilda_sum_reg(data: GaoData, g_l: float) -> np.ndarray:
    return normalize(calc_v_tilda(data))
