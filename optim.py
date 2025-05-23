from tqdm.auto import tqdm
import numpy as np
import warnings


def GradientDescent(
    grad_f: callable,
    x0: np.ndarray,
    alpha: float = 1e-3,
    max_iter: int = 1000,
    f: callable = None,
    verbose_iter: int | None = None,
    tol: float = 1e-5,
) -> np.ndarray:
    x = x0.copy()
    init_grad_norm = np.linalg.norm(grad_f(x))
    f_log, iter_log = [], []
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - alpha * grad
        print(f"iter {i}, grad norm {np.linalg.norm(grad)}, f {f(x)}, x {x}")
        if f is not None and verbose_iter is not None and i % verbose_iter == 0:
            f_log.append(f(x))
            iter_log.append(i)
        if np.linalg.norm(grad) < tol:
            break
    else:
        warnings.warn(
            f"GD did not converge in {max_iter} iterations. Final grad norm: {np.linalg.norm(grad)}; Initial grad norm: {init_grad_norm}.",
            RuntimeWarning,
        )
    return x, f_log, iter_log
