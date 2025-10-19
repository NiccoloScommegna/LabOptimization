import pycutest
import numpy as np
from typing import Callable, Tuple, Optional, List


# Funzione di norma vettoriale
def vecnorm(x: np.ndarray) -> float:
    return np.linalg.norm(x)


# Metodo di Armijo per la ricerca del passo
def armijo_line_search(f: Callable[[np.ndarray], float],
                       g: Callable[[np.ndarray], np.ndarray],
                       xk: np.ndarray,
                       dk: np.ndarray,
                       alpha0: float = 1.0,
                       sigma: float = 0.5,
                       gamma: float = 1e-4) -> float:
    fk = f(xk)
    gk = g(xk)
    alpha = alpha0
    while f(xk + alpha * dk) > fk + gamma * alpha * np.dot(gk, dk):
        alpha *= sigma
    return alpha


# Metodo di discesa del gradiente con ricerca del passo di Armijo
def gradient_descent_armijo(f: Callable[[np.ndarray], float],
                            g: Callable[[np.ndarray], np.ndarray],
                            x0: np.ndarray,
                            tol: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
    xk = x0
    f_values = [f(xk)]
    while (vecnorm(g(xk)) > tol):
        dk = -g(xk)
        alpha = armijo_line_search(f, g, xk, dk)
        xk = xk + alpha * dk
        f_values.append(f(xk))
    return xk, f_values


if __name__ == "__main__":
    p = pycutest.import_problem('ROSENBR')

    print("Rosenbrock function in %gD" % p.n)

    # Esecuzione del metodo di discesa del gradiente con Armijo
    x0 = p.x0
    f = lambda x: p.obj(x)
    g = lambda x: p.grad(x)
    xmin, f_values = gradient_descent_armijo(f, g, x0)
    print("Minimum found at x =", xmin)
    print("Function value at minimum f(x) =", f(xmin))
    print("Number of function evaluations =", len(f_values))
    print("Done")
