import pycutest
import numpy as np
from typing import List
from scipy.optimize import line_search

import optim_utils


def find_suitable_problems(num_problems: int = 6) -> List[str]:
    """
    Trova i problemi in PyCUTEst che hanno:
    - objective: 'other'
    - constraints: 'unconstrained' 
    - regular: 'True

    Restituisce una lista di nomi di problemi.    
    """
    # Trova tutti i problemi con i criteri specificati
    problems = pycutest.find_problems(
        objective = 'other',
        constraints = 'unconstrained',
        regular = True
    )
    
    # Restituisce al massimo il numero di problemi richiesto
    problems.sort()  # Ordina i problemi per consistenza
    return problems[:num_problems]


if __name__ == "__main__":
    
    # Trova problemi adatti in PyCUTEst 
    problems = find_suitable_problems()
    print("Found problems:", problems)
    for prob in problems:
        print(pycutest.problem_properties(prob))

    
    p = pycutest.import_problem('ROSENBR')
    # print("Rosenbrock function in %gD" % p.n)

    # # Esecuzione del metodo di discesa del gradiente con Armijo
    x0 = p.x0
    f = lambda x: p.obj(x)
    g = lambda x: p.grad(x)
    xmin, f_values = optim_utils.gradient_descent_armijo(f, g, x0)
    print("Gradient descent with Armijo:")
    print("Minimum found at x =", xmin)
    print("Function value at minimum f(x) =", f(xmin))
    print("Number of function evaluations =", len(f_values))

    print("\n---\n")

    # Esecuzione del metodo BFGS con Wolfe forti
    xmin_bfgs, info = optim_utils.bfgs_strong_wolfe(f, g, x0)
    print("BFGS with strong Wolfe conditions:")
    print("Minimum found at x =", xmin_bfgs)
    print("Function value at minimum f(x) =", info['f_history'][-1])
    print("Number of function evaluations =", info['nit'])

    print("\n---\n")

    # Per ripetere i test
    rng = np.random.default_rng(seed=42)
    eps_f = 1e-3
    eps_g = 1e-3

    # Esecuzione del metodo BFGS con Wolfe forti e tollerante al rumore
    xmin_bfgs_noisy, info_noisy = optim_utils.bfgs_strong_wolfe_noise_tolerant(f, g, x0, tol=1e-10, eps_f=eps_f, eps_g=eps_g, rng=rng)
    print("BFGS with strong Wolfe conditions (noisy):")
    print("Minimum found at x =", xmin_bfgs_noisy)
    print("Function value at minimum f(x) =", info_noisy['f_history'][-1])
    print("Number of function evaluations =", info_noisy['nit'])

    print("\n---\n")

    # XXX: La line search con Armijo su funzione rumorosa non riesce a convergere e raggiunge il numero massimo di iterazioni
    # # Esecuzione del metodo di discesa del gradiente con Armijo ma con funzione rumorosa
    # xmin_noisy, f_values_noisy = optim_utils.gradient_descent_armijo(f, g, x0, eps_f=1e-5, eps_g=1e-5, rng=rng)
    # print("Gradient descent with Armijo (noisy function):")
    # print("Minimum found at x =", xmin_noisy)
    # print("Function value at minimum f(x) =", f(xmin_noisy))
    # print("Number of function evaluations =", len(f_values_noisy))

    # print("\n---\n")

    # Esecuzione del metodo BFGS con Wolfe forti ma con funzione rumorosa
    xmin_bfgs_noisy2, info_noisy2 = optim_utils.bfgs_strong_wolfe(f, g, x0, eps_f=eps_f, eps_g=eps_g, rng=rng)
    print("BFGS with strong Wolfe conditions (noisy function):")
    print("Minimum found at x =", xmin_bfgs_noisy2)
    print("Function value at minimum f(x) =", info_noisy2['f_history'][-1])
    print("Number of function evaluations =", info_noisy2['nit'])

    print("\n---\n")

    # # Cerca il passo con le condizioni di Wolfe forti su i problemi trovati e controllo se soddisfa le condizioni
    # for name in problems:
    #     print(f"\n=== Problema: {name} ===")
    #     prob = pycutest.import_problem(name)
    #     x0 = np.asarray(prob.x0, dtype=float)
    #     # funzioni wrapper
    #     f = lambda x, prob=prob: float(prob.obj(x))
    #     g = lambda x, prob=prob: np.asarray(prob.grad(x), dtype=float)

    #     g0 = g(x0)
    #     pk = -g0
    #     # our strong_wolfe
    #     try:
    #         alpha_my, info = optim_utils.strong_wolfe_line_search(f, g, x0, pk)
    #         check_my = optim_utils.check_strong_wolfe(f, g, x0, pk, alpha_my, c1=1e-4, c2=0.9)
    #         print(f"strong_wolfe found alpha = {alpha_my}, checks: Armijo={check_my[0]}, strong_grad={check_my[1]}")
    #         print(f"  info: {info}")
    #     except Exception as e:
    #         print(f"strong_wolfe error on {name}: {e}") 
    
    # akiva = pycutest.import_problem('AKIVA')
    
    # # cerca il passo con line search di scipy
    # x0 = np.asarray(akiva.x0, dtype=float)
    # f = lambda x, prob=akiva: float(prob.obj(x))
    # g = lambda x, prob=akiva: np.asarray(prob.grad(x), dtype=float)
    # g0 = g(x0)
    # pk = -g0
    # try:
    #     res = line_search(f, g, x0, pk, gfk=g0, old_fval=f(x0), c1=1e-4, c2=0.9)
    #     alpha_scipy = res[0]  # None or float
    #     print(f"SciPy line_search found alpha = {alpha_scipy}")
    #     # controlla che soddisfa le condizioni di Wolfe forti
    #     if alpha_scipy is not None:
    #         check_scipy = optim_utils.check_strong_wolfe(f, g, x0, pk, alpha_scipy, c1=1e-4, c2=0.9)
    #         print(f"  checks: Armijo={check_scipy[0]}, strong_grad={check_scipy[1]}")
    # except Exception as e:
    #     print(f"scipy line_search error on AKIVA: {e}")

