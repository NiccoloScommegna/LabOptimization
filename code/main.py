import pycutest
import numpy as np
from typing import List
from scipy.optimize import line_search

import optim_utils
import test


def find_suitable_problems(num_problems: int) -> List[str]:
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


def choose_tol_from_noise(eps_g, n, factor=10, tol_min=1e-12):
    """
    Ritorna un tol coerente con il rumore sul gradiente.
    factor: moltiplicatore empirico (5..20 => più conservativo)
    """
    if eps_g is None or eps_g <= 0:
        return tol_min
    noise_est = np.sqrt(n) * eps_g
    return max(tol_min, factor * noise_est)


if __name__ == "__main__":
    
    # # Trova problemi adatti in PyCUTEst 
    # problems = find_suitable_problems(num_problems=8)
    # print("Found problems:", problems)
    # for prob in problems:
    #     print(pycutest.problem_properties(prob))

    # print("\n---\n")

    # # p = pycutest.import_problem('ROSENBR')
    
    # # p = pycutest.import_problem('AKIVA')  # problema con cui non si riesce a far convergere la line search con Armijo
    
    # # p = pycutest.import_problem('ALLINITU')
    
    # # pycutest.clear_cache('ARWHEAD')
    # # pycutest.print_available_sif_params('ARWHEAD')
    # # p = pycutest.import_problem('ARWHEAD', sifParams={'N': 100})  # Possibili valori per N: 100, 500, 1000, 5000

    # # pycutest.print_available_sif_params('BOX')
    # # p = pycutest.import_problem('BOX', sifParams={'N': 10})  # Possibili valori per N: 10, 100, 1000, 10000

    # # pycutest.print_available_sif_params('BOXPOWER')
    # # p = pycutest.import_problem('BOXPOWER', sifParams={'N': 10})  # Possibili valori per N: 10, 100, 1000, 10000, 20000

    # # p = pycutest.import_problem('BRKMCC')

    # # pycutest.print_available_sif_params('BROYDN7D')
    # p = pycutest.import_problem('BROYDN7D', sifParams={'N/2': 25})  # Possibili valori per N/2: 5, 25, 50, 250, 500

    # # problemi = ['ALLINITU', 'ARWHEAD', 'BOX', 'BOXPOWER', 'BRKMCC', 'BROYDN7D']


    # # Parametri per il rumore e la tolleranza da usare nei test
    # n = p.n
    # rng = np.random.default_rng(seed=42)  # generatore di numeri casuali per il rumore con seed fisso per riproducibilità
    # eps_f = 1e-7
    # eps_g = 1e-7
    # tol = choose_tol_from_noise(eps_g, n, factor=10, tol_min=1e-12)
    # print(f"Using noise levels eps_f={eps_f}, eps_g={eps_g}, tol={tol}, n={n}")

    # print("\n---\n")

    # # # Esecuzione del metodo di discesa del gradiente con Armijo
    # x0 = p.x0
    # f = lambda x: p.obj(x)
    # g = lambda x: p.grad(x)
    # xmin, f_values = optim_utils.gradient_descent_armijo(f, g, x0, tol=tol)
    # print("Gradient descent with Armijo (base function and base method):")
    # print("Minimum found at x =", xmin)
    # print("Function value at minimum f(x) =", f(xmin))
    # print("Number of function evaluations =", len(f_values))

    # print("\n---\n")

    # # Esecuzione del metodo BFGS con Wolfe forti
    # xmin_bfgs, info = optim_utils.bfgs_strong_wolfe(f, g, x0, tol=tol)
    # print("BFGS with strong Wolfe conditions (base function and base method):")
    # print("Minimum found at x =", xmin_bfgs)
    # print("Function value at minimum f(x) =", info['f_history'][-1])
    # print("Number of function evaluations =", info['nit'])

    # print("\n---\n")

    # # Esecuzione del metodo BFGS con Wolfe forti e tollerante al rumore
    # xmin_bfgs_noisy, info_noisy = optim_utils.bfgs_strong_wolfe_noise_tolerant(f, g, x0, tol=tol, eps_f=eps_f, eps_g=eps_g, rng=rng)
    # print("BFGS with strong Wolfe conditions (noisy function and noise-tolerant method):")
    # print("Minimum found at x =", xmin_bfgs_noisy)
    # print("Function value at minimum f(x) =", info_noisy['f_history'][-1])
    # print("Number of function evaluations =", info_noisy['nit'])

    # print("\n---\n")

    # # XXX: La line search con Armijo su funzione rumorosa non riesce a convergere e raggiunge il numero massimo di iterazioni
    # # # Esecuzione del metodo di discesa del gradiente con Armijo ma con funzione rumorosa
    # # xmin_noisy, f_values_noisy = optim_utils.gradient_descent_armijo(f, g, x0, eps_f=1e-5, eps_g=1e-5, rng=rng)
    # # print("Gradient descent with Armijo (noisy function):")
    # # print("Minimum found at x =", xmin_noisy)
    # # print("Function value at minimum f(x) =", f(xmin_noisy))
    # # print("Number of function evaluations =", len(f_values_noisy))

    # # print("\n---\n")

    # # Esecuzione del metodo BFGS con Wolfe forti ma con funzione rumorosa
    # xmin_bfgs_noisy2, info_noisy2 = optim_utils.bfgs_strong_wolfe(f, g, x0, tol=tol, eps_f=eps_f, eps_g=eps_g, rng=rng)
    # print("BFGS with strong Wolfe conditions (noisy function and base method):")
    # print("Minimum found at x =", xmin_bfgs_noisy2)
    # print("Function value at minimum f(x) =", info_noisy2['f_history'][-1])
    # print("Number of function evaluations =", info_noisy2['nit'])

    # print("\n---\n")

    methods = [# 'gd_armijo_with_base_function', 
               # 'gd_armijo_with_noisy_function', 
               'bfgs_with_base_function', 
               'bfgs_with_noisy_function', 
               'bfgs_noisy_with_noisy_function',
               'scipy_bfgs_with_base_function',
               'scipy_bfgs_with_noisy_function'
               ]
    
    test.run_and_print(problem_name='ALLINITU', methods=methods, eps_f=1e-4, eps_g=1e-4, tol_factor=10, max_iter=10000)
    # summary = test.repeat_timing(problem_name='ALLINITU', methods=methods, eps_f=1e-4, eps_g=1e-4, tol_factor=10, max_iter=10000, n_runs=5, quiet=False)
    
    # test.run_and_print(problem_name='BRKMCC', methods=methods, eps_f=1e-7, eps_g=1e-7, tol_factor=10, max_iter=10000)

    # test.run_and_print(problem_name='BROYDN7D', methods=methods, eps_f=1e-5, eps_g=1e-5, tol_factor=10, max_iter=10000, sif_params={'N/2': 25})

    
