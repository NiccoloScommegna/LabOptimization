from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pycutest
except Exception as e:
    pycutest = None  # gestito più avanti

import optim_utils


# -----------------------
# Helper
# -----------------------


def list_suitable_problems(num_problems: int = 8) -> List[str]:
    """
    Trova i problemi in PyCUTEst che hanno:
    - objective: 'other'
    - constraints: 'unconstrained' 
    - regular: 'True'
    Restituisce una lista di nomi di problemi.

    Se pycutest non è disponibile solleva RuntimeError.
    """
    if pycutest is None:
        raise RuntimeError("pycutest non è disponibile. Installa pycutest per usare questa funzione.")

    problems = pycutest.find_problems(objective="other", constraints="unconstrained", regular=True)
    problems = sorted(problems)
    return problems[:num_problems]


def choose_tol_from_noise(eps_g: Optional[float], n: int, factor: float = 10.0, tol_min: float = 1e-12) -> float:
    """
    Sceglie una tolleranza coerente con il rumore sul gradiente.
    Ritorna almeno tol_min.
    factor: moltiplicatore empirico (5..20 => più conservativo)
    """
    if eps_g is None or eps_g <= 0:
        return tol_min
    noise_est = np.sqrt(n) * eps_g
    return max(tol_min, float(factor * noise_est))


# -----------------------
# Funzioni per eseguire test
# -----------------------


def import_problem(problem_name: str, sif_params: Optional[Dict] = None):
    """Importa un problema PyCUTEst con gestione errori.
    Restituisce l'oggetto problema.
    """
    if pycutest is None:
        raise RuntimeError("pycutest non è disponibile. Installa pycutest per usare questo script.")

    try:
        if sif_params:
            p = pycutest.import_problem(problem_name, sifParams=sif_params)
        else:
            p = pycutest.import_problem(problem_name)
    except Exception as e:
        raise RuntimeError(f"Errore importando il problema {problem_name}: {e}")
    return p


def run_method_on_problem(p, method: str, eps_f: float, eps_g: float, rng: np.random.Generator, tol_factor: float, max_iter: int) -> Dict:
    """
    Esegue uno dei metodi implementati in optim_utils sul problema PyCUTEst `p`.

    method in {'gd_armijo', 'bfgs', 'bfgs_noisy'}
    """
    x0 = p.x0
    n = p.n
    f = lambda x: p.obj(x)
    g = lambda x: p.grad(x)

    tol = choose_tol_from_noise(eps_g, n, factor=tol_factor, tol_min=1e-12)

    info: Dict = {
        'method': method,
        # 'problem': p.name,
        'problem': getattr(p, 'name', None) or getattr(p, 'probname', None) or str(p),
        'n': n,
        'eps_f': eps_f,
        'eps_g': eps_g,
        'tol': tol,
        # 'start_time': time.time(),
    }


    start = time.perf_counter()
    try:
        if method == 'gd_armijo':
            xmin, f_values = optim_utils.gradient_descent_armijo(f=f, g=g, x0=x0, 
                                                                 tol=tol, maxiter=max_iter, 
                                                                 eps_f=eps_f, eps_g=eps_g, rng=rng
                                                                 )
            
            elapsed = time.perf_counter() - start
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),
                
                'f_val': float(_safe_eval(f, xmin)),
                
                # 'f_history_len': len(f_values),
                'n_iter': max(0, len(f_values) - 1),
                
                'elapsed': elapsed,
            })

        elif method == 'bfgs':
            xmin, out = optim_utils.bfgs_strong_wolfe(f=f, g=g, x0=x0, 
                                                      tol=tol, max_iter=max_iter, 
                                                      eps_f=eps_f, eps_g=eps_g, rng=rng
                                                      )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            fval = out.get('f_history', [None])[-1] if out.get('f_history') else _safe_eval(f, xmin)
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),

                # 'f_val': float(out['f_history'][-1]) if out.get('f_history') else float(_safe_eval(f, xmin)),
                'f_val': float(fval) if fval is not None else float('nan'),

                # 'nit': out.get('nit'),
                'n_iter': nit,

                'elapsed': elapsed,

                # 'f_history_len': len(out.get('f_history', [])),
                # 'grad_norms': [float(v) for v in out.get('grad_norms', [])],
                'ls_info': out
            })

        elif method == 'bfgs_noisy':
            xmin, out = optim_utils.bfgs_strong_wolfe_noise_tolerant(f=f, g=g, x0=x0, 
                                                                     tol=tol, max_iter=max_iter, 
                                                                     eps_f=eps_f, eps_g=eps_g, rng=rng
                                                                     )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            fval = out.get('f_history', [None])[-1] if out.get('f_history') else _safe_eval(f, xmin)
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),

                # 'f_val': float(out['f_history'][-1]) if out.get('f_history') else float(_safe_eval(f, xmin)),
                'f_val': float(fval) if fval is not None else float('nan'),

                # 'nit': out.get('nit'),
                'n_iter': nit,

                'elapsed': elapsed,

                # 'f_history_len': len(out.get('f_history', [])),
                # 'grad_norms': [float(v) for v in out.get('grad_norms', [])],
                # 'ls_history': out.get('ls_history', []),
                'ls_info': out.get('ls_history', []),
            })

        else:
            raise ValueError(f"Metodo non riconosciuto: {method}")

        info['status'] = 'success'

    except Exception as e:
        elapsed = time.perf_counter() - start
        logging.exception("Errore durante l'esecuzione del metodo")
        info['status'] = 'error'
        info['error'] = str(e)
        info['elapsed'] = elapsed

    return info


def _safe_eval(f_callable, x):
    try:
        return float(f_callable(x))
    except Exception:
        return float('nan')


def run_and_print(problem_name: str, methods: List[str], eps_f: float = 1e-7, eps_g: float = 1e-7, seed: int = 42, tol_factor: float = 10.0, max_iter: int = 10000, sif_params: Optional[Dict] = None) -> None:
    """
    Esegue i metodi su ciascun problema e stampa a schermo alcuni risultati.


    Output stampato per ciascuna coppia (problema, metodo):
    - Metodo usato
    - Problema (nome), dimensione n, eventuali sif_params
    - Valori di rumore eps_f, eps_g
    - Punto di minimo trovato (x)
    - Numero di iterazioni
    - Tempo impiegato (s)

    """
    rng_master = np.random.default_rng(seed)

    try:
        p = import_problem(problem_name, sif_params=sif_params)
    except Exception as e:
        print(f"Impossibile importare problema {problem_name}: {e}")
        return

    # Informazioni generali sul problema
    prob_id = getattr(p, 'name', None) or getattr(p, 'probname', None) or problem_name
    n = getattr(p, 'n', None)

    for method in methods:
        subrng = np.random.default_rng(rng_master.integers(1 << 30))
        print("----------------------------------------")
        print(f"Metodo: {method}")
        print(f"Problema: {prob_id}")
        if sif_params:
            print(f"SIF params: {sif_params}")
        print(f"Dimensione n: {n}")
        print(f"Rumore: eps_f={eps_f}, eps_g={eps_g}")

        res = run_method_on_problem(p, method=method, eps_f=eps_f, eps_g=eps_g, rng=subrng, tol_factor=tol_factor, max_iter=max_iter)

        if res.get('status') == 'success':
            print(f"Punto minimo trovato (x): {res.get('x_min')}")
            print(f"Valore funzione: {res.get('f_val')}")
            print(f"Numero iterazioni: {res.get('n_iter')}")
            print(f"Tempo impiegato: {res.get('elapsed'):.6f} s")
        else:
            print(f"Metodo fallito con errore: {res.get('error')}")
        print("----------------------------------------")