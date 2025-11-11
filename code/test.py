from __future__ import annotations
import argparse
import json
import logging
import os
import time
import contextlib
import io
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

import plotting


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


_VALID_METHODS = {
    'gd_armijo_with_base_function',
    'gd_armijo_with_noisy_function',
    'bfgs_with_base_function',
    'bfgs_with_noisy_function',
    'bfgs_noisy_with_noisy_function',
    'scipy_bfgs_with_base_function',
    'scipy_bfgs_with_noisy_function',
}


def run_method_on_problem(p, method: str, eps_f: float, eps_g: float, rng: np.random.Generator, tol_factor: float, max_iter: int) -> Dict:
    """
    Esegue uno dei metodi implementati in optim_utils sul problema PyCUTEst `p`.

    method in _VALID_METHODS
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Metodo non riconosciuto: {method}")

    x0 = p.x0
    n = p.n
    # f = lambda x: p.obj(x)
    # g = lambda x: p.grad(x)

    f = lambda x: float(p.obj(x))
    g = lambda x: np.asarray(p.grad(x), dtype=float)

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
    
    # backup_start esiste sempre per evitare errori in caso di eccezioni prima di settare start 
    backup_start = time.perf_counter()
    start = None

    try:
        if method == 'gd_armijo_with_base_function':
            print(f"Tolleranza usata: {tol}")
            start = time.perf_counter()
            xmin, out = optim_utils.gradient_descent_armijo(f=f, g=g, x0=x0, 
                                                            tol=tol, maxiter=max_iter
                                                            )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            f_history = [float(v) for v in out.get('f_history', [])] if out.get('f_history') else []
            fval = float(f_history[-1]) if f_history else _safe_eval(f, xmin)
            grad_norms = [float(v) for v in out.get('grad_norms', [])] if out.get('grad_norms') else []
            
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),
                
                'f_val': fval,

                'f_history': f_history,

                'grad_norms': grad_norms,
                
                # 'f_history_len': len(f_values),
                'n_iter': nit,
                
                'elapsed': elapsed,
            })

        elif method == 'gd_armijo_with_noisy_function':
            print(f"Rumore eps_f: {eps_f}, eps_g: {eps_g}")
            print(f"Tolleranza usata: {tol}")
            start = time.perf_counter()
            xmin, out = optim_utils.gradient_descent_armijo(f=f, g=g, x0=x0, 
                                                                 tol=tol, maxiter=max_iter, 
                                                                 eps_f=eps_f, eps_g=eps_g, rng=rng
                                                                 )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            f_history = [float(v) for v in out.get('f_history', [])] if out.get('f_history') else []
            fval = float(f_history[-1]) if f_history else _safe_eval(f, xmin)
            grad_norms = [float(v) for v in out.get('grad_norms', [])] if out.get('grad_norms') else []
            
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),
                
                'f_val': fval,

                'f_history': f_history,

                'grad_norms': grad_norms,
                
                # 'f_history_len': len(f_values),
                'n_iter': nit,
                
                'elapsed': elapsed,
            })

        elif method == 'bfgs_with_base_function':
            print(f"Tolleranza usata: {tol}")
            start = time.perf_counter()
            xmin, out = optim_utils.bfgs_strong_wolfe(f=f, g=g, x0=x0, 
                                                     tol=tol, max_iter=max_iter
                                                     )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            f_history = [float(v) for v in out.get('f_history', [])] if out.get('f_history') else []
            fval = float(f_history[-1]) if f_history else _safe_eval(f, xmin)
            grad_norms = [float(v) for v in out.get('grad_norms', [])] if out.get('grad_norms') else []
            
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),

                # 'f_val': float(out['f_history'][-1]) if out.get('f_history') else float(_safe_eval(f, xmin)),
                'f_val': fval,

                'f_history': f_history,

                'grad_norms': grad_norms,

                # 'nit': out.get('nit'),
                'n_iter': nit,

                'elapsed': elapsed,

                # 'f_history_len': len(out.get('f_history', [])),
                # 'grad_norms': [float(v) for v in out.get('grad_norms', [])],
                'ls_info': out
            })

        elif method == 'bfgs_with_noisy_function':
            print(f"Rumore eps_f: {eps_f}, eps_g: {eps_g}")
            print(f"Tolleranza usata: {tol}")
            start = time.perf_counter()
            xmin, out = optim_utils.bfgs_strong_wolfe(f=f, g=g, x0=x0, 
                                                      tol=tol, max_iter=max_iter, 
                                                      eps_f=eps_f, eps_g=eps_g, rng=rng
                                                      )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            f_history = [float(v) for v in out.get('f_history', [])] if out.get('f_history') else []
            fval = float(f_history[-1]) if f_history else _safe_eval(f, xmin)
            grad_norms = [float(v) for v in out.get('grad_norms', [])] if out.get('grad_norms') else []
            
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),

                # 'f_val': float(out['f_history'][-1]) if out.get('f_history') else float(_safe_eval(f, xmin)),
                'f_val': fval,

                'f_history': f_history,

                'grad_norms': grad_norms,

                # 'nit': out.get('nit'),
                'n_iter': nit,

                'elapsed': elapsed,

                # 'f_history_len': len(out.get('f_history', [])),
                # 'grad_norms': [float(v) for v in out.get('grad_norms', [])],
                'ls_info': out
            })

        elif method == 'bfgs_noisy_with_noisy_function':
            print(f"Rumore eps_f: {eps_f}, eps_g: {eps_g}")
            print(f"Tolleranza usata: {tol}")
            start = time.perf_counter()
            xmin, out = optim_utils.bfgs_strong_wolfe_noise_tolerant(f=f, g=g, x0=x0, 
                                                                     tol=tol, max_iter=max_iter, 
                                                                     eps_f=eps_f, eps_g=eps_g, rng=rng
                                                                     )
            
            elapsed = time.perf_counter() - start
            nit = out.get('nit', None)
            f_history = [float(v) for v in out.get('f_history', [])] if out.get('f_history') else []
            fval = float(f_history[-1]) if f_history else _safe_eval(f, xmin)
            grad_norms = [float(v) for v in out.get('grad_norms', [])] if out.get('grad_norms') else []
            
            info.update({
                # 'x': xmin.tolist(),
                'x_min': np.asarray(xmin).tolist(),

                # 'f_val': float(out['f_history'][-1]) if out.get('f_history') else float(_safe_eval(f, xmin)),
                'f_val': fval,

                'f_history': f_history,

                'grad_norms': grad_norms,

                # 'nit': out.get('nit'),
                'n_iter': nit,

                'elapsed': elapsed,

                # 'f_history_len': len(out.get('f_history', [])),
                # 'grad_norms': [float(v) for v in out.get('grad_norms', [])],
                # 'ls_history': out.get('ls_history', []),
                'ls_info': out.get('ls_history', []),
            })

        elif method in ('scipy_bfgs_with_base_function', 'scipy_bfgs_with_noisy_function'):
            noisy = (method == 'scipy_bfgs_with_noisy_function')
            print(f"SciPy BFGS (noisy={noisy}), eps_f={eps_f}, eps_g={eps_g}, tol={tol}")
            # costruttori wrappers che usano la rng passata quando noisy=True
            f_hist = []
            grad_norms = []

            def f_wrapper(x):
                # x potrebbe essere array-like
                x_arr = np.asarray(x, dtype=float)
                val = float(p.obj(x_arr))
                if noisy and eps_f and eps_f > 0:
                    val = val + float(rng.normal(loc=0.0, scale=eps_f))
                return val

            def g_wrapper(x):
                x_arr = np.asarray(x, dtype=float)
                grad = np.asarray(p.grad(x_arr), dtype=float)
                if noisy and eps_g and eps_g > 0:
                    grad = grad + rng.normal(loc=0.0, scale=eps_g, size=grad.shape)
                return grad

            # callback per registrare la storia (viene chiamato con xk)
            def cb(xk):
                try:
                    # registra valore funzione (con eventuale rumore se noisy=True)
                    f_hist.append(float(f_wrapper(xk)))
                except Exception:
                    # ignora errori interni alla callback
                    pass
                try:
                    # registra norma del gradiente (con eventuale rumore se noisy=True)
                    gk = g_wrapper(xk)
                    grad_norms.append(float(np.linalg.norm(gk)))
                except Exception:
                    pass

            # opzioni: maxiter come da input
            start = time.perf_counter()
            res = minimize(fun=f_wrapper, x0=x0, jac=g_wrapper, method='BFGS',
                           callback=cb, options={'maxiter': max_iter, 'disp': False})
            elapsed = time.perf_counter() - start

            # estrai informazioni
            xmin = np.asarray(res.x, dtype=float)
            nit = getattr(res, 'nit', None) or res.get('nit', None) if isinstance(res, dict) else getattr(res, 'nit', None)

            # se callback non è stato chiamato mai, almeno registra il valore iniziale e finale
            if len(f_hist) == 0:
                try:
                    f_hist = [float(f_wrapper(x0)), float(f_wrapper(xmin))]
                except Exception:
                    f_hist = []
            
            # se non abbiamo norme dei gradienti registrate, calcoliamo almeno quelle in x0 e xmin
            if len(grad_norms) == 0:
                try:
                    g0 = g_wrapper(x0)
                    gmin = g_wrapper(xmin)
                    grad_norms = [float(np.linalg.norm(g0)), float(np.linalg.norm(gmin))]
                except Exception:
                    grad_norms = []
            
            fval = float(f_hist[-1]) if f_hist else float(_safe_eval(lambda x: f_wrapper(x), xmin))

            info.update({
                'x_min': xmin.tolist(),
                'f_val': fval,
                'f_history': [float(v) for v in f_hist],
                'grad_norms': [float(v) for v in grad_norms],
                'n_iter': nit,
                'elapsed': elapsed,
                'scipy_result': res
            })

        else:
            raise ValueError(f"Metodo non riconosciuto: {method}")

        info['status'] = 'success'

    except Exception as e:
        elapsed = time.perf_counter() - (start if start is not None else backup_start)
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

    Inoltre, raccoglie le storie dei valori di f e delle norme del gradiente per ciascun metodo
    e disegna i grafici usando plotting.plot_function_histories e plotting.plot_gradient_norm_histories.

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

    # raccolte per il plotting
    f_histories = []       # lista di liste di valori di f per ogni metodo
    grad_norms_histories = []  # lista di liste di norme del gradiente per ogni metodo
    method_labels = []   # etichette corrispondenti

    for method in methods:
        subrng = np.random.default_rng(rng_master.integers(1 << 30))
        print("----------------------------------------")
        print(f"Metodo: {method}")
        print(f"Problema: {prob_id}")
        if sif_params:
            print(f"SIF params: {sif_params}")
        print(f"Dimensione n: {n}")
        # print(f"Rumore: eps_f={eps_f}, eps_g={eps_g}")

        res = run_method_on_problem(p, method=method, eps_f=eps_f, eps_g=eps_g, rng=subrng, tol_factor=tol_factor, max_iter=max_iter)

        if res.get('status') == 'success':
            print(f"Punto minimo trovato (x): {res.get('x_min')}")
            print(f"Valore funzione: {res.get('f_val')}")
            print(f"Numero iterazioni: {res.get('n_iter')}")
            print(f"Tempo impiegato: {res.get('elapsed'):.6f} s")
        else:
            print(f"Metodo fallito con errore: {res.get('error')}")
        print("----------------------------------------")

        # raccolta dati per il plotting
        f_hist = None
        grad_norms_hist = None

        if 'f_history' in res and isinstance(res['f_history'], list):
            f_hist = [float(v) for v in res['f_history']]
        else:
            f_hist = []

        if 'grad_norms' in res and isinstance(res['grad_norms'], list):
            grad_norms_hist = [float(v) for v in res['grad_norms']]
        else:
            grad_norms_hist = []
            
        f_histories.append(f_hist)
        grad_norms_histories.append(grad_norms_hist)
        method_labels.append(method)

    # Disegna il grafico con le storie dei valori di f per i metodi eseguiti sul problema
    all_empty_f = all(len(h) == 0 for h in f_histories)
    if all_empty_f:
        print("Attenzione: nessuna storia dei valori della funzione disponibile per i metodi eseguiti. Niente da plottare.")
    else:
        plotting.plot_function_histories(f_histories, method_labels, problem_name=prob_id)

    # Disegna il grafico con le storie delle norme del gradiente per i metodi eseguiti sul problema
    all_empty_grad = all(len(h) == 0 for h in grad_norms_histories)
    if all_empty_grad:
        print("Attenzione: nessuna storia delle norme del gradiente disponibile per i metodi eseguiti. Niente da plottare.")
    else:
        plotting.plot_gradient_norm_histories(grad_norms_histories, method_labels, problem_name=prob_id, logy=True)



def repeat_timing(problem_name: str,
                  methods: List[str],
                  n_runs: int = 5,
                  eps_f: float = 1e-7,
                  eps_g: float = 1e-7,
                  seed: int = 42,
                  tol_factor: float = 10.0,
                  max_iter: int = 10000,
                  sif_params: Optional[Dict] = None,
                  quiet: bool = False) -> Dict[str, Dict]:
    """
    Esegue `n_runs` volte ogni metodo su `problem_name`, raccoglie i tempi di esecuzione
    e stampa la media (e deviazione standard) dei tempi per metodo.

    Parametri:
      - problem_name, methods, eps_f, eps_g, seed, tol_factor, max_iter, sif_params:
        come in run_and_print(...)
      - n_runs: numero di ripetizioni per metodo (intero >= 1)
      - quiet: se True sopprime l'output prodotto da run_method_on_problem durante le ripetizioni

    Ritorna:
      dict mapping method -> {
         'times': [t1, t2, ...],
         'mean': mean_time,
         'std': std_time,
         'success_count': number_of_successful_runs
      }
    """
    if n_runs < 1:
        raise ValueError("n_runs deve essere >= 1")

    rng_master = np.random.default_rng(seed)
    results_summary: Dict[str, Dict] = {}

    # Importa problema qui per evitare di ricaricarlo molte volte
    try:
        p = import_problem(problem_name, sif_params=sif_params)
    except Exception as e:
        print(f"Impossibile importare problema {problem_name}: {e}")
        return {}

    print(f"Ripetizione tempi: problema={problem_name}, runs per metodo={n_runs}")
    print(f"Metodi: {methods}")
    print("-----------------------------------------------------------")

    for method in methods:
        times = []
        success_count = 0

        for run_idx in range(n_runs):
            # generiamo un sub-seed per ogni run per garantire indipendenza e riproducibilità
            seed_run = rng_master.integers(1 << 30)
            subrng = np.random.default_rng(seed_run)

            # opzionalmente sopprimiamo l'output delle singole esecuzioni
            if quiet:
                fnull = io.StringIO()
                ctx = contextlib.redirect_stdout(fnull)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                # misuriamo tempo esternamente come fallback; run_method_on_problem restituisce comunque 'elapsed'
                t0 = time.perf_counter()
                res = run_method_on_problem(p, method=method, eps_f=eps_f, eps_g=eps_g,
                                            rng=subrng, tol_factor=tol_factor, max_iter=max_iter)
                t1 = time.perf_counter()

            # prendi preferibilmente il tempo riportato da run_method_on_problem, altrimenti usa la misura esterna
            t_measured = None
            if isinstance(res, dict) and 'elapsed' in res and res['elapsed'] is not None:
                try:
                    t_measured = float(res['elapsed'])
                except Exception:
                    t_measured = None
            if t_measured is None:
                t_measured = t1 - t0

            times.append(t_measured)
            if res.get('status') == 'success':
                success_count += 1

            # breve log per run (se non quiet)
            if not quiet:
                print(f"[{method}] run {run_idx+1}/{n_runs}: time={t_measured:.6f}s, status={res.get('status')}")

        times_arr = np.asarray(times, dtype=float)
        mean_t = float(np.mean(times_arr)) if times_arr.size > 0 else float('nan')
        std_t = float(np.std(times_arr, ddof=0)) if times_arr.size > 0 else float('nan')

        # stampa riepilogo per metodo
        print("-----------------------------------------------------------")
        print(f"Metodo: {method}")
        print(f"  runs = {n_runs}, success = {success_count}")
        print(f"  tempi (s): mean = {mean_t:.6f}, std = {std_t:.6f}")
        print(f"  singoli tempi: {[f'{t:.6f}' for t in times]}")
        print("-----------------------------------------------------------")

        results_summary[method] = {
            'times': [float(t) for t in times],
            'mean': mean_t,
            'std': std_t,
            'success_count': success_count,
        }

    return results_summary