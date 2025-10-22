import numpy as np
from typing import Callable, Tuple, List, Dict


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
                       gamma: float = 1e-4,
                       maxiter: int = 1000) -> float:
    """
    Ricerca del passo secondo la condizione di Armijo:
    f(xk + alpha * dk) <= f(xk) + gamma * alpha * g(xk)^T * dk
    Ritorna il passo alpha trovato.
    dk deve essere una direzione di discesa (g(xk)^T * dk < 0).
    """

    fk = f(xk)
    gk = g(xk)
    alpha = alpha0
    # Verifica che dk sia una direzione di discesa
    if np.dot(gk, dk) >= 0:
        raise ValueError("dk non è una direzione di discesa")
    
    iter = 0
    while f(xk + alpha * dk) > fk + gamma * alpha * np.dot(gk, dk) and iter < maxiter:
        alpha *= sigma
        iter += 1
    if iter == maxiter:
        print("Warning: line search di Armijo ha raggiunto il numero massimo di iterazioni")
    return alpha


# Metodo di discesa del gradiente con ricerca del passo di Armijo
def gradient_descent_armijo(f: Callable[[np.ndarray], float],
                            g: Callable[[np.ndarray], np.ndarray],
                            x0: np.ndarray,
                            tol: float = 1e-6,
                            maxiter: int = 100000) -> Tuple[np.ndarray, List[float]]:
    """
    Metodo di discesa del gradiente con ricerca del passo secondo la condizione di Armijo.
    Ritorna il punto minimo trovato e la lista dei valori della funzione obiettivo ad ogni iterazione.
    """
    xk = x0
    f_values = [f(xk)]
    iter = 0
    while (vecnorm(g(xk)) > tol) and (iter < maxiter):
        dk = -g(xk)
        alpha = armijo_line_search(f, g, xk, dk)
        xk = xk + alpha * dk
        f_values.append(f(xk))
        iter += 1
    if iter == maxiter:
        print("Warning: discesa del gradiente ha raggiunto il numero massimo di iterazioni")
    return xk, f_values


def check_strong_wolfe(f, g, xk, dk, alpha, c1=1e-4, c2=0.9) -> Tuple[bool, bool]:
    """
    Controlla se le condizioni di strong Wolfe sono soddisfatte per il passo alpha.
    Ritorna una tupla di booleani (Armijo_ok, strong_grad_ok).

    Condizioni di strong Wolfe:
    1. Armijo: f(xk + alpha * dk) <= f(xk) + c1 * alpha * g(xk)^T * dk
    2. Strong gradient: |g(xk + alpha * dk)^T * dk| <= c2 * |g(xk)^T * dk|
    con c1 in (0, 1/2) e c2 in (c1, 1).

    Le condizioni di strong Wolfe possono anche essere riscritte come:
    1. Armijo: φ(alpha) <= φ(0) + c1 * alpha * φ'(0)
    2. Strong gradient: |φ'(alpha)| <= c2 * |φ'(0)|

    Quindi: 
    φ(alpha) = f(xk + alpha * dk),
    φ(0) = f(xk),
    φ'(0) = g(xk)^T * dk,
    φ'(alpha) = g(xk + alpha * dk)^T * dk

    """
    phi0 = f(xk)
    phi_alpha = f(xk + alpha * dk)
    g0 = g(xk)
    dphi0 = float(np.dot(g0, dk))
    dphi_alpha = float(np.dot(g(xk + alpha * dk), dk))
    
    armijo_ok = phi_alpha <= phi0 + c1 * alpha * dphi0
    strong_grad_ok = abs(dphi_alpha) <= c2 * abs(dphi0)
    
    return armijo_ok, strong_grad_ok


def strong_wolfe_line_search(f: Callable[[np.ndarray], float],
                             g: Callable[[np.ndarray], np.ndarray],
                             xk: np.ndarray,
                             dk: np.ndarray,
                             c1: float = 1e-4,
                             c2: float = 0.9,
                             alpha_l: float = 0.0,
                             alpha_u: float = 1.0,
                             max_iter: int = 500) -> Tuple[float, Dict]:
    """
    Ricerca del passo che usa il punto centrale dell'intervallo
    [alpha_l, alpha_u] ad ogni iterazione e aggiorna i bound secondo le regole:
      1) Se φ(alpha) > φ(0) + c1 alpha φ'(0) -> alpha_u = alpha
      2) Se φ(alpha) ≤ φ(0) + c1 alpha φ'(0)  e  φ'(alpha) < c2 φ'(0) -> alpha_l = alpha
      3) Se φ(alpha) ≤ φ(0) + c1 alpha φ'(0)  e  φ'(alpha) > c2 |φ'(0)| -> alpha_u = alpha

    Ritorna (alpha_star, info) dove info contiene:
      - 'status' : 'found' | 'maxiter' | 'bad_direction'
      - 'nit' : numero di iterazioni
      - 'alpha_history' : lista dei candidate alpha provati
      - 'armijo' / 'strong_grad' : ultimo controllo booleano (se applicabile)
    """
    xk = np.asarray(xk, dtype=float)
    dk = np.asarray(dk, dtype=float)

    info = {
        'status': 'maxiter',
        'nit': 0,
        'alpha_history': [],
        'armijo': False,
        'strong_grad': False,
    }

    # assicurati direzione di discesa; se non lo è, forziamo -g
    g0 = g(xk)
    dphi0 = float(np.dot(g0, dk))
    if dphi0 >= 0:
        # non è una direzione di discesa: forziamo dk = -g
        dk = -g0
        dphi0 = float(np.dot(g0, dk))
        info['status'] = 'bad_direction'
        # continuiamo comunque; l'algoritmo può ancora cercare alpha
        # TODO: potremmo anche scegliere di uscire qui

    # ciclo principale: prendere il punto centrale (midpoint)
    for j in range(max_iter):
        alpha = 0.5 * (alpha_l + alpha_u)
        info['alpha_history'].append(alpha)

        # Controllo forte-Wolfe con la funzione di verifica
        armijo_ok, strong_grad_ok = check_strong_wolfe(f, g, xk, dk, alpha, c1=c1, c2=c2)
        info['armijo'] = bool(armijo_ok)
        info['strong_grad'] = bool(strong_grad_ok)
        info['nit'] = j + 1

        if armijo_ok and strong_grad_ok:
            info['status'] = 'found'
            return alpha, info

        # calcola phi e derivata esplicitamente (per le regole di aggiornamento)
        phi0 = f(xk)
        phi_alpha = f(xk + alpha * dk)
        dphi_alpha = float(np.dot(g(xk + alpha * dk), dk))

        # 1) Se φ(α) > φ(0) + c1 α φ'(0) -> αu = α
        if phi_alpha > phi0 + c1 * alpha * dphi0:
            alpha_u = alpha
            # next iter
            continue

        # 2) φ(α) ≤ φ(0) + c1 α φ'(0)  e  φ'(α) < c2 φ'(0) -> αl = α
        #    (attenzione: dphi0 può essere negativo; seguiamo il tuo pseudocodice)
        if (phi_alpha <= phi0 + c1 * alpha * dphi0) and (dphi_alpha < c2 * dphi0):
            alpha_l = alpha
            continue

        # 3) φ(α) ≤ φ(0) + c1 α φ'(0)  e  φ'(α) > c2 |φ'(0)| -> αu = α
        if (phi_alpha <= phi0 + c1 * alpha * dphi0) and (dphi_alpha > c2 * abs(dphi0)):
            alpha_u = alpha
            continue

        # Se nessuna delle regole è scattata (caso raro), interrompiamo e restituiamo il candidato attuale
        info['status'] = 'no_rule_matched'
        return alpha, info

    # se non abbiamo trovato alpha soddisfacente entro max_iter
    info['status'] = 'maxiter'
    return 0.5 * (alpha_l + alpha_u), info


def bfgs_strong_wolfe(f: Callable[[np.ndarray], float],
                      g: Callable[[np.ndarray], np.ndarray],
                      x0: np.ndarray,
                      c1: float = 1e-4,
                      c2: float = 0.9,
                      tol: float = 1e-6,
                      max_iter: int = 10000) -> Tuple[np.ndarray, Dict]:
    """
    Implementa il metodo BFGS seguendo il seguente pseudocodice:
    
    Dati: x0 ∈ Rn, B0 definita positiva, 0<c1<c2<1, tol.
    Poni k=0.
    While |∇f(xk)| > tol
        dk = -(Bk^-1) ∇f(xk)
        Determina alpha_k con la funzione strong_wolfe_line_search()
        xk+1 = xk + alpha_k dk
        yk = ∇f(xk+1) - ∇f(xk)
        sk = xk+1 - xk
        Bk+1 = Bk + (yk yk^T)/(sk^T yk) - (Bk sk sk^T Bk)/(sk^T Bk sk)
        k = k + 1
    End While
    """
    xk = np.asarray(x0, dtype=float)
    n = xk.size
    Bk = np.eye(n)  # B0 definita positiva (matrice identità)
    gk = g(xk)
    fk = f(xk)

    k = 0
    x_history = [xk.copy()]
    f_history = [fk]
    grad_norms = [np.linalg.norm(gk)]
    
    info = {
        'status': None,
        'nit': 0,
        'x_history': x_history,
        'f_history': f_history,
        'grad_norms': grad_norms
    }

    while vecnorm(gk) > tol and k < max_iter:
        # direzione di discesa
        dk = -np.linalg.solve(Bk, gk)

        # ricerca del passo
        alpha, ls_info = strong_wolfe_line_search(f, g, xk, dk, c1=c1, c2=c2)

        # aggiornamento
        x_next = xk + alpha * dk
        g_next = g(x_next)
        yk = g_next - gk
        sk = x_next - xk

        # aggiornamento di Bk (formula standard)
        sy = float(np.dot(sk, yk))
        Bs = Bk @ sk
        sBs = float(np.dot(sk, Bs))
        if sy <= 1e-12 or sBs <= 1e-12:
            # evita divisioni per zero o valori non curvati
            info['status'] = 'curvature_condition_failed'
            break

        term1 = np.outer(yk, yk) / sy
        term2 = np.outer(Bs, Bs) / sBs
        Bk = Bk + term1 - term2

        # aggiorna valori
        xk = x_next
        gk = g_next
        fk = f(xk)

        k += 1
        info['nit'] = k
        info['x_history'].append(xk.copy())
        info['f_history'].append(fk)
        info['grad_norms'].append(vecnorm(gk))

    if vecnorm(gk) <= tol:
        info['status'] = 'converged'
    elif k >= max_iter:
        info['status'] = 'max_iter_reached'

    return xk, info