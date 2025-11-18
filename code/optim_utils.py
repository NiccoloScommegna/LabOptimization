import numpy as np
from typing import Callable, Tuple, List, Dict, Optional
import logging


# -----------------------
# METODI HELPER PER VALUTAZIONI E NORMA
# -----------------------


def _eval_f_noisy(f: Callable[[np.ndarray], float], x: np.ndarray, eps_f: float, rng: np.random.Generator) -> float:
    """
    Valuta f(x) e aggiunge rumore uniforme in [-eps_f, eps_f].
    Valuta f(x) e aggiunge rumore gaussiano N(0, eps_f^2).
    Se eps_f <= 0 o eps_f è None, ritorna f(x) senza rumore.
    """
    val = float(f(x))
    if eps_f and eps_f > 0:
        # noise = rng.uniform(-eps_f, eps_f)  # Rumore uniforme
        noise = rng.normal(loc=0.0, scale=eps_f)  # Rumore gaussiano
        return val + noise
    return val


def _eval_g_noisy(g: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps_g: float, rng: np.random.Generator) -> np.ndarray:
    """
    Valuta g(x) e aggiunge rumore vettoriale uniforme su ogni componente in [-eps_g, eps_g].
    Valuta g(x) e aggiunge rumore gaussiano N(0, eps_g^2 I).
    Se eps_g <= 0 o eps_g è None, ritorna g(x) senza rumore.
    """
    gx = np.asarray(g(x), dtype=float)
    if eps_g and eps_g > 0:
        # noise = rng.uniform(-eps_g, eps_g, size=gx.shape)  # Rumore uniforme per ogni componente
        noise = rng.normal(loc=0.0, scale=eps_g, size=gx.shape)  # Rumore gaussiano per ogni componente
        return gx + noise
    return gx


# Funzione di norma vettoriale
def vecnorm(x: np.ndarray) -> float:
    return np.linalg.norm(x)



# -----------------------
# METODI BASE DI OTTIMIZZAZIONE
# -----------------------



# Metodo di Armijo per la ricerca del passo
def armijo_line_search(f: Callable[[np.ndarray], float],
                       g: Callable[[np.ndarray], np.ndarray],
                       xk: np.ndarray,
                       dk: np.ndarray,
                       alpha0: float = 1.0,
                       sigma: float = 0.5,
                       c1: float = 1e-4,
                       maxiter: int = 1000,
                       eps_f: Optional[float] = None,
                       eps_g: Optional[float] = None,
                       rng: Optional[np.random.Generator] = None) -> float:
    """
    Ricerca del passo secondo la condizione di Armijo:
    f(xk + alpha * dk) <= f(xk) + c1 * alpha * g(xk)^T * dk
    Ritorna il passo alpha trovato.
    dk deve essere una direzione di discesa (g(xk)^T * dk < 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    fk = _eval_f_noisy(f, xk, eps_f, rng)
    gk = _eval_g_noisy(g, xk, eps_g, rng)
    alpha = alpha0
    # Verifica che dk sia una direzione di discesa
    if np.dot(gk, dk) >= 0:
        raise ValueError("dk non è una direzione di discesa")
    
    iter = 0
    while _eval_f_noisy(f, xk + alpha * dk, eps_f, rng) > fk + c1 * alpha * np.dot(gk, dk) and iter < maxiter:
        alpha *= sigma
        iter += 1
    if iter == maxiter:
        print("Warning: line search di Armijo ha raggiunto il numero massimo di iterazioni")
    return alpha


def gradient_descent_armijo(f: Callable[[np.ndarray], float],
                            g: Callable[[np.ndarray], np.ndarray],
                            x0: np.ndarray,
                            tol: float = 1e-6,
                            maxiter: int = 100000,
                            alpha0: float = 1.0,
                            sigma: float = 0.5,
                            c1: float = 1e-4,
                            max_line_search_iter: int = 1000,
                            eps_f: Optional[float] = None,
                            eps_g: Optional[float] = None,
                            rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Dict]:
    """
    Metodo di discesa del gradiente con ricerca del passo secondo la condizione di Armijo.
    Ritorna (x_min, info) dove info è un dict con:
      - 'status': 'converged' | 'max_iter_reached' | 'bad_direction' | 'error'
      - 'nit' : numero di iterazioni completate
      - 'x_history' : lista di np.ndarray (punti)
      - 'f_history' : lista dei valori di f (float)
      - 'grad_norms' : lista delle norme del gradiente (float)
    """
    if rng is None:
        rng = np.random.default_rng()

    xk = np.asarray(x0, dtype=float).copy()

    # storici
    x_history: List[np.ndarray] = [xk.copy()]
    f_history: List[float] = [float(_eval_f_noisy(f, xk, eps_f, rng))]
    gk = _eval_g_noisy(g, xk, eps_g, rng)
    grad_norms: List[float] = [vecnorm(gk)]

    k = 0
    status = 'max_iter_reached'  # valore di default, sovrascritto se converge o altro

    try:
        while (vecnorm(gk) > tol) and (k < maxiter):
            dk = -gk
            # verifica che dk sia direzione di discesa
            if float(np.dot(gk, dk)) >= 0:
                status = 'bad_direction'
                break

            alpha = armijo_line_search(f=f, g=g, xk=xk, dk=dk, alpha0=alpha0,
                                       sigma=sigma, c1=c1, maxiter=max_line_search_iter,
                                       eps_f=eps_f, eps_g=eps_g, rng=rng)

            xk = xk + alpha * dk

            # aggiorna storici
            fx = float(_eval_f_noisy(f, xk, eps_f, rng))
            f_history.append(fx)
            x_history.append(xk.copy())

            gk = _eval_g_noisy(g, xk, eps_g, rng)
            grad_norms.append(vecnorm(gk))

            k += 1

        # stato finale
        if vecnorm(gk) <= tol:
            status = 'converged'
        elif status != 'bad_direction' and k >= maxiter:
            status = 'max_iter_reached'

    except Exception as e:
        # in caso di eccezione registriamo lo stato e rilanciamo l'errore nel dict
        status = 'error'
        logging.exception("Errore nella discesa del gradiente con Armijo: %s", e)

    info: Dict = {
        'status': status,
        'nit': k,
        'x_history': x_history,
        'f_history': f_history,
        'grad_norms': grad_norms
    }

    return xk, info


def check_strong_wolfe(f: Callable[[np.ndarray], float], 
                       g: Callable[[np.ndarray], np.ndarray], 
                       xk: np.ndarray, 
                       dk: np.ndarray, 
                       alpha: float, 
                       c1: float = 1e-4, 
                       c2: float = 0.9,
                       eps_f: Optional[float] = None,
                       eps_g: Optional[float] = None,
                       rng: Optional[np.random.Generator] = None) -> Tuple[bool, bool]:
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
    if rng is None:
        rng = np.random.default_rng()

    xk = np.asarray(xk, dtype=float)
    dk = np.asarray(dk, dtype=float)

    phi0 = _eval_f_noisy(f, xk, eps_f, rng)
    phi_alpha = _eval_f_noisy(f, xk + alpha * dk, eps_f, rng)

    g0 = _eval_g_noisy(g, xk, eps_g, rng)
    g_alpha = _eval_g_noisy(g, xk + alpha * dk, eps_g, rng)

    dphi0 = float(np.dot(g0, dk))
    dphi_alpha = float(np.dot(g_alpha, dk))

    armijo_ok = phi_alpha <= phi0 + c1 * alpha * dphi0
    strong_grad_ok = abs(dphi_alpha) <= c2 * abs(dphi0)

    return bool(armijo_ok), bool(strong_grad_ok)


def strong_wolfe_line_search(f: Callable[[np.ndarray], float],
                             g: Callable[[np.ndarray], np.ndarray],
                             xk: np.ndarray,
                             dk: np.ndarray,
                             c1: float = 1e-4,
                             c2: float = 0.9,
                             alpha_l: float = 0.0,
                             alpha_u: float = 1.0,
                             max_iter: int = 500,
                             eps_f: Optional[float] = None,
                             eps_g: Optional[float] = None,
                             rng: Optional[np.random.Generator] = None) -> Tuple[Optional[float], Dict]:
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
    if rng is None:
        rng = np.random.default_rng()

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
    g0 = _eval_g_noisy(g, xk, eps_g, rng)
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
        armijo_ok, strong_grad_ok = check_strong_wolfe(f=f, g=g, xk=xk, dk=dk, alpha=alpha, c1=c1, c2=c2, eps_f=eps_f, eps_g=eps_g, rng=rng)
        info['armijo'] = bool(armijo_ok)
        info['strong_grad'] = bool(strong_grad_ok)
        info['nit'] = j + 1

        if armijo_ok and strong_grad_ok:
            info['status'] = 'found'
            return alpha, info

        # calcola phi e derivata esplicitamente (per le regole di aggiornamento)
        phi0 = _eval_f_noisy(f, xk, eps_f, rng)
        phi_alpha = _eval_f_noisy(f, xk + alpha * dk, eps_f, rng)
        dphi_alpha = float(np.dot(_eval_g_noisy(g, xk + alpha * dk, eps_g, rng), dk))

        # 1) Se φ(α) > φ(0) + c1 α φ'(0) -> αu = α
        if phi_alpha > phi0 + c1 * alpha * dphi0:
            alpha_u = alpha
            # next iter
            continue

        # 2) φ(α) ≤ φ(0) + c1 α φ'(0)  e  φ'(α) < c2 φ'(0) -> αl = α
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
    return None, info


def bfgs_strong_wolfe(f: Callable[[np.ndarray], float],
                      g: Callable[[np.ndarray], np.ndarray],
                      x0: np.ndarray,
                      c1: float = 1e-4,
                      c2: float = 0.9,
                      tol: float = 1e-6,
                      max_iter: int = 10000,
                      alpha_l: float = 0.0,
                      alpha_u: float = 1.0,
                      max_line_search_iter: int = 500,
                      eps_f: Optional[float] = None,
                      eps_g: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Dict]:
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
    if rng is None:
        rng = np.random.default_rng()
    
    xk = np.asarray(x0, dtype=float)
    n = xk.size
    Bk = np.eye(n)  # B0 definita positiva (matrice identità)
    gk = _eval_g_noisy(g, xk, eps_g, rng)
    fk = _eval_f_noisy(f, xk, eps_f, rng)

    k = 0
    x_history = [xk.copy()]
    f_history = [fk]
    grad_norms = [vecnorm(gk)]
    
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
        alpha, ls_info = strong_wolfe_line_search(f=f, g=g, xk=xk, dk=dk, c1=c1, c2=c2, alpha_l=alpha_l, alpha_u=alpha_u, max_iter=max_line_search_iter, eps_f=eps_f, eps_g=eps_g, rng=rng)
        # se non trovato, usa Armijo come fallback
        if alpha is None:
            alpha = armijo_line_search(f=f, g=g, xk=xk, dk=dk, eps_f=eps_f, eps_g=eps_g, rng=rng)
            print(f"Warning: strong Wolfe line search failed at iteration {k}, using Armijo fallback with alpha={alpha}")

        # aggiornamento
        x_next = xk + alpha * dk
        g_next = _eval_g_noisy(g, x_next, eps_g, rng)
        fk_next = _eval_f_noisy(f, x_next, eps_f, rng)

        yk = g_next - gk
        sk = x_next - xk

        # aggiornamento di Bk (formula standard)
        sy = float(np.dot(sk, yk))
        Bs = Bk @ sk
        sBs = float(np.dot(sk, Bs))
        # if sy <= 1e-30 or sBs <= 1e-30:
        #     # evita divisioni per zero o valori non curvati
        #     info['status'] = 'curvature_condition_failed'
        #     print(f"Warning: curvature condition failed at iteration {k}. sy={sy}, sBs={sBs}")
        #     break

        term1 = np.outer(yk, yk) / sy
        term2 = np.outer(Bs, Bs) / sBs
        Bk = Bk + term1 - term2

        # aggiorna valori
        xk = x_next
        gk = g_next
        fk = fk_next

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



# -----------------------
# METODI NOISE TOLERANT
# -----------------------



def check_strong_wolfe_noise_tolerant(f: Callable[[np.ndarray], float], 
                                      g: Callable[[np.ndarray], np.ndarray],
                                      xk: np.ndarray, 
                                      dk: np.ndarray, 
                                      alpha: float, 
                                      beta: float,
                                      c1: float = 1e-4, c2: float = 0.9, c3: float = 0.5,
                                      eps_f: float = 1e-5,
                                      eps_g: float = 1e-5,
                                      rng: Optional[np.random.Generator] = None) -> Tuple[bool, bool, bool]:
    """
    Controlla se le condizioni di strong Wolfe noise tolerant sono soddisfatte per il passo alpha e beta.
    Ritorna una tupla di booleani (Armijo_ok, strong_grad_ok, noise_ok).

    Condizioni di strong Wolfe noise tolerant:
    1. Armijo: f(xk + alpha * dk) <= f(xk) + c1 * alpha * g(xk)^T * dk
    2. Strong gradient: |g(xk + alpha * dk)^T * dk| <= c2 * |g(xk)^T * dk|
    3. Noise tolerant: (g(xk + beta * dk) - g(xk))^T * dk >= 2 * (1 + c3) * epsilon_g * ||dk||
    con c1 in (0, 1/2), c2 in (c1, 1) e c3 > 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    xk = np.asarray(xk, dtype=float)
    dk = np.asarray(dk, dtype=float)

    # valori rumorosi
    phi0 = _eval_f_noisy(f, xk, eps_f, rng)
    phi_alpha = _eval_f_noisy(f, xk + alpha * dk, eps_f, rng)

    g0 = _eval_g_noisy(g, xk, eps_g, rng)
    g_alpha = _eval_g_noisy(g, xk + alpha * dk, eps_g, rng)

    dphi0 = float(np.dot(g0, dk))
    dphi_alpha = float(np.dot(g_alpha, dk))

    armijo_ok = phi_alpha <= phi0 + c1 * alpha * dphi0
    strong_grad_ok = abs(dphi_alpha) <= c2 * abs(dphi0)

    # noise control uses beta (se None -> false)
    if beta is None:
        noise_ok = False
    else:
        g_beta = _eval_g_noisy(g, xk + beta * dk, eps_g, rng)
        lhs = float(np.dot(g_beta - g0, dk))
        rhs = 2.0 * (1.0 + c3) * float(eps_g) * vecnorm(dk)
        noise_ok = lhs >= rhs

    return bool(armijo_ok), bool(strong_grad_ok), bool(noise_ok)


def split_phase(f: Callable[[np.ndarray], float],
                g: Callable[[np.ndarray], np.ndarray],
                x: np.ndarray,
                d: np.ndarray,
                alpha_init: float,
                beta_init: Optional[float] = None,
                eps_f: float = 1e-5,
                eps_g: float = 1e-5,
                rng: Optional[np.random.Generator] = None,
                c1: float = 1e-4,
                c3: float = 0.5,
                max_backtrack: int = 50,
                max_doublings: int = 30,
                alpha_min: float = 1e-20,
                beta_max: float = 1e20) -> Tuple[float, float, Dict]:
    """
    Algoritmo 4.2 - Split Phase (lengthening).
    

    Input:
      f, g      : callable per la funzione obiettivo e il gradiente (vettori)
      x, d      : punto corrente e direzione di ricerca (array numpy)
      alpha_init: lunghezza iniziale del passo (scalare)
      beta_init : parametro di allungamento iniziale (se None, beta = 1.0)
      eps_f     : livello di rumore per le valutazioni della funzione (scalare)
      eps_g     : livello di rumore per le valutazioni del gradiente (scalare)
      c1, c3    : costanti dell'algoritmo (c1 non utilizzato rigorosamente tranne che per Armijo RHS)
      max_backtrack : numero massimo di iterazioni per dividere alpha per 10
      max_doublings  : numero massimo di raddoppi per beta
      alpha_min, beta_max : limiti di sicurezza per evitare under/overflow

    Output:
      alpha (float), beta (float), info (dict):
        info contiene 'status', 'n_f', 'n_g', 'alpha_history', 'beta_history'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    # Diagnostica
    n_f = 0
    n_g = 0
    alpha_history = []
    beta_history = []

    # Precalcola valori iniziali
    fx = _eval_f_noisy(f, x, eps_f, rng); n_f += 1
    gx = _eval_g_noisy(g, x, eps_g, rng); n_g += 1
    gp0 = float(np.dot(gx, d))
    d_norm = float(np.linalg.norm(d))

    # Controllo che alpha sia positivo
    alpha = float(alpha_init if alpha_init is not None else 1.0)
    if alpha <= 0:
        alpha = 1.0

    # BACKTRACK: while f(x+alpha p) > f(x) + c1 alpha g(x)^T p : alpha = alpha / 10
    bt_iters = 0
    status = 'ok'
    while bt_iters < max_backtrack:
        alpha_history.append(alpha)
        f_alpha = _eval_f_noisy(f, x + alpha * d, eps_f, rng); n_f += 1
        # Condizione di Armijo
        if f_alpha <= fx + c1 * alpha * gp0:
            break
        # Altrimenti, riduci alpha di 1/10
        alpha /= 10.0
        bt_iters += 1
        if alpha < alpha_min:
            status = 'alpha_too_small'
            break

    if bt_iters >= max_backtrack and status == 'ok':
        status = 'backtrack_maxiters'

    # LENGTHEN: raddoppia beta while (g(x+beta p)-g(x))^T p < 2(1+c3) eps_g ||p||
    if beta_init is None:
        beta = 1.0
    else:
        beta = float(beta_init)
        if beta <= 0:
            beta = 1.0

    doublings = 0
    lhs = None  # left-hand side dell'ultima valutazione
    rhs = None  # right-hand side dell'ultima valutazione
    while doublings < max_doublings:
        beta_history.append(beta)
        g_beta = _eval_g_noisy(g, x + beta * d, eps_g, rng); n_g += 1
        lhs = float(np.dot(g_beta - gx, d))
        rhs = 2.0 * (1.0 + c3) * float(eps_g) * d_norm
        if lhs >= rhs:
            # Condizione di soddisfazione del rumore
            break
        # Altrimenti, raddoppia beta
        beta *= 2.0
        doublings += 1
        if beta > beta_max or not np.isfinite(beta):
            status = 'beta_too_large'
            break

    if doublings >= max_doublings and status == 'ok':
        status = 'doubling_maxiters'

    info = {
        'status': status,
        'n_f': n_f,
        'n_g': n_g,
        'alpha_history': alpha_history,
        'beta_history': beta_history,
        'final_alpha': alpha,
        'final_beta': beta,
        'lhs_last': lhs if 'lhs' in locals() else None,
        'rhs_thresh': rhs if 'rhs' in locals() else None,
    }
    return alpha, beta, info


def strong_wolfe_noise_tolerant_line_search(f: Callable[[np.ndarray], float],
                                            g: Callable[[np.ndarray], np.ndarray],
                                            x: np.ndarray,
                                            d: np.ndarray,
                                            alpha_init: float = 1.0,
                                            beta_init: Optional[float] = None,
                                            eps_f: float = 1e-5,
                                            eps_g: float = 1e-5,
                                            rng: Optional[np.random.Generator] = None,
                                            c1: float = 1e-4,
                                            c2: float = 0.9,
                                            c3: float = 0.5,
                                            Nsplit: int = 30,
                                            max_backtrack: int = 50,
                                            max_doublings: int = 30,
                                            alpha_min: float = 1e-20,
                                            beta_max: float = 1e20) -> Tuple[float, Optional[float], Dict]:
    """
    Algoritmo 4.1 - Fase iniziale della ricerca del passo Armijo-Wolfe a due fasi con allungamento.

    Input:
      f, g        : callable per la funzione obiettivo e il gradiente (vettori)
      x, d        : punto corrente e direzione di ricerca (array numpy)
      alpha_init  : passo iniziale (default 1.0)
      beta_init   : parametro di allungamento iniziale (default None)
      eps_f       : livello di rumore per le valutazioni della funzione (scalare)
      eps_g       : livello di rumore per le valutazioni del gradiente (scalare)
      rng         : generatore di numeri casuali (se None, ne viene creato uno nuovo)
      c1, c2, c3  : costanti (0 < c1 < c2 < 1, c3 > 0)
      Nsplit      : numero massimo di iterazioni prima della split-phase
      max_backtrack : massimo numero di backtrack nella split_phase
      max_doublings  : massimo numero di raddoppi nella split_phase
      alpha_min, beta_max : limiti di sicurezza per evitare under/overflow nella split_phase

    Restituisce:
      alpha (float) : alpha finale restituito dalla fase iniziale o dalla split_phase
      beta  (float o None) : beta, se la fase iniziale ha trovato un alpha accettabile (alpha = beta),
                             altrimenti il valore restituito dalla split_phase, o None se non determinato.
      info  (dict)  : informazioni diagnostiche che includono 'status', 'n_iter', 'l', 'u',
                      l'ultima phi calcolata e le derivate direzionali.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    # Precalcola f(x) e g(x)
    fx = _eval_f_noisy(f, x, eps_f, rng)
    gx = _eval_g_noisy(g, x, eps_g, rng)
    gp0 = float(np.dot(gx, d))  # g(x)^T p
    d_norm = float(np.linalg.norm(d))

    # Inizializza
    l = 0.0
    u = np.inf
    alpha = float(alpha_init)
    beta = None if beta_init is None else float(beta_init)

    info: Dict = {
        'status': None,
        'n_iter': 0,
        'l': l,
        'u': u,
        'last_phi': None,
        'last_gdotp': None,
        'gp0': gp0
    }

    # Main loop (Fase Iniziale)
    for i in range(Nsplit):
        info['n_iter'] = i + 1
        # Valuta phi(alpha) e il gradiente in x+alpha p
        x_alpha = x + alpha * d
        phialpha = _eval_f_noisy(f, x_alpha, eps_f, rng)
        g_alpha = _eval_g_noisy(g, x_alpha, eps_g, rng)
        gdiff_dot_d = float(np.dot(g_alpha - gx, d))  # (g(x + alpha p)-g(x))^T p
        g_alpha_dot_d = float(np.dot(g_alpha, d))     # g(x + alpha p)^T p

        info['last_phi'] = phialpha
        info['last_gdotp'] = g_alpha_dot_d
        info['l'] = l; info['u'] = u

        # Se la condizione di Armijo fallisce
        if phialpha > fx + c1 * alpha * gp0:
            # Armijo fallisce -> riduci upper bound
            u = alpha
            alpha = 0.5 * (u + l)
            continue

        # Controllo del rumore
        if abs(gdiff_dot_d) < 2.0 * (1.0 + c3) * eps_g * d_norm:
            # Controllo del rumore fallisce -> entra nella Fase di Split
            info['status'] = 'split_required'
            alpha_sp, beta_sp, split_info = split_phase(f=f, g=g,
                                                        x=x, d=d,
                                                        alpha_init=alpha,
                                                        beta_init=beta_init,
                                                        eps_f=eps_f,
                                                        eps_g=eps_g,
                                                        rng=rng,
                                                        c1=c1, c3=c3,
                                                        max_backtrack=max_backtrack,
                                                        max_doublings=max_doublings,
                                                        alpha_min=alpha_min,
                                                        beta_max=beta_max)
            # Aggiorna info con i dettagli della split phase
            info.update(split_info if isinstance(split_info, dict) else {})
            info['status'] = 'split_called'
            return alpha_sp, beta_sp, info
        # Se la condizione di Wolfe fallisce (condizione del gradiente insufficiente)
        if g_alpha_dot_d < c2 * gp0:
            l = alpha
            if np.isinf(u):
                # Aggiorna alpha raddoppiandolo
                alpha = 2.0 * alpha
            else:
                alpha = 0.5 * (u + l)
            continue

        # Soddisfa tutte le condizioni (Armijo, Wolfe, Noise)
        beta = alpha
        info['status'] = 'initial_phase_success'
        info['l'] = l; info['u'] = u
        return alpha, beta, info

    # Dopo Nsplit iterazioni, entra nella split phase
    info['status'] = 'split_required_maxiters'
    alpha_sp, beta_sp, split_info = split_phase(f=f, g=g,
                                                x=x, d=d,
                                                alpha_init=alpha,
                                                beta_init=beta_init,
                                                eps_f=eps_f,
                                                eps_g=eps_g,
                                                rng=rng,
                                                c1=c1, c3=c3,
                                                max_backtrack=max_backtrack,
                                                max_doublings=max_doublings,
                                                alpha_min=alpha_min,
                                                beta_max=beta_max)
    # Aggiorna info con i dettagli della split phase
    info.update(split_info if isinstance(split_info, dict) else {})
    info['status'] = 'split_called_maxiters'
    return alpha_sp, beta_sp, info


def bfgs_strong_wolfe_noise_tolerant(f: Callable[[np.ndarray], float],
                                     g: Callable[[np.ndarray], np.ndarray],
                                     x0: np.ndarray,
                                     c1: float = 1e-4,
                                     c2: float = 0.9,
                                     c3: float = 0.5,
                                     tol: float = 1e-6,
                                     max_iter: int = 10000,
                                     eps_f: float = 1e-5,
                                     eps_g: float = 1e-5,
                                     rng: Optional[np.random.Generator] = None,
                                     alpha_init: float = 1.0,
                                     beta_init: Optional[float] = 1.0,
                                     Nsplit: int = 30,
                                     max_backtrack: int = 50,
                                     max_doublings: int = 30,
                                     alpha_min: float = 1e-20,
                                     beta_max: float = 1e20) -> Tuple[np.ndarray, Dict]:
    """
    Implementa il metodo BFGS tollerante al rumore seguendo il seguente pseudocodice:

    Dati: x0 ∈ Rn, B0 definita positiva, 0<c1<c2<1, c3>0, tol, eps_f, eps_g.
    Poni k=0.
    While |∇f(xk)| > tol
        dk = -(Bk^-1) ∇f(xk)
        Determina alpha_k con la funzione strong_wolfe_noise_tolerant_line_search()
        xk+1 = xk + alpha_k dk
        yk = ∇f(xk + beta_k dk) - ∇f(xk) [equivalente a yk = g(xk + beta_k dk) - g(xk)]
        sk = beta_k dk  # con beta_k parametro di allungamento
        Bk+1 = Bk + (yk yk^T)/(sk^T yk) - (Bk sk sk^T Bk)/(sk^T Bk sk)
        k = k + 1
    End While
    """
    if rng is None:
        rng = np.random.default_rng()

    # Inizializzazioni
    xk = np.asarray(x0, dtype=float)
    n = xk.size
    Bk = np.eye(n)  # B0 definita positiva (matrice identità)
    fk = _eval_f_noisy(f, xk, eps_f, rng)
    gk = _eval_g_noisy(g, xk, eps_g, rng)


    k = 0
    x_history = [xk.copy()]
    f_history = [fk]
    grad_norms = [vecnorm(gk)]

    info = {
        'status': None,
        'nit': 0,
        'x_history': x_history,
        'f_history': f_history,
        'grad_norms': grad_norms,
        'ls_history': []
    }

    while vecnorm(gk) > tol and k < max_iter:
        # direzione di discesa
        dk = -np.linalg.solve(Bk, gk)

        # ricerca del passo
        alpha, beta, ls_info = strong_wolfe_noise_tolerant_line_search(f=f, g=g, x=xk, d=dk, 
                                                                       alpha_init=alpha_init, 
                                                                       beta_init=beta_init,
                                                                       eps_f=eps_f,
                                                                       eps_g=eps_g,
                                                                       rng=rng,
                                                                       c1=c1, c2=c2, c3=c3,
                                                                       Nsplit=Nsplit, 
                                                                       max_backtrack=max_backtrack, 
                                                                       max_doublings=max_doublings, 
                                                                       alpha_min=alpha_min, 
                                                                       beta_max=beta_max)
        info['ls_history'].append(ls_info if isinstance(ls_info, dict) else {})

        # se non trovato, usa Armijo come fallback
        if alpha is None:
            alpha_local = alpha_init
            fx = _eval_f_noisy(f, xk, eps_f, rng)
            gp = float(np.dot(gk, dk))
            armijo_found = False
            for _ in range(1000):
                f_trial = _eval_f_noisy(f, xk + alpha_local * dk, eps_f, rng)
                if f_trial <= fx + c1 * alpha_local * gp + eps_f:
                    armijo_found = True
                    break
                alpha_local *= 0.5
                if alpha_local < 1e-16:
                    break
            alpha = float(alpha_local) if armijo_found else None

        # se alpha è ancora None -> piccolo fallback e salto dell'aggiornamento
        if alpha is None:
            info['status'] = 'line_search_failed'
            # prendo uno step molto piccolo per aggiornare xk e gk
            tiny = 1e-12
            x_next = xk + tiny * dk
            fk_next = _eval_f_noisy(f, x_next, eps_f, rng)
            g_next = _eval_g_noisy(g, x_next, eps_g, rng)
            # non aggiorno Bk
            info['status_detail'] = 'tiny_step_and_skip_update'
            # aggiornamento iterazione per i criteri di terminazione
            xk = x_next
            gk = g_next
            fk = fk_next
            info['nit'] = k + 1
            info['x_history'].append(xk.copy()); info['f_history'].append(fk); info['grad_norms'].append(vecnorm(gk))
            break


        # aggiornamento
        x_next = xk + alpha * dk
        fk_next = _eval_f_noisy(f, x_next, eps_f, rng)
        g_next = _eval_g_noisy(g, x_next, eps_g, rng)
        
        # Se beta non è stato trovato, usiamo l'aggiornamento classico di BFGS come fallback
        if beta is None:
            print("Warning: beta not found in line search; using classic BFGS update.")
            yk = g_next - gk
            sk = x_next - xk
        else:
            # XXX: da pseudocodice sembra non usare il nuovo punto calcolato x_next per aggiornare yk e sk

            g_beta = _eval_g_noisy(g, xk + beta * dk, eps_g, rng)
            yk = g_beta - gk
            sk = beta * dk

            # Personalmente, trovo più coerente usare x_next come fa BFGS classico, per aggiornare yk e sk
            # g_beta = _eval_g_noisy(g, x_next + beta * dk, eps_g, rng)
            # yk = g_beta - gk  # yk calcolato con il parametro di allungamento beta
            # sk = beta * dk  # sk calcolato con il parametro di allungamento beta

        # aggiornamento di Bk (formula standard)
        sy = float(np.dot(sk, yk))
        Bs = Bk @ sk
        sBs = float(np.dot(sk, Bs))
        # if sy <= 1e-30 or sBs <= 1e-30:
        #     # evita divisioni per zero o valori non curvati
        #     info['status'] = 'curvature_condition_failed'
        #     print(f"Warning: curvature condition failed at iteration {k}. sy={sy}, sBs={sBs}")
        #     break

        term1 = np.outer(yk, yk) / sy
        term2 = np.outer(Bs, Bs) / sBs
        Bk = Bk + term1 - term2

        # aggiorna valori
        xk = x_next
        gk = g_next
        fk = fk_next

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