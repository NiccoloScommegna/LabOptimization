import numpy as np
from typing import Callable, Tuple, List, Dict, Optional


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
                       c1: float = 1e-4,
                       maxiter: int = 1000) -> float:
    """
    Ricerca del passo secondo la condizione di Armijo:
    f(xk + alpha * dk) <= f(xk) + c1 * alpha * g(xk)^T * dk
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
    while f(xk + alpha * dk) > fk + c1 * alpha * np.dot(gk, dk) and iter < maxiter:
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
                             max_iter: int = 500) -> Tuple[Optional[float], Dict]:
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
    return None, info


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
        # se non trovato, usa Armijo come fallback
        if alpha is None:
            alpha, armijo_line_search_info = armijo_line_search(f, g, xk, dk)

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



# -----------------------
# METODI NOISE TOLERANT
# -----------------------

def check_strong_wolfe_noise_tolerant(f: callable[[np.ndarray], float], 
                                      g: callable[[np.ndarray], np.ndarray],
                                      xk: np.ndarray, 
                                      dk: np.ndarray, 
                                      alpha: float, 
                                      beta: float,
                                      c1: float = 1e-4, c2: float = 0.9, c3: float = 0.5,
                                      epsilon_g: float = 1e-8) -> Tuple[bool, bool, bool]:
    """
    Controlla se le condizioni di strong Wolfe noise tolerant sono soddisfatte per il passo alpha e beta.
    Ritorna una tupla di booleani (Armijo_ok, strong_grad_ok, noise_ok).

    Condizioni di strong Wolfe noise tolerant:
    1. Armijo: f(xk + alpha * dk) <= f(xk) + c1 * alpha * g(xk)^T * dk
    2. Strong gradient: |g(xk + alpha * dk)^T * dk| <= c2 * |g(xk)^T * dk|
    3. Noise tolerant: (g(xk + beta * dk) - g(xk))^T * dk >= 2 * (1 + c3) * epsilon_g * ||dk||
    con c1 in (0, 1/2), c2 in (c1, 1) e c3 > 0.
    """
    armijo_ok, strong_grad_ok = check_strong_wolfe(f, g, xk, dk, alpha, c1=c1, c2=c2)

    # Controllo della condizione di noise tolerant
    noise_ok = float(np.dot(g(xk + beta * dk) - g(xk), dk)) >= 2 * (1 + c3) * epsilon_g * np.linalg.norm(dk)

    return bool(armijo_ok), bool(strong_grad_ok), bool(noise_ok)


def split_phase(f: Callable[[np.ndarray], float],
                g: Callable[[np.ndarray], np.ndarray],
                x: np.ndarray,
                d: np.ndarray,
                eps_g: float,
                alpha_init: float,
                beta_init: Optional[float] = None,
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
      eps_g     : livello di rumore per i gradienti (scalare)
      alpha     : lunghezza iniziale del passo (scalare)
      beta      : parametro di allungamento iniziale (se None, beta = 1.0)
      c1, c3    : costanti dell'algoritmo (c1 non utilizzato rigorosamente tranne che per Armijo RHS)
      max_backtrack : numero massimo di iterazioni per dividere alpha per 10
      max_doublings  : numero massimo di raddoppi per beta
      alpha_min, beta_max : limiti di sicurezza per evitare under/overflow

    Output:
      alpha (float), beta (float), info (dict):
        info contiene 'status', 'n_f', 'n_g', 'alpha_history', 'beta_history'
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    # Diagnostica
    n_f = 0
    n_g = 0
    alpha_history = []
    beta_history = []

    # Precalcola valori iniziali
    fx = float(f(x)); n_f += 1
    gx = np.asarray(g(x), dtype=float); n_g += 1
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
        f_alpha = float(f(x + alpha * d)); n_f += 1
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
        g_beta = np.asarray(g(x + beta * d), dtype=float); n_g += 1
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
                                            eps_g: float,
                                            alpha_init: float = 1.0,
                                            beta_init: Optional[float] = None,
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
      eps_g       : livello di rumore per i gradienti (scalare)
      alpha_init  : passo iniziale (default 1.0)
      beta_init   : parametro di allungamento iniziale (se None, beta = 1.0)
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
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    # Precalcola f(x) e g(x)
    fx = float(f(x))
    gx = np.asarray(g(x), dtype=float)
    gp0 = float(np.dot(gx, d))  # g(x)^T p
    d_norm = float(np.linalg.norm(d))

    # Inizializza
    l = 0.0
    u = np.inf
    alpha = float(alpha_init)
    beta = 1.0 if beta_init is None else float(beta_init)

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
        phialpha = float(f(x_alpha))
        g_alpha = np.asarray(g(x_alpha), dtype=float)
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
                                                        eps_g=eps_g,
                                                        alpha_init=alpha,
                                                        beta_init=beta_init,
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
                                                eps_g=eps_g,
                                                alpha_init=alpha,
                                                beta_init=beta_init,
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
                                     eps_f: float = 1e-8,
                                     eps_g: float = 1e-8,
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
        yk = ∇f(xk + beta_k dk) - ∇f(xk) [equivalente a yk = (xk + beta_k dk) - g(xk)]
        sk = beta_k dk  # con beta_k parametro di allungamento
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
                                                                       eps_g=eps_g, 
                                                                       alpha_init=alpha_init, 
                                                                       beta_init=beta_init, 
                                                                       c1=c1, c2=c2, c3=c3,
                                                                       Nsplit=Nsplit, 
                                                                       max_backtrack=max_backtrack, 
                                                                       max_doublings=max_doublings, 
                                                                       alpha_min=alpha_min, 
                                                                       beta_max=beta_max)
        info['ls_history'].append(ls_info if isinstance(ls_info, dict) else {})

        # se non trovato, usa Armijo come fallback
        if alpha is None:
            alpha, armijo_line_search_info = armijo_line_search(f=f, g=g, x=xk, d=dk)

        # se alpha è ancora None -> piccolo fallback e salto dell'aggiornamento
        if alpha is None:
            info['status'] = 'line_search_failed'
            # prendo uno step molto piccolo per aggiornare xk e gk
            tiny = 1e-12
            x_next = xk + tiny * dk
            g_next = np.asarray(g(x_next), dtype=float)
            # non aggiorno Bk
            info['status_detail'] = 'tiny_step_and_skip_update'
            # aggiornamento iterazione per i criteri di terminazione
            xk = x_next
            gk = g_next
            fk = float(f(xk))
            info['nit'] = k + 1
            info['x_history'].append(xk.copy()); info['f_history'].append(fk); info['grad_norms'].append(vecnorm(gk))
            break


        # aggiornamento
        x_next = xk + alpha * dk
        g_next = np.asarray(g(x_next), dtype=float)
        fk_next = float(f(x_next))

        yk = g(x_next + beta * dk) - gk  # yk calcolato con il parametro di allungamento beta
        sk = beta * dk  # sk calcolato con il parametro di allungamento beta

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