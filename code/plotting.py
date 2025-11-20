from typing import Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_function_histories(histories: Sequence[Sequence[float]],
                            method_names: Sequence[str],
                            problem_name: Optional[str] = None,
                            tol: Optional[float] = None,
                            xlabel: str = "Iterazione",
                            ylabel: str = "Valore funzione",
                            logy: bool = False,
                            show_legend: bool = True) -> None:
    """
    Disegna un singolo grafico contenente gli andamenti dei valori
    della funzione per diversi metodi.
    ...
    """
    if len(histories) != len(method_names):
        raise ValueError("`histories` e `method_names` devono avere la stessa lunghezza.")

    plt.figure(figsize=(8, 5))  # singolo plot e dimensione leggermente più ampia per evitare tagli
    tol_sci = format(float(tol), ".2e") if tol is not None else None  # formato scientifico per il titolo
    title_kwargs = {"fontweight": "bold", "pad": 12}  # pad aggiuntivo per non tagliare il titolo
    if problem_name is not None and tol is not None:
        plt.title(f"Andamento valori funzione - problema {problem_name} (tol={tol_sci})", **title_kwargs)
    elif problem_name is not None:
        plt.title(f"Andamento valori funzione - problema {problem_name}", **title_kwargs)
    elif tol is not None:
        plt.title(f"Andamento valori funzione (tol={tol_sci})", **title_kwargs)
    else:
        plt.title("Andamento valori funzione", **title_kwargs)

    for hist, name in zip(histories, method_names):
        arr = np.asarray(hist, dtype=float)
        if arr.size == 0:
            plt.plot([], [], label=f"{name} (vuoto)")
            continue
        x = np.arange(arr.shape[0])
        if logy:
            plt.semilogy(x, arr, label=name)
        else:
            plt.plot(x, arr, label=name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if show_legend:
        plt.legend()

    # Riserva spazio in alto per il titolo e usa bbox_inches='tight' al salvataggio
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Salva il grafico in forma PDF nella cartella plots
    os.makedirs("plots", exist_ok=True)
    if problem_name is not None:
        filename = f"plots/{problem_name}_function_histories.pdf"
    else:
        filename = "plots/function_histories.pdf"
    plt.savefig(filename, bbox_inches="tight")


def plot_gradient_norm_histories(histories: Sequence[Sequence[float]],
                                 method_names: Sequence[str],
                                 problem_name: Optional[str] = None,
                                 tol: Optional[float] = None,
                                 xlabel: str = "Iterazione",
                                 ylabel: str = "Norma del gradiente",
                                 logy: bool = False,
                                 show_legend: bool = True) -> None:
    """
    Disegna un singolo grafico contenente gli andamenti delle norme del gradiente
    per diversi metodi.

    Parametri
    ---------
    histories : sequenza di sequenze di float
        Ogni elemento è la storia delle norme del gradiente (in ordine di iterazione)
        prodotta da un metodo.
    method_names : sequenza di str
        Nomi dei metodi; deve avere la stessa lunghezza di `histories`.
    problem_name : str | None
        Nome del problema (opzionale), usato per titolo del grafico.
    tol : float | None
        Tolleranza usata (opzionale), mostrata nel titolo del grafico.
    xlabel, ylabel : str
        Etichette degli assi.
    logy : bool
        Se True imposta scala logaritmica sull'asse y (utile quando i valori variano molto).
    show_legend : bool
        Se True visualizza la legenda.

    Note
    ----
    - Non vengono impostati colori espliciti: matplotlib userà la palette di default.
    - Ogni curva è disegnata su un unico plot; non si usano subplots.
    """
    if len(histories) != len(method_names):
        raise ValueError("`histories` e `method_names` devono avere la stessa lunghezza.")

    plt.figure(figsize=(8, 5))  # singolo plot e dimensione leggermente più ampia per evitare tagli
    tol_sci = format(float(tol), ".2e") if tol is not None else None  # formato scientifico per il titolo
    title_kwargs = {"fontweight": "bold", "pad": 12}  # pad aggiuntivo per non tagliare il titolo
    if problem_name is not None and tol is not None:
        plt.title(f"Andamento norme del gradiente - problema {problem_name} (tol={tol_sci})", **title_kwargs)
    elif problem_name is not None:
        plt.title(f"Andamento norme del gradiente - problema {problem_name}", **title_kwargs)
    elif tol is not None:
        plt.title(f"Andamento norme del gradiente (tol={tol_sci})", **title_kwargs)
    else:   
        plt.title("Andamento norme del gradiente", **title_kwargs)
        
    for hist, name in zip(histories, method_names):
        # converto in array per essere robusto (lunghezze diverse gestite automaticamente)
        arr = np.asarray(hist, dtype=float)
        if arr.size == 0:
            # evita errori per liste vuote — disegna un punto vuoto (o salta)
            plt.plot([], [], label=f"{name} (vuoto)")
            continue
        x = np.arange(arr.shape[0])
        # disegno: niente colori espliciti, lasciamo la palette di matplotlib
        if logy:
            # usa semilogy per scala logaritmica in y
            plt.semilogy(x, arr, label=name)
        else:
            plt.plot(x, arr, label=name)

    # Disegna linea orizzontale tratteggiata rossa alla quota tol (se valida)
    if tol is not None:
        try:
            tol_val = float(tol)
        except Exception:
            tol_val = None
        if tol_val is not None:
            if not (logy and tol_val <= 0):
                # aggiungiamo la label solo se verrà mostrata la legenda
                lbl = f"tol={tol_sci}" if show_legend and tol_sci is not None else None
                plt.axhline(y=tol_val, color="red", linestyle="--", linewidth=1, label=lbl)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if show_legend:
        plt.legend()
    
    # Riserva spazio in alto per il titolo e usa bbox_inches='tight' al salvataggio
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Salva il grafico in forma PDF nella cartella plots
    os.makedirs("plots", exist_ok=True)
    if problem_name is not None:
        filename = f"plots/{problem_name}_gradient_norm_histories.pdf"
    else:
        filename = "plots/gradient_norm_histories.pdf"
    plt.savefig(filename, bbox_inches="tight")