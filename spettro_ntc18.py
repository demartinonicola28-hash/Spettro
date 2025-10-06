# gui_spettro_step12_plot.py
# Step1: scelte base + regolarità. Step2: SL, ξ, a_g/g, F0, T_c*.
# Calcolo parametri NTC18 e grafico Se(T) con etichette TB, TC, TD e Sd/g plateau.

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont
import numpy as np
import matplotlib.pyplot as plt
import os


# --- costante gravitazionale ---
g0 = 9.80665

# --- opzioni combobox ---
SUOLO_OPZ = ["A", "B", "C", "D", "E"]
TOPO_OPZ = ["T1", "T2", "T3", "T4"]
DUTT_OPZ = ["CD B", "CD A", "NON DISSIPATIVO"]
REG_OPZ = ["SI", "NO"]
KR_MAP = {"SI": 1.0, "NO": 0.8}

# --- SL e tempi di ritorno (anni) ---
SL_OPZ = ["SLO", "SLD", "SLV", "SLC"]
TR_MAP = {"SLO": 30, "SLD": 50, "SLV": 475, "SLC": 975}

# --- ST (Tab. 3.2.V) ---
ST_MAP = {"T1": 1.0, "T2": 1.2, "T3": 1.2, "T4": 1.4}

def set_fonts_courier(root: tk.Tk):
    """Imposta Courier New per Tkinter e Matplotlib."""
    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family="Courier New")
        except tk.TclError:
            pass
    plt.rcParams["font.family"] = "Courier New"

def label_pedice(master, base, pedice, r, c):
    """Label 'base' con pedice piccolo a fianco, tutto in Courier."""
    f1, f2 = ("Courier New", 10), ("Courier New", 7)
    frame = ttk.Frame(master)
    ttk.Label(frame, text=base, font=f1).pack(side="left")
    ttk.Label(frame, text=pedice, font=f2).pack(side="left", anchor="s")
    frame.grid(row=r, column=c, sticky="e", padx=6, pady=3)

# ----------------- fattori e parametri NTC18 -----------------
def eta_corr(xi_pct: float) -> float:
    """η(ξ) = max( sqrt(10/(5+ξ)), 0.55 ) con ξ [%]."""
    from math import sqrt
    return max(sqrt(10.0 / (5.0 + xi_pct)), 0.55)

def cc_da_suolo(suolo: str, Tc_star: float) -> float:
    """C_C in funzione della categoria e di T_c*."""
    if suolo == "A": return 1.0
    if suolo == "B": return 1.10 * (Tc_star ** -0.20)
    if suolo == "C": return 1.05 * (Tc_star ** -0.33)
    if suolo == "D": return 1.25 * (Tc_star ** -0.50)
    if suolo == "E": return 1.15 * (Tc_star ** -0.40)
    raise ValueError("Suolo non valido")

def ss_da_suolo(suolo: str, F0: float, ag_over_g: float) -> float:
    """S_S in funzione della categoria e di F0*(ag/g) con limiti tabellari."""
    x = F0 * ag_over_g
    if suolo == "A":
        return 1.0
    elif suolo == "B":
        return min(1.40, max(1.00, 1.40 - 0.40 * x))
    elif suolo == "C":
        return min(1.70, max(1.00, 1.70 - 0.60 * x))
    elif suolo == "D":
        return min(2.40, max(0.90, 2.40 - 1.50 * x))
    elif suolo == "E":
        return min(2.00, max(1.00, 2.00 - 1.10 * x))
    else:
        raise ValueError("Suolo non valido")

def params_spettro(suolo: str, topo: str, ag_over_g: float, F0: float, Tc_star: float):
    """Ritorna ST, SS, S, CC, TB, TC, TD secondo NTC18."""
    ST = ST_MAP[topo]
    SS = ss_da_suolo(suolo, F0, ag_over_g)
    S = SS * ST
    CC = cc_da_suolo(suolo, Tc_star)
    TC = CC * Tc_star
    TB = TC / 3.0
    TD = 1.6 + 4.0 * ag_over_g
    return dict(ST=ST, SS=SS, S=S, CC=CC, TB=TB, TC=TC, TD=TD)

def se_elastico(T: np.ndarray, ag: float, S: float, F0: float, TB: float, TC: float, TD: float, xi_pct: float):
    """Spettro elastico Se(T) [m/s²] a tratti NTC18 con η(ξ)."""
    eta = eta_corr(xi_pct)
    Se = np.zeros_like(T, dtype=float)

    # 0 <= T <= TB
    i1 = T <= TB
    if np.any(i1):
        r = T[i1] / max(TB, 1e-9)
        Se[i1] = ag * S * (1.0 + (eta * F0 - 1.0) * r)

    # TB < T <= TC
    i2 = (T > TB) & (T <= TC)
    Se[i2] = ag * S * eta * F0

    # TC < T <= TD
    i3 = (T > TC) & (T <= TD)
    Se[i3] = ag * S * eta * F0 * (TC / T[i3])

    # T > TD
    i4 = T > TD
    Se[i4] = ag * S * eta * F0 * (TC * TD / (T[i4] ** 2))

    return Se

def sd_progetto(T: np.ndarray, ag: float, S: float, F0: float,
                TB: float, TC: float, TD: float, xi_pct: float, q_lim: float) -> np.ndarray:
    """
    Spettro di progetto Sd(T) NTC18 con richiesta utente:
    - 1° tratto (0–TB): NON diviso per q_lim. Interpola da Se(0)=ag*S a Se(TB)=ag*S*eta*F0/q_lim.
    - 2° e 3° tratto: divisi per q_lim.
    - 4° tratto: diviso per q_lim (coerenza e continuità oltre TD).
    """
    eta = eta_corr(xi_pct)
    Sd = np.zeros_like(T, dtype=float)

    # 1) 0 <= T <= TB  (join tra valore iniziale elastico e plateau/q_lim)
    i1 = T <= TB
    if np.any(i1):
        r = T[i1] / max(TB, 1e-9)
        Sd0    = ag * S
        Sd_TB  = ag * S * eta * F0 / q_lim
        Sd[i1] = Sd0 + (Sd_TB - Sd0) * r

    # 2) TB < T <= TC  (plateau / q_lim)
    i2 = (T > TB) & (T <= TC)
    Sd[i2] = ag * S * eta * F0 / q_lim

    # 3) TC < T <= TD  (decadimento 1/T / q_lim)
    i3 = (T > TC) & (T <= TD)
    Sd[i3] = ag * S * eta * F0 * (TC / T[i3]) / q_lim

    # 4) T > TD  (decadimento 1/T^2 / q_lim)
    i4 = T > TD
    Sd[i4] = ag * S * eta * F0 * (TC * TD / (T[i4] ** 2)) / q_lim

    return Sd



# =========================
# STEP 1: base + regolarità
# =========================

CTRL_W = 10  # caratteri: stessa lunghezza per tutti i campi

class Step1(tk.Tk):
    def __init__(self):
        super().__init__()
        set_fonts_courier(self)
        self.title("Parametri della Struttura e del Sito - NTC18")
        self.resizable(False, False)
        pad = dict(padx=12, pady=6)

        ttk.Label(self, text="Categoria di suolo").grid(row=0, column=0, sticky="e", **pad)
        self.suolo = tk.StringVar(value="B")
        ttk.Combobox(self, textvariable=self.suolo, values=SUOLO_OPZ,
                     state="readonly", width=CTRL_W).grid(row=0, column=1, **pad)

        ttk.Label(self, text="Categoria topografica").grid(row=1, column=0, sticky="e", **pad)
        self.topo = tk.StringVar(value="T1")
        ttk.Combobox(self, textvariable=self.topo, values=TOPO_OPZ,
                     state="readonly", width=CTRL_W).grid(row=1, column=1, **pad)

        ttk.Label(self, text="Classe di duttilità").grid(row=2, column=0, sticky="e", **pad)
        self.dutt = tk.StringVar(value="CD B")
        self.dutt_cb = ttk.Combobox(self, textvariable=self.dutt, values=DUTT_OPZ,
                                    state="readonly", width=CTRL_W)
        self.dutt_cb.grid(row=2, column=1, **pad)
        self.dutt_cb.bind("<<ComboboxSelected>>", self._on_dutt_change)

        ttk.Label(self, text="Fattore di comportamento base q₀").grid(row=3, column=0, sticky="e", **pad)
        self.q0_var = tk.StringVar(value="4.0")
        self.q0_entry = ttk.Entry(self, textvariable=self.q0_var, width=CTRL_W+2)
        self.q0_entry.grid(row=3, column=1, **pad)

        ttk.Label(self, text="Regolarità in altezza").grid(row=4, column=0, sticky="e", **pad)
        self.reg = tk.StringVar(value="SI")
        cb_reg = ttk.Combobox(self, textvariable=self.reg, values=REG_OPZ,
                              state="readonly", width=CTRL_W)
        cb_reg.grid(row=4, column=1, **pad)
        cb_reg.bind("<<ComboboxSelected>>", self._update_KR)

        label_pedice(self, "K", "R", 4, 2)
        self.kr = tk.StringVar(value=str(KR_MAP[self.reg.get()]))
        ttk.Entry(self, textvariable=self.kr, width=6, state="disabled").grid(row=4, column=3, **pad)

        ttk.Button(self, text="OK", command=self._ok).grid(row=5, column=0, columnspan=4, pady=10)
        self._on_dutt_change()

    def _on_dutt_change(self, *_):
        if self.dutt.get() == "NON DISSIPATIVO":
            self.q0_var.set("1.0")
            self.q0_entry.state(["disabled"])
        else:
            self.q0_entry.state(["!disabled"])
            if self.q0_var.get() == "1.0":
                self.q0_var.set("4.0")

    def _update_KR(self, *_):
        self.kr.set(str(KR_MAP[self.reg.get()]))

    def _ok(self):
        try:
            q0 = float(self.q0_var.get().replace(",", "."))
            if q0 <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Errore", "q₀ deve essere numerico positivo.")
            return
        dati1 = {
            "suolo": self.suolo.get(),
            "topografia": self.topo.get(),
            "classe_duttilita": self.dutt.get(),
            "q0": q0,
            "reg_in_altezza": self.reg.get(),
            "KR": float(self.kr.get()),
        }
        self.destroy()
        Step2(dati1).mainloop()

# =========================
# STEP 2: input + grafico
# =========================
class Step2(tk.Tk):
    def __init__(self, dati1: dict):
        super().__init__()
        set_fonts_courier(self)
        self.title("Parametri dell'Azione Sismica - NTC18")
        self.resizable(False, False)
        self.dati = dati1
        pad = dict(padx=22, pady=6)

        ttk.Label(self, text="Stato limite").grid(row=0, column=0, sticky="e", **pad)
        self.sl = tk.StringVar(value="SLV")
        cb_sl = ttk.Combobox(self, textvariable=self.sl, values=SL_OPZ, state="readonly", width=CTRL_W-2)
        cb_sl.grid(row=0, column=1, **pad)
        cb_sl.bind("<<ComboboxSelected>>", self._update_TR)

        label_pedice(self, "T", "R", 0, 2)
        self.tr = tk.StringVar(value=str(TR_MAP[self.sl.get()]))
        ttk.Entry(self, textvariable=self.tr, width=6, state="disabled").grid(row=0, column=3, **pad)

        ttk.Label(self, text="ξ [%]").grid(row=1, column=0, sticky="e", **pad)
        self.xi = tk.StringVar(value="5.0")
        ttk.Entry(self, textvariable=self.xi, width=CTRL_W).grid(row=1, column=1, **pad)

        ttk.Label(self, text="a₉ / g").grid(row=2, column=0, sticky="e", **pad)
        self.ag_over_g = tk.StringVar(value="0.25")
        ttk.Entry(self, textvariable=self.ag_over_g, width=CTRL_W).grid(row=2, column=1, **pad)

        ttk.Label(self, text="F₀").grid(row=3, column=0, sticky="e", **pad)
        self.f0 = tk.StringVar(value="2.5")
        ttk.Entry(self, textvariable=self.f0, width=CTRL_W).grid(row=3, column=1, **pad)

        label_pedice(self, "T", "c*", 4, 0)
        self.tc_star = tk.StringVar(value="0.5")
        ttk.Entry(self, textvariable=self.tc_star, width=CTRL_W).grid(row=4, column=1, **pad)

        ttk.Button(self, text="Disegna spettro", command=self._plot).grid(row=5, column=0, columnspan=4, pady=10)

    def _update_TR(self, *_):
        self.tr.set(str(TR_MAP[self.sl.get()]))

    def _plot(self):
        try:
            xi = float(self.xi.get().replace(",", "."))
            ag_over_g = float(self.ag_over_g.get().replace(",", "."))
            F0 = float(self.f0.get().replace(",", "."))
            Tc_star = float(self.tc_star.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Errore", "Inserisci numeri validi per ξ, a₉/g, F₀, T꜀*.")
            return

        # Parametri spettrali e accelerazione di riferimento
        par = params_spettro(self.dati["suolo"], self.dati["topografia"], ag_over_g, F0, Tc_star)
        ag = ag_over_g * g0

        # q_lim = q0 * K_R (spettro di progetto)
        q_lim = self.dati["q0"] * self.dati["KR"]

        # Griglia tempi e spettro elastico Se(T)
        T = np.linspace(0.0, max(4.0, par["TD"] * 1.2), 2000)
        T[0] = max(T[1]*1e-6, 1e-9)  # evita T=0 esatto
        Se = se_elastico(T, ag, par["S"], F0, par["TB"], par["TC"], par["TD"], xi)

        # q_lim = q0 * K_R (spettro di progetto)
        q_lim = self.dati["q0"] * self.dati["KR"]

        # Griglia tempi
        T = np.linspace(0.0, max(4.0, par["TD"] * 1.2), 2000)
        T[0] = max(T[1]*1e-6, 1e-9)

        # Sd di progetto secondo le regole richieste
        Sd = sd_progetto(T, ag, par["S"], F0, par["TB"], par["TC"], par["TD"], xi, q_lim)
        Sd_over_g = Sd / g0  # Sd/g

        # Valore di plateau Sd/g nel tratto TB–TC (già diviso per q_lim)
        eta = eta_corr(xi)
        Sd_plateau_over_g = (ag * par["S"] * eta * F0) / (g0 * q_lim)


        # Stampa su terminale
        #print(f"TB = {par['TB']:.3f} s, TC = {par['TC']:.3f} s, TD = {par['TD']:.3f} s, q_lim = {q_lim:.3f}")

        # Plot Sd/g - T
        plt.figure(figsize=(12, 6))
        # Impostare lo sfondo grigio chiaro
        plt.gca().set_facecolor('whitesmoke')
        ax = plt.gca()
        ax.plot(T, Sd_over_g, color="black")          # curva Sd/g in nero
        ax.set_xlabel("T [s]", fontsize=14)
        ax.set_ylabel("Sd/g [m/s²]", fontsize=14)
        ax.set_title("Spettro di Progetto NTC18", fontsize=18, pad=10)
        ax.grid(True)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, )

        # Ottimizzare il layout
        plt.tight_layout()

        # Rette verticali rosse con etichette a metà
        ymin, ymax = ax.get_ylim()
        ymid = 0.5*(ymin + ymax)

        def vlabel(x, pedice, valore):
            """Linea verticale rossa e label 'T' con pedice piccolo + valore."""
            ax.axvline(x, color="red", linestyle="--", linewidth=0.9)
            ax.annotate(f"{pedice} = {valore:.3f} s", (x, ymid), xycoords="data",
                        textcoords="offset points", xytext=(-12, -80),
                        ha="left", va="center", rotation=90,
                        fontsize=12, color="red")

        vlabel(par["TB"], "TB", par["TB"])
        vlabel(par["TC"], "TC", par["TC"])
        vlabel(par["TD"], "TD", par["TD"])

        # Testo del plateau Sd/g posizionato tra TB e TC
        #xmid = 0.65*(par["TB"] + par["TC"])
        xmid = par["TC"]+0.05
        ax.text(xmid, Sd_plateau_over_g*1.00, f"Sd/g = {Sd_plateau_over_g:.3f} m/s²",
                ha="left", va="center", fontsize=12, color="black")

        # Salvataggio JPG nella stessa cartella del .py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, "spettro_ntc18.jpg")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")

        self.destroy()
        plt.show()
        

        # --- salvataggio .txt T - Sd/g (per import Straus7) ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(script_dir, "spettro_ntc18.txt")

        # ordina per T e rimuove duplicati
        arr = np.column_stack([T, Sd_over_g])
        arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
        arr = arr[arr[:, 0] > 0.0]
        arr = arr[np.concatenate(([True], np.diff(arr[:, 0]) > 0.0))]

        if arr.size == 0:
            raise ValueError("Spettro vuoto dopo pulizia.")

        # salva con separatore SPAZIO, punto decimale, nessuna intestazione
        np.savetxt(
            txt_path,
            arr,
            fmt="%.6f",
            delimiter=" ",   # spazio singolo
            newline="\n"     # LF
        )


# Avvio
if __name__ == "__main__":
    Step1().mainloop()
