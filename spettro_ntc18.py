# spettro_ntc18.py  # GUI unica, annulla con cleanup file, immagine laterale

# =============== IMPORT ===============
import os                              # gestione percorsi e file
import threading                       # thread per analisi annullabile
from typing import Optional, Dict, Any # annotazioni tipi
import tempfile                        # file temporanei per salvataggio atomico

import tkinter as tk                   # GUI base
from tkinter import ttk, messagebox    # widget themed + popup
import tkinter.font as tkfont          # misure font per layout

# Matplotlib in backend non interattivo (evita lock UI e problemi thread)
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt        # plotting

import numpy as np                     # numerica

# Pillow opzionale per caricare jpg/png in Tk
try:
    from PIL import Image, ImageTk     # immagini e adattamento
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# =============== COSTANTI NTC/GUI ===============
g0 = 9.80665

SUOLO_OPZ = ["A", "B", "C", "D", "E"]
TOPO_OPZ  = ["T1", "T2", "T3", "T4"]
DUTT_OPZ  = ["CD B", "CD A", "NON DISSIPATIVO"]
REG_OPZ   = ["SI", "NO"]
KR_MAP    = {"SI": 1.0, "NO": 0.8}

SL_OPZ = ["SLO", "SLD", "SLV", "SLC"]
TR_MAP = {"SLO": 30, "SLD": 50, "SLV": 475, "SLC": 975}

ST_MAP = {"T1": 1.0, "T2": 1.2, "T3": 1.2, "T4": 1.4}

# Layout coerente
LBL_WCH   = 42     # etichette a sinistra, monospace
CTRL_W    = 18     # larghezza Entry/Combobox in caratteri
PADX_FORM = 12     # padding orizzontale uniforme

# =============== UTILI GRAFICI/GUI ===============
def set_fonts_courier(root: tk.Tk) -> None:
    """Forza Courier New in Tk e Matplotlib; ignora se assente."""
    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family="Courier New")
        except tk.TclError:
            pass
    plt.rcParams["font.family"] = "Courier New"

def label_pedice(master, base, pedice, r, c):
    """Stampa 'base' con pedice piccolo a fianco."""
    f1, f2 = ("Courier New", 10), ("Courier New", 7)
    fr = ttk.Frame(master)
    ttk.Label(fr, text=base,  font=f1).pack(side="left")
    ttk.Label(fr, text=pedice, font=f2).pack(side="left", anchor="s")
    fr.grid(row=r, column=c, sticky="w", padx=6, pady=3)  # <-- allineato a SINISTRA

# =============== FORMULE NTC18 ===============
def eta_corr(xi_pct: float) -> float:
    from math import sqrt
    return max(sqrt(10.0 / (5.0 + xi_pct)), 0.55)

def cc_da_suolo(suolo: str, Tc_star: float) -> float:
    if suolo == "A": return 1.0
    if suolo == "B": return 1.10 * (Tc_star ** -0.20)
    if suolo == "C": return 1.05 * (Tc_star ** -0.33)
    if suolo == "D": return 1.25 * (Tc_star ** -0.50)
    if suolo == "E": return 1.15 * (Tc_star ** -0.40)
    raise ValueError("Suolo non valido")

def ss_da_suolo(suolo: str, F0: float, ag_over_g: float) -> float:
    x = F0 * ag_over_g
    if suolo == "A":   return 1.0
    if suolo == "B":   return min(1.40, max(1.00, 1.40 - 0.40 * x))
    if suolo == "C":   return min(1.70, max(1.00, 1.70 - 0.60 * x))
    if suolo == "D":   return min(2.40, max(0.90, 2.40 - 1.50 * x))
    if suolo == "E":   return min(2.00, max(1.00, 2.00 - 1.10 * x))
    raise ValueError("Suolo non valido")

def params_spettro(suolo: str, topo: str, ag_over_g: float, F0: float, Tc_star: float) -> Dict[str, float]:
    ST = ST_MAP[topo]
    SS = ss_da_suolo(suolo, F0, ag_over_g)
    S  = SS * ST
    CC = cc_da_suolo(suolo, Tc_star)
    TC = CC * Tc_star
    TB = TC / 3.0
    TD = 1.6 + 4.0 * ag_over_g
    return dict(ST=ST, SS=SS, S=S, CC=CC, TB=TB, TC=TC, TD=TD)

def sd_progetto(T: np.ndarray, ag: float, S: float, F0: float,
                TB: float, TC: float, TD: float, xi_pct: float, q_lim: float) -> np.ndarray:
    eta = eta_corr(xi_pct)
    Sd = np.zeros_like(T, float)
    i1 = T <= TB
    if np.any(i1):
        r = T[i1] / max(TB, 1e-9)
        Sd0   = ag * S
        Sd_TB = ag * S * eta * F0 / q_lim
        Sd[i1] = Sd0 + (Sd_TB - Sd0) * r
    i2 = (T > TB) & (T <= TC)
    Sd[i2] = ag * S * eta * F0 / q_lim
    i3 = (T > TC) & (T <= TD)
    Sd[i3] = ag * S * eta * F0 * (TC / T[i3]) / q_lim
    i4 = T > TD
    Sd[i4] = ag * S * eta * F0 * (TC * TD / (T[i4] ** 2)) / q_lim
    return Sd

def _merge_T_arrays(T_base: np.ndarray, T_extra: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    T_all = np.concatenate([T_base, T_extra])
    T_all.sort(kind="mergesort")
    keep = [True]
    for i in range(1, T_all.size):
        keep.append(not np.isclose(T_all[i], T_all[i-1], rtol=0.0, atol=tol))
    T_all = T_all[np.array(keep, bool)]
    if T_all[0] <= tol:
        T_all[0] = max(tol, (T_all[1] if T_all.size > 1 else tol) * 1e-6)
    return T_all

# =============== STATO OUTPUT E UTILI FILE ===============
_OUTPUT_DIR: Optional[str] = None   # cartella per salvataggi
_RESULT: Dict[str, Any] = {}        # risultati da ritornare

def _get_output_dir() -> str:
    if _OUTPUT_DIR:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        return _OUTPUT_DIR
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def _unique_path(path: str) -> str:
    """Crea un nome non esistente aggiungendo (n)."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        cand = f"{base} ({n}){ext}"
        if not os.path.exists(cand):
            return cand
        n += 1

def _find_hazard_map(dirpath: str) -> Optional[str]:
    """Cerca 'mappa_pericolosità_sismica' con estensioni note nella cartella."""
    names = [
        "mappa_pericolosità_sismica.png",
        "mappa_pericolosità_sismica.jpg",
        "mappa_pericolosità_sismica.jpeg",
        "mappa_pericolosita_sismica.png",
        "mappa_pericolosita_sismica.jpg",
        "mappa_pericolosita_sismica.jpeg",
    ]
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p):
            return p
    return None

# =============== APP GUI ===============
class App(tk.Tk):
    """Finestra unica con riquadri form a sinistra e immagine a destra."""
    def __init__(self):
        super().__init__()
        self._cancel_event = threading.Event()  # flag annullamento
        self._worker: Optional[threading.Thread] = None  # thread analisi
        self._last_jpg: Optional[str] = None     # ultimo JPG creato in questa esecuzione
        self._last_txt: Optional[str] = None     # ultimo TXT creato in questa esecuzione
        self._img_ref = None                     # riferimento PhotoImage per Tk

        self.protocol("WM_DELETE_WINDOW", self._on_click_annulla)  # X = Annulla
        set_fonts_courier(self)
        self.title("Spettro di Progetto NTC18")
        self.resizable(False, False)

        # Contenitore 2 colonne: sx = form, dx = immagine
        container = ttk.Frame(self, padding=(10, 10))
        container.grid(row=0, column=0, sticky="nsew")

        # ------- Colonna sinistra: due riquadri -------
        self.left_frame = ttk.Frame(container)          # <-- mantieni riferimento per misurare altezza
        self.left_frame.grid(row=0, column=0, sticky="n")

        lf1 = ttk.LabelFrame(self.left_frame, text="Parametri della Struttura e del Sito - NTC18", padding=(12, 8))
        lf1.grid(row=0, column=0, sticky="n")
        pad = dict(padx=PADX_FORM, pady=6)

        ttk.Label(lf1, text="Categoria di suolo", width=LBL_WCH).grid(row=0, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.suolo = tk.StringVar(value="B")
        ttk.Combobox(lf1, textvariable=self.suolo, values=SUOLO_OPZ, state="readonly", width=CTRL_W)\
            .grid(row=0, column=1, sticky="we", **pad)

        ttk.Label(lf1, text="Categoria topografica", width=LBL_WCH).grid(row=1, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.topo = tk.StringVar(value="T1")
        ttk.Combobox(lf1, textvariable=self.topo, values=TOPO_OPZ, state="readonly", width=CTRL_W)\
            .grid(row=1, column=1, sticky="we", **pad)

        ttk.Label(lf1, text="Classe di duttilità", width=LBL_WCH).grid(row=2, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.dutt = tk.StringVar(value="CD B")
        self.dutt_cb = ttk.Combobox(lf1, textvariable=self.dutt, values=DUTT_OPZ, state="readonly", width=CTRL_W)
        self.dutt_cb.grid(row=2, column=1, sticky="we", **pad)
        self.dutt_cb.bind("<<ComboboxSelected>>", self._on_dutt_change)

        ttk.Label(lf1, text="Fattore di comportamento base q₀", width=LBL_WCH).grid(row=3, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.q0_var = tk.StringVar(value="4.0")
        self.q0_entry = ttk.Entry(lf1, textvariable=self.q0_var, width=CTRL_W)
        self.q0_entry.grid(row=3, column=1, sticky="we", **pad)

        ttk.Label(lf1, text="Regolarità in altezza", width=LBL_WCH).grid(row=4, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.reg = tk.StringVar(value="SI")
        cb_reg = ttk.Combobox(lf1, textvariable=self.reg, values=REG_OPZ, state="readonly", width=CTRL_W)
        cb_reg.grid(row=4, column=1, sticky="we", **pad)
        cb_reg.bind("<<ComboboxSelected>>", self._update_KR)

        ttk.Label(lf1, text="Fattore per regolarità in altezza Kr", width=LBL_WCH).grid(row=5, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.kr = tk.StringVar(value=str(KR_MAP[self.reg.get()]))
        ttk.Entry(lf1, textvariable=self.kr, width=CTRL_W, state="disabled").grid(row=5, column=1, sticky="we", **pad)

        lf2 = ttk.LabelFrame(self.left_frame, text="Parametri dell'Azione Sismica - NTC18", padding=(12, 8))
        lf2.grid(row=1, column=0, sticky="n", pady=(8, 0))

        ttk.Label(lf2, text="Stato limite", width=LBL_WCH).grid(row=0, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.sl = tk.StringVar(value="SLV")
        self.cb_sl = ttk.Combobox(lf2, textvariable=self.sl, values=SL_OPZ, state="readonly", width=CTRL_W)
        self.cb_sl.grid(row=0, column=1, sticky="we", **pad)
        self.cb_sl.bind("<<ComboboxSelected>>", self._update_TR)

        ttk.Label(lf2, text="Tempo di ritorno Tr", width=LBL_WCH).grid(row=1, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.tr = tk.StringVar(value=str(TR_MAP[self.sl.get()]))
        ttk.Entry(lf2, textvariable=self.tr, width=CTRL_W, state="disabled").grid(row=1, column=1, sticky="we", **pad)

        ttk.Label(lf2, text="Coefficiente di smorzamento viscoso ξ [%]", width=LBL_WCH).grid(row=2, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.xi = tk.StringVar(value="5.0")
        ttk.Entry(lf2, textvariable=self.xi, width=CTRL_W).grid(row=2, column=1, sticky="we", **pad)

        ttk.Label(lf2, text="Accelerazione massima al sito a₉ / g", width=LBL_WCH).grid(row=3, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.ag_over_g = tk.StringVar(value="0.255")
        ttk.Entry(lf2, textvariable=self.ag_over_g, width=CTRL_W).grid(row=3, column=1, sticky="we", **pad)

        ttk.Label(lf2, text="Fattore di amplificazione massima F₀", width=LBL_WCH).grid(row=4, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.f0 = tk.StringVar(value="2.376")
        ttk.Entry(lf2, textvariable=self.f0, width=CTRL_W).grid(row=4, column=1, sticky="we", **pad)
        
        ttk.Label(lf2, text="Periodo di inizio del plateau T*c", width=LBL_WCH).grid(row=5, column=0, sticky="w", **pad)  # <-- a SINISTRA
        self.tc_star = tk.StringVar(value="0.335")
        ttk.Entry(lf2, textvariable=self.tc_star, width=CTRL_W).grid(row=5, column=1, sticky="we", **pad)

        btns = ttk.Frame(lf2)
        btns.grid(row=6, column=0, columnspan=2, pady=10)
        self.btn_disegna = ttk.Button(btns, text="Calcola spettro", command=self._on_click_disegna)
        self.btn_disegna.pack(side="left", padx=6)
        self.btn_annulla = ttk.Button(btns, text="Annulla", command=self._on_click_annulla)
        self.btn_annulla.pack(side="left", padx=6)

        # Allineamento larghezze box in entrambi i riquadri
        box_font = tkfont.nametofont("TkTextFont")
        BOX_PX = int(box_font.measure("0" * CTRL_W))
        for lf in (lf1, lf2):
            lf.grid_columnconfigure(0, weight=0)
            lf.grid_columnconfigure(1, weight=1, minsize=BOX_PX)  # solo due colonne

        # ------- Colonna destra: immagine riquadrata -------
        right = ttk.LabelFrame(container, text="Mappa di Pericolosità Sismica", padding=(8, 8))
        right.grid(row=0, column=1, rowspan=2, sticky="n", padx=15)

        # Canvas con bordo disegnato per riquadro
        self.img_canvas = tk.Canvas(right, width=320, height=320, bg="white", highlightthickness=0)
        self.img_canvas.pack()

        # Rimanda il caricamento per conoscere l'altezza reale dei riquadri a sinistra
        self.after(0, self._load_hazard_map_into_canvas)  # tenta caricamento immagine e ridimensionamento

        # Stato iniziale coerente
        self._on_dutt_change()
        self._update_KR()
        self._update_TR()

    # ---------- Solo gestione file output ----------
    def _delete_output_files(self) -> None:
        """Elimina i soli file di output standard JPG/TXT se presenti + eventuali temporanei residui."""
        base_dir = _get_output_dir()
        targets = [
            os.path.join(base_dir, "spettro_ntc18.jpg"),
            os.path.join(base_dir, "spettro_ntc18.txt"),
        ]
        # rimuove anche temporanei creati con tempfile.mkstemp
        try:
            for n in os.listdir(base_dir):
                if n.startswith("tmp_") and (n.endswith(".jpg") or n.endswith(".txt")):
                    targets.append(os.path.join(base_dir, n))
        except Exception:
            pass
        for p in targets:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
            except Exception:
                pass
        # azzera riferimenti locali
        self._last_jpg = None
        self._last_txt = None

    # ---------- Immagine a destra ----------
    def _draw_image_border(self):
        """Disegna una cornice sottile attorno all'area immagine."""
        self.img_canvas.delete("border")
        w = int(self.img_canvas["width"]); h = int(self.img_canvas["height"])
        self.img_canvas.create_rectangle(2, 2, w-2, h-2, outline="#666666", width=1, tags="border")

    def _load_hazard_map_into_canvas(self):
        """Carica mappa_pericolosità_sismica.* e adatta il CANVAS:
        - altezza = altezza totale dei riquadri a sinistra FINO AI PULSANTI
        - larghezza = proporzionale all'altezza (mantiene aspect ratio dell'immagine)
        L'immagine VIENE SCALATA per stare tutta nel riquadro.
        """
        self.update_idletasks()  # assicura geometria calcolata
        left_h = self.left_frame.winfo_height() or 320  # altezza target (due riquadri + pulsanti)
        base_dir = _get_output_dir()                    # cartella di output
        path = _find_hazard_map(base_dir)               # cerca file

        # -------- nuova logica: scala l'immagine all'altezza left_h mantenendo il rapporto --------
        if path:
            try:
                if _PIL_OK:
                    img = Image.open(path).convert("RGBA")      # carica immagine
                    src_w, src_h = img.size                     # dimensioni originali
                    scale = left_h / max(1, src_h)              # fattore di scala su altezza
                    new_w = int(round(src_w * scale))           # larghezza proporzionale
                    new_h = left_h                               # nuova altezza = target
                    img = img.resize((new_w, new_h), Image.LANCZOS)  # ridimensiona con antialias
                    self._img_ref = ImageTk.PhotoImage(img)     # PhotoImage da PIL
                    canvas_w, canvas_h = new_w, new_h           # dimensioni canvas = immagine
                else:
                    # Fallback senza PIL: usa PhotoImage nativo e subsample per ridurre
                    ph = tk.PhotoImage(file=path)                # carica
                    src_w, src_h = ph.width(), ph.height()
                    # calcola fattore intero di riduzione in altezza
                    k = max(1, int(round(src_h / max(1, left_h))))  # fattore intero
                    ph = ph.subsample(k, k)                      # riduzione approssimata
                    self._img_ref = ph                           # assegna
                    canvas_h = ph.height()                       # altezza effettiva ottenuta
                    canvas_w = ph.width()                        # larghezza proporzionale
            except Exception:
                self._img_ref = None
                canvas_w, canvas_h = 320, left_h
        else:
            self._img_ref = None
            canvas_w, canvas_h = 320, left_h

        # ridimensiona canvas alle dimensioni calcolate e ridisegna
        self.img_canvas.config(width=canvas_w, height=canvas_h)
        self.img_canvas.delete("all")
        if self._img_ref:
            self.img_canvas.create_image(0, 0, image=self._img_ref, anchor="nw", tags="img")  # ancorata in alto-sx
        else:
            self.img_canvas.create_text(canvas_w//2, canvas_h//2, text="Immagine non trovata", tags="img")

    # ---------- Pulsanti ----------
    def _on_click_disegna(self) -> None:
        """Valida input e lancia l'analisi in thread."""
        # reset annulla e disabilita pulsante
        self._cancel_event.clear()
        self.btn_disegna.state(["disabled"])
        # validazione minima
        try:
            q0 = float(self.q0_var.get().replace(",", ".")); assert q0 > 0
            xi = float(self.xi.get().replace(",", "."))
            ag_over_g = float(self.ag_over_g.get().replace(",", "."))
            F0 = float(self.f0.get().replace(",", "."))
            Tc_star = float(self.tc_star.get().replace(",", "."))
        except Exception:
            messagebox.showerror("Errore", "Controlla i valori numerici.")
            self.btn_disegna.state(["!disabled"])
            return
        # parametri per worker
        args = dict(
            suolo=self.suolo.get(), topo=self.topo.get(),
            q0=q0, KR=float(self.kr.get()),
            xi=xi, ag_over_g=ag_over_g, F0=F0, Tc_star=Tc_star,
        )
        # avvio thread
        self._worker = threading.Thread(target=self._run_analysis, args=(args,), daemon=True)
        self._worker.start()
        self.after(100, self._poll_worker)

    def _on_click_annulla(self) -> None:
        """Segnala annullamento e rimuove file creati se presenti."""
        self._cancel_event.set()
        # elimina subito e soltanto i file di output
        self._delete_output_files()
        # se nessun worker attivo, chiudi subito
        if not self._worker or not self._worker.is_alive():
            self.destroy()

    # ---------- Polling thread ----------
    def _poll_worker(self) -> None:
        """Controlla avanzamento; riabilita UI o chiude a fine corsa."""
        if self._worker and self._worker.is_alive():
            self.after(100, self._poll_worker)
        else:
            # worker terminato: riabilita e chiudi se annullato
            self.btn_disegna.state(["!disabled"])
            if self._cancel_event.is_set():
                # elimina e basta i file di output
                self._delete_output_files()
                self.destroy()

    # ---------- Sincronizzazione campi ----------
    def _on_dutt_change(self, *_):
        if self.dutt.get() == "NON DISSIPATIVO":
            self.q0_var.set("1.0"); self.q0_entry.state(["disabled"])
        else:
            self.q0_entry.state(["!disabled"])
            if self.q0_var.get() == "1.0": self.q0_var.set("4.0")

    def _update_KR(self, *_):
        self.kr.set(str(KR_MAP[self.reg.get()]))

    def _update_TR(self, *_):
        self.tr.set(str(TR_MAP[self.sl.get()]))

    # ---------- Analisi in worker ----------
    def _run_analysis(self, a: Dict[str, Any]) -> None:
        """Esegue i calcoli e salva file; rispetta annulla con checkpoint frequenti."""
        try:
            # parametri spettro
            par = params_spettro(a["suolo"], a["topo"], a["ag_over_g"], a["F0"], a["Tc_star"])
            ag = a["ag_over_g"] * g0
            q_lim = a["q0"] * a["KR"]
            if self._cancel_event.is_set(): return

            # griglia T e Sd
            T_base = np.linspace(0.0, max(4.0, par["TD"] * 1.2), 100); T_base[0] = max(T_base[1]*1e-6, 1e-9)
            T = _merge_T_arrays(T_base, np.array([par["TB"], par["TC"], par["TD"]], float))
            if self._cancel_event.is_set(): return

            Sd = sd_progetto(T, ag, par["S"], a["F0"], par["TB"], par["TC"], par["TD"], a["xi"], q_lim)
            Sd_over_g = Sd / g0
            eta = eta_corr(a["xi"])
            Sd_plateau_over_g = (ag * par["S"] * eta * a["F0"]) / (g0 * q_lim)
            if self._cancel_event.is_set(): return

            # figura in memoria
            fig = plt.figure(figsize=(12, 6), facecolor="white")  # faccia bianca per JPG
            ax = plt.gca()
            ax.set_facecolor('whitesmoke')
            ax.plot(T, Sd_over_g)
            ax.set_xlabel("T [s]", fontsize=14)
            ax.set_ylabel("Sd/g [m/s²]", fontsize=14)
            ax.set_title("Spettro di Progetto NTC18", fontsize=18, pad=10)
            ax.grid(True); ax.set_xlim(0, 4); ax.set_ylim(bottom=0)  # bottom fisso, top auto
            ymin, ymax = ax.get_ylim(); ymid = 0.5*(ymin + ymax)
            def vlabel(x, t, v):
                ax.axvline(x, linestyle="--", linewidth=0.9)
                ax.annotate(f"{t} = {v:.3f} s",(x, ymid),
                            xycoords="data", textcoords="offset points", xytext=(-12,-80),
                            ha="left", va="center", rotation=90, fontsize=12)
            vlabel(par["TB"], "TB", par["TB"]); vlabel(par["TC"], "TC", par["TC"]); vlabel(par["TD"], "TD", par["TD"])
            ax.text(par["TC"] + 0.05, Sd_plateau_over_g, f"Sd/g = {Sd_plateau_over_g:.3f} m/s²",
                    ha="left", va="center", fontsize=12)
            if self._cancel_event.is_set(): plt.close(fig); return

            # percorsi file standard + temporanei per sostituzione atomica
            base_dir = _get_output_dir()
            jpg_path = os.path.join(base_dir, "spettro_ntc18.jpg")
            txt_path = os.path.join(base_dir, "spettro_ntc18.txt")

            # crea path temporanei nello stesso folder
            tmp_jpg_fd, tmp_jpg = tempfile.mkstemp(prefix="tmp_", suffix=".jpg", dir=base_dir)
            os.close(tmp_jpg_fd)  # chiudi descrittore per non bloccare Windows
            tmp_txt_fd, tmp_txt = tempfile.mkstemp(prefix="tmp_", suffix=".txt", dir=base_dir)
            os.close(tmp_txt_fd)

            # salva JPG su temporaneo
            if not self._cancel_event.is_set():
                fig.savefig(tmp_jpg, dpi=300, bbox_inches="tight", format="jpg")
            plt.close(fig)
            if self._cancel_event.is_set():
                # pulizia temporanei e uscita
                try: os.remove(tmp_jpg)
                except Exception: pass
                try: os.remove(tmp_txt)
                except Exception: pass
                return

            # prepara array TXT
            arr = np.column_stack([T, Sd_over_g])
            arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
            keep = [True]
            for i in range(1, arr.shape[0]):
                keep.append(not np.isclose(arr[i,0], arr[i-1,0], rtol=0.0, atol=1e-9))
            arr = arr[np.array(keep, bool)]
            arr = arr[arr[:, 0] > 0.0]

            # salva TXT su temporaneo
            if not self._cancel_event.is_set():
                np.savetxt(tmp_txt, arr, fmt="%.6f", delimiter=" ", newline="\n")
            if self._cancel_event.is_set():
                # pulizia temporanei e uscita
                try: os.remove(tmp_jpg)
                except Exception: pass
                try: os.remove(tmp_txt)
                except Exception: pass
                return

            # sostituzioni atomiche; se esistono, vengono sovrascritte
            try:
                os.replace(tmp_jpg, jpg_path)
                os.replace(tmp_txt, txt_path)
            except Exception as e:
                # pulizia residui e rilancio
                try: os.remove(tmp_jpg)
                except Exception: pass
                try: os.remove(tmp_txt)
                except Exception: pass
                raise e

            self._last_jpg = jpg_path
            self._last_txt = txt_path

            # aggiorna risultati e chiudi GUI
            _RESULT.update(dict(
                TB=par["TB"], TC=par["TC"], TD=par["TD"], S=par["S"], SS=par["SS"], ST=par["ST"],
                q_lim=q_lim, eta=eta, Sd_plateau_over_g=Sd_plateau_over_g,
                Sd_over_g_TB=sd_progetto(np.array([par["TB"]]), ag, par["S"], a["F0"],
                                         par["TB"], par["TC"], par["TD"], a["xi"], q_lim)[0] / g0,
                Sd_over_g_TC=sd_progetto(np.array([par["TC"]]), ag, par["S"], a["F0"],
                                         par["TB"], par["TC"], par["TD"], a["xi"], q_lim)[0] / g0,
                Sd_over_g_TD=sd_progetto(np.array([par["TD"]]), ag, par["S"], a["F0"],
                                         par["TB"], par["TC"], par["TD"], a["xi"], q_lim)[0] / g0,
                jpg_path=self._last_jpg, txt_path=self._last_txt
            ))
            self.after(0, self.destroy)
        except Exception as e:
            # errore: mostra popup nel main thread
            self.after(0, lambda: messagebox.showerror("Errore", str(e)))

# =============== API PUBBLICA ===============
def run_spettro_ntc18_gui(output_dir: Optional[str] = None, show_plot: bool = False) -> Dict[str, Any]:
    """Apre la GUI, salva JPG/TXT, ritorna un dict con i risultati. Il grafico non viene mostrato."""
    global _OUTPUT_DIR, _RESULT
    _OUTPUT_DIR = output_dir
    _RESULT = {}
    App().mainloop()
    return dict(_RESULT)

# =============== AVVIO DIRETTO ===============
if __name__ == "__main__":
    info = run_spettro_ntc18_gui(output_dir=None, show_plot=False)
    if info:
        print("TB = {TB:.3f} s, TC = {TC:.3f} s, TD = {TD:.3f} s, q_lim = {q_lim:.3f}".format(**info))
        print("Sd/g @TB = {Sd_over_g_TB:.6f}".format(**info))
        print("Sd/g @TC = {Sd_over_g_TC:.6f}".format(**info))
        print("Sd/g @TD = {Sd_over_g_TD:.6f}".format(**info))
        print("File:", info["jpg_path"])
        print("File:", info["txt_path"])
