"""
Aplicação principal do Projeto PDI.
Interface gráfica Tkinter com layout em 3 colunas:
  - Esquerda:  lista de processos + painel de parâmetros
  - Centro:    imagem original
  - Direita:   imagem resultado + dados acessórios
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
from PIL import Image

# Tenta importar tkinterdnd2 para suporte a Drag & Drop nativo
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

from gui.panels import ImagePanel, MultiImagePanel, HistogramPanel, ROIImagePanel
from gui.widgets import LabeledEntry, LabeledSlider, LabeledCheckbox, LabeledOptionMenu, StyledButton

import processing.color as pcolor
import processing.intensity as pintensity
import processing.histogram as phist
import processing.spatial as pspatial
import processing.frequency as pfreq
import processing.adaptive as padaptive
import processing.noise as pnoise


# ─────────────────────────────────────────────────────────────────────────────
# Definição dos processos: nome, categoria, se exige grayscale, se exige RGB
# ─────────────────────────────────────────────────────────────────────────────
PROCESSES = [
    # (id, nome exibido, requer_gray, requer_rgb)
    ("rgb_decomp",       "1. Decomposição RGB",                False, True),
    ("hsv_decomp",       "2. Decomposição HSV",                False, True),
    ("threshold",        "3. Limiarização",                    True,  False),
    ("log_transform",    "4. Trafo. Logarítmica",              False, False),
    ("power_transform",  "5. Trafo. de Potência (Gamma)",      False, False),
    ("hist_equalize",    "6. Equalização de Histograma",       True,  False),
    ("intensity_slice",  "7. Fatiamento por Intensidade",      True,  False),
    ("gaussian_blur",    "8. Filtro Gaussiano (Média)",        False, False),
    ("median_filter",    "9a. Filtro de Mediana",              False, False),
    ("min_filter",       "9b. Filtro de Mínimo",               False, False),
    ("max_filter",       "9c. Filtro de Máximo",               False, False),
    ("unsharp_mask",     "10. Máscara de Aguçamento",          False, False),
    ("laplacian",        "11. Realce por Laplaciano",          True,  False),
    ("sobel",            "12. Gradiente de Sobel",             True,  False),
    ("gauss_lpf",        "13a. Passa-Baixa Gaussiano",         False, False),
    ("gauss_hpf",        "13b. Passa-Alta Gaussiano",          False, False),
    ("butter_lpf",       "14a. Passa-Baixa Butterworth",       False, False),
    ("butter_hpf",       "14b. Passa-Alta Butterworth",        False, False),
    ("adaptive_median",  "15. Mediana Adaptativa",             True,  False),
    ("gauss_noise",      "16. Ruído Gaussiano Aditivo",        False, False),
    ("sp_noise",         "17. Ruído Sal-e-Pimenta",            False, False),
    ("pseudo_color",     "18. Pseudo-coloração",               True,  False),
]


class App(TkinterDnD.Tk if _DND_AVAILABLE else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Processamento Digital de Imagens — UFPB")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self.minsize(1200, 720)

        self._img_original: Image.Image | None = None
        self._img_result: Image.Image | None = None
        self._img_first: Image.Image | None = None   # cópia da imagem recém-aberta
        self._is_gray: bool = False
        self._process_id: str = ""
        self._param_widgets: list = []

        # ROI
        self._roi_active: bool = False   # True = modo de desenho ativo no painel

        # Histórico de estados do resultado (pilha com limite)
        self._history: list[tuple[Image.Image, str]] = []
        self._history_index: int = -1
        self._MAX_HISTORY: int = 30

        self._build_menu()
        self._build_layout()
        self._update_process_list()
        self._setup_drag_drop()
        self._update_history_buttons()

    # ──────── Menu ────────
    def _build_menu(self):
        menubar = tk.Menu(self, bg="#313244", fg="#cdd6f4",
                          activebackground="#89b4fa", activeforeground="#1e1e2e",
                          relief="flat", bd=0)

        file_menu = tk.Menu(menubar, tearoff=0, bg="#313244", fg="#cdd6f4",
                            activebackground="#89b4fa", activeforeground="#1e1e2e")
        file_menu.add_command(label="Abrir imagem PNG…", command=self._open_image,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="Salvar resultado PNG…", command=self._save_result,
                              accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.quit)
        menubar.add_cascade(label="Arquivo", menu=file_menu)

        self.config(menu=menubar)
        self.bind_all("<Control-o>", lambda e: self._open_image())
        self.bind_all("<Control-s>", lambda e: self._save_result())
        self.bind_all("<Control-z>", lambda e: self._undo())
        self.bind_all("<Control-y>", lambda e: self._redo())
        self.bind_all("<Control-Z>", lambda e: self._redo())  # Ctrl+Shift+Z

    # ──────── Layout principal ────────
    def _build_layout(self):
        # Título
        tk.Label(
            self, text="Processamento Digital de Imagens",
            font=("Segoe UI", 15, "bold"),
            fg="#89b4fa", bg="#1e1e2e"
        ).pack(side="top", pady=(10, 0))
        tk.Label(
            self, text="UFPB — DSC — Prof. Augusto Tavares",
            font=("Segoe UI", 9),
            fg="#585b70", bg="#1e1e2e"
        ).pack(side="top", pady=(0, 8))

        # Container principal
        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=1)

        # ── Painel esquerdo: lista + parâmetros + botões ──
        left = tk.Frame(main, bg="#181825", width=310, relief="flat")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        tk.Label(left, text="Processos", font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#181825").grid(row=0, column=0,
                                                   padx=8, pady=(8, 4), sticky="w")

        # Lista de processos com scrollbar
        list_frame = tk.Frame(left, bg="#181825")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=8)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical", bg="#313244",
                                  troughcolor="#181825", activebackground="#45475a")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self._listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 9), bg="#1e1e2e", fg="#cdd6f4",
            selectbackground="#89b4fa", selectforeground="#1e1e2e",
            activestyle="none", relief="flat", bd=0,
            yscrollcommand=scrollbar.set,
            exportselection=False
        )
        self._listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_process_select)

        # Painel de parâmetros com scrollbar
        tk.Label(left, text="Parâmetros", font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#181825").grid(row=2, column=0,
                                                   padx=8, pady=(8, 2), sticky="w")

        # Frame scrollable para parâmetros
        param_outer = tk.Frame(left, bg="#181825")
        param_outer.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 4))
        param_outer.columnconfigure(0, weight=1)

        self._param_canvas = tk.Canvas(param_outer, bg="#181825", highlightthickness=0,
                                        height=260)
        param_scroll = tk.Scrollbar(param_outer, orient="vertical",
                                     command=self._param_canvas.yview,
                                     bg="#313244", troughcolor="#181825")
        self._param_canvas.configure(yscrollcommand=param_scroll.set)
        self._param_canvas.grid(row=0, column=0, sticky="ew")
        param_scroll.grid(row=0, column=1, sticky="ns")

        self._param_frame = tk.Frame(self._param_canvas, bg="#181825")
        self._param_frame_id = self._param_canvas.create_window(
            (0, 0), window=self._param_frame, anchor="nw"
        )
        self._param_frame.bind("<Configure>", self._on_param_frame_resize)
        self._param_canvas.bind("<Configure>", self._on_param_canvas_resize)

        # ── Ferramentas de manipulação básica ──
        tk.Label(left, text="Ferramentas", font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#181825").grid(row=4, column=0,
                                                   padx=8, pady=(8, 2), sticky="w")

        tools_frame = tk.Frame(left, bg="#181825")
        tools_frame.grid(row=5, column=0, padx=8, pady=(0, 4), sticky="ew")
        tools_frame.columnconfigure(0, weight=1)

        # Resize
        resize_lf = tk.LabelFrame(tools_frame, text="Redimensionar",
                                  font=("Segoe UI", 8, "bold"),
                                  fg="#a6e3a1", bg="#181825",
                                  bd=1, relief="groove")
        resize_lf.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        resize_lf.columnconfigure(0, weight=1)
        resize_lf.columnconfigure(1, weight=1)
        resize_lf.columnconfigure(2, weight=1)

        tk.Label(resize_lf, text="W:", font=("Segoe UI", 8),
                 fg="#cdd6f4", bg="#181825").grid(row=0, column=0, padx=(4, 0), pady=4)
        self._resize_w_var = tk.IntVar(value=512)
        self._resize_w_entry = tk.Entry(
            resize_lf, textvariable=self._resize_w_var, width=5,
            font=("Segoe UI", 9, "bold"),
            bg="#313244", fg="#a6e3a1",
            insertbackground="#a6e3a1",
            relief="flat", bd=4, justify="center"
        )
        self._resize_w_entry.grid(row=0, column=1, padx=2, pady=4)

        tk.Label(resize_lf, text="H:", font=("Segoe UI", 8),
                 fg="#cdd6f4", bg="#181825").grid(row=1, column=0, padx=(4, 0), pady=2)
        self._resize_h_var = tk.IntVar(value=512)
        self._resize_h_entry = tk.Entry(
            resize_lf, textvariable=self._resize_h_var, width=5,
            font=("Segoe UI", 9, "bold"),
            bg="#313244", fg="#a6e3a1",
            insertbackground="#a6e3a1",
            relief="flat", bd=4, justify="center"
        )
        self._resize_h_entry.grid(row=1, column=1, padx=2, pady=2)

        self._keep_aspect_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            resize_lf, text="Manter proporção",
            variable=self._keep_aspect_var,
            font=("Segoe UI", 8),
            bg="#181825", fg="#cdd6f4",
            selectcolor="#313244",
            activebackground="#181825",
            activeforeground="#89b4fa",
            relief="flat"
        ).grid(row=2, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="w")

        StyledButton(resize_lf, text="↔  Aplicar Resize",
                     command=self._apply_resize).grid(
            row=3, column=0, columnspan=3, padx=4, pady=(0, 6), sticky="ew")

        # Bind: quando W muda e manter proporção está ativo, atualiza H
        self._resize_w_var.trace_add("write", self._on_resize_w_changed)
        self._resize_h_var.trace_add("write", self._on_resize_h_changed)
        self._resize_aspect_lock = False  # evita recursão

        # Botão converter para cinza
        StyledButton(tools_frame, text="⬜  Converter para Escala de Cinza",
                     command=self._convert_to_gray).grid(
            row=1, column=0, pady=(0, 4), sticky="ew")

        # ── ROI ──
        roi_lf = tk.LabelFrame(tools_frame, text="Região de Interesse (ROI)",
                               font=("Segoe UI", 8, "bold"),
                               fg="#89b4fa", bg="#181825",
                               bd=1, relief="groove")
        roi_lf.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        roi_lf.columnconfigure(0, weight=1)
        roi_lf.columnconfigure(1, weight=1)

        self._btn_roi_toggle = tk.Button(
            roi_lf, text="✏  Selecionar ROI",
            command=self._toggle_roi_mode,
            font=("Segoe UI", 8, "bold"),
            bg="#313244", fg="#cdd6f4",
            activebackground="#45475a", activeforeground="#89b4fa",
            relief="flat", bd=0, padx=6, pady=5, cursor="hand2"
        )
        self._btn_roi_toggle.grid(row=0, column=0, sticky="ew", padx=(4, 2), pady=(4, 2))

        tk.Button(
            roi_lf, text="🗑  Limpar ROI",
            command=self._clear_roi,
            font=("Segoe UI", 8, "bold"),
            bg="#313244", fg="#f38ba8",
            activebackground="#45475a", activeforeground="#f38ba8",
            relief="flat", bd=0, padx=6, pady=5, cursor="hand2"
        ).grid(row=0, column=1, sticky="ew", padx=(2, 4), pady=(4, 2))

        self._roi_status_var = tk.StringVar(value="ROI: nenhuma")
        tk.Label(
            roi_lf, textvariable=self._roi_status_var,
            font=("Segoe UI", 8, "italic"),
            fg="#a6e3a1", bg="#181825",
            wraplength=240, justify="left"
        ).grid(row=1, column=0, columnspan=2, padx=4, pady=(0, 4), sticky="w")

        # ── Histórico: Desfazer / Refazer / Restaurar ──
        tk.Label(left, text="Histórico", font=("Segoe UI", 10, "bold"),
                 fg="#cdd6f4", bg="#181825").grid(row=6, column=0,
                                                   padx=8, pady=(8, 2), sticky="w")

        hist_frame = tk.Frame(left, bg="#181825")
        hist_frame.grid(row=7, column=0, padx=8, pady=(0, 4), sticky="ew")
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.columnconfigure(1, weight=1)

        self._btn_undo = tk.Button(
            hist_frame, text="\u21a9 Desfazer",
            command=self._undo,
            font=("Segoe UI", 8, "bold"),
            bg="#313244", fg="#cdd6f4",
            activebackground="#45475a", activeforeground="#cdd6f4",
            relief="flat", bd=0, padx=6, pady=5, cursor="hand2",
            state="disabled", disabledforeground="#45475a"
        )
        self._btn_undo.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self._btn_redo = tk.Button(
            hist_frame, text="\u21aa Refazer",
            command=self._redo,
            font=("Segoe UI", 8, "bold"),
            bg="#313244", fg="#cdd6f4",
            activebackground="#45475a", activeforeground="#cdd6f4",
            relief="flat", bd=0, padx=6, pady=5, cursor="hand2",
            state="disabled", disabledforeground="#45475a"
        )
        self._btn_redo.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        self._btn_restore = tk.Button(
            hist_frame, text="\u21ba Restaurar Original",
            command=self._restore_original,
            font=("Segoe UI", 8, "bold"),
            bg="#45475a", fg="#f38ba8",
            activebackground="#585b70", activeforeground="#f38ba8",
            relief="flat", bd=0, padx=6, pady=5, cursor="hand2",
            state="disabled", disabledforeground="#585b70"
        )
        self._btn_restore.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # Botão Aplicar processo
        btn_frame = tk.Frame(left, bg="#181825")
        btn_frame.grid(row=8, column=0, padx=8, pady=8, sticky="ew")
        StyledButton(btn_frame, text="▶  Aplicar Processo",
                     command=self._apply_process, accent=True).pack(fill="x")

        # Status bar
        self._status_var = tk.StringVar(value="Abra uma imagem PNG para começar.")
        tk.Label(left, textvariable=self._status_var,
                 font=("Segoe UI", 8), fg="#585b70", bg="#181825",
                 wraplength=290, justify="left"
                 ).grid(row=9, column=0, padx=8, pady=(0, 8), sticky="w")

        # ── Central: imagem original ──
        center = tk.Frame(main, bg="#1e1e2e")
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=0)

        self._panel_orig = ROIImagePanel(
            center,
            title="Imagem Original  (arraste uma imagem aqui 🖼)",
            on_roi_set=self._on_roi_set
        )
        self._panel_orig.grid(row=0, column=0, sticky="nsew")

        StyledButton(center, text="Abrir imagem…",
                     command=self._open_image).grid(row=1, column=0, pady=(6, 0))

        # ── Direita: resultado + acessórios ──
        right = tk.Frame(main, bg="#1e1e2e")
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=0)

        self._panel_result = ImagePanel(right, title="Imagem Resultado")
        self._panel_result.grid(row=0, column=0, sticky="nsew")

        # Sub-painel de acessórios (histograma OU decomposição)
        self._panel_hist = HistogramPanel(right)
        self._panel_hist.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self._panel_multi = MultiImagePanel(right, title="Informações Complementares")
        self._panel_multi.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self._panel_multi.grid_remove()  # Oculto por padrão

        StyledButton(right, text="Salvar resultado…",
                     command=self._save_result).grid(row=2, column=0, pady=(6, 0))

    def _on_param_frame_resize(self, _=None):
        self._param_canvas.configure(scrollregion=self._param_canvas.bbox("all"))

    def _on_param_canvas_resize(self, event):
        self._param_canvas.itemconfig(self._param_frame_id, width=event.width)

    # ──────── Lista de processos ────────
    def _update_process_list(self):
        self._listbox.delete(0, "end")
        self._process_items = []
        for pid, name, req_gray, req_rgb in PROCESSES:
            self._listbox.insert("end", f"  {name}")
            self._process_items.append((pid, name, req_gray, req_rgb))
        self._refresh_list_state()

    def _refresh_list_state(self):
        """Habilita/desabilita itens conforme tipo da imagem carregada."""
        for i, (pid, name, req_gray, req_rgb) in enumerate(self._process_items):
            if self._img_original is None:
                color = "#585b70"
            elif req_gray and not self._is_gray:
                color = "#45475a"  # Desabilitado (imagem colorida)
            elif req_rgb and self._is_gray:
                color = "#45475a"  # Desabilitado (imagem grayscale)
            else:
                color = "#cdd6f4"  # Habilitado
            self._listbox.itemconfig(i, fg=color)

    def _is_process_enabled(self, idx: int) -> bool:
        if self._img_original is None:
            return False
        _, _, req_gray, req_rgb = self._process_items[idx]
        if req_gray and not self._is_gray:
            return False
        if req_rgb and self._is_gray:
            return False
        return True

    # ──────── Seleção de processo ────────
    def _on_process_select(self, _=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if not self._is_process_enabled(idx):
            self._status_var.set("⚠ Este processo não está disponível para o tipo de imagem atual.")
            return
        pid, name, _, _ = self._process_items[idx]
        self._process_id = pid
        self._build_param_panel(pid)
        self._status_var.set(f"Processo selecionado: {name}")

    # ──────── Painel de parâmetros dinâmico ────────
    def _build_param_panel(self, pid: str):
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_widgets.clear()

        cfg = self._param_configs()
        if pid in cfg:
            for widget in cfg[pid]:
                widget.pack(fill="x", pady=3, padx=4)
                self._param_widgets.append(widget)
        else:
            tk.Label(self._param_frame, text="(sem parâmetros)",
                     font=("Segoe UI", 9, "italic"),
                     fg="#585b70", bg="#181825").pack(pady=8)

        # Atualiza scrollregion após construir
        self._param_frame.update_idletasks()
        self._param_canvas.configure(scrollregion=self._param_canvas.bbox("all"))

    def _param_configs(self) -> dict:
        """Retorna dicionário pid -> lista de widgets de parâmetro."""
        F = self._param_frame
        BG = "#181825"

        def entry(label, from_, to, default, res=1.0):
            return LabeledEntry(F, label, from_=from_, to=to,
                                default=default, resolution=res, bg=BG)

        def check(label, default=False):
            return LabeledCheckbox(F, label, default=default, bg=BG)

        def option(label, options, default=None):
            return LabeledOptionMenu(F, label, options=options, default=default, bg=BG)

        # Kernel size compartilhado (filtros espaciais)
        ksize = lambda: entry("Tamanho do kernel (ímpar)", 3, 31, 5, 2)

        return {
            "threshold":       [entry("Limiar k", 0, 255, 128, 1)],
            "log_transform":   [entry("Ganho c", 0.1, 10.0, 1.0, 0.1)],
            "power_transform": [entry("Ganho c", 0.1, 5.0, 1.0, 0.1),
                                entry("Gama γ", 0.1, 5.0, 1.0, 0.1)],
            "intensity_slice": [entry("Mínimo A", 0, 255, 100, 1),
                                entry("Máximo B", 0, 255, 200, 1),
                                check("Preservar fundo", True)],
            "gaussian_blur":   [entry("Desvio padrão σ", 0.1, 20.0, 2.0, 0.1),
                                ksize()],
            "median_filter":   [ksize()],
            "min_filter":      [ksize()],
            "max_filter":      [ksize()],
            "unsharp_mask":    [entry("Ganho k", 0.1, 5.0, 1.5, 0.1),
                                entry("Desvio padrão σ", 0.1, 10.0, 2.0, 0.1),
                                ksize()],
            "laplacian":       [check("Usar diagonais (8-viz.)", True)],
            "gauss_lpf":       [entry("Frequência de corte D₀", 1, 200, 30, 1)],
            "gauss_hpf":       [entry("Frequência de corte D₀", 1, 200, 30, 1)],
            "butter_lpf":      [entry("Frequência de corte D₀", 1, 200, 30, 1),
                                entry("Ordem n", 1, 10, 2, 1)],
            "butter_hpf":      [entry("Frequência de corte D₀", 1, 200, 30, 1),
                                entry("Ordem n", 1, 10, 2, 1)],
            "adaptive_median": [entry("Tamanho máx. janela", 3, 25, 7, 2)],
            "gauss_noise":     [entry("Média μ", -50, 50, 0, 1),
                                entry("Desvio padrão σ", 1, 100, 20, 1)],
            "sp_noise":        [option("Tipo de ruído", ["ambos", "sal", "pimenta"], "ambos"),
                                entry("Prob. sal (%)", 0.1, 20.0, 2.0, 0.1),
                                entry("Prob. pimenta (%)", 0.1, 20.0, 2.0, 0.1)],
            "pseudo_color":    [option("Colormap", pcolor.COLORMAPS, "jet")],
        }

    def _get_param_values(self) -> list:
        """Retorna valores atuais dos widgets de parâmetro."""
        vals = []
        for w in self._param_widgets:
            if isinstance(w, (LabeledEntry, LabeledSlider)):
                vals.append(w.get())
            elif isinstance(w, LabeledCheckbox):
                vals.append(w.get())
            elif isinstance(w, LabeledOptionMenu):
                vals.append(w.get())
        return vals

    # ──────── Drag & Drop ────────
    def _setup_drag_drop(self):
        """Registra o painel original como alvo de Drag & Drop."""
        if _DND_AVAILABLE:
            self._panel_orig.drop_target_register(DND_FILES)
            self._panel_orig.dnd_bind("<<Drop>>", self._on_drop)
            self._panel_orig._canvas.drop_target_register(DND_FILES)
            self._panel_orig._canvas.dnd_bind("<<Drop>>", self._on_drop)
        # Destaca visualmente o painel como drop target
        self._panel_orig.config(relief="groove")

    def _on_drop(self, event):
        """Callback chamado quando um arquivo é soltado sobre o painel."""
        # O caminho pode vir entre chaves em sistemas Linux: {/path/to/file}
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            path = raw[1:-1]
        else:
            path = raw.split()[0]  # Pega o primeiro arquivo se houver múltiplos
        self._load_image_from_path(path)

    # ──────── Abrir / Salvar ────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Todos os arquivos", "*.*")
            ]
        )
        if not path:
            return
        self._load_image_from_path(path)

    def _load_image_from_path(self, path: str):
        """Carrega uma imagem a partir do caminho e atualiza o estado da UI."""
        try:
            img = Image.open(path)
            # Detecta se é grayscale
            if img.mode == "L":
                self._is_gray = True
                self._img_original = img
            elif img.mode in ("RGB", "RGBA", "P"):
                converted = img.convert("RGB")
                # Verifica se todos os canais são iguais (grayscale disfarçado)
                arr = np.array(converted)
                if np.all(arr[:, :, 0] == arr[:, :, 1]) and \
                   np.all(arr[:, :, 1] == arr[:, :, 2]):
                    self._img_original = img.convert("L")
                    self._is_gray = True
                else:
                    self._img_original = converted
                    self._is_gray = False
            else:
                self._img_original = img.convert("RGB")
                self._is_gray = False

            self._img_result = None
            self._panel_orig.set_image(self._img_original)
            self._panel_result.clear()
            self._panel_hist.clear()
            self._panel_multi.clear()
            self._refresh_list_state()

            # Guarda cópia da imagem original e limpa histórico
            self._img_first = self._img_original.copy()
            self._history.clear()
            self._history_index = -1

            # Atualiza campos de resize com dimensões reais
            w, h = self._img_original.size
            self._resize_aspect_lock = True
            self._resize_w_var.set(w)
            self._resize_h_var.set(h)
            self._resize_aspect_lock = False

            tipo = "Escala de cinza (8 bits)" if self._is_gray else "Colorida RGB (24 bits)"
            nome = path.split("/")[-1]
            self._status_var.set(f"✓ {nome} | {w}×{h} | {tipo}")
            self._update_history_buttons()
        except Exception as e:
            messagebox.showerror("Erro ao abrir imagem", str(e))

    def _save_result(self):
        if self._img_result is None:
            messagebox.showwarning("Nenhum resultado", "Aplique um processo antes de salvar.")
            return
        path = filedialog.asksaveasfilename(
            title="Salvar resultado",
            defaultextension=".png",
            filetypes=[("Imagem PNG", "*.png")]
        )
        if not path:
            return
        try:
            self._img_result.save(path)
            self._status_var.set(f"✓ Salvo em: {path.split('/')[-1]}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

    # ──────── Resize ────────
    def _on_resize_w_changed(self, *_):
        """Atualiza H proporcionalmente quando W muda (se 'manter proporção' ativo)."""
        if self._resize_aspect_lock or not self._keep_aspect_var.get():
            return
        if self._img_original is None:
            return
        try:
            new_w = self._resize_w_var.get()
            if new_w <= 0:
                return
            orig_w, orig_h = self._img_original.size
            new_h = max(1, round(new_w * orig_h / orig_w))
            self._resize_aspect_lock = True
            self._resize_h_var.set(new_h)
            self._resize_aspect_lock = False
        except (tk.TclError, ValueError):
            pass

    def _on_resize_h_changed(self, *_):
        """Atualiza W proporcionalmente quando H muda (se 'manter proporção' ativo)."""
        if self._resize_aspect_lock or not self._keep_aspect_var.get():
            return
        if self._img_original is None:
            return
        try:
            new_h = self._resize_h_var.get()
            if new_h <= 0:
                return
            orig_w, orig_h = self._img_original.size
            new_w = max(1, round(new_h * orig_w / orig_h))
            self._resize_aspect_lock = True
            self._resize_w_var.set(new_w)
            self._resize_aspect_lock = False
        except (tk.TclError, ValueError):
            pass

    def _apply_resize(self):
        """Redimensiona a imagem (resultado no painel direito; original inalterada)."""
        if self._img_original is None:
            messagebox.showwarning("Sem imagem", "Carregue uma imagem primeiro.")
            return
        try:
            new_w = int(self._resize_w_var.get())
            new_h = int(self._resize_h_var.get())
            if new_w <= 0 or new_h <= 0:
                raise ValueError("Dimensões devem ser positivas.")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Dimensões inválidas",
                                 f"Informe valores inteiros positivos para W e H.\n{e}")
            return

        orig_w, orig_h = self._img_original.size
        resized = self._img_original.resize((new_w, new_h), Image.LANCZOS)

        # Coloca no painel resultado; original permanece inalterada
        self._img_result = resized
        self._panel_result.set_image(resized)
        self._panel_hist.grid()
        self._panel_multi.grid_remove()
        if resized.mode == "L":
            hist = phist.compute_histogram(resized)
            self._panel_hist.show_histograms(hist)
        else:
            self._panel_hist.clear()

        desc = f"Resize {orig_w}×{orig_h}→{new_w}×{new_h}"
        self._history_push(resized, desc)
        tipo = "Escala de cinza (8 bits)" if self._is_gray else "Colorida RGB (24 bits)"
        self._status_var.set(
            f"✓ {desc} | {tipo}  (Ctrl+Z para desfazer)"
        )

    # ──────── Conversão para Escala de Cinza ────────
    def _convert_to_gray(self):
        """Converte para cinza 8-bit (resultado no painel direito; original inalterada)."""
        if self._img_original is None:
            messagebox.showwarning("Sem imagem", "Carregue uma imagem primeiro.")
            return
        if self._is_gray:
            messagebox.showinfo("Já em cinza",
                                "A imagem já está em escala de cinza (8 bits).")
            return

        gray = self._img_original.convert("L")

        # Coloca no painel resultado; original permanece colorida e inalterada
        self._img_result = gray
        self._panel_result.set_image(gray)
        self._panel_hist.grid()
        self._panel_multi.grid_remove()
        hist = phist.compute_histogram(gray)
        self._panel_hist.show_histograms(hist)

        self._history_push(gray, "Converter para Cinza")
        w, h = gray.size
        self._status_var.set(
            f"✓ Escala de Cinza (8 bits) | {w}×{h}  — original preservada  (Ctrl+Z para desfazer)"
        )

    # ──────── ROI ────────
    def _toggle_roi_mode(self):
        """Ativa/desativa o modo de seleção de ROI no painel original."""
        self._roi_active = not self._roi_active
        self._panel_orig.set_roi_mode(self._roi_active)
        if self._roi_active:
            self._btn_roi_toggle.config(
                text="❌  Cancelar ROI",
                fg="#f38ba8",
                bg="#45475a"
            )
            self._roi_status_var.set("ROI: clique e arraste na imagem…")
        else:
            self._btn_roi_toggle.config(
                text="✏  Selecionar ROI",
                fg="#cdd6f4",
                bg="#313244"
            )
            roi = self._panel_orig.get_roi()
            if roi:
                x0, y0, x1, y1 = roi
                self._roi_status_var.set(f"ROI: ({x0},{y0}) → ({x1},{y1})")
            else:
                self._roi_status_var.set("ROI: nenhuma")

    def _clear_roi(self):
        """Remove a ROI e desativa o modo de seleção."""
        self._roi_active = False
        self._panel_orig.set_roi_mode(False)
        self._panel_orig.clear_roi()
        self._btn_roi_toggle.config(
            text="✏  Selecionar ROI",
            fg="#cdd6f4",
            bg="#313244"
        )
        self._roi_status_var.set("ROI: nenhuma")

    def _on_roi_set(self, roi):
        """Callback chamado pelo ROIImagePanel quando a seleção termina."""
        # Desativa automaticamente o modo de desenho após finalizar
        self._roi_active = False
        self._panel_orig.set_roi_mode(False)
        self._btn_roi_toggle.config(
            text="✏  Selecionar ROI",
            fg="#cdd6f4",
            bg="#313244"
        )
        if roi:
            x0, y0, x1, y1 = roi
            self._roi_status_var.set(f"ROI: ({x0},{y0}) → ({x1},{y1})")
            self._status_var.set(
                f"✓ ROI definida: ({x0},{y0}) → ({x1},{y1})  — aplique um processo para usar"
            )
        else:
            self._roi_status_var.set("ROI: inválida, tente novamente")

    # ──────── Sistema de Histórico ────────
    def _history_push(self, img: Image.Image, description: str):
        """Empilha o estado atual após processar; descarta estados 'futuros' se houver."""
        if img is None:
            return
        # Descarta tudo após o índice atual (branch de redo)
        self._history = self._history[: self._history_index + 1]
        # Limita tamanho da pilha
        if len(self._history) >= self._MAX_HISTORY:
            self._history.pop(0)
        self._history.append((img.copy(), description))
        self._history_index = len(self._history) - 1
        self._update_history_buttons()

    def _undo(self):
        """Desfaz a última operação, recuando um estado no histórico."""
        if self._history_index < 0:
            return  # Nada para desfazer
        if self._history_index == 0:
            # Volta para o estado "sem resultado" (antes do primeiro processo)
            self._history_index = -1
            self._img_result = None
            self._panel_result.clear()
            self._panel_hist.clear()
            self._panel_multi.clear()
            self._status_var.set("↩ Desfeito — sem resultado (imagem original no painel)")
        else:
            self._history_index -= 1
            img, desc = self._history[self._history_index]
            self._img_result = img
            self._panel_result.set_image(img)
            # Atualiza histograma se grayscale
            self._panel_multi.grid_remove()
            self._panel_hist.grid()
            if img.mode == "L":
                self._panel_hist.show_histograms(phist.compute_histogram(img))
            else:
                self._panel_hist.clear()
            self._status_var.set(f"↩ Desfeito → voltou para: {desc}")
        self._update_history_buttons()

    def _redo(self):
        """Refaz a operação desfeita, avançando um estado no histórico."""
        if self._history_index >= len(self._history) - 1:
            return  # Não há nada para refazer
        self._history_index += 1
        img, desc = self._history[self._history_index]
        self._img_result = img
        self._panel_result.set_image(img)
        self._panel_multi.grid_remove()
        self._panel_hist.grid()
        if img.mode == "L":
            self._panel_hist.show_histograms(phist.compute_histogram(img))
        else:
            self._panel_hist.clear()
        self._status_var.set(f"↪ Refeito → {desc}")
        self._update_history_buttons()

    def _restore_original(self):
        """Restaura a imagem para o estado em que foi aberta, descartando todo o histórico."""
        if self._img_first is None:
            messagebox.showwarning("Sem imagem original", "Nenhuma imagem foi carregada.")
            return
        confirm = messagebox.askyesno(
            "Restaurar Original",
            "Isso descartará TODO o histórico de processamento.\n"
            "A imagem retornará ao estado em que foi aberta.\n\n"
            "Deseja continuar?"
        )
        if not confirm:
            return
        # Restaura estado
        self._img_original = self._img_first.copy()
        self._is_gray = (self._img_first.mode == "L")
        self._img_result = None
        self._history.clear()
        self._history_index = -1
        # Atualiza UI
        self._panel_orig.set_image(self._img_original)
        self._panel_result.clear()
        self._panel_hist.clear()
        self._panel_multi.clear()
        self._refresh_list_state()
        w, h = self._img_original.size
        self._resize_aspect_lock = True
        self._resize_w_var.set(w)
        self._resize_h_var.set(h)
        self._resize_aspect_lock = False
        tipo = "Escala de cinza (8 bits)" if self._is_gray else "Colorida RGB (24 bits)"
        self._status_var.set(f"↺ Imagem original restaurada | {w}×{h} | {tipo}")
        self._update_history_buttons()

    def _update_history_buttons(self):
        """Habilita/desabilita os botões de Undo, Redo e Restaurar conforme o estado."""
        can_undo    = self._history_index >= 0
        can_redo    = self._history_index < len(self._history) - 1
        can_restore = self._img_first is not None
        self._btn_undo.config(   state="normal"   if can_undo    else "disabled")
        self._btn_redo.config(   state="normal"   if can_redo    else "disabled")
        self._btn_restore.config(state="normal"   if can_restore else "disabled")

    # ──────── Aplicar processo ────────
    def _apply_process(self):
        if self._img_original is None:
            messagebox.showwarning("Sem imagem", "Carregue uma imagem primeiro.")
            return

        # Verifica se há item selecionado na lista
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showwarning("Sem processo", "Selecione um processo na lista.")
            return

        # Verifica se o item selecionado está habilitado para o tipo de imagem atual
        idx = sel[0]
        if not self._is_process_enabled(idx):
            pid, name, req_gray, req_rgb = self._process_items[idx]
            if req_rgb:
                motivo = "requer imagem colorida RGB"
            elif req_gray:
                motivo = "requer imagem em escala de cinza"
            else:
                motivo = "não disponível para este tipo de imagem"
            messagebox.showwarning(
                "Processo não disponível",
                f"'{name}' {motivo}.\n\nPor favor, selecione um processo compatível com a imagem carregada."
            )
            self._status_var.set(f"⚠ '{name}' não disponível: {motivo}.")
            return

        # Garante que o process_id está atualizado com a seleção atual
        pid, name, _, _ = self._process_items[idx]
        if self._process_id != pid:
            self._process_id = pid
            self._build_param_panel(pid)

        self._status_var.set("⏳ Processando…")
        self.update_idletasks()

        # Roda em thread separada para não travar a UI
        threading.Thread(target=self._run_process, daemon=True).start()

    def _run_process(self):
        try:
            result = self._dispatch()
            self.after(0, self._on_process_done, result)
        except Exception as e:
            self.after(0, self._on_process_error, str(e))

    def _on_process_done(self, result):
        pid = self._process_id
        name = next(n for p, n, _, _ in self._process_items if p == pid)

        # Reset painéis acessórios
        self._panel_hist.grid()
        self._panel_multi.grid_remove()

        if pid in ("rgb_decomp", "hsv_decomp"):
            # result é dict de imagens
            self._panel_multi.set_images(result)
            self._panel_hist.grid_remove()
            self._panel_multi.grid()
            # Usa o primeiro canal como resultado principal
            first_key = list(result.keys())[0]
            self._img_result = result[first_key]
            self._panel_result.set_image(self._img_result)

        elif pid == "hist_equalize":
            img_eq, hist_before, hist_after = result
            self._img_result = img_eq
            self._panel_result.set_image(img_eq)
            self._panel_hist.show_histograms(hist_before, hist_after)

        elif pid == "laplacian":
            img_enhanced, img_lap = result
            self._img_result = img_enhanced
            self._panel_result.set_image(img_enhanced)
            # Exibe o laplaciano no painel acessório
            self._panel_multi.set_images({"Laplaciano ajustado": img_lap})
            self._panel_hist.grid_remove()
            self._panel_multi.grid()

        elif pid == "sobel":
            # result é (img_gradient, img_gx, img_gy)
            img_grad, img_gx, img_gy = result
            self._img_result = img_grad
            self._panel_result.set_image(img_grad)
            self._panel_multi.set_images({"Gx (horizontal)": img_gx, "Gy (vertical)": img_gy})
            self._panel_hist.grid_remove()
            self._panel_multi.grid()

        else:
            self._img_result = result
            self._panel_result.set_image(result)
            # Mostra histograma da imagem resultado (grayscale)
            if result.mode == "L":
                hist = phist.compute_histogram(result)
                self._panel_hist.show_histograms(hist)
            else:
                self._panel_hist.clear()

        self._status_var.set(f"✓ {name} aplicado com sucesso.  (Ctrl+Z para desfazer)")
        self._history_push(self._img_result, name)

    def _on_process_error(self, msg: str):
        messagebox.showerror("Erro no processamento", msg)
        self._status_var.set(f"✗ Erro: {msg[:60]}")

    # ──────── Dispatch de processos ────────

    # Processos cujo resultado é multi-imagem (dict) — ROI não aplicável
    _ROI_INCOMPATIBLE = {"rgb_decomp", "hsv_decomp", "sobel", "laplacian"}

    def _dispatch(self):
        pid = self._process_id
        img = self._img_original
        p = self._get_param_values()

        # Verifica ROI
        roi = self._panel_orig.get_roi()
        use_roi = (roi is not None) and (pid not in self._ROI_INCOMPATIBLE)

        if use_roi:
            # Recorta apenas a região da ROI e processa sobre ela
            x0, y0, x1, y1 = roi
            ow, oh = img.size
            x0c = max(0, min(x0, ow)); x1c = max(0, min(x1, ow))
            y0c = max(0, min(y0, oh)); y1c = max(0, min(y1, oh))
            img_crop = img.crop((x0c, y0c, x1c, y1c))
            roi_clipped = (x0c, y0c, x1c, y1c)
            result = self._dispatch_core(pid, img_crop, p)
            # Se o resultado é uma tupla (ex: hist_equalize), a imagem é o
            # primeiro elemento; composita de volta e retorna nova tupla.
            if isinstance(result, tuple):
                img_roi = result[0]
                composed = self._apply_roi_result(img, img_roi, roi_clipped)
                return (composed,) + result[1:]
            return self._apply_roi_result(img, result, roi_clipped)
        else:
            return self._dispatch_core(pid, img, p)

    def _apply_roi_result(self, original: Image.Image, roi_result: Image.Image,
                          roi: tuple) -> Image.Image:
        """
        Recorta `roi_result` para o tamanho da ROI e cola sobre uma cópia
        da imagem original. Garante compatibilidade de modo entre os dois.
        """
        x0, y0, x1, y1 = roi
        # Clipa coordenadas à imagem
        ow, oh = original.size
        x0 = max(0, min(x0, ow)); x1 = max(0, min(x1, ow))
        y0 = max(0, min(y0, oh)); y1 = max(0, min(y1, oh))
        rw, rh = x1 - x0, y1 - y0
        if rw <= 0 or rh <= 0:
            return roi_result  # Fallback: retorna resultado completo

        # Redimensiona roi_result para o tamanho exato da ROI (caso difira)
        roi_crop = roi_result.resize((rw, rh), Image.LANCZOS)

        # Cópia do original no modo correto
        out_mode = roi_result.mode  # O resultado pode mudar de modo (ex: L→RGB)
        base = original.convert(out_mode).copy()
        base.paste(roi_crop, (x0, y0))
        return base

    def _dispatch_core(self, pid: str, img: Image.Image, p: list):
        """Executa o processo puro (sem ROI) e retorna o resultado."""

        if pid == "rgb_decomp":
            return pcolor.decompose_rgb(img)

        elif pid == "hsv_decomp":
            return pcolor.decompose_hsv(img)

        elif pid == "threshold":
            return pintensity.threshold(img, int(p[0]))

        elif pid == "log_transform":
            return pintensity.log_transform(img, float(p[0]))

        elif pid == "power_transform":
            return pintensity.power_transform(img, float(p[0]), float(p[1]))

        elif pid == "hist_equalize":
            img_eq = phist.equalize_histogram(img)
            hist_before = phist.compute_histogram(img)
            hist_after = phist.compute_histogram(img_eq)
            return img_eq, hist_before, hist_after

        elif pid == "intensity_slice":
            return pintensity.intensity_slicing(img, int(p[0]), int(p[1]), bool(p[2]))

        elif pid == "gaussian_blur":
            sigma = float(p[0])
            ksize = int(p[1])
            if ksize % 2 == 0:
                ksize += 1
            return pspatial.gaussian_blur(img, sigma, ksize)

        elif pid == "median_filter":
            ksize = int(p[0])
            if ksize % 2 == 0:
                ksize += 1
            return pspatial.apply_median_filter(img, ksize)

        elif pid == "min_filter":
            ksize = int(p[0])
            if ksize % 2 == 0:
                ksize += 1
            return pspatial.apply_min_filter(img, ksize)

        elif pid == "max_filter":
            ksize = int(p[0])
            if ksize % 2 == 0:
                ksize += 1
            return pspatial.apply_max_filter(img, ksize)

        elif pid == "unsharp_mask":
            gain = float(p[0])
            sigma = float(p[1])
            ksize = int(p[2])
            if ksize % 2 == 0:
                ksize += 1
            return pspatial.unsharp_masking(img, gain, ksize, sigma)

        elif pid == "laplacian":
            use_diag = bool(p[0])
            return pspatial.laplacian_enhance(img, use_diag)

        elif pid == "sobel":
            return pspatial.sobel_gradient(img)

        elif pid == "gauss_lpf":
            return pfreq.gaussian_lpf(img, float(p[0]))

        elif pid == "gauss_hpf":
            return pfreq.gaussian_hpf(img, float(p[0]))

        elif pid == "butter_lpf":
            return pfreq.butterworth_lpf(img, float(p[0]), int(p[1]))

        elif pid == "butter_hpf":
            return pfreq.butterworth_hpf(img, float(p[0]), int(p[1]))

        elif pid == "adaptive_median":
            max_sz = int(p[0])
            if max_sz % 2 == 0:
                max_sz += 1
            return padaptive.adaptive_median_filter(img, max_sz)

        elif pid == "gauss_noise":
            return pnoise.add_gaussian_noise(img, float(p[0]), float(p[1]))

        elif pid == "sp_noise":
            # p[0] = tipo ("ambos"/"sal"/"pimenta"), p[1] = prob_sal, p[2] = prob_pimenta
            noise_type = str(p[0])
            prob_salt = float(p[1]) / 100.0
            prob_pepper = float(p[2]) / 100.0
            return pnoise.add_salt_pepper_unified(img, prob_salt, prob_pepper, noise_type)

        elif pid == "pseudo_color":
            colormap = str(p[0])
            return pcolor.pseudo_colorize(img, colormap)

        else:
            raise ValueError(f"Processo desconhecido: {pid}")
