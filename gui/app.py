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

from gui.panels import ImagePanel, MultiImagePanel, HistogramPanel
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
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Processamento Digital de Imagens — UFPB")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self.minsize(1200, 720)

        self._img_original: Image.Image | None = None
        self._img_result: Image.Image | None = None
        self._is_gray: bool = False
        self._process_id: str = ""
        self._param_widgets: list = []

        self._build_menu()
        self._build_layout()
        self._update_process_list()

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

        # Botão Aplicar
        btn_frame = tk.Frame(left, bg="#181825")
        btn_frame.grid(row=4, column=0, padx=8, pady=8, sticky="ew")
        StyledButton(btn_frame, text="▶  Aplicar Processo",
                     command=self._apply_process, accent=True).pack(fill="x")

        # Status bar
        self._status_var = tk.StringVar(value="Abra uma imagem PNG para começar.")
        tk.Label(left, textvariable=self._status_var,
                 font=("Segoe UI", 8), fg="#585b70", bg="#181825",
                 wraplength=290, justify="left"
                 ).grid(row=5, column=0, padx=8, pady=(0, 8), sticky="w")

        # ── Central: imagem original ──
        center = tk.Frame(main, bg="#1e1e2e")
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=0)

        self._panel_orig = ImagePanel(center, title="Imagem Original")
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

    # ──────── Abrir / Salvar ────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagem PNG",
            filetypes=[("Imagens PNG", "*.png"), ("Todos os arquivos", "*.*")]
        )
        if not path:
            return
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

            tipo = "Escala de cinza (8 bits)" if self._is_gray else "Colorida RGB (24 bits)"
            w, h = self._img_original.size
            self._status_var.set(f"✓ {path.split('/')[-1]} | {w}×{h} | {tipo}")
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

        self._status_var.set(f"✓ {name} aplicado com sucesso.")

    def _on_process_error(self, msg: str):
        messagebox.showerror("Erro no processamento", msg)
        self._status_var.set(f"✗ Erro: {msg[:60]}")

    # ──────── Dispatch de processos ────────
    def _dispatch(self):
        pid = self._process_id
        img = self._img_original
        p = self._get_param_values()

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

        else:
            raise ValueError(f"Processo desconhecido: {pid}")
