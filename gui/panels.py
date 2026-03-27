"""
Painéis da interface gráfica do projeto PDI.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImagePanel(tk.LabelFrame):
    """
    Painel interativo para exibir uma imagem PIL com:
      - Zoom via scroll do mouse (centrado no cursor)
      - Pan via arrasto com botão esquerdo
      - Conta-gotas: exibe coordenadas e valores de pixel ao passar o mouse
      - Duplo-clique para ajustar zoom ao tamanho do painel
    """
    MIN_ZOOM   = 0.02
    MAX_ZOOM   = 30.0
    ZOOM_STEP  = 1.15   # Fator de ampliação/redução por passo de scroll

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(
            parent, text=title,
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa", bg="#1e1e2e",
            bd=2, relief="groove",
            **kwargs
        )
        # ── Estado interno ──
        self._pil_image: Image.Image | None = None
        self._zoom      = 1.0
        self._offset_x  = 0.0   # posição no canvas onde o canto TL da imagem é desenhado
        self._offset_y  = 0.0
        self._pan_last  = None   # última posição do mouse durante arrasto
        self._tk_image  = None   # mantém referência para evitar GC

        # ── Canvas de exibição ──
        self._canvas = tk.Canvas(
            self, bg="#181825",
            highlightthickness=0,
            cursor="crosshair"
        )
        self._canvas.pack(fill="both", expand=True, padx=6, pady=(6, 2))

        # ── Barra de informação (conta-gotas + zoom) ──
        self._info_var = tk.StringVar(value=self._HINT)
        self._info_bar = tk.Label(
            self, textvariable=self._info_var,
            font=("Courier New", 8),
            fg="#a6e3a1", bg="#1e1e2e",
            anchor="w", padx=6
        )
        self._info_bar.pack(fill="x", pady=(0, 4))

        # ── Bindings ──
        cv = self._canvas
        cv.bind("<Configure>",        self._on_configure)
        # Zoom — Linux (Button-4/5) e Windows/Mac (MouseWheel)
        cv.bind("<Button-4>",         lambda e: self._zoom_at(e.x, e.y, self.ZOOM_STEP))
        cv.bind("<Button-5>",         lambda e: self._zoom_at(e.x, e.y, 1.0 / self.ZOOM_STEP))
        cv.bind("<MouseWheel>",       self._on_mousewheel)
        # Pan
        cv.bind("<ButtonPress-1>",    self._on_pan_start)
        cv.bind("<B1-Motion>",        self._on_pan_drag)
        cv.bind("<ButtonRelease-1>",  self._on_pan_end)
        # Conta-gotas (movimento do mouse)
        cv.bind("<Motion>",           self._on_mouse_move)
        cv.bind("<Leave>",            lambda e: self._info_var.set(self._zoom_text()))
        # Duplo-clique → ajustar zoom
        cv.bind("<Double-Button-1>",  lambda e: self._fit_and_center())

    # ────────────────────────── textos auxiliares ──────────────────────────
    _HINT = "scroll: zoom  ·  arrastar: pan  ·  duplo-clique: ajustar"

    def _zoom_text(self) -> str:
        return f"🔍 {self._zoom * 100:.0f}%  —  {self._HINT}"

    # ────────────────────────── API pública ──────────────────────────
    def set_image(self, img: Image.Image):
        """Carrega uma nova imagem e ajusta o zoom para caber no painel."""
        if img is None:
            self.clear()
            return
        self._pil_image = img.copy()
        self._fit_and_center()
        self._info_var.set(self._zoom_text())

    def clear(self):
        """Remove a imagem e exibe o placeholder."""
        self._pil_image = None
        self._zoom      = 1.0
        self._offset_x  = self._offset_y = 0.0
        self._tk_image  = None
        self._info_var.set(self._HINT)
        self._canvas.delete("all")
        self._draw_placeholder()

    # ────────────────────────── placeholder ──────────────────────────
    def _draw_placeholder(self):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        cx = cw // 2 if cw > 1 else 200
        cy = ch // 2 if ch > 1 else 150
        self._canvas.create_text(
            cx, cy,
            text="Sem imagem",
            fill="#45475a",
            font=("Segoe UI", 13, "italic"),
            tags="placeholder",
            anchor="center"
        )

    # ────────────────────────── zoom / fit ──────────────────────────
    def _fit_and_center(self):
        """Ajusta zoom para que a imagem caiba no canvas e a centraliza."""
        if self._pil_image is None:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            # Canvas ainda não foi redimensionado — tenta novamente mais tarde
            self._canvas.after(50, self._fit_and_center)
            return
        iw, ih = self._pil_image.size
        self._zoom     = min(cw / iw, ch / ih)
        self._offset_x = (cw - iw * self._zoom) / 2.0
        self._offset_y = (ch - ih * self._zoom) / 2.0
        self._render()
        self._info_var.set(self._zoom_text())

    def _zoom_at(self, cx: float, cy: float, factor: float):
        """Aplica fator de zoom mantendo o ponto (cx, cy) fixo no canvas."""
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        scale         = new_zoom / self._zoom
        self._offset_x = cx - scale * (cx - self._offset_x)
        self._offset_y = cy - scale * (cy - self._offset_y)
        self._zoom     = new_zoom
        self._render()
        self._info_var.set(self._zoom_text())

    # ────────────────────────── renderização ──────────────────────────
    def _render(self):
        """
        Renderiza apenas a região visível da imagem no canvas.
        Eficiente para imagens grandes ou zoom elevado.
        """
        if self._pil_image is None:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        iw, ih   = self._pil_image.size
        zoom     = self._zoom
        ox, oy   = self._offset_x, self._offset_y

        # Região da imagem visível (coordenadas de pixel da imagem)
        src_x0 = max(0.0, -ox / zoom)
        src_y0 = max(0.0, -oy / zoom)
        src_x1 = min(float(iw), (cw - ox) / zoom)
        src_y1 = min(float(ih), (ch - oy) / zoom)

        if src_x1 <= src_x0 + 0.5 or src_y1 <= src_y0 + 0.5:
            # Imagem fora da área visível
            self._canvas.delete("all")
            return

        # Tamanho de destino (pixels do canvas a pintar)
        dst_w = max(1, round((src_x1 - src_x0) * zoom))
        dst_h = max(1, round((src_y1 - src_y0) * zoom))

        # Recorta e redimensiona apenas o trecho visível
        crop_box = (
            max(0,  int(src_x0)),
            max(0,  int(src_y0)),
            min(iw, int(src_x1) + 1),
            min(ih, int(src_y1) + 1),
        )
        crop     = self._pil_image.crop(crop_box)
        resample = Image.NEAREST if zoom >= 4.0 else Image.LANCZOS
        try:
            rendered = crop.resize((dst_w, dst_h), resample)
        except Exception:
            return

        self._tk_image = ImageTk.PhotoImage(rendered)
        dst_x = max(0, round(ox))
        dst_y = max(0, round(oy))

        self._canvas.delete("all")
        self._canvas.create_image(dst_x, dst_y, anchor="nw", image=self._tk_image)

    # ────────────────────────── eventos de canvas ──────────────────────────
    def _on_configure(self, _event=None):
        if self._pil_image is None:
            self._canvas.delete("all")
            self._draw_placeholder()
        else:
            self._render()

    # ── Scroll / Zoom ──
    def _on_mousewheel(self, event):
        """Windows/Mac: event.delta positivo = zoom in."""
        factor = self.ZOOM_STEP if event.delta > 0 else 1.0 / self.ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)

    # ── Pan ──
    def _on_pan_start(self, event):
        self._pan_last = (event.x, event.y)
        self._canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        if self._pan_last is None:
            return
        dx, dy       = event.x - self._pan_last[0], event.y - self._pan_last[1]
        self._offset_x += dx
        self._offset_y += dy
        self._pan_last = (event.x, event.y)
        self._render()

    def _on_pan_end(self, _event=None):
        self._pan_last = None
        self._canvas.config(cursor="crosshair")

    # ── Conta-gotas ──
    def _canvas_to_image(self, cx: float, cy: float):
        """Converte coordenadas do canvas para pixel da imagem original."""
        if self._pil_image is None or self._zoom == 0:
            return None, None
        px = int((cx - self._offset_x) / self._zoom)
        py = int((cy - self._offset_y) / self._zoom)
        iw, ih = self._pil_image.size
        if 0 <= px < iw and 0 <= py < ih:
            return px, py
        return None, None

    def _on_mouse_move(self, event):
        if self._pil_image is None:
            return
        px, py = self._canvas_to_image(event.x, event.y)
        if px is None:
            # Cursor fora da imagem → mostra só zoom
            self._info_var.set(self._zoom_text())
            return
        try:
            pixel = self._pil_image.getpixel((px, py))
        except Exception:
            return

        mode = self._pil_image.mode
        if mode == "L":
            pixel_info = f"L={pixel}"
        elif mode == "RGB":
            r, g, b    = pixel
            pixel_info = f"R={r:>3}  G={g:>3}  B={b:>3}"
        elif mode == "RGBA":
            r, g, b, a = pixel
            pixel_info = f"R={r:>3}  G={g:>3}  B={b:>3}  A={a:>3}"
        else:
            pixel_info = str(pixel)

        zoom_pct = f"  │  🔍{self._zoom * 100:.0f}%"
        self._info_var.set(f"x={px:<5} y={py:<5} │  {pixel_info}{zoom_pct}")


# ══════════════════════════════════════════════════════════════════════
# As classes abaixo permanecem inalteradas
# ══════════════════════════════════════════════════════════════════════

class MultiImagePanel(tk.LabelFrame):
    """
    Painel para exibir múltiplas imagens lado a lado (ex: decomposições RGB/HSV).
    """
    THUMB_W = 130
    THUMB_H = 120

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(
            parent, text=title,
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa", bg="#1e1e2e",
            bd=2, relief="groove",
            **kwargs
        )
        self._inner = tk.Frame(self, bg="#1e1e2e")
        self._inner.pack(expand=True, fill="both", padx=4, pady=4)
        self._photos = []
        self._labels = []

    def set_images(self, images: dict):
        """
        Exibe dicionário de {nome: imagem_PIL}.
        Limpa painéis anteriores antes de exibir.
        """
        for w in self._labels:
            w.destroy()
        self._labels.clear()
        self._photos.clear()

        for name, img in images.items():
            frame = tk.Frame(self._inner, bg="#1e1e2e")
            frame.pack(side="left", padx=4, pady=4)

            thumb = img.copy()
            thumb.thumbnail((self.THUMB_W, self.THUMB_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self._photos.append(photo)

            lbl_img = tk.Label(frame, image=photo, bg="#181825", relief="flat")
            lbl_img.pack()
            lbl_name = tk.Label(frame, text=name,
                                font=("Segoe UI", 9, "bold"),
                                fg="#a6e3a1", bg="#1e1e2e")
            lbl_name.pack()
            self._labels.extend([frame, lbl_img, lbl_name])

    def clear(self):
        for w in self._labels:
            w.destroy()
        self._labels.clear()
        self._photos.clear()


class HistogramPanel(tk.LabelFrame):
    """
    Painel para exibir 1 ou 2 histogramas via matplotlib.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent, text="Histograma",
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa", bg="#1e1e2e",
            bd=2, relief="groove",
            **kwargs
        )
        self._fig = Figure(figsize=(5, 2.2), dpi=90,
                           facecolor="#1e1e2e")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=4, pady=4)
        self._fig.text(0.5, 0.5, "Selecione 'Equalização de Histograma'",
                       ha='center', va='center',
                       color="#585b70", fontsize=9)
        self._canvas.draw()

    def show_histograms(self, hist_before: np.ndarray, hist_after: np.ndarray = None):
        """Exibe histogramas antes e (opcionalmente) depois da equalização."""
        self._fig.clear()
        x = np.arange(256)

        if hist_after is not None:
            ax1 = self._fig.add_subplot(1, 2, 1)
            ax2 = self._fig.add_subplot(1, 2, 2)
            axes = [(ax1, hist_before, "Antes",  "#89b4fa"),
                    (ax2, hist_after,  "Depois", "#a6e3a1")]
        else:
            ax1  = self._fig.add_subplot(1, 1, 1)
            axes = [(ax1, hist_before, "Histograma", "#89b4fa")]

        for ax, hist, title, color in axes:
            ax.bar(x, hist, color=color, alpha=0.85, width=1.0)
            ax.set_title(title, color="#cdd6f4", fontsize=8, pad=2)
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#585b70", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#313244")
            ax.set_xlim(0, 255)

        self._fig.patch.set_facecolor("#1e1e2e")
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()


    def clear(self):
        self._fig.clear()
        self._fig.text(0.5, 0.5, "Selecione 'Equalização de Histograma'",
                       ha='center', va='center',
                       color="#585b70", fontsize=9)
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════
# ROIImagePanel — ImagePanel com seleção de retângulo de interesse
# ══════════════════════════════════════════════════════════════════════

class ROIImagePanel(ImagePanel):
    """
    Extensão de ImagePanel que permite ao usuário desenhar um retângulo de
    interesse (ROI) sobre a imagem.

    Modos
    ------
    ROI desativado (padrão): comportamento idêntico ao ImagePanel (pan, zoom).
    ROI ativado: clique+arrasto desenha um retângulo tracejado; ao soltar o
    botão, as coordenadas em pixels da imagem são armazenadas e o callback
    on_roi_set(x0, y0, x1, y1) é chamado (se definido).
    """

    ROI_COLOR  = "#89b4fa"   # Azul claro (paleta Catppuccin)
    ROI_FILL   = ""           # Sem preenchimento sólido
    ROI_DASH   = (6, 3)       # Tracejado

    def __init__(self, parent, title: str, on_roi_set=None, **kwargs):
        super().__init__(parent, title, **kwargs)
        self._roi_mode    = False
        self._roi_start   = None
        self._roi_rect_id = None
        self._roi_img     = None          # (x0, y0, x1, y1) em coords da imagem
        self._on_roi_set  = on_roi_set

        self._rebind()

    # ── API pública ──────────────────────────────────────────────────

    def set_roi_mode(self, enabled: bool):
        """Ativa/desativa o modo de seleção de ROI."""
        self._roi_mode = enabled
        self._rebind()

    def get_roi(self):
        """Retorna a ROI como (x0, y0, x1, y1) em coords de imagem, ou None."""
        return self._roi_img

    def clear_roi(self):
        """Remove o retângulo de ROI do canvas e reseta o estado."""
        self._roi_img   = None
        self._roi_start = None
        if self._roi_rect_id:
            self._canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def set_image(self, img):
        """Override: limpa ROI ao carregar nova imagem."""
        self.clear_roi()
        super().set_image(img)

    def clear(self):
        """Override: limpa ROI junto com a imagem."""
        self.clear_roi()
        super().clear()

    # ── Bindings dinâmicos ───────────────────────────────────────────

    def _rebind(self):
        """Reconfigura bindings do mouse conforme o modo atual."""
        cv = self._canvas
        cv.unbind("<ButtonPress-1>")
        cv.unbind("<B1-Motion>")
        cv.unbind("<ButtonRelease-1>")

        if self._roi_mode:
            cv.bind("<ButtonPress-1>",   self._roi_start_draw)
            cv.bind("<B1-Motion>",       self._roi_drag)
            cv.bind("<ButtonRelease-1>", self._roi_end_draw)
        else:
            cv.bind("<ButtonPress-1>",   self._on_pan_start)
            cv.bind("<B1-Motion>",       self._on_pan_drag)
            cv.bind("<ButtonRelease-1>", self._on_pan_end)

    # ── Desenho de ROI ────────────────────────────────────────────────

    def _roi_start_draw(self, event):
        self._roi_start = (event.x, event.y)
        if self._roi_rect_id:
            self._canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _roi_drag(self, event):
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        if self._roi_rect_id:
            self._canvas.delete(self._roi_rect_id)
        self._roi_rect_id = self._canvas.create_rectangle(
            x0, y0, x1, y1,
            outline=self.ROI_COLOR,
            fill=self.ROI_FILL,
            dash=self.ROI_DASH,
            width=2,
            tags="roi_rect"
        )

    def _roi_end_draw(self, event):
        if self._roi_start is None:
            return
        cx0, cy0 = self._roi_start
        cx1, cy1 = event.x, event.y

        # Normaliza ordem
        cx0, cx1 = min(cx0, cx1), max(cx0, cx1)
        cy0, cy1 = min(cy0, cy1), max(cy0, cy1)

        # Converte para coordenadas da imagem
        ix0, iy0 = self._canvas_to_image(cx0, cy0)
        ix1, iy1 = self._canvas_to_image(cx1, cy1)

        if ix0 is None or ix1 is None or ix0 >= ix1 or iy0 >= iy1:
            self.clear_roi()
            if self._on_roi_set:
                self._on_roi_set(None)
            return

        self._roi_img = (ix0, iy0, ix1, iy1)
        if self._on_roi_set:
            self._on_roi_set(self._roi_img)

    # ── Re-render: mantém retângulo sobre a imagem ────────────────────

    def _render(self):
        """Override: redesenha imagem e coloca overlay de ROI por cima."""
        super()._render()
        self._redraw_roi_overlay()

    def _redraw_roi_overlay(self):
        if self._roi_img is None:
            return
        ix0, iy0, ix1, iy1 = self._roi_img
        cx0 = self._offset_x + ix0 * self._zoom
        cy0 = self._offset_y + iy0 * self._zoom
        cx1 = self._offset_x + ix1 * self._zoom
        cy1 = self._offset_y + iy1 * self._zoom

        if self._roi_rect_id:
            self._canvas.delete(self._roi_rect_id)
        self._roi_rect_id = self._canvas.create_rectangle(
            cx0, cy0, cx1, cy1,
            outline=self.ROI_COLOR,
            fill=self.ROI_FILL,
            dash=self.ROI_DASH,
            width=2,
            tags="roi_rect"
        )
