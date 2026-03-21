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
    Painel para exibir uma imagem PIL com título.
    Redimensiona automaticamente para caber no painel.
    """
    MAX_W = 420
    MAX_H = 380

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(
            parent, text=title,
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa", bg="#1e1e2e",
            bd=2, relief="groove",
            **kwargs
        )
        self._label = tk.Label(
            self, bg="#181825",
            relief="flat", cursor="crosshair"
        )
        self._label.pack(expand=True, fill="both", padx=6, pady=6)
        self._photo = None
        self._show_placeholder()

    def _show_placeholder(self):
        placeholder = Image.new("RGB", (self.MAX_W, self.MAX_H), color="#181825")
        self._photo = ImageTk.PhotoImage(placeholder)
        self._label.config(image=self._photo)

    def set_image(self, img: Image.Image):
        """Exibe a imagem redimensionada para caber no painel."""
        if img is None:
            self._show_placeholder()
            return
        img_copy = img.copy()
        img_copy.thumbnail((self.MAX_W, self.MAX_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_copy)
        self._label.config(image=self._photo)

    def clear(self):
        self._show_placeholder()


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
            axes = [(ax1, hist_before, "Antes", "#89b4fa"),
                    (ax2, hist_after, "Depois", "#a6e3a1")]
        else:
            ax1 = self._fig.add_subplot(1, 1, 1)
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
