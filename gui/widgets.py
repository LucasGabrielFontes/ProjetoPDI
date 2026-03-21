"""
Widgets reutilizáveis para a interface do projeto PDI.
"""
import tkinter as tk
from tkinter import ttk


class LabeledSlider(tk.Frame):
    """
    Widget com Label + Scale (slider) + Entry sincronizados.
    Permite ajuste preciso de parâmetros numéricos.
    """
    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, resolution: float = 1.0,
                 callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._callback = callback
        self._var = tk.DoubleVar(value=default)
        self._resolution = resolution

        tk.Label(self, text=label, anchor="w", width=22,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")

        self._scale = tk.Scale(
            self, from_=from_, to=to, resolution=resolution,
            orient="horizontal", variable=self._var,
            showvalue=False, length=160,
            command=self._on_scale, bg="#1e1e2e",
            troughcolor="#313244", fg="#cdd6f4",
            highlightthickness=0, activebackground="#89b4fa"
        )
        self._scale.grid(row=0, column=1, padx=(4, 4))

        self._entry = tk.Entry(self, textvariable=self._var, width=7,
                               font=("Segoe UI", 9),
                               bg="#313244", fg="#cdd6f4",
                               insertbackground="#cdd6f4",
                               relief="flat", bd=4)
        self._entry.grid(row=0, column=2)
        self._entry.bind("<Return>", self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

    def _on_scale(self, _=None):
        if self._callback:
            self._callback()

    def _on_entry(self, _=None):
        try:
            val = float(self._var.get())
            self._var.set(val)
            if self._callback:
                self._callback()
        except (tk.TclError, ValueError):
            pass

    def get(self) -> float:
        return self._var.get()

    def set(self, value: float):
        self._var.set(value)


class LabeledCheckbox(tk.Frame):
    """Checkbox com label."""
    def __init__(self, parent, label: str, default: bool = False,
                 callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._callback = callback
        self._var = tk.BooleanVar(value=default)

        self._check = tk.Checkbutton(
            self, text=label, variable=self._var,
            command=self._on_change,
            font=("Segoe UI", 9),
            bg="#1e1e2e", fg="#cdd6f4",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            activeforeground="#89b4fa",
            relief="flat"
        )
        self._check.pack(anchor="w")

    def _on_change(self):
        if self._callback:
            self._callback()

    def get(self) -> bool:
        return self._var.get()


class StyledButton(tk.Button):
    """Botão estilizado."""
    def __init__(self, parent, text: str, command=None,
                 accent=False, **kwargs):
        color = "#89b4fa" if accent else "#313244"
        fg = "#1e1e2e" if accent else "#cdd6f4"
        super().__init__(
            parent, text=text, command=command,
            font=("Segoe UI", 9, "bold"),
            bg=color, fg=fg,
            activebackground="#b4befe",
            activeforeground="#1e1e2e",
            relief="flat", bd=0,
            padx=12, pady=6,
            cursor="hand2",
            **kwargs
        )
        self.bind("<Enter>", lambda e: self.config(bg="#b4befe" if accent else "#45475a"))
        self.bind("<Leave>", lambda e: self.config(bg=color))
