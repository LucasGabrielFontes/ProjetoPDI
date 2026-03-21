"""
Widgets reutilizáveis para a interface do projeto PDI.
"""
import tkinter as tk
from tkinter import ttk


class LabeledEntry(tk.Frame):
    """
    Widget com Label + Entry para entrada numérica manual.
    Inclui slider auxiliar que se sincroniza com o valor digitado.
    """
    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, resolution: float = 1.0,
                 callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._callback = callback
        self._from = from_
        self._to = to
        self._resolution = resolution
        self._var = tk.DoubleVar(value=default)
        self._updating = False

        # Linha 0: label
        tk.Label(self, text=label, anchor="w",
                 font=("Segoe UI", 9, "bold"),
                 fg="#cdd6f4", bg=kwargs.get("bg", "#181825")
                 ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # Linha 1: range min label, entry (centro), range max label
        tk.Label(self, text=f"{from_}", anchor="e", width=6,
                 font=("Segoe UI", 8),
                 fg="#585b70", bg=kwargs.get("bg", "#181825")
                 ).grid(row=1, column=0, sticky="e")

        self._entry = tk.Entry(
            self, textvariable=self._var, width=8,
            font=("Segoe UI", 10, "bold"),
            bg="#313244", fg="#89b4fa",
            insertbackground="#89b4fa",
            relief="flat", bd=6,
            justify="center"
        )
        self._entry.grid(row=1, column=1, padx=4)
        self._entry.bind("<Return>", self._on_entry_commit)
        self._entry.bind("<FocusOut>", self._on_entry_commit)
        self._entry.bind("<Up>", self._on_arrow_up)
        self._entry.bind("<Down>", self._on_arrow_down)

        tk.Label(self, text=f"{to}", anchor="w", width=6,
                 font=("Segoe UI", 8),
                 fg="#585b70", bg=kwargs.get("bg", "#181825")
                 ).grid(row=1, column=2, sticky="w")

        # Linha 2: slider auxiliar
        self._scale = tk.Scale(
            self, from_=from_, to=to, resolution=resolution,
            orient="horizontal", variable=self._var,
            showvalue=False, length=200,
            command=self._on_scale,
            bg=kwargs.get("bg", "#181825"),
            troughcolor="#313244", fg="#cdd6f4",
            highlightthickness=0, activebackground="#89b4fa",
            sliderlength=14
        )
        self._scale.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 4))

    def _on_scale(self, _=None):
        if not self._updating:
            if self._callback:
                self._callback()

    def _on_entry_commit(self, _=None):
        try:
            raw = self._entry.get()
            val = float(raw)
            val = max(self._from, min(self._to, val))
            # Arredonda para a resolução
            if self._resolution >= 1:
                val = round(val / self._resolution) * self._resolution
            self._updating = True
            self._var.set(val)
            self._updating = False
            if self._callback:
                self._callback()
        except (tk.TclError, ValueError):
            # Restaura valor anterior
            self._var.set(self._var.get())

    def _on_arrow_up(self, _=None):
        val = self._var.get() + self._resolution
        val = min(self._to, val)
        self._var.set(val)
        if self._callback:
            self._callback()
        return "break"

    def _on_arrow_down(self, _=None):
        val = self._var.get() - self._resolution
        val = max(self._from, val)
        self._var.set(val)
        if self._callback:
            self._callback()
        return "break"

    def get(self) -> float:
        return self._var.get()

    def set(self, value: float):
        self._var.set(value)


# Alias para compatibilidade (LabeledSlider agora é LabeledEntry aprimorado)
LabeledSlider = LabeledEntry


class LabeledCheckbox(tk.Frame):
    """Checkbox com label."""
    def __init__(self, parent, label: str, default: bool = False,
                 callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._callback = callback
        self._var = tk.BooleanVar(value=default)
        bg = kwargs.get("bg", "#181825")

        self._check = tk.Checkbutton(
            self, text=label, variable=self._var,
            command=self._on_change,
            font=("Segoe UI", 9),
            bg=bg, fg="#cdd6f4",
            selectcolor="#313244",
            activebackground=bg,
            activeforeground="#89b4fa",
            relief="flat"
        )
        self._check.pack(anchor="w", pady=(4, 0))

    def _on_change(self):
        if self._callback:
            self._callback()

    def get(self) -> bool:
        return self._var.get()


class LabeledOptionMenu(tk.Frame):
    """OptionMenu com label para escolha entre opções."""
    def __init__(self, parent, label: str, options: list,
                 default: str = None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._callback = callback
        bg = kwargs.get("bg", "#181825")
        self._var = tk.StringVar(value=default or options[0])

        tk.Label(self, text=label, anchor="w",
                 font=("Segoe UI", 9, "bold"),
                 fg="#cdd6f4", bg=bg
                 ).pack(anchor="w", pady=(2, 0))

        self._menu = tk.OptionMenu(
            self, self._var, *options,
            command=lambda _: self._on_change()
        )
        self._menu.config(
            font=("Segoe UI", 9),
            bg="#313244", fg="#cdd6f4",
            activebackground="#45475a",
            activeforeground="#cdd6f4",
            relief="flat", bd=0,
            highlightthickness=0,
            cursor="hand2"
        )
        self._menu["menu"].config(
            font=("Segoe UI", 9),
            bg="#313244", fg="#cdd6f4",
            activebackground="#89b4fa",
            activeforeground="#1e1e2e",
            relief="flat"
        )
        self._menu.pack(fill="x", pady=(2, 4))

    def _on_change(self):
        if self._callback:
            self._callback()

    def get(self) -> str:
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
