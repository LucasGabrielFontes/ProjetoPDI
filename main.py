"""
main.py — Ponto de entrada do Projeto PDI
Disciplina: Introdução ao Processamento Digital de Imagens — UFPB
"""
import sys
import tkinter as tk
from gui.app import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
