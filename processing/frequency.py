"""
Módulo de filtros no domínio da frequência.
Implementa filtros Gaussiano e Butterworth (passa-baixa e passa-alta)
conforme slides do professor (Seção 3 - Domínio da Frequência).

Fórmulas:
  Butterworth LP: H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
  Butterworth HP: H(u,v) = 1 / (1 + (D0/D(u,v))^(2n))
  Gaussiano LP:   H(u,v) = exp(-D(u,v)^2 / (2*D0^2))
  Gaussiano HP:   H(u,v) = 1 - Gaussiano_LP
"""
import numpy as np
from PIL import Image


def _distance_matrix(rows: int, cols: int) -> np.ndarray:
    """
    Calcula matriz de distâncias D(u,v) ao centro do espectro.
    Usado nos filtros de frequência.
    """
    u = np.fft.fftfreq(rows) * rows
    v = np.fft.fftfreq(cols) * cols
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2)
    return D


def _apply_frequency_filter(arr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Aplica filtro H no domínio da frequência:
      1. DFT da imagem
      2. Multiplica pelo filtro H(u,v)
      3. DFT inversa
    """
    F = np.fft.fft2(arr)
    G = F * H
    g = np.real(np.fft.ifft2(G))
    return g


def _freq_filter_image(img: Image.Image, H_builder) -> Image.Image:
    """
    Aplica filtro de frequência a imagem (grayscale ou RGB canal a canal).
    H_builder(rows, cols) -> H matriz de filtro.
    """
    if img.mode == "L":
        arr = np.array(img, dtype=np.float64)
        H = H_builder(arr.shape[0], arr.shape[1])
        result = _apply_frequency_filter(arr, H)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode="L")
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        H = H_builder(arr.shape[0], arr.shape[1])
        channels = []
        for c in range(3):
            result = _apply_frequency_filter(arr[:, :, c], H)
            channels.append(np.clip(result, 0, 255))
        result = np.stack(channels, axis=2).astype(np.uint8)
        return Image.fromarray(result, mode="RGB")


def butterworth_lpf(img: Image.Image, cutoff: float, order: int) -> Image.Image:
    """
    Filtro Butterworth Passa-Baixa:
      H(u,v) = 1 / (1 + (D(u,v) / D0)^(2n))
    """
    def _build(rows, cols):
        D = _distance_matrix(rows, cols)
        D[D == 0] = 1e-10  # Evita divisão por zero
        H = 1.0 / (1.0 + (D / cutoff) ** (2 * order))
        return H

    return _freq_filter_image(img, _build)


def butterworth_hpf(img: Image.Image, cutoff: float, order: int) -> Image.Image:
    """
    Filtro Butterworth Passa-Alta:
      H(u,v) = 1 / (1 + (D0 / D(u,v))^(2n))
    """
    def _build(rows, cols):
        D = _distance_matrix(rows, cols)
        D[D == 0] = 1e-10
        H = 1.0 / (1.0 + (cutoff / D) ** (2 * order))
        return H

    return _freq_filter_image(img, _build)


def gaussian_lpf(img: Image.Image, cutoff: float) -> Image.Image:
    """
    Filtro Gaussiano Passa-Baixa:
      H(u,v) = exp(-D(u,v)^2 / (2 * D0^2))
    """
    def _build(rows, cols):
        D = _distance_matrix(rows, cols)
        H = np.exp(-D ** 2 / (2 * cutoff ** 2))
        return H

    return _freq_filter_image(img, _build)


def gaussian_hpf(img: Image.Image, cutoff: float) -> Image.Image:
    """
    Filtro Gaussiano Passa-Alta:
      H(u,v) = 1 - Gaussiano_LP(u,v)
    """
    def _build(rows, cols):
        D = _distance_matrix(rows, cols)
        H_lp = np.exp(-D ** 2 / (2 * cutoff ** 2))
        return 1.0 - H_lp

    return _freq_filter_image(img, _build)
