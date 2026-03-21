"""
Módulo de histograma e equalização.
Fórmula de equalização conforme slides do professor (Seção 2):
  s_k = (L-1) / (M*N) * Σ_{j=0}^{k} n_j
"""
import numpy as np
from PIL import Image


def compute_histogram(img: Image.Image) -> np.ndarray:
    """
    Calcula o histograma de intensidade de uma imagem grayscale.
    Retorna array de 256 posições com contagem de pixels.
    """
    arr = np.array(img.convert("L"), dtype=np.uint8)
    hist = np.zeros(256, dtype=np.int64)
    for val in range(256):
        hist[val] = np.sum(arr == val)
    return hist


def equalize_histogram(img: Image.Image) -> Image.Image:
    """
    Equalização de histograma pela transformação cumulativa:
      s_k = (L-1) / (M*N) * Σ_{j=0}^{k} n_j
    Somente para imagens em escala de cinza.
    """
    arr = np.array(img.convert("L"), dtype=np.uint8)
    M, N = arr.shape
    L = 256

    # Histograma
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))

    # CDF (soma cumulativa)
    cdf = hist.cumsum()

    # Transformação: s_k = (L-1) / (M*N) * CDF(k)
    lut = np.round((L - 1) / (M * N) * cdf).astype(np.uint8)

    # Aplica LUT
    result = lut[arr]
    return Image.fromarray(result, mode="L")
