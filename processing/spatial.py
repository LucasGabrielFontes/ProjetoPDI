"""
Módulo de filtros espaciais.
Implementa filtros de convolução conforme slides do professor (Seção 2):
  - Filtro Gaussiano (média gaussiana)
  - Filtro de Mediana, Mínimo e Máximo
  - Unsharp Masking
  - Realce por Laplaciano
  - Gradiente de Sobel
  - Fatiamento por Intensidade
"""
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, median_filter, minimum_filter, maximum_filter


def _make_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Cria kernel gaussiano 2D normalizado."""
    k = kernel_size // 2
    x = np.arange(-k, k + 1)
    y = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _apply_to_channels(img: Image.Image, func, **kwargs):
    """Aplica função a uma imagem, canal a canal se RGB."""
    if img.mode == "L":
        arr = np.array(img, dtype=np.float64)
        result = func(arr, **kwargs)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L")
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        channels = [func(arr[:, :, c], **kwargs) for c in range(3)]
        result = np.stack(channels, axis=2)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def gaussian_blur(img: Image.Image, sigma: float, kernel_size: int) -> Image.Image:
    """
    Filtro de média gaussiana com desvio padrão σ.
    g(x,y) = convolução de f(x,y) com kernel gaussiano.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Garante tamanho ímpar
    kernel = _make_gaussian_kernel(kernel_size, sigma)

    def _blur(arr):
        return convolve(arr, kernel, mode='reflect')

    return _apply_to_channels(img, _blur)


def apply_median_filter(img: Image.Image, kernel_size: int) -> Image.Image:
    """Filtro de mediana."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    def _med(arr):
        return median_filter(arr, size=kernel_size, mode='reflect')

    return _apply_to_channels(img, _med)


def apply_min_filter(img: Image.Image, kernel_size: int) -> Image.Image:
    """Filtro de mínimo (erosão morfológica)."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    def _min(arr):
        return minimum_filter(arr, size=kernel_size, mode='reflect')

    return _apply_to_channels(img, _min)


def apply_max_filter(img: Image.Image, kernel_size: int) -> Image.Image:
    """Filtro de máximo (dilatação morfológica)."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    def _max(arr):
        return maximum_filter(arr, size=kernel_size, mode='reflect')

    return _apply_to_channels(img, _max)


def unsharp_masking(img: Image.Image, gain: float, kernel_size: int, sigma: float) -> Image.Image:
    """
    Máscara de aguçamento (Unsharp Masking):
      máscara = f - f_suavizado
      g = f + k * máscara
    Funciona para grayscale e RGB.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = _make_gaussian_kernel(kernel_size, sigma)

    def _unsharp(arr):
        blurred = convolve(arr, kernel, mode='reflect')
        mask = arr - blurred
        sharpened = arr + gain * mask
        return sharpened

    return _apply_to_channels(img, _unsharp)


def laplacian_enhance(img: Image.Image, use_diagonal: bool = True) -> Image.Image:
    """
    Realce por Laplaciano conforme slides:
      Kernel Laplaciano com diagonal: [[1,1,1],[1,-8,1],[1,1,1]]
      Kernel sem diagonal:            [[0,1,0],[1,-4,1],[0,1,0]]
      g = f - laplaciano(f)   (subtrai pois kernel tem centro negativo)
    Apenas para imagens grayscale.
    """
    arr = np.array(img.convert("L"), dtype=np.float64)

    if use_diagonal:
        # Kernel com diagonais (8-vizinhos) — slides Seção 2
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], dtype=np.float64)
    else:
        # Kernel sem diagonais (4-vizinhos)
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float64)

    lap = convolve(arr, kernel, mode='reflect')
    # Aguçamento: g = f - laplaciano (pois kernel tem centro negativo)
    enhanced = arr - lap

    enhanced_img = Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8), mode="L")
    lap_display = lap - lap.min()
    if lap_display.max() > 0:
        lap_display = lap_display / lap_display.max() * 255
    lap_img = Image.fromarray(np.clip(lap_display, 0, 255).astype(np.uint8), mode="L")

    return enhanced_img, lap_img


def sobel_gradient(img: Image.Image) -> Image.Image:
    """
    Gradiente de Sobel: magnitude do gradiente.
      Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
      Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
      |G| = sqrt(Gx² + Gy²)
    Apenas para imagens grayscale.
    """
    arr = np.array(img.convert("L"), dtype=np.float64)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]], dtype=np.float64)

    Gx = convolve(arr, Kx, mode='reflect')
    Gy = convolve(arr, Ky, mode='reflect')
    G = np.sqrt(Gx ** 2 + Gy ** 2)

    # Normaliza para [0, 255]
    if G.max() > 0:
        G = G / G.max() * 255.0

    return Image.fromarray(np.clip(G, 0, 255).astype(np.uint8), mode="L")
