"""
Módulo de transformações de intensidade.
Implementa limiarização, transformação logarítmica e de potência
conforme slides do professor (Seção 2).
"""
import numpy as np
from PIL import Image


def _to_gray_array(img: Image.Image) -> np.ndarray:
    """Converte imagem PIL para array float64 de 0 a 255."""
    return np.array(img.convert("L"), dtype=np.float64)


def _clip_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def threshold(img: Image.Image, k: int) -> Image.Image:
    """
    Limiarização binária.
    s = 0   se r < k
    s = 255 se r >= k
    Apenas para imagens em escala de cinza.
    """
    arr = _to_gray_array(img)
    result = np.where(arr >= k, 255, 0).astype(np.uint8)
    return Image.fromarray(result, mode="L")


def log_transform(img: Image.Image, c: float) -> Image.Image:
    """
    Transformação logarítmica: s = c * log(1 + r)
    O ganho c influencia diretamente o brilho da saída.
    Valores de c maiores saturam mais pixels em 255 (imagem mais clara);
    valores menores produzem uma imagem mais escura.
    O valor "neutro" que mapeia exatamente 255 -> 255 é c ≈ 45.9.
    Funciona para imagens grayscale e coloridas (canal a canal).
    """
    def _apply_log(arr: np.ndarray) -> np.ndarray:
        result = c * np.log1p(arr)  # s = c * log(1 + r)
        return _clip_to_uint8(result)  # clip direto, sem normalização

    if img.mode == "L":
        arr = _to_gray_array(img)
        return Image.fromarray(_apply_log(arr), mode="L")
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        result = np.stack([_apply_log(arr[:, :, c_]) for c_ in range(3)], axis=2)
        return Image.fromarray(result.astype(np.uint8), mode="RGB")


def power_transform(img: Image.Image, c: float, gamma: float) -> Image.Image:
    """
    Transformação de potência (lei de potência): s = c * r^γ
    Resultado normalizado para [0, 255].
    Funciona para imagens grayscale e coloridas (canal a canal).
    """
    def _apply_power(arr: np.ndarray) -> np.ndarray:
        # Normaliza entrada para [0, 1]
        normalized = arr / 255.0
        result = c * np.power(normalized, gamma) * 255.0
        return _clip_to_uint8(result)

    if img.mode == "L":
        arr = _to_gray_array(img)
        return Image.fromarray(_apply_power(arr), mode="L")
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        result = np.stack([_apply_power(arr[:, :, c_]) for c_ in range(3)], axis=2)
        return Image.fromarray(result.astype(np.uint8), mode="RGB")


def intensity_slicing(img: Image.Image, A: int, B: int, preserve_bg: bool) -> Image.Image:
    """
    Fatiamento por intensidade na faixa [A, B].
    Se preserve_bg=True: pixels in [A,B] -> 255, fora mantém valor original.
    Se preserve_bg=False: pixels in [A,B] -> 255, fora -> 0.
    Apenas para imagens em escala de cinza.
    """
    arr = _to_gray_array(img)
    mask = (arr >= A) & (arr <= B)

    if preserve_bg:
        result = arr.copy()
        result[mask] = 255
    else:
        result = np.zeros_like(arr)
        result[mask] = 255

    return Image.fromarray(_clip_to_uint8(result), mode="L")
