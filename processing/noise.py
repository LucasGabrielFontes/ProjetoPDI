"""
Módulo de adição de ruídos.
Implementa ruídos gaussiano aditivo, sal, pimenta e sal-e-pimenta
conforme slides do professor (Seção 5 - Restauração/Degradação).
"""
import numpy as np
from PIL import Image


def _apply_noise_to_channels(img: Image.Image, func, **kwargs):
    """Aplica função de ruído a imagem, canal a canal se RGB."""
    if img.mode == "L":
        arr = np.array(img, dtype=np.float64)
        result = func(arr, **kwargs)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L")
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        channels = [func(arr[:, :, c], **kwargs) for c in range(3)]
        result = np.stack(channels, axis=2)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def add_gaussian_noise(img: Image.Image, mean: float = 0.0, sigma: float = 25.0) -> Image.Image:
    """
    Ruído gaussiano aditivo: g = f + η, onde η ~ N(mean, sigma²).
    """
    def _add(arr):
        noise = np.random.normal(mean, sigma, arr.shape)
        return arr + noise

    return _apply_noise_to_channels(img, _add)


def add_salt_noise(img: Image.Image, prob: float = 0.02) -> Image.Image:
    """
    Ruído sal (pixels aleatórios → 255) com probabilidade prob.
    """
    def _add(arr):
        result = arr.copy()
        mask = np.random.random(arr.shape) < prob
        result[mask] = 255.0
        return result

    return _apply_noise_to_channels(img, _add)


def add_pepper_noise(img: Image.Image, prob: float = 0.02) -> Image.Image:
    """
    Ruído pimenta (pixels aleatórios → 0) com probabilidade prob.
    """
    def _add(arr):
        result = arr.copy()
        mask = np.random.random(arr.shape) < prob
        result[mask] = 0.0
        return result

    return _apply_noise_to_channels(img, _add)


def add_salt_pepper_noise(img: Image.Image, prob: float = 0.02) -> Image.Image:
    """
    Ruído sal-e-pimenta: metade dos pixels afetados → 255 (sal),
    outra metade → 0 (pimenta). Probabilidade total = prob.
    """
    def _add(arr):
        result = arr.copy()
        rnd = np.random.random(arr.shape)
        result[rnd < prob / 2] = 255.0        # Sal
        result[(rnd >= prob / 2) & (rnd < prob)] = 0.0  # Pimenta
        return result

    return _apply_noise_to_channels(img, _add)


def add_salt_pepper_unified(
    img: Image.Image,
    prob_salt: float = 0.02,
    prob_pepper: float = 0.02,
    noise_type: str = "ambos"
) -> Image.Image:
    """
    Ruído sal-e-pimenta com controle individual de probabilidade.
    noise_type: 'sal' | 'pimenta' | 'ambos'
    """
    def _add(arr):
        result = arr.copy()
        rnd = np.random.random(arr.shape)
        if noise_type in ("sal", "ambos"):
            result[rnd < prob_salt] = 255.0
        if noise_type in ("pimenta", "ambos"):
            # Usar outra amostra para pimenta, independente do sal
            rnd2 = np.random.random(arr.shape)
            result[rnd2 < prob_pepper] = 0.0
        return result

    return _apply_noise_to_channels(img, _add)
