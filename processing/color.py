"""
Módulo de decomposição de canais de cor.
Implementa decomposição RGB e HSV conforme slides do professor (Seção 4).
"""
import numpy as np
from PIL import Image


def decompose_rgb(img: Image.Image) -> dict:
    """
    Decompõe imagem colorida RGB nos 3 canais.
    Retorna dicionário com imagens grayscale dos canais R, G e B.
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    r = Image.fromarray(arr[:, :, 0], mode="L")
    g = Image.fromarray(arr[:, :, 1], mode="L")
    b = Image.fromarray(arr[:, :, 2], mode="L")
    return {"R": r, "G": g, "B": b}


def _rgb_to_hsv_arrays(arr: np.ndarray):
    """
    Converte array RGB (uint8) para H, S, V normalizados em [0,1].
    Fórmulas conforme slides (Seção 4 - Modelo HSV).
    """
    r = arr[:, :, 0].astype(np.float64) / 255.0
    g = arr[:, :, 1].astype(np.float64) / 255.0
    b = arr[:, :, 2].astype(np.float64) / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # --- Hue ---
    H = np.zeros_like(r)
    mask = delta != 0
    # cmax == r
    m = mask & (cmax == r)
    H[m] = (60.0 * ((g[m] - b[m]) / delta[m])) % 360.0
    # cmax == g
    m = mask & (cmax == g)
    H[m] = (60.0 * ((b[m] - r[m]) / delta[m]) + 120.0) % 360.0
    # cmax == b
    m = mask & (cmax == b)
    H[m] = (60.0 * ((r[m] - g[m]) / delta[m]) + 240.0) % 360.0
    H = H / 360.0  # Normaliza para [0, 1]

    # --- Saturation ---
    S = np.where(cmax == 0, 0.0, delta / cmax)

    # --- Value ---
    V = cmax

    return H, S, V


def decompose_hsv(img: Image.Image) -> dict:
    """
    Decompõe imagem colorida RGB nos canais H, S, V.
    Retorna dicionário com imagens grayscale (0-255).
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    H, S, V = _rgb_to_hsv_arrays(arr)

    h_img = Image.fromarray((H * 255).clip(0, 255).astype(np.uint8), mode="L")
    s_img = Image.fromarray((S * 255).clip(0, 255).astype(np.uint8), mode="L")
    v_img = Image.fromarray((V * 255).clip(0, 255).astype(np.uint8), mode="L")

    return {"H": h_img, "S": s_img, "V": v_img}
