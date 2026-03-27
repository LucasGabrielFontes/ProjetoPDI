"""
Módulo de decomposição de canais de cor e pseudo-coloração.
Implementa decomposição RGB e HSV conforme slides do professor (Seção 4)
e pseudo-coloração por mapeamento de colormap.
"""
import numpy as np
from PIL import Image

# ── Pseudo-coloração ────────────────────────────────────────────────────────

# Colormaps disponíveis (nome exibido → nome matplotlib)
COLORMAPS = [
    "jet", "hot", "cool", "hsv", "rainbow",
    "viridis", "plasma", "inferno", "magma", "turbo",
]

# Cache de CLUT: {nome → array (256,3) uint8}
_CLUT_CACHE: dict = {}


def _build_clut(colormap: str) -> np.ndarray:
    """Gera CLUT de 256 entradas RGB para o colormap dado (usa matplotlib)."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap, 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)  # (256, 3)
    return lut


def _get_clut(colormap: str) -> np.ndarray:
    if colormap not in _CLUT_CACHE:
        _CLUT_CACHE[colormap] = _build_clut(colormap)
    return _CLUT_CACHE[colormap]


def pseudo_colorize(img: Image.Image, colormap: str = "jet") -> Image.Image:
    """
    Mapeia uma imagem em escala de cinza para uma falsa imagem colorida (RGB)
    aplicando o colormap especificado como tabela de pesquisa (CLUT).

    Parâmetros
    ----------
    img : Image.Image
        Imagem de entrada (qualquer modo; convertida para 'L' internamente).
    colormap : str
        Nome do colormap. Deve ser um dos valores em COLORMAPS.

    Retorna
    -------
    Image.Image
        Imagem PIL no modo 'RGB'.
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)          # (H, W)
    lut  = _get_clut(colormap)                                  # (256, 3)
    rgb  = lut[gray]                                            # (H, W, 3)
    return Image.fromarray(rgb, mode="RGB")


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
