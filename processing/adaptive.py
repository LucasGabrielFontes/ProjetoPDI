"""
Módulo de filtro adaptativo de mediana.
Algoritmo conforme slides do professor (Seção 5 - Restauração).

O filtro adaptativo de mediana aumenta o tamanho da janela dinamicamente:
  Etapa A: zmed - zmin > 0 e zmed - zmax < 0 → vai para Etapa B
           Caso contrário, aumenta tamanho da janela
  Etapa B: z_xy - zmin > 0 e z_xy - zmax < 0 → retorna z_xy
           Caso contrário, retorna zmed

Implementação vetorizada: para cada tamanho de janela possível, computa
globalmente min/med/max e decide pixel a pixel quais pixels ainda não foram
resolvidos, até cobrir todos ou atingir max_size.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, minimum_filter, maximum_filter


def adaptive_median_filter(img: Image.Image, max_size: int) -> Image.Image:
    """
    Filtro adaptativo de mediana vetorizado.
    max_size: tamanho máximo da janela (ímpar, >= 3).
    Somente para imagens em escala de cinza.
    """
    if max_size % 2 == 0:
        max_size += 1
    if max_size < 3:
        max_size = 3

    arr = np.array(img.convert("L"), dtype=np.float64)
    z_xy = arr.copy()

    # Output começa com o valor original; substituição feita progressivamente
    output = arr.copy()
    resolved = np.zeros(arr.shape, dtype=bool)

    sxy = 3
    while sxy <= max_size:
        zmed = median_filter(arr, size=sxy, mode='reflect')
        zmin = minimum_filter(arr, size=sxy, mode='reflect')
        zmax = maximum_filter(arr, size=sxy, mode='reflect')

        # Etapa A: verifica se mediana é impulso
        A1 = zmed - zmin  # > 0 → mediana não é mínimo
        A2 = zmed - zmax  # < 0 → mediana não é máximo
        stage_a_ok = (A1 > 0) & (A2 < 0)

        # Dentro de stage_a_ok: Etapa B
        B1 = z_xy - zmin
        B2 = z_xy - zmax
        pixel_not_impulse = (B1 > 0) & (B2 < 0)

        # Pixels com Etapa A OK: usa z_xy se não for impulso, senão zmed
        resolved_now = stage_a_ok & ~resolved
        output[resolved_now & pixel_not_impulse] = z_xy[resolved_now & pixel_not_impulse]
        output[resolved_now & ~pixel_not_impulse] = zmed[resolved_now & ~pixel_not_impulse]
        resolved |= resolved_now

        # Pixels com Etapa A FALHOU e chegamos ao limite: usa zmed
        if sxy == max_size:
            still_unresolved = ~resolved
            output[still_unresolved] = zmed[still_unresolved]

        sxy += 2

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="L")
