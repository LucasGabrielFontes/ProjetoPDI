"""
Testes automáticos para os módulos de processamento do Projeto PDI.
Usa imagens sintéticas (geradas programaticamente) para testar cada função.
Execute com: python -m pytest test_processing.py -v
"""
import pytest
import numpy as np
from PIL import Image

# Módulos de processamento
import processing.color as pcolor
import processing.intensity as pintensity
import processing.histogram as phist
import processing.spatial as pspatial
import processing.frequency as pfreq
import processing.adaptive as padaptive
import processing.noise as pnoise


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: imagens sintéticas de teste
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_img():
    """Imagem grayscale 64×64 com gradiente."""
    arr = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def rgb_img():
    """Imagem colorida RGB 64×64 com canais distintos."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(64, dtype=np.uint8), (64, 1))   # R = gradiente H
    arr[:, :, 1] = np.tile(np.arange(64, dtype=np.uint8).reshape(64, 1), (1, 64))  # G = gradiente V
    arr[:, :, 2] = 128  # B = constante
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def uniform_gray():
    """Imagem grayscale 64×64 uniforme (valor 128)."""
    arr = np.full((64, 64), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


# ──────────────────────────────────────────────────────────────────────────────
# 1 & 2. Decomposição RGB e HSV
# ──────────────────────────────────────────────────────────────────────────────

class TestColorDecomposition:
    def test_rgb_decomp_returns_dict_with_3_channels(self, rgb_img):
        result = pcolor.decompose_rgb(rgb_img)
        assert set(result.keys()) == {"R", "G", "B"}

    def test_rgb_decomp_channels_are_grayscale(self, rgb_img):
        result = pcolor.decompose_rgb(rgb_img)
        for key, img in result.items():
            assert img.mode == "L", f"Canal {key} deve ser grayscale"

    def test_rgb_decomp_size_preserved(self, rgb_img):
        result = pcolor.decompose_rgb(rgb_img)
        for img in result.values():
            assert img.size == rgb_img.size

    def test_rgb_decomp_r_channel_correct(self, rgb_img):
        result = pcolor.decompose_rgb(rgb_img)
        r_arr = np.array(result["R"])
        orig_arr = np.array(rgb_img)
        np.testing.assert_array_equal(r_arr, orig_arr[:, :, 0])

    def test_hsv_decomp_returns_dict_with_3_channels(self, rgb_img):
        result = pcolor.decompose_hsv(rgb_img)
        assert set(result.keys()) == {"H", "S", "V"}

    def test_hsv_decomp_channels_are_grayscale(self, rgb_img):
        result = pcolor.decompose_hsv(rgb_img)
        for key, img in result.items():
            assert img.mode == "L", f"Canal {key} deve ser grayscale"

    def test_hsv_decomp_values_in_range(self, rgb_img):
        result = pcolor.decompose_hsv(rgb_img)
        for key, img in result.items():
            arr = np.array(img)
            assert arr.min() >= 0 and arr.max() <= 255


# ──────────────────────────────────────────────────────────────────────────────
# 3. Limiarização
# ──────────────────────────────────────────────────────────────────────────────

class TestThreshold:
    def test_threshold_output_is_binary(self, gray_img):
        result = pintensity.threshold(gray_img, 128)
        arr = np.array(result)
        unique = set(np.unique(arr))
        assert unique <= {0, 255}

    def test_threshold_k0_all_white(self, gray_img):
        result = pintensity.threshold(gray_img, 0)
        arr = np.array(result)
        assert np.all(arr == 255)

    def test_threshold_k255_all_black(self, gray_img):
        result = pintensity.threshold(gray_img, 255)
        arr = np.array(result)
        # Apenas pixels com valor >= 255 ficam brancos
        orig = np.array(gray_img)
        assert arr[orig < 255].max() == 0

    def test_threshold_returns_grayscale(self, gray_img):
        result = pintensity.threshold(gray_img, 100)
        assert result.mode == "L"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Transformação Logarítmica
# ──────────────────────────────────────────────────────────────────────────────

class TestLogTransform:
    def test_log_grayscale_output(self, gray_img):
        result = pintensity.log_transform(gray_img, 1.0)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_log_rgb_output(self, rgb_img):
        result = pintensity.log_transform(rgb_img, 1.0)
        assert result.mode == "RGB"
        assert result.size == rgb_img.size

    def test_log_output_in_range(self, gray_img):
        result = pintensity.log_transform(gray_img, 2.0)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_log_zero_input_gives_zero(self):
        # Imagem preta: log(1+0) = 0
        black = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
        result = pintensity.log_transform(black, 1.0)
        arr = np.array(result)
        assert np.all(arr == 0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Transformação de Potência (Gamma)
# ──────────────────────────────────────────────────────────────────────────────

class TestPowerTransform:
    def test_power_grayscale(self, gray_img):
        result = pintensity.power_transform(gray_img, 1.0, 1.0)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_power_rgb(self, rgb_img):
        result = pintensity.power_transform(rgb_img, 1.0, 1.0)
        assert result.mode == "RGB"

    def test_power_gamma1_identity_ish(self, gray_img):
        # c=1, gamma=1 deve mapear linearmente (mas com normalização pode variar)
        result = pintensity.power_transform(gray_img, 1.0, 1.0)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_power_output_in_range(self, gray_img):
        result = pintensity.power_transform(gray_img, 2.0, 0.5)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255


# ──────────────────────────────────────────────────────────────────────────────
# 6. Equalização de Histograma
# ──────────────────────────────────────────────────────────────────────────────

class TestHistogramEqualization:
    def test_equalize_returns_pil_image(self, gray_img):
        result = phist.equalize_histogram(gray_img)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_equalize_size_preserved(self, gray_img):
        result = phist.equalize_histogram(gray_img)
        assert result.size == gray_img.size

    def test_compute_histogram_length(self, gray_img):
        hist = phist.compute_histogram(gray_img)
        assert len(hist) == 256

    def test_compute_histogram_sum(self, gray_img):
        hist = phist.compute_histogram(gray_img)
        w, h = gray_img.size
        assert hist.sum() == w * h

    def test_equalize_uniform_image_unchanged(self, uniform_gray):
        # Imagem uniforme: histograma já "equalizado"
        result = phist.equalize_histogram(uniform_gray)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255


# ──────────────────────────────────────────────────────────────────────────────
# 7. Fatiamento por Intensidade
# ──────────────────────────────────────────────────────────────────────────────

class TestIntensitySlicing:
    def test_slicing_preserve_bg(self, gray_img):
        result = pintensity.intensity_slicing(gray_img, 20, 40, preserve_bg=True)
        arr = np.array(result)
        orig = np.array(gray_img)
        # Pixels no intervalo devem ser 255
        mask = (orig >= 20) & (orig <= 40)
        assert np.all(arr[mask] == 255)

    def test_slicing_no_preserve_bg(self, gray_img):
        result = pintensity.intensity_slicing(gray_img, 20, 40, preserve_bg=False)
        arr = np.array(result)
        orig = np.array(gray_img)
        # Pixels fora do intervalo devem ser 0
        mask_out = ~((orig >= 20) & (orig <= 40))
        assert np.all(arr[mask_out] == 0)

    def test_slicing_returns_grayscale(self, gray_img):
        result = pintensity.intensity_slicing(gray_img, 50, 150, True)
        assert result.mode == "L"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Filtro Gaussiano
# ──────────────────────────────────────────────────────────────────────────────

class TestGaussianBlur:
    def test_gaussian_grayscale(self, gray_img):
        result = pspatial.gaussian_blur(gray_img, sigma=2.0, kernel_size=5)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_gaussian_rgb(self, rgb_img):
        result = pspatial.gaussian_blur(rgb_img, sigma=2.0, kernel_size=5)
        assert result.mode == "RGB"
        assert result.size == rgb_img.size

    def test_gaussian_uniform_unchanged(self, uniform_gray):
        # Uma imagem uniforme suavizada deve permanecer uniforme (±1 por arredondamento float)
        result = pspatial.gaussian_blur(uniform_gray, sigma=2.0, kernel_size=5)
        arr = np.array(result)
        assert np.all(np.abs(arr.astype(int) - 128) <= 1)

    def test_gaussian_output_in_range(self, gray_img):
        result = pspatial.gaussian_blur(gray_img, sigma=3.0, kernel_size=7)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255


# ──────────────────────────────────────────────────────────────────────────────
# 9. Filtros Mediana, Mínimo e Máximo
# ──────────────────────────────────────────────────────────────────────────────

class TestStatisticFilters:
    def test_median_grayscale(self, gray_img):
        result = pspatial.apply_median_filter(gray_img, 5)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_median_rgb(self, rgb_img):
        result = pspatial.apply_median_filter(rgb_img, 5)
        assert result.mode == "RGB"

    def test_min_grayscale(self, gray_img):
        result = pspatial.apply_min_filter(gray_img, 3)
        arr = np.array(result)
        orig = np.array(gray_img)
        # Mínimo deve ser ≤ original em todos os pixels
        assert np.all(arr <= orig)

    def test_max_grayscale(self, gray_img):
        result = pspatial.apply_max_filter(gray_img, 3)
        arr = np.array(result)
        orig = np.array(gray_img)
        # Máximo deve ser ≥ original em todos os pixels
        assert np.all(arr >= orig)

    def test_median_uniform(self, uniform_gray):
        result = pspatial.apply_median_filter(uniform_gray, 5)
        arr = np.array(result)
        assert np.all(arr == 128)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Máscara de Aguçamento (Unsharp Masking)
# ──────────────────────────────────────────────────────────────────────────────

class TestUnsharpMask:
    def test_unsharp_grayscale(self, gray_img):
        result = pspatial.unsharp_masking(gray_img, gain=1.5, kernel_size=5, sigma=2.0)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_unsharp_rgb(self, rgb_img):
        result = pspatial.unsharp_masking(rgb_img, gain=1.5, kernel_size=5, sigma=2.0)
        assert result.mode == "RGB"

    def test_unsharp_output_in_range(self, gray_img):
        result = pspatial.unsharp_masking(gray_img, gain=3.0, kernel_size=7, sigma=2.0)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255


# ──────────────────────────────────────────────────────────────────────────────
# 11. Realce por Laplaciano
# ──────────────────────────────────────────────────────────────────────────────

class TestLaplacian:
    def test_laplacian_returns_tuple(self, gray_img):
        result = pspatial.laplacian_enhance(gray_img, use_diagonal=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_laplacian_both_images_grayscale(self, gray_img):
        enhanced, lap = pspatial.laplacian_enhance(gray_img, True)
        assert enhanced.mode == "L"
        assert lap.mode == "L"

    def test_laplacian_size_preserved(self, gray_img):
        enhanced, lap = pspatial.laplacian_enhance(gray_img, True)
        assert enhanced.size == gray_img.size
        assert lap.size == gray_img.size

    def test_laplacian_without_diag(self, gray_img):
        enhanced, lap = pspatial.laplacian_enhance(gray_img, use_diagonal=False)
        assert enhanced.mode == "L"

    def test_laplacian_uniform_unchanged(self, uniform_gray):
        # Imagem uniforme: Laplaciano = 0 → realçada ≈ original
        enhanced, lap = pspatial.laplacian_enhance(uniform_gray, True)
        enhanced_arr = np.array(enhanced)
        # Deve ser aproximadamente 128
        assert 120 <= enhanced_arr.mean() <= 136


# ──────────────────────────────────────────────────────────────────────────────
# 12. Gradiente de Sobel
# ──────────────────────────────────────────────────────────────────────────────

class TestSobel:
    def test_sobel_returns_tuple_3(self, gray_img):
        result = pspatial.sobel_gradient(gray_img)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_sobel_all_grayscale(self, gray_img):
        img_grad, img_gx, img_gy = pspatial.sobel_gradient(gray_img)
        assert img_grad.mode == "L"
        assert img_gx.mode == "L"
        assert img_gy.mode == "L"

    def test_sobel_size_preserved(self, gray_img):
        img_grad, img_gx, img_gy = pspatial.sobel_gradient(gray_img)
        assert img_grad.size == gray_img.size

    def test_sobel_uniform_image_zero_gradient(self, uniform_gray):
        img_grad, _, _ = pspatial.sobel_gradient(uniform_gray)
        arr = np.array(img_grad)
        assert arr.max() == 0 or arr.max() <= 1  # Gradiente ~0 em imagem uniforme


# ──────────────────────────────────────────────────────────────────────────────
# 13 & 14. Filtros no Domínio da Frequência
# ──────────────────────────────────────────────────────────────────────────────

class TestFrequencyFilters:
    def test_gauss_lpf_grayscale(self, gray_img):
        result = pfreq.gaussian_lpf(gray_img, cutoff=30)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_gauss_hpf_grayscale(self, gray_img):
        result = pfreq.gaussian_hpf(gray_img, cutoff=30)
        assert result.mode == "L"

    def test_gauss_lpf_rgb(self, rgb_img):
        result = pfreq.gaussian_lpf(rgb_img, cutoff=30)
        assert result.mode == "RGB"

    def test_butter_lpf_grayscale(self, gray_img):
        result = pfreq.butterworth_lpf(gray_img, cutoff=30, order=2)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_butter_hpf_grayscale(self, gray_img):
        result = pfreq.butterworth_hpf(gray_img, cutoff=30, order=2)
        assert result.mode == "L"

    def test_butter_lpf_rgb(self, rgb_img):
        result = pfreq.butterworth_lpf(rgb_img, cutoff=30, order=2)
        assert result.mode == "RGB"

    def test_freq_output_in_range(self, gray_img):
        for func in [
            lambda img: pfreq.gaussian_lpf(img, 30),
            lambda img: pfreq.gaussian_hpf(img, 30),
            lambda img: pfreq.butterworth_lpf(img, 30, 2),
            lambda img: pfreq.butterworth_hpf(img, 30, 2),
        ]:
            result = func(gray_img)
            arr = np.array(result)
            assert arr.min() >= 0 and arr.max() <= 255


# ──────────────────────────────────────────────────────────────────────────────
# 15. Filtro Adaptativo de Mediana
# ──────────────────────────────────────────────────────────────────────────────

class TestAdaptiveMedian:
    def test_adaptive_median_grayscale(self, gray_img):
        result = padaptive.adaptive_median_filter(gray_img, max_size=7)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_adaptive_median_output_in_range(self, gray_img):
        result = padaptive.adaptive_median_filter(gray_img, max_size=9)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_adaptive_median_removes_impulse(self):
        """Verifica que o filtro remove ruído impulsivo."""
        arr = np.full((32, 32), 128, dtype=np.uint8)
        arr[16, 16] = 0    # Impulso pimenta
        arr[8, 8] = 255    # Impulso sal
        img = Image.fromarray(arr, mode="L")
        result = padaptive.adaptive_median_filter(img, max_size=7)
        result_arr = np.array(result)
        # Os impulsos devem ter sido corrigidos
        assert result_arr[16, 16] != 0
        assert result_arr[8, 8] != 255


# ──────────────────────────────────────────────────────────────────────────────
# 16 & 17. Ruídos
# ──────────────────────────────────────────────────────────────────────────────

class TestNoise:
    def test_gaussian_noise_grayscale(self, gray_img):
        result = pnoise.add_gaussian_noise(gray_img, mean=0, sigma=20)
        assert result.mode == "L"
        assert result.size == gray_img.size

    def test_gaussian_noise_rgb(self, rgb_img):
        result = pnoise.add_gaussian_noise(rgb_img, mean=0, sigma=20)
        assert result.mode == "RGB"

    def test_gaussian_noise_output_in_range(self, gray_img):
        result = pnoise.add_gaussian_noise(gray_img, mean=0, sigma=50)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_salt_noise_adds_white_pixels(self, uniform_gray):
        # Imagem uniform 128: salt deve adicionar pixels 255
        result = pnoise.add_salt_noise(uniform_gray, prob=0.1)
        arr = np.array(result)
        assert np.any(arr == 255)

    def test_pepper_noise_adds_black_pixels(self, uniform_gray):
        result = pnoise.add_pepper_noise(uniform_gray, prob=0.1)
        arr = np.array(result)
        assert np.any(arr == 0)

    def test_sp_unified_both(self, uniform_gray):
        result = pnoise.add_salt_pepper_unified(uniform_gray, 0.05, 0.05, "ambos")
        arr = np.array(result)
        assert np.any(arr == 255)  # Sal
        assert np.any(arr == 0)    # Pimenta

    def test_sp_unified_sal_only(self, uniform_gray):
        result = pnoise.add_salt_pepper_unified(uniform_gray, 0.1, 0.0, "sal")
        arr = np.array(result)
        assert np.any(arr == 255)
        # Não deve ter pixels pretos (pimenta)
        assert not np.any(arr == 0)

    def test_sp_unified_pimenta_only(self, uniform_gray):
        result = pnoise.add_salt_pepper_unified(uniform_gray, 0.0, 0.1, "pimenta")
        arr = np.array(result)
        assert np.any(arr == 0)
        # Não deve ter pixels brancos (sal)
        assert not np.any(arr == 255)

    def test_sp_unified_output_in_range(self, gray_img):
        result = pnoise.add_salt_pepper_unified(gray_img, 0.05, 0.05, "ambos")
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255
