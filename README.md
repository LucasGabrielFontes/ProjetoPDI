# Projeto PDI — Processamento Digital de Imagens

**UFPB — DSC | Disciplina: Introdução ao Processamento Digital de Imagens**
**Professor:** Augusto de Holanda B. M. Tavares

---

## Requisitos

- Python 3.10+
- pip

## Instalação e Execução

```bash
# 1. Clonar / baixar o projeto
cd ProjetoPDI

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar
python main.py
```

## Processos Implementados

| # | Processo | Parâmetros |
|---|---|---|
| 1 | Decomposição RGB | — |
| 2 | Decomposição HSV | — |
| 3 | Limiarização | k |
| 4 | Transformação Logarítmica | c |
| 5 | Transformação de Potência (Gamma) | c, γ |
| 6 | Equalização de Histograma | — |
| 7 | Fatiamento por Intensidade | A, B, preservar fundo |
| 8 | Filtro Gaussiano | σ, tamanho do kernel |
| 9a | Filtro de Mediana | tamanho do kernel |
| 9b | Filtro de Mínimo | tamanho do kernel |
| 9c | Filtro de Máximo | tamanho do kernel |
| 10 | Máscara de Aguçamento (Unsharp) | gain, σ, tamanho kernel |
| 11 | Realce por Laplaciano | usar diagonais |
| 12 | Gradiente de Sobel | — |
| 13a | Passa-Baixa Gaussiano | frequência de corte D₀ |
| 13b | Passa-Alta Gaussiano | frequência de corte D₀ |
| 14a | Passa-Baixa Butterworth | D₀, ordem n |
| 14b | Passa-Alta Butterworth | D₀, ordem n |
| 15 | Mediana Adaptativa | tamanho máx. janela |
| 16 | Ruído Gaussiano | μ, σ |
| 17a | Ruído Sal | probabilidade |
| 17b | Ruído Pimenta | probabilidade |
| 17c | Ruído Sal-e-Pimenta | probabilidade |

## Formato de Imagens Suportado

- **Grayscale:** PNG 8 bits (escala de cinza)
- **Coloridas:** PNG 24 bits (RGB)

Operações indisponíveis para o tipo de imagem são automaticamente desabilitadas.
