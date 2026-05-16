# Benchmark: BirdNET nativo vs ONNX fine-tuned

**Fecha:** 2026-05-16  
**Modelo:** `wa-drop3-v1` (ver `models/classifier.json`)  
**BirdNET:** V2.4 GLOBAL 6K (6522 clases, 18 non-bird filtradas)  
**Inference time total:** 361.6 s

## TL;DR

- **test_clean** (n=351): baseline 70.4% -> pipeline 94.6% top-1 micro. Delta absoluto: **+24.2pp**, relativo: **1.34x**.
- **test_hard** (n=208): baseline 38.9% -> pipeline 81.2% top-1 micro. Delta absoluto: **+42.3pp**, relativo: **2.09x**.

## Mapeo BirdNET -> mis 20 clases

- 19/20 con match exacto por nombre cientifico.
- **Faltantes en BirdNET V2.4:** ['Pipile jacutinga']
  - Probados sinonimos para Pipile jacutinga: `Aburria jacutinga`, `Penelope jacutinga`. Tampoco existen.
  - Implica: para audios de `Pipile jacutinga`, baseline NUNCA puede acertar (top-1 baseline = wrong garantizado). Justifica el fine-tuning.

## Fold: `test_clean` (n=351)

### Metricas globales

| metric | baseline (BirdNET nativo) | pipeline (BirdNET emb + ONNX) | delta |
|---|--:|--:|--:|
| top-1 micro accuracy | 70.37% | 94.59% | +24.22pp |
| top-1 macro accuracy | 63.59% | 93.14% | +29.55pp |
| top-3 micro accuracy | 82.34% | 98.01% | +15.67pp |
| confianza promedio | 0.772 (max sigmoid) | 0.911 (softmax top-1) | (escalas distintas, no compar.) |
| baseline no_match (cayo fuera de mis 20) | 99 | — | — |

### Sub-vista: solo clases que BirdNET conoce (19/20)

Excluye Pipile jacutinga. Baseline accuracy comparable sin la penalidad estructural.
- n = 342
- baseline top-1 micro: 72.22%
- pipeline top-1 micro: 94.44%

### Per-species

| especie | n | baseline top-1 | pipeline top-1 | baseline top-3 | pipeline top-3 | gana |
|---|--:|--:|--:|--:|--:|:--|
| Actitis macularius | 31 | 74% | 94% | 87% | 97% | **pipe** |
| Ara ararauna | 29 | 72% | 100% | 97% | 100% | **pipe** |
| Calidris bairdii | 12 | 67% | 83% | 75% | 100% | **pipe** |
| Calidris canutus | 35 | 63% | 100% | 69% | 100% | **pipe** |
| Calidris fuscicollis | 7 | 71% | 86% | 71% | 100% | **pipe** |
| Calidris melanotos | 14 | 64% | 100% | 86% | 100% | **pipe** |
| Calidris minutilla | 22 | 55% | 82% | 68% | 91% | **pipe** |
| Calidris pusilla | 17 | 76% | 100% | 88% | 100% | **pipe** |
| Chauna torquata | 17 | 100% | 100% | 100% | 100% | **tie** |
| Columba livia | 8 | 75% | 100% | 100% | 100% | **pipe** |
| Heliornis fulica | 7 | 57% | 100% | 57% | 100% | **pipe** |
| Jabiru mycteria | 10 | 10% | 90% | 30% | 90% | **pipe** |
| Ortalis canicollis | 17 | 88% | 94% | 100% | 100% | **pipe** |
| Pipile jacutinga | 9 | 0% | 100% | 0% | 100% | **pipe** |
| Pluvialis dominica | 12 | 58% | 92% | 75% | 92% | **pipe** |
| Pluvialis squatarola | 10 | 80% | 90% | 90% | 90% | **pipe** |
| Ramphastos toco | 20 | 90% | 95% | 100% | 100% | **pipe** |
| Rhea americana | 6 | 0% | 67% | 17% | 100% | **pipe** |
| Tringa flavipes | 33 | 79% | 91% | 94% | 97% | **pipe** |
| Tringa melanoleuca | 35 | 91% | 100% | 100% | 100% | **pipe** |

## Fold: `test_hard` (n=208)

### Metricas globales

| metric | baseline (BirdNET nativo) | pipeline (BirdNET emb + ONNX) | delta |
|---|--:|--:|--:|
| top-1 micro accuracy | 38.94% | 81.25% | +42.31pp |
| top-1 macro accuracy | 34.47% | 87.67% | +53.20pp |
| top-3 micro accuracy | 60.10% | 89.42% | +29.33pp |
| confianza promedio | 0.486 (max sigmoid) | 0.759 (softmax top-1) | (escalas distintas, no compar.) |
| baseline no_match (cayo fuera de mis 20) | 123 | — | — |

### Sub-vista: solo clases que BirdNET conoce (19/20)

Excluye Pipile jacutinga. Baseline accuracy comparable sin la penalidad estructural.
- n = 207
- baseline top-1 micro: 39.13%
- pipeline top-1 micro: 81.16%

### Per-species

| especie | n | baseline top-1 | pipeline top-1 | baseline top-3 | pipeline top-3 | gana |
|---|--:|--:|--:|--:|--:|:--|
| Actitis macularius | 21 | 33% | 67% | 48% | 81% | **pipe** |
| Ara ararauna | 4 | 100% | 100% | 100% | 100% | **tie** |
| Calidris bairdii | 4 | 25% | 75% | 25% | 75% | **pipe** |
| Calidris canutus | 30 | 23% | 97% | 43% | 97% | **pipe** |
| Calidris fuscicollis | 2 | 50% | 100% | 50% | 100% | **pipe** |
| Calidris melanotos | 15 | 27% | 40% | 40% | 67% | **pipe** |
| Calidris minutilla | 11 | 27% | 100% | 64% | 100% | **pipe** |
| Calidris pusilla | 9 | 11% | 67% | 44% | 89% | **pipe** |
| Chauna torquata | 5 | 60% | 100% | 60% | 100% | **pipe** |
| Columba livia | 30 | 50% | 93% | 63% | 100% | **pipe** |
| Heliornis fulica | 1 | 0% | 100% | 0% | 100% | **pipe** |
| Jabiru mycteria | 1 | 0% | 100% | 0% | 100% | **pipe** |
| Ortalis canicollis | 2 | 0% | 100% | 0% | 100% | **pipe** |
| Pipile jacutinga | 1 | 0% | 100% | 0% | 100% | **pipe** |
| Pluvialis dominica | 8 | 38% | 75% | 62% | 75% | **pipe** |
| Pluvialis squatarola | 30 | 60% | 67% | 83% | 80% | **pipe** |
| Ramphastos toco | 2 | 100% | 100% | 100% | 100% | **tie** |
| Rhea americana | 4 | 0% | 100% | 25% | 100% | **pipe** |
| Tringa flavipes | 15 | 47% | 73% | 93% | 87% | **pipe** |
| Tringa melanoleuca | 13 | 38% | 100% | 77% | 100% | **pipe** |

## Decisiones metodologicas

- **Single-pass por audio**: una sola pasada por BirdNET captura embedding (penultima capa, 1024-d) y logits nativos (~6522). Sin doble forward — eficiencia y paridad numerica.
- **Agregacion baseline = MAX per-class across windows** (mirrora `lambda/handler.py` para gating; justificable por Wood & Kahl 2024 — BirdNET fue disenado para deteccion puntual en ventanas de 3s, max captura el peak).
- **Agregacion pipeline = MEAN embedding across windows** (igual que produccion y precompute_embeddings.py).
- **Filtro non-bird**: 18 clases de BirdNET excluidas antes del argmax (Engine, Noise, Dog, Human-varios, Power tools, ranas de generos Acris/Eleutherodactylus/Hyliola/Lithobates, etc).
- **Mapeo conservador**: solo nombre cientifico exacto + sinonimos verificados. No se inventan matches.
- **Confianza**: baseline (max sigmoid multi-label) y pipeline (softmax top-1) NO son la misma escala. Se reportan con su nombre real, sin claim de 'mejora de confianza'.
- **Sin overlap train/test verificado**: 0 archivos compartidos por basename entre `data/raw/metadata.parquet` (2392 audios, 1675 train+356 val+351 test_clean) y `data/test_sets/xc_hard/` (208).
