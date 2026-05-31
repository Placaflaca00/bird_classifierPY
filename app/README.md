---
title: Clasificador de aves de Paraguay
emoji: 🦜
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: Identifica 20 especies de aves de Paraguay por su canto
---

# Conoce Tu Ave Py

Clasificador de aves de Paraguay por audio. Subi o graba un audio corto y el modelo
te dice las 3 especies mas probables, con foto y descripcion de la mas probable.

## Como funciona

Pipeline en dos etapas:

1. **BirdNET V2.4** (Cornell Lab) extrae un embedding 1024-dim del audio.
2. **Clasificador MLP propio** sobre ese embedding devuelve probabilidades para
   las 20 especies en scope.

El frontend Gradio hace POST a una Lambda en AWS (codigo en el repo de GitHub).

## 20 especies en scope

Charata, Chaja, Nandu, Tucan toco, Guacamayo azul y amarillo, Yacutinga, Jabiru,
Ipequi, Paloma domestica, 8 especies de playeros (Calidris, Tringa, Pluvialis,
Actitis).

## Configuracion

Esta app necesita la variable de entorno `API_GATEWAY_URL` apuntando al endpoint
de Lambda. Se configura en _Settings -> Variables and secrets_ del Space.

## Creditos

- **Audio**: BirdNET (Cornell Lab of Ornithology), modelo TFLite V2.4.
- **Fotos**: iNaturalist (16) y Wikimedia Commons (4). Atribucion individual en
  cada card del tab "Las 20 aves".
- **Descripciones**: Wikipedia ES (via iNat API).

Repo del proyecto: https://github.com/Placaflaca00/bird_classifierPY
