# D20 — Política de hold-out inmutable versionado en W&B Artifacts

- **Estado:** aceptada
- **Fecha:** 2026-05-25
- **Contexto**

  Fase 6 va a ejecutar un loop de retraining: cada candidato producido por GitHub Actions se compara contra el modelo en producción (`wa-drop3-v1`) usando McNemar's test pareado sobre un hold-out fijo (ver `memory/project-phase6-plan-discussed-2026-05-24`). La condición de promoción es `accuracy_delta ≥ +1pp AND p_value < 0.05` sobre `test_hard`.

  El test pareado de McNemar **exige que ambos modelos vean exactamente los mismos inputs**. Si entre el entrenamiento de `wa-drop3-v1` (Fase 5) y un retrain futuro alguien:

  - agrega audios nuevos al test_hard,
  - dropea audios sospechosos de label noise,
  - re-genera el split estratificado con otro seed,
  - cambia el mapping `filepath → fold`,

  los modelos viejos pierden su punto de referencia. Sus métricas históricas (e.g., `wa-drop3-v1: 0.8125 hard acc`) dejan de ser comparables con cualquier nuevo número producido. El loop de promoción se rompe silenciosamente: una "mejora" reportada podría ser solo cambio de denominador.

  El proyecto ya generó al cierre de Fase 5 los artefactos del hold-out actual:

  - `splits:v0` (W&B Artifact, type=dataset): 2590 filas mapeando `filepath → fold` en `{train: 1675, val: 356, test_clean: 351, test_hard: 208}`. Política de assignment: tiered por especie (N≥50 → 70/15/15; N≥20 → min 3 val/test; N≥10 → min 2; N<10 → todo a train), seed=42.
  - `test_hard:v0` (W&B Artifact, type=dataset, ~200 MB): los 208 audios MP3 físicos curados de Xeno-Canto con rating D/E ("worst") + `metadata.parquet`. 20 especies — todas las del clasificador post-drop Fase 5 — presentes.
  - `embeddings:v0` (W&B Artifact, type=dataset): embeddings BirdNET 1024-d con columna `split ∈ {train, test_hard}` ya marcada (los `train` se sub-particionan en `splits:v0` a train/val/test_clean).

  Necesitamos una regla explícita y aplicable que blinde estos artefactos contra mutación, válida no solo para Fase 6 sino para cualquier futura iteración del proyecto.

- **Decisión**

  **Los artefactos del hold-out son inmutables write-once. La versión `:v0` nunca se modifica. Cualquier cambio en el hold-out se materializa como `:v1` (nuevo artefacto) y queda explícitamente registrado.**

  Aplicado a este proyecto:

  1. **Alcance de la inmutabilidad** — cubre los tres artefactos encadenados que definen el hold-out:
     - `test_hard:v0` → audios + metadata.
     - `embeddings:v0` → embeddings BirdNET (incluye el fold `test_hard` marcado).
     - `splits:v0` → asignación filepath → fold para train / val / test_clean / test_hard.

     Las dos partes del hold-out son `fold=test_clean` (351 filas, derivadas por split) y `fold=test_hard` (208 filas, dataset separado). **Ambas son hold-out, ambas son inmutables.**

  2. **Lo que constituye una "modificación" prohibida sobre `:v0`** (cualquiera de estas requiere bumpear a `:v1`):
     - Agregar audios al fold `test_clean` o `test_hard`.
     - Eliminar audios del fold `test_clean` o `test_hard`.
     - Re-asignar un audio a otro fold (e.g., mover una fila de `test_clean` a `train`).
     - Cambiar la columna `species` de un audio (corrección de label noise).
     - Cambiar el seed del split estratificado.
     - Cambiar el contenido binario de un audio del `test_hard:v0` (reencodear, recortar, denoiser).
     - Cambiar las dimensiones del embedding (lo cual de hecho rompería todo, pero queda explícito).

  3. **Lo que NO es modificación** (no requiere bump):
     - Aliases adicionales sobre `:v0` (e.g., `--alias wa-drop3-baseline`). W&B permite múltiples aliases por versión sin alterar el contenido.
     - Cambios al `metadata` campo del artifact (descripción, tags). Solo afecta búsqueda en W&B, no el contenido.
     - Cambios en el código que **lee** los artefactos (e.g., `eval_grouped.py` agregando una vista nueva). El código de eval evoluciona; los datos congelados no.

  4. **Cuándo está permitido bumpear a `:v1`** — solo con justificación documentada:
     - Bug crítico en `:v0` (e.g., un audio mal-labeled que un revisor experto confirmó incorrecto, y el bug afecta sustancialmente las métricas).
     - Cambio de scope del proyecto que invalida el hold-out actual (e.g., expandir de 20 a 30 especies — el hold-out de 20 deja de ser representativo).
     - Cambio de modelo de extracción de features (e.g., BirdNET V2.4 → V3.0). Requiere bumpear `embeddings:v1` (re-computar) que arrastra cascada a `splits:v1`.

     En ningún caso se hace "in-place edit" de `:v0`. Siempre artefacto nuevo.

  5. **Procedimiento de bump**:
     - Documentar en este D20 (sección "Cambios al hold-out") la razón, fecha, y qué se cambió.
     - Subir el artefacto nuevo con W&B (alias `:v1` se asigna automáticamente al usar `log_artifact`).
     - Los modelos previos retienen su métrica histórica reportada **siempre con el sufijo de versión** (e.g., `wa-drop3-v1: 0.8125 hard acc sobre test_hard:v0`). Si se reentrenan modelos sobre `:v1`, su métrica se reporta con el nuevo sufijo. No hay re-evaluación retroactiva.
     - Todo modelo nuevo entrenado después del bump usa `:v1` para entrenar (vía `splits:v1`) y evaluar.
     - **La comparación de promoción de Fase 6 SIEMPRE usa la versión más reciente:** candidato vs prod, ambos evaluados sobre `:vN` actual. Si el bump ocurrió entre el último train de `wa-drop3-v1` (sobre `:v0`) y un retrain nuevo, el `wa-drop3-v1` debe re-evaluarse sobre `:v1` antes del McNemar para que la comparación tenga sentido.

  6. **Auditabilidad obligatoria**: toda evaluación de modelo registra en W&B vía `use_artifact("splits:vN")` y `use_artifact("embeddings:vN")` ANTES de `trainer.fit()` o `trainer.test()`. Sin esto, no hay lineage y la comparación no es defensible. El código de training (`src/training/train.py:148-150`) ya lo hace para el path no-local; el script `evaluate.py` de Fase 6 debe hacerlo igual.

- **Consecuencias**

  Trade-offs aceptados:

  - **Comparabilidad cross-version garantizada.** Cualquier número del proyecto (`wa-drop3-v1: 0.8125 hard acc`, `wa-drop4-vN: ???`) es interpretable sin ambigüedad mientras se cite la versión del hold-out.
  - **Costo: cero.** W&B Artifacts es gratis dentro del free tier de uso personal. El proyecto ya está versionando otros artefactos (`raw_audios`, `metadata`, `splits`, `embeddings`), agregar versiones de hold-out es marginal.
  - **Techo artificial aceptado.** Si en `test_hard:v0` hay audios mal-etiquetados, ningún modelo puede superar ese techo aunque sea perfecto. **Es aceptable**: la única forma de superarlo "honestamente" es bumpear a `:v1` con razón documentada, no en silencio. La opcionalidad de hacerlo a futuro existe; lo prohibido es hacerlo de paso.
  - **Hold-out pequeño es vulnerable a varianza.** Con 208 audios en test_hard distribuidos en 20 especies (~10 por especie), un cambio de label en 1 audio mueve la accuracy ~0.5pp. Es la razón principal por la que el threshold de promoción de Fase 6 es `+1pp AND p<0.05` y no solo `+1pp` (ver Plan Fase 6). El test estadístico controla la varianza; la inmutabilidad controla el sesgo cross-version.
  - **Trabajo extra al bumpear.** Cuando llegue el momento de `:v1`, hay que: re-evaluar modelos previos sobre la nueva versión si se quieren comparar con candidatos nuevos. Es esfuerzo computacional pequeño (~minutos en CPU) pero requiere disciplina.
  - **El annotator (Fase 5) NO contamina hold-out.** Los items `review_status=approved` con `reviewed_label` corregido por el ornitólogo son ground truth nuevo, valioso para training — pero NO entran al hold-out. Si entraran, los modelos previos (sin acceso a esos audios al entrenar) tendrían desventaja injusta. El flujo definido en Fase 6 (`export_dataset.py`) explícitamente filtra approved y los agrega al **fold train**, jamás al test_clean ni al test_hard.

  Lo que se pierde (asumido conscientemente):

  - **No se puede "actualizar progresivamente"** el hold-out con audios reales del usuario que validó el ornitólogo. Si quisiéramos un hold-out que evolucione con el dominio real (los audios que la app realmente recibe), no podríamos hacerlo dentro de esta política sin re-baselines de todos los modelos. Aceptado: la estabilidad de comparación pesa más que el "frescor" del hold-out.
  - **Hold-out puede quedar desactualizado** si la distribución del dominio real cambia mucho (e.g., usuarios graban con dispositivos de nueva generación con perfil acústico distinto). Mitigación: monitoreo de drift en Etapa 7 del `RETRAINING_STRATEGY.txt`. Si se detecta drift relevante, bumpear `:v1` con justificación.

- **Alternativas consideradas**

  1. **Hold-out mutable, "siempre la última versión"** — agregar audios approved del annotator al `test_hard` para hacerlo crecer sobre el tiempo. **Descartado**: rompe comparabilidad de raíz. La métrica reportada de `wa-drop3-v1` dejaría de ser reproducible la semana siguiente.

  2. **Solo `test_hard` inmutable, `test_clean` mutable** — atractivo porque `test_clean` es derivado por split, "se puede regenerar". **Descartado**: regenerar el split con otro seed (o sobre nuevos audios en `embeddings:v1`) cambia qué audios caen en `test_clean` vs `train`. Los modelos viejos pierden referencia igual. La derivación no es una excusa, los outputs son lo que importa.

  3. **Snapshot por modelo** — cada modelo guarda su propio test set congelado en el momento de entrenarse. **Descartado**: hace imposible comparar modelos entre sí. Cada uno se evalúa sobre su propio hold-out → solo se puede decir "el modelo X tuvo 0.81 sobre su hold-out, el modelo Y tuvo 0.83 sobre el suyo", sin poder afirmar que Y > X. McNemar's test colapsa porque exige inputs pareados.

  4. **Versionado manual en S3** (en vez de W&B Artifacts) — copias `s3://bucket/holdout/v0/`, `v1/`, etc. **Descartado**: sin lineage. Un modelo entrenado en W&B no quedaría enganchado automáticamente con la versión de hold-out usada. Habría que mantener el mapping manualmente, fuente de errores. W&B Artifacts es estándar de la industria precisamente por el lineage automático vía `use_artifact()`.

  5. **DVC + Git LFS** para versionar los archivos del hold-out. **Descartado**: el proyecto ya usa W&B para versionar otros datasets (`raw_audios`, `metadata`, `embeddings`, `splits`). Agregar DVC duplica el stack de versionado sin aportar funcionalidad nueva. KISS.

  6. **Política "lo borramos y empezamos de cero cada N meses"**. **Descartado**: incompatible con la comparación cross-version requerida por el loop de Fase 6. Cuando se necesite refrescar, se bumpea `:v1` con la razón documentada, no se borra `:v0`.

- **Notas de implementación**

  1. **Auditar antes de cada eval que `splits:v0` y `test_hard:v0` están subidos a W&B.** Los scripts `scripts/upload_splits_to_wandb.py` y `scripts/upload_test_hard_to_wandb.py` ya soportan dry-run (default) y `--upload`. Si por alguna razón los artefactos no están en W&B (e.g., entorno nuevo sin acceso), el lineage del eval falla loudly. **NO permitir fallback silencioso a archivos locales** — eso es exactamente el bug que esta política previene.

  2. **`evaluate.py` (Fase 6) debe declarar `use_artifact` explícito al inicio, antes de cargar cualquier dato**. Mismo patrón que `src/training/train.py:148-150` con `WandbLogger.experiment.use_artifact(cfg["splits_artifact"])`. Sin esa línea, el run de W&B no muestra el lineage, y la auditabilidad se rompe.

  3. **El script `scripts/upload_test_hard_to_wandb.py:95` declara `metadata.missing_species = ["Eudromia formosa", "Phoenicopterus chilensis"]`** — es información histórica antes del drop de Fase 5 (cuando el clasificador apuntaba a 22-23 especies). Al cierre de Fase 5, el clasificador final apunta a 20 especies y todas están presentes en `test_hard:v0` (verificado: 208 filas / 20 species unique sobre `data/test_sets/xc_hard/metadata.parquet`). El `missing_species` del metadata refleja la motivación original del drop, no un gap actual del hold-out vs las 20 clases. NO "corregir" el metadata de `:v0` por esto — es histórico válido.

  4. **El embedding del fold `test_hard` ya está pre-computado en `embeddings:v0`**. No hay paso de re-cómputo en cada eval. Esto es importante para Fase 6: `evaluate.py` solo necesita descargar `embeddings:v0` + `splits:v0` + el modelo candidato, NO `test_hard:v0` (los MP3 físicos). El artefacto `test_hard:v0` se descarga solo cuando se necesita re-computar embeddings (caso raro: cuando se bumpea `embeddings:v1` por cambio de modelo BirdNET, no por cambio de hold-out).

  5. **Backup de seguridad**: el proyecto mantiene `data/processed/embeddings.bak_pre_phase5_2026-05-09.parquet` y `data/processed/splits.bak_pre_phase5_2026-05-09.parquet` localmente. Estos son backups OPERACIONALES, no son la "versión `:v0`" oficial. Si W&B se cayera permanentemente y hubiera que re-subir, estos backups permiten reconstruir. Pero la fuente de verdad sigue siendo W&B.

  6. **Si en algún momento llega un bump a `:v1`** (e.g., 2026-08 después de detectar un audio mal-labeled crítico):
     - Agregar al final de este D20 una subsección "Cambios al hold-out / v0 → v1" con fecha + razón + qué cambió + impacto en modelos previos.
     - Re-evaluar `wa-drop3-v1` (o el modelo en prod en ese momento) sobre `:v1` y dejar la nueva métrica documentada al lado de la histórica de `:v0`.
     - El modelo en prod NO se re-entrena por el bump del hold-out — solo se re-evalúa. El re-train se hace cuando lo dispara el loop de Fase 6 normal.

  7. **Esta política aplica solo al hold-out**. Otros artefactos del proyecto (`raw_audios:vN`, `metadata:vN`) tienen otras políticas de versionado (en particular `metadata` cambia con frecuencia razonable cuando se re-curan audios). Lo que distingue al hold-out es que es la **referencia inmóvil** contra la que se compara todo. Ningún otro artefacto tiene ese rol.

- **Referencias cruzadas**

  - D18 — Estrategia de splits (pendiente): define cómo se construyó `splits:v0`. Esta D20 es la política de inmutabilidad sobre los outputs de D18.
  - D19 — Métrica primaria: macro-F1 (pendiente): qué se mide sobre el hold-out inmutable. La métrica puede evolucionar (reportar más métricas, refinarlas), el hold-out no.
  - D26 — Reentrenamiento manual (pendiente): el ADR de Fase 6, donde el hold-out de D20 es prerequisito.
  - `docs/RETRAINING_STRATEGY.txt` — Etapa 2 (Data Collection & Labeling) menciona la política de inmutabilidad; este D20 es la referencia formal.
  - `memory/project-phase6-plan-discussed-2026-05-24.md` — Plan Fase 6 que consume esta política para el loop de promoción (McNemar pareado).
  - `scripts/build_splits.py` — código que generó `splits:v0` (tiered per-species, seed=42).
  - `scripts/upload_splits_to_wandb.py` — código que sube `splits:v0` a W&B con lineage hacia `embeddings:v0`.
  - `scripts/upload_test_hard_to_wandb.py` — código que sube `test_hard:v0` a W&B.
