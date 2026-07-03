# Memory - OpusClip Clone Project

## Arquitectura del Proyecto

```
opus-clip-clone/
├── app.py              # UI Gradio (web interface)
├── transcriber.py      # Whisper transcription con CUDA
├── llm_analyzer.py     # Gemini API integration (key en .env)
├── video_editor.py     # FFmpeg/moviepy video processing
├── utils.py            # Helper functions
├── config.py           # Configuration y constantes
├── claude.md           # Reglas de estilo de código
├── memory.md           # Este archivo
├── requirements.txt    # Dependencias Python
├── .env               # Variables de entorno (API keys)
├── output/            # Videos generados
└── temp/              # Archivos temporales
```

## Stack Tecnológico

| Componente | Librería | Versión | Notas |
|------------|----------|---------|-------|
| UI | gradio | 4.x | Web interface local |
| Transcripción | openai-whisper | latest | Modelo "base", CUDA obligatorio |
| LLM | google-generativeai | latest | Gemini API - key en .env, NO hardcodeada |
| Video | ffmpeg-python | latest | Requiere FFmpeg preinstalado |
| Video | moviepy | latest | Para edición más compleja |
| Config | python-dotenv | latest | Variables de entorno |
| Tipado | mypy (dev) | latest | Type checking opcional |

## Estado Actual

### ✅ Completado
- [x] Crear `claude.md` con reglas de estilo
- [x] Crear `memory.md` con arquitectura
- [x] Generar `requirements.txt`
- [x] Módulo `transcriber.py` - Whisper con CUDA
- [x] Módulo `llm_analyzer.py` - Gemini API con key vía .env
- [x] Módulo `video_editor.py` - FFmpeg/moviepy (crop 9:16 + subtítulos)
- [x] Módulo `config.py` - Configuración centralizada
- [x] Módulo `state_manager.py` - Gestión de estado del proyecto
- [x] Módulo `subtitle_editor.py` - Editor de subtítulos con estilos
- [x] Módulo `app.py` - **UI OpusClip Pro** (3 paneles: Timeline | Preview | Editor)

### � Bugs Críticos Fixed (Fase 1)
- [x] **UI Threading**: Mejor progress tracking con callbacks
- [x] **Gemini Retry Logic**: Exponential backoff (1s, 2s, 4s) y mejor error handling
- [x] **MoviePy Memory Leak**: Context managers con try/finally y garbage collection
- [x] **Validación de sliders**: Min < Max duration

### ✨ Features MVP Implementadas (Fase 2)
- [x] **Filler Word Removal**: Elimina "eh", "um", "pues", "bueno", "sabes", etc.
- [x] **Auto-Emoji**: 80+ emojis mapeados a palabras clave (emociones, objetos, conceptos)
- [x] **UI Buttons**: Botones "🗑️ Quitar Muletillas" y "😊 Agregar Emojis"

### 📋 Pendiente (corregido tras auditoría 2026-07-02, ver sección "Auditoría 2026-07-02" más abajo)
- [x] ~~Animated Captions~~ → **YA IMPLEMENTADO**: `burn_karaoke_subtitles` + modos karaoke/highlight/pop en app.py (verificar pulido de la animación, ver punchlist)
- [x] ~~AI Object Tracking~~ → **YA IMPLEMENTADO**: `face_tracker.py` conectado a `video_editor.py` vía parámetro `track_faces`, checkbox en UI (app.py:2794)
- [ ] Testing end-to-end con video real
- [ ] Ver roadmap priorizado completo en sección "Auditoría 2026-07-02" más abajo

## Dependencias Instaladas
*Nota: Lista vacía hasta que se ejecute `pip install`*

## Configuración Específica

### Hardware
- **GPU**: NVIDIA GTX 1080 (8GB VRAM)
- **RAM**: 16GB sistema
- **CUDA**: Disponible, se detecta automáticamente

### Whisper
- Modelo: `base` (precargado)
- Dispositivo: `cuda` si disponible, else `cpu`
- Idioma: `es` (español)

### Gemini API
- **ESTADO**: ✅ Pool de 8 API keys en `.env` (`GEMINI_API_KEYS`, separadas por coma). Ya NO están
  hardcodeadas en el código ni en este archivo (ver sección "Auditoría 2026-07-02" más abajo).
- **Rotación automática**: `GeminiAnalyzer._generate_with_rotation()` en `llm_analyzer.py` detecta
  errores 429/quota/rate-limit y rota a la siguiente key del pool automáticamente, sin intervención
  del usuario. Si todas las keys agotan cuota, recién ahí falla.
- Modelo: `gemini-2.5-flash`
- Tarea: Analizar transcripción → JSON con timestamps + score viralidad
- Archivo: `@/d:/TODO/opus clip v2/llm_analyzer.py`
- **Nota (no urgente)**: `google.generativeai` (el SDK usado) está deprecado por Google en favor de
  `google.genai`. Sigue funcionando pero conviene migrar en algún momento — no es prioridad ahora.

### Formato de Salida
- Aspect ratio: 9:16 (vertical)
- Resolución: 1080x1920 (Full HD vertical)
- Subtítulos: Quemados en video, estilo moderno

## Formato JSON del LLM

```json
{
  "clips": [
    {
      "start": 12.5,
      "end": 45.2,
      "virality_score": 8.7,
      "reason": "Momentos graciosos con reacción inesperada"
    }
  ]
}
```

## Notas Importantes

> **⚠️ ADVERTENCIA**: `GEMINI_API_KEY` se carga desde `.env` (nunca se commitea, ver `.gitignore`).
> Copiar `.env.example` → `.env` y pegar la key real ahí en cada máquina nueva.

> **ℹ️ INFO**: FFmpeg debe estar preinstalado y en PATH de Windows.

## Estructura de Archivos

```
opus-clip-v2/
├── claude.md              # ✅ Reglas de estilo de código
├── memory.md              # ✅ Este archivo
├── requirements.txt       # ✅ Dependencias Python
├── config.py             # ✅ Configuración centralizada + estilos
├── transcriber.py        # ✅ Whisper + CUDA (GTX 1080)
├── llm_analyzer.py       # ✅ Gemini API (key vía .env)
├── video_editor.py       # ✅ FFmpeg/moviepy 9:16 + subtítulos + preview
├── state_manager.py      # ✅ Gestión de proyectos y estado
├── subtitle_editor.py    # ✅ Editor de subtítulos + estilos (Modern, TikTok, Minimal, Classic)
├── app.py                # ✅ UI OpusClip PRO - 3 paneles (Timeline | Preview | Editor)
├── output/               # 📁 Clips exportados
└── temp/                 # 📁 Archivos temporales y previews
```

## Actualizaciones

| Fecha | Cambio | Autor |
|-------|--------|-------|
| 2026-04-21 | Creación inicial de archivos de contexto | Claude |
| 2026-04-21 | Módulos transcriber, llm_analyzer, video_editor, config, app | Claude |
| 2026-04-21 | UI OpusClip Pro: state_manager, subtitle_editor, app.py rediseñado | Claude |
| 2026-07-02 | Auditoría seguridad/bugs/UX + remediación de 8 API keys expuestas + .gitignore + primer push a GitHub | Claude |

## Características de OpusClip Pro

### 🎯 Flujo de trabajo profesional
1. **Importar & Analizar**: Upload video + transcripción Whisper + análisis Gemini
2. **Editor de Clips**: Timeline visual + ajuste de timestamps + selección de clips
3. **Exportar**: Estilos de subtítulos + calidad configurable

### 🛠️ Panel de 3 secciones
- **Timeline (izquierda)**: Cards de clips con score viral, hook preview, controles de inicio/fin
- **Preview (centro)**: Reproductor de video 9:16 con previsualización de 15 segundos
- **Editor de Subtítulos (derecha)**: DataFrame editable, auto-corrección, estilos visuales

### 🎨 Estilos de subtítulos incluidos
- **Modern**: Arial-Bold, 64px, blanco con borde negro (estilo por defecto)
- **TikTok Style**: Impact, 72px, cyan #00f2ea (estilo TikTok)
- **Minimal**: Helvetica, 56px, fondo semitransparente
- **Classic**: Arial, 48px, amarillo (estilo tradicional)

### 💾 Gestión de estado
- Auto-guardado de ediciones en `temp/states/`
- Edición de subtítulos persistente por clip
- Selección/deselección de clips para exportación personalizada

---

## 🔍 Auditoría 2026-07-02 (agent-context, actualizar en cada sesión)

> Este bloque es el "agente de contexto vivo" del proyecto: cada vez que se encuentre un bug,
> hueco de seguridad o de UX, se documenta aquí con fecha. No se crean archivos nuevos de
> contexto — `memory.md` (estado/arquitectura) + `claude.md` (reglas de estilo) son la única
> fuente de verdad para futuras sesiones de Claude Code en este proyecto.

### 🔐 Seguridad — hallazgos y estado
| Hallazgo | Severidad | Estado |
|---|---|---|
| Key de Gemini hardcodeada en `llm_analyzer.py` (fallback) | Crítico | ✅ Corregido — ahora solo lee `.env` (`GEMINI_API_KEYS`), falla explícito si falta |
| 7 keys de Gemini adicionales, reales y funcionales, hardcodeadas en `test_api.py` | Crítico | ✅ Corregido — reescrito para leer de `.env`. Las 8 keys nunca llegaron a subirse a ningún lado (confirmado por el usuario), así que no había fuga real — el problema era solo el hardcodeo. Ahora forman un **pool con rotación automática** (`_generate_with_rotation` en `llm_analyzer.py`) que cambia de key sola si una pega un 429/rate-limit. |
| Key en texto plano dentro de `memory.md` | Alto | ✅ Corregido — removida de este archivo |
| No existía `.gitignore` (venv/, temp/, output/, videos con copyright, __pycache__ se hubieran subido) | Alto | ✅ Corregido — `.gitignore` creado |
| `videos para editar/` y `temp/` contienen videos de terceros con copyright (~700MB, contenido de YouTube) | Medio | ✅ Excluidos vía `.gitignore`, nunca deben entrar al historial de git |
| `GRADIO_SERVER_NAME = "0.0.0.0"` en `config.py:52` — expone la app a toda la red local, no solo localhost | Bajo (hoy) / Medio (si la PC se conecta a red no confiable) | ⚠️ Pendiente de decisión del usuario — cambiar a `127.0.0.1` por defecto salvo que se quiera acceso desde el celular/otro dispositivo en la LAN |
| Sin `shell=True` / f-strings en llamadas a ffmpeg (`subprocess.run` con listas) | — | ✅ Sin riesgo de command injection, ya está bien hecho |
| Sin uso de `pickle`/`eval`/`exec` | — | ✅ Sin riesgo de deserialización insegura |

**Resuelto 2026-07-02**: las 8 keys nunca se subieron a git/GitHub (el repo no existía hasta este
mismo día), así que no se consideran comprometidas — no hace falta rotarlas. Quedan como pool en
`.env` (`GEMINI_API_KEYS`), fuera del historial de git.

### 🐛 Bugs / calidad de código — hallazgos (auditoría profunda 2026-07-02, re-ejecutada)
- `video_editor.py:493,498,503,680,685,690` — seis `except: pass` desnudos en bloques `finally`
  de limpieza de recursos moviepy (cerrar `final_video`, `video`, `subtitle_clips`). Es un patrón
  defendible (best-effort cleanup) pero viola la regla propia de `claude.md` #2 (nunca `except:`
  vacío). **Pendiente**: `except Exception: logger.debug(...)` en vez de `except: pass`.
- `app_old.py` (22KB) — código muerto casi duplicado de `app.py` (misma clase `OpusClipPro`),
  confirmado también por graphify (nodo "surprising connection" `app_old.py` ↔ `app.py`).
  **Pendiente**: eliminarlo (regla #10 de `claude.md`).
- **🔴 CRÍTICO (impacto bajo hoy) — estado compartido entre sesiones de Gradio** (`app.py:3284`):
  la app crea **una sola instancia** de `OpusClipPro()` y todos los usuarios/pestañas comparten sus
  atributos mutables (`self.current_state`, `self.current_video`, `self.cancel_requested`,
  `self.subtitle_editors`). Si dos pestañas/usuarios usan la app a la vez, uno puede pisar el
  video/clips del otro, o cancelar el análisis del otro.
  **Decisión del usuario (2026-07-02)**: no se arregla por ahora — uso personal de una sola persona
  en localhost, no vale el riesgo de un refactor grande (mover `current_state`/`current_video`/
  `subtitle_editors`/`cancel_requested` a `gr.State()` por sesión) sin poder probarlo en vivo.
  Revisar si esto cambia si `GRADIO_SERVER_NAME` deja de ser `127.0.0.1`/uso mono-usuario.
- ✅ **Corregido**: export de clips (`app.py`, `_export_single_clip`) — antes solo el paso de zoom
  cues tenía try/except propio; si ducking/mood-grade/branding fallaban, se perdía el clip completo
  (incluyendo el crop+subtítulos que ya habían salido bien). Ahora los 3 pasos tienen try/except
  individual con `logger.warning`, igual que zoom cues — un fallo en un paso solo lo omite, no tira
  el clip entero.
- ✅ **Corregido**: `StateManager.save_state()` (`state_manager.py`) ahora usa `threading.Lock()` +
  escritura atómica (`os.replace`) para evitar JSON corrupto por escrituras concurrentes.
- ✅ **Corregido**: `on_video_select` (`app.py`, precheck de video) ya loguea el error en vez de
  descartarlo silenciosamente.
- ✅ **Corregido**: `toggle_clip_selection`/`on_select_clip` y los `clip_choices` de los dropdowns
  (`app.py`, 3 sitios) ahora usan `clip.id` de forma consistente en vez de mezclar índice de lista
  con id — evita que seleccionar/togglear el clip equivocado una vez se implemente reordenar
  (`state_manager.reorder_clips` ya existe pero no está cableado a la UI todavía).
- ✅ **Corregido**: type hint faltante en `load_saved_project` (`app.py`).
- Quedan **17** bloques `except Exception as e` que devuelven strings planos a un textbox en vez
  de `gr.Error`/`gr.Warning` — ya priorizado como roadmap UX #3 más abajo, esfuerzo bajo.

### 🎨 UX/UI — realidad vs. documentación
El proyecto está más avanzado de lo que `memory.md` indicaba antes de esta auditoría:
ya existen face-tracking (crop inteligente), subtítulos animados karaoke/highlight/pop,
6 presets de exportación por plataforma (TikTok/Reels/Shorts 1080x1920, LinkedIn/Twitter
1080x1080, Landscape 1920x1080), grading de color por mood, zoom cues automáticos, audio
ducking, y overlay de marca/color — ver "Pendiente" arriba, corregido.

**Roadmap priorizado (impacto vs. esfuerzo para creador solo):**
1. **Pulir animación de subtítulos karaoke** — ya existe el scaffolding (`animation_mode`),
   revisar si el swap de TextClip se siente fluido o crudo comparado a Opus Clip real.
2. **Carga por lotes (batch) de varios videos** — hoy `gr.File` es de un solo archivo;
   mayor ausencia para un flujo de creador solo (subir 5 episodios y dejarlo correr).
3. **Errores visibles vía `gr.Error`/`gr.Warning`** en vez de texto plano en un textbox —
   esfuerzo bajo, mejora percepción de calidad.
4. **Timeline con drag-to-trim** — hoy son campos numéricos de inicio/fin; el gap de UX
   más "se siente como Opus Clip real" pero requiere componente HTML/JS custom.
5. **Explicabilidad del score de viralidad** — verificar si el campo `reason` que ya
   devuelve Gemini se muestra en las cards del timeline o solo el número.
6. Undo/redo de ediciones (state_manager.py ya persiste estado, extenderlo es esfuerzo medio).
7. Atajos de teclado (espacio=play/pause, flechas=nudge) — después del timeline (#4).
8. Librería de música de fondo (hoy solo hay ducking de una pista ya agregada, no inserción).
9. B-roll/stock footage — baja prioridad, ni el Opus Clip real lo hace mucho.

### 👁️ Revisión visual real (Playwright, 2026-07-02) — no solo código
Se lanzó la app y se tomaron screenshots de las 3 pantallas (Importar/Editar/Exportar) para
revisar la UI tal como la ve el usuario, no solo leyendo el código de `app.py`.

**Bugs de UX confirmados visualmente:**
- **🔴 Tab "Exportar" se ve completamente vacío** cuando no hay clips seleccionados — ni un
  mensaje de estado. Contrasta con "Editar", que sí muestra "Analiza un video para ver clips
  detectados". Un usuario que hace click en "Exportar" antes de tiempo ve una pantalla en blanco
  y puede pensar que la app está rota. **Fix sugerido**: mismo patrón de placeholder que "Editar".
- **Nav lateral "Recursos 🔒" y "Ajustes 🔒"** están permanentemente bloqueados/grises — prometen
  funciones que no existen. Para uso 100% personal es inofensivo, pero si el proyecto se muestra
  a alguien más (o se piensa compartir), da sensación de producto a medio terminar. O se ocultan
  hasta que existan, o se quitan del nav.
- **Footer expone "Construido con Gradio" + "Usar vía API"** — rompe la ilusión de producto
  pulido tipo SaaS que el resto del diseño (header "OpusClip Pro V2.4 Powered by AI", badge de
  tokens, campana de notificaciones) sí logra. Fix de una línea: `footer{display:none}` en el CSS
  custom que ya existe, o `show_api=False` en `ui.launch()`.
- Header tiene badge "1,200 Tokens", campana y avatar de perfil — decorativos, no hacen nada.
  Coherente con la estética "SaaS" que se buscó, pero si en algún momento confunden al usuario
  (¿por qué no cambia el contador de tokens?), vale la pena quitarlos o cablearlos a algo real.
- Timeline confirma visualmente el hallazgo de código: son barras estáticas + campos numéricos
  Inicio(s)/Fin(s), no hay drag-to-trim (ya en el roadmap #4).

**Hallazgo de contenido viral (experto UX + growth IG/YT/TT) — el mayor gap real de producto:**
`app.py:_generate_clip_metadata` (línea ~908) **ya genera** título, descripción, hashtags y CTA
por clip — una feature core de Opus Clip que YA EXISTE en este proyecto. Pero:
1. **Está mal expuesta**: se escribe a un `.json` que se agrega a la lista de descargas del
   `gr.File`, no se muestra como texto legible/copiable en la UI. Un creador necesita copiar y
   pegar el caption directo a TikTok/IG al momento de publicar — obligarlo a abrir un JSON
   descargado mata el ahorro de tiempo que la feature debería dar. **Fix sugerido**: agregar un
   `gr.Textbox` (o Markdown con botón de copiar) por clip en el panel de exportación mostrando
   título + caption + hashtags listos para pegar.
2. **Calidad de los hashtags es débil**: se extraen de `hook + reason` (el texto que Gemini
   genera para *explicar por qué el clip es viral*, ej. "Ojos desorbitados de sorpresa"), no del
   *tema* real del clip. Esto puede producir hashtags como `#desorbitados` en vez de hashtags de
   descubrimiento reales (`#storytime`, `#viral`, tema del video). **Fix sugerido**: pedirle a
   Gemini hashtags explícitos en el mismo prompt de análisis (ya devuelve JSON estructurado, es
   agregar un campo `hashtags` al schema) en vez de derivarlos con regex del texto de rationale.
- **Zonas seguras de plataforma no consideradas**: TikTok/Reels/Shorts tapan con su propia UI
  (botones de interacción, perfil, caption nativo) las zonas inferior-derecha y a veces superior
  del video 9:16. Los subtítulos con `position: bottom` (estilo Minimal/Classic en `config.py`)
  pueden quedar tapados por la UI nativa de la plataforma al publicar. No es urgente pero es una
  optimización real de creador de contenido: dejar un margen inferior mayor (~20% de la altura)
  libre de texto importante.
- **Sin corte de silencios/pausas largas dentro de un clip** (distinto de "eliminar muletillas"):
  Opus Clip real acorta pausas largas de más de ~1-2s para mantener el ritmo. Hay
  `snap_to_silence` en `transcriber.py` pero es para no cortar a mitad de palabra al definir los
  bordes del clip, no para comprimir silencios internos. Posible feature futura, no urgente.

### Próximos pasos sugeridos (loop de mejora continua)
- Cada vez que Claude Code trabaje en este proyecto y encuentre un bug/duda/decisión de
  producto, agregarlo a este archivo con fecha antes de cerrar la sesión.
- Re-ejecutar la auditoría de bugs pendiente (arriba) cuando haya presupuesto de sesión.
- Decidir sobre `GRADIO_SERVER_NAME` (arriba) y actualizar esta tabla cuando se resuelva.
