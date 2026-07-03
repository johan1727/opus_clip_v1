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
| 2026-07-03 | Creado `AGENT.md` como índice del loop de mejora continua (apunta a este archivo y a claude.md, no duplica contenido) | Claude |
| 2026-07-03 | UX pantalla "Importar": presets de 1 clic (Rápido/Balanceado/Calidad) + sliders técnicos movidos a acordeón "Ajustes avanzados" (cerrado por defecto) + tira de onboarding de 3 pasos + nav "Recursos/Ajustes" con badge "Pronto" en vez de 🔒 crudo. De paso se sacaron 2 badges de HTML estático (`stitch-slider-value` "15" y "15s - 60s") que nunca se actualizaban y ahora quedaban más visibles junto a los sliders funcionales. Motivo: usuario quiere vender el producto por suscripción a futuro, hoy la pantalla de import abrumaba con 4 sliders + 2 dropdowns antes de poder analizar nada. Verificado con Playwright (screenshots + click real de presets, valores confirmados). | Claude |
| 2026-07-03 | **Auditoría completa (comando "encuentra bugs reales, no arregles hasta que diga go") + arreglo de 21 bugs reales tras luz verde ("procede con todo").** Los 3 más graves: (1) `app.py::get_subtitle_data` — `KeyError: 'Start'` porque leía claves en inglés cuando `subtitle_editor.get_entries_for_dataframe()` devuelve claves en español ('Inicio'/'Texto Original'/etc) — **rompía la pantalla Editar completa** (info de clip, tabla de subtítulos, y los 3 botones de Mejoras IA) apenas un clip tenía subtítulos reales. (2) `analyze_with_frames` (el camino multimodal, preferido por defecto) mandaba a Gemini `text[:2000]` SIN timestamps y solo frames de los primeros ~45s de cualquier video — ahora manda transcripción completa por ventanas de 20s + frames distribuidos en toda la duración + le dice a Gemini a qué segundo corresponde cada frame. (3) **Bug oculto descubierto en vivo**: `extract_keyframes` llamaba a `config.TEMP_DIR` (el módulo, no `config.config.TEMP_DIR` la instancia) → `AttributeError` en cada llamada sin `output_dir` explícito → el camino multimodal NUNCA se había ejecutado de verdad en producción, siempre caía en silencio al análisis solo-texto. Al arreglarlo apareció un SEGUNDO bug (nunca antes alcanzado): el crop 9:16 se hacía después de escalar en vez de antes, lo que rompía la extracción de frames en TODO video landscape (16:9) con "Invalid too big... size for height". Ambos arreglados y verificados con un análisis real de punta a punta (15/15 frames extraídos, Gemini multimodal respondió con razones que mencionan expresiones faciales visuales — confirma que sí está usando los frames). Otros 18 bugs arreglados (ver lista completa en el chat/PR): selector de modelo Whisper que no tenía efecto tras la primera carga; ediciones de subtítulos que se perdían al recargar un proyecto guardado (+ claves int→str por el round-trip JSON); crash de `max()` en engagement scoring cuando no hay cambio de escena cerca; `snap_to_nearest` que nunca ajustaba nada (comparaba contra distancia 0); estilo "Minimal" que crasheaba en silencio (`ColorClip` no acepta strings de ningún color, ni CSS ni hex) y caía al fallback ffmpeg perdiendo posición/fondo/hook; reporte de "✅" en videos fallidos durante análisis por lote; SRT/VTT desincronizados del video cuando F4 (comprimir pausas) estaba activo; zoom dinámico (F2.3) que **nunca se había aplicado ni una vez** (variable `n` inválida en zoompan, debía ser `on`) y además solo usaba el primer cue de la lista; apóstrofes en el fallback drawtext que se comían la letra en silencio (arreglado quemando cada segmento a un `.txt` temporal con `textfile=` en vez de escapar `text='...'` inline, que no tiene forma confiable de manejar comillas simples en el parser de ffmpeg); duración 0 en videos MKV/WebM; tabla de subtítulos "editable" que no guardaba nada tecleado directo en la grilla; emojis automáticos con matching por substring (❌ aparecía dentro de "noche", "nosotros") — arreglado con `\b`; "como" sacado de la lista de muletillas (palabra demasiado común y legítima); clips ligeramente más largos que el máximo pedido ahora se recortan en vez de descartarse enteros; vista previa de clip crasheaba con videos sin pista de audio; caché de transcripción para modo "Calidad" podía servir en silencio datos de una corrida "Rápida" anterior del mismo video; thumbnails deformados en exports landscape/cuadrados; audio temporal de Whisper escribía junto al video fuente en vez de en `temp/`; mensaje de "Guardar" tiempos mostraba lo que el usuario tipeó en vez del valor real guardado (por los clamps de start/end). Cada arreglo se probó con render real de ffmpeg/MoviePy o ejecución directa de la función (no solo lectura de código) antes de darlo por bueno. | Claude |
| 2026-07-03 | Round 2 de la misma sesión, tras preguntarle al usuario explícitamente (decidió "ocultar" en vez de "Pronto", y "mismo tema oscuro + patrones Opus Clip" en vez de reescritura a tema claro): (1) se sacaron del todo los botones de nav "Recursos"/"Ajustes" (y su CSS `.nav-btn-locked`) — ya no prometen nada que no exista; (2) **acento único**: se reemplazó el esquema de 2 colores (cian `#00f2ea` + púrpura `#8523dd`, ~28 usos entre hex literal y rgba) por un solo violeta (`#7C5CFF` acento, `#5B3FD9` variante oscura para gradientes) en TODO `app.py` — botones, nav activo, logo, texto de marca, glows, bordes de foco. Los colores semánticos de `_score_color()` (verde/amarillo/rojo por calidad de score) no se tocaron a propósito, son funcionales no de marca. (3) El badge circular de "Virality Score" (antes 52-56px, decía "/10") se agrandó a 72px con glow y label "VIRALITY" en mayúsculas, más parecido al de Opus Clip real (investigado por web search: Opus Clip real es tema CLARO con acento único violeta — el usuario prefirió mantener el dark mode existente y solo adoptar el patrón de acento único + score protagonista, no migrar a tema claro). Verificado con Playwright en las 3 pantallas (Importar/Editar/Exportar), sin regresiones. Pendiente: si en algún futuro se decide ir a tema claro real estilo Opus Clip, es una reescritura grande del CSS (~1300 líneas), no incremental. | Claude |
| 2026-07-03 | **Feature nueva: overlay del "hook" quemado en pantalla** (`video_editor.py::_build_hook_clip`, enganchado en `burn_subtitles_moviepy` y `burn_karaoke_subtitles` vía nuevo param `hook_text`, y expuesto en `create_viral_clip(hook_text, show_hook, hook_duration)`). Motivo: Gemini ya calculaba `clip.hook` (frase gancho) para cada clip y se mostraba en la UI, pero nunca se quemaba en el video exportado — es el patrón #1 de retención que usan TikTok/CapCut/Opus Clip (texto grande los primeros ~2s). Reutiliza el mismo `SubtitleStyle` (fuente/color) que el usuario ya elegía para subtítulos (Moderno/TikTok/Minimal/Clásico) en vez de crear un picker de estilo aparte — ya existía suficiente variedad de "tipo de letra" en el proyecto (4 estilos visuales × 4 modos de animación Estático/Karaoke/Highlight/Pop), no hacía falta duplicarla. Checkbox nuevo "🪝 Hook al Inicio" (default ON) en el panel "Mejoras IA" de Exportar, threadeado por `_export_single_clip`/`export_clips`/`export_btn.click`. **Bug real encontrado y arreglado durante el testing**: `bg_color="rgba(0,0,0,0.55)"` (sintaxis CSS) no es válido para `TextClip` de MoviePy — solo acepta hex con alfa (`#RRGGBBAA`, como ya se usaba en el highlight de karaoke). Con el valor CSS, MoviePy fallaba en silencio y caía al fallback ffmpeg drawtext (que no soporta el hook), o sea el feature "funcionaba" pero nunca se veía. Se corrigió a `#0000008C`. Verificado renderizando de verdad con un video real (no solo lectura de código): frames extraídos confirman el hook visible en el tercio superior, con fade in/out, desapareciendo a los 2.2s, sin chocar con los subtítulos normales (estático y karaoke/pop probados). | Claude |

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
- ✅ **Corregido (2026-07-02)**: `requirements.txt` pineaba `gradio==4.44.0` pero el venv real
  tiene `gradio==6.13.0` instalado (detectado al usar `buttons=["copy"]` en un `Textbox`, API que
  no existe en 4.44 — ahí salió a la luz el drift). Se actualizó el pin en `requirements.txt` a
  `6.13.0` para que coincida con lo que realmente corre. Un `pip install -r requirements.txt` en
  limpio antes de este fix habría instalado una versión vieja e incompatible con el código actual.

### 🎨 UX/UI — realidad vs. documentación
El proyecto está más avanzado de lo que `memory.md` indicaba antes de esta auditoría:
ya existen face-tracking (crop inteligente), subtítulos animados karaoke/highlight/pop,
6 presets de exportación por plataforma (TikTok/Reels/Shorts 1080x1920, LinkedIn/Twitter
1080x1080, Landscape 1920x1080), grading de color por mood, zoom cues automáticos, audio
ducking, y overlay de marca/color — ver "Pendiente" arriba, corregido.

**Roadmap actualizado (2026-07-03) — solo lo que sigue pendiente de verdad:**

✅ Ya resueltos desde la versión original de esta lista (no repetir como pendientes):
carga por lotes, errores visibles vía `gr.Warning`, explicabilidad del score, corte de pausas.

Pendiente, de mayor a menor impacto para un creador solo:
1. **Timeline con drag-to-trim** — hoy son campos numéricos de inicio/fin en vez de arrastrar
   para cortar. El gap de UX más "se siente como Opus Clip real", pero requiere un componente
   HTML/JS custom (Gradio no tiene uno nativo) — el más grande y riesgoso de verificar sin que
   el usuario lo prueba interactuando en el navegador. Deliberadamente pospuesto.
2. **Pulir animación de subtítulos karaoke** — el scaffolding (`animation_mode`) ya funciona
   (verificado con MoviePy real tras el fix de `video.close()`), pero no se evaluó si el
   swap de `TextClip` se siente fluido o crudo comparado a Opus Clip real.
3. Undo/redo de ediciones (state_manager.py ya persiste estado, extenderlo es esfuerzo medio).
4. Atajos de teclado (espacio=play/pause, flechas=nudge) — depende de tener timeline (#1) primero.
5. Librería de música de fondo (hoy solo hay ducking de una pista ya agregada, no inserción).
6. B-roll/stock footage — baja prioridad, ni el Opus Clip real lo hace mucho.
7. Los 4 `except Exception` que quedaron sin `gr.Warning` a propósito (ver sección de limpieza
   más abajo) — bajo riesgo, no urgente.

### 👁️ Revisión visual real (Playwright, 2026-07-02) — no solo código
Se lanzó la app y se tomaron screenshots de las 3 pantallas (Importar/Editar/Exportar) para
revisar la UI tal como la ve el usuario, no solo leyendo el código de `app.py`.

**⚠️ Corrección (2026-07-02, mismo día)**: el hallazgo original "tab Exportar vacío" fue un
**falso positivo** del propio test de Playwright — el primer intento usó un selector de texto
(`get_by_text("Exportar", exact=True)`) que apuntó al breadcrumb del stepper en vez del botón real
del nav lateral (`button.nav-btn`). Con el selector correcto, la pantalla de Exportar está completa:
selector de Estilo Visual, toggles de Mejoras IA, panel "Listo para Renderizar" (plataforma destino,
post-procesamiento, Brand Kit, botón Exportar Video), galería de "Exports Recientes" y botones de
compartir. **No hay bug acá** — queda como nota para no repetir el mismo error de test.

**Hallazgo nuevo (válido) de esa misma pantalla**: la sección "Compartir" tiene botones para
Facebook/Twitter/LinkedIn — las plataformas menos relevantes para clips verticales cortos — pero
no para TikTok ni Instagram, que son el público objetivo real de esta herramienta. Probablemente
son decorativos (como el badge de tokens), no prioritario arreglarlo, pero vale la pena saberlo.

**Bugs de UX confirmados visualmente:**
- **Nav lateral "Recursos 🔒" y "Ajustes 🔒"** están permanentemente bloqueados/grises — prometen
  funciones que no existen. Para uso 100% personal es inofensivo, pero si el proyecto se muestra
  a alguien más (o se piensa compartir), da sensación de producto a medio terminar. O se ocultan
  hasta que existan, o se quitan del nav.
- ✅ **Corregido (2026-07-02)**: footer con "Construido con Gradio" / "Usar vía API" oculto vía
  `footer{display:none!important}` agregado al CSS custom de `app.py`. Verificado con Playwright.
- Header tiene badge "1,200 Tokens", campana y avatar de perfil — decorativos, no hacen nada.
  Coherente con la estética "SaaS" que se buscó, pero si en algún momento confunden al usuario
  (¿por qué no cambia el contador de tokens?), vale la pena quitarlos o cablearlos a algo real.
- Timeline confirma visualmente el hallazgo de código: son barras estáticas + campos numéricos
  Inicio(s)/Fin(s), no hay drag-to-trim (ya en el roadmap #4).

**Hallazgo de contenido viral (experto UX + growth IG/YT/TT) — el mayor gap real de producto:**
`app.py:_generate_clip_metadata` (línea ~908) **ya genera** título, descripción, hashtags y CTA
por clip — una feature core de Opus Clip que YA EXISTE en este proyecto. Pero:
1. ✅ **Corregido (2026-07-02)**: antes solo se escribía a un `.json` descargable, invisible para
   el creador al momento de publicar. Se extrajo la lógica compartida a `_build_social_metadata()`
   (usada tanto por `_generate_clip_metadata` para el JSON como por la nueva `_build_captions_text()`),
   y se agregó un `gr.Textbox` de solo lectura ("📋 Captions listos para publicar") con botón de
   copiar (`buttons=["copy"]`) en el panel de exportación, debajo de "Descargar archivos". Muestra
   título + descripción + hashtags + CTA de cada clip exportado, listo para pegar en TikTok/IG/YT.
   Verificado en vivo con Playwright (aparece correctamente, con placeholder cuando no hay exports).
2. ✅ **Corregido (2026-07-02)**: los hashtags ahora los devuelve Gemini directamente en el mismo
   JSON del análisis (`ViralClip.hashtags`, campo nuevo en el schema/few-shot de `llm_analyzer.py`,
   4-6 hashtags sobre el tema real del clip, mezclando nicho + alcance genérico). `_parse_hashtags()`
   normaliza y valida la respuesta (minúsculas, con `#`, dedupe, máx 6, descarta valores no-string).
   `_build_social_metadata()` en `app.py` usa `clip_state.hashtags` si Gemini los devolvió, y solo
   cae al regex viejo (`hook+reason`) para proyectos guardados antes de este cambio.
   **Verificado con una llamada real a Gemini**: para un clip sobre "se quemó la cena y el gato se
   comió todo", devolvió `#storytime #gatostiktok #cenadesastre #viral #fyp #parati` — hashtags de
   tema real, no derivados del texto de "por qué es viral" como antes.
   **Bug encontrado y corregido de paso**: la prueba reveló que la key #1 del pool (la original)
   está **suspendida por Google** (`403 CONSUMER_SUSPENDED`), y la rotación automática de keys
   (agregada antes en esta misma sesión) solo detectaba errores de cuota/429, no de key
   inválida/suspendida (403) — así que con la key #1 muerta, la rotación no se activaba y fallaba
   directo. Se amplió `_is_quota_error()` en `llm_analyzer.py` para también rotar en 403/permission
   denied/suspended/invalid api key. Verificado: ahora rota de la key #1 (suspendida) a la #2
   automáticamente y el análisis funciona.
   **⚠️ Pendiente del usuario**: la 1ª key en `GEMINI_API_KEYS` dentro de `.env` (la que era el
   fallback hardcodeado original) está suspendida por Google — no hace daño dejarla (el pool la
   salta sola), pero conviene reemplazarla por una key nueva o quitarla de `.env` para no gastar el
   intento fallido en cada análisis. (Ver `.env` local — nunca en este archivo ni en git.)
- ✅ **Corregido (2026-07-02)**: zonas seguras de plataforma. Además de agregar el margen, se
  encontró y arregló un bug mucho más grande en el camino: **el selector "Estilo Visual"
  (Moderno/TikTok/Minimal/Clásico) en la UI de exportación no tenía ningún efecto real** —
  `export_clips()` recibía `style_name` pero nunca lo pasaba a `_export_single_clip()` ni se
  construía nunca un `SubtitleStyle` a partir de él; `create_viral_clip()` ni siquiera aceptaba un
  parámetro `style`. Todos los clips exportados usaban el `SubtitleStyle()` default de
  `VideoEditor` (`position="center"`) sin importar qué estilo eligiera el usuario. Fix:
  - `video_editor.create_viral_clip()` ahora acepta `style: Optional[SubtitleStyle]` y lo pasa a
    `burn_subtitles_moviepy`/`burn_karaoke_subtitles`.
  - Nuevo `app.py:_subtitle_style_from_name()` convierte `style_name` → `SubtitleStyle` real leyendo
    `config.SUBTITLE_STYLES`, y aplica `margin_vertical=220` (antes 100) para estilos con
    `position="bottom"` (Minimal, Clásico) — la zona segura pedida originalmente.
  - `style_name` ahora se propaga `export_clips()` → `_export_single_clip()` (secuencial y
    paralelo vía `ThreadPoolExecutor`) → `create_viral_clip()`.
  - Diseño thread-safe a propósito: cada llamada construye un `SubtitleStyle` nuevo en vez de
    mutar `self.editor.subtitle_style` compartido, evitando una race condition con exportación
    paralela de varios clips.
  - Verificado: `_subtitle_style_from_name('minimal')` → `position=bottom, margin=220`;
    `_subtitle_style_from_name('modern')` → `position=center, margin=100`; nombre inválido cae a
    "modern" sin crashear.
- ✅ **Corregido (2026-07-02)**: explicabilidad del score — el campo `reason` de Gemini (por qué el
  clip tiene ese puntaje) nunca se mostraba en las cards del timeline, solo el hook y el número.
  Se agregó una línea `💡 {reason}` (truncada a 110 chars) en `_build_clips_summary()`.
- ✅ **Corregido (2026-07-02)**: corte de silencios/pausas largas dentro de un clip — implementado
  como Feature F4 (ver sección más abajo), ya no es un pendiente.

### 🧹 Limpieza 2026-07-02
- ✅ `app_old.py` eliminado (código muerto confirmado, sin referencias en ningún lado).
- ✅ Los 6 `except: pass` desnudos de `video_editor.py` ahora son `except Exception as _ce:
  logger.debug(...)`, cumpliendo la regla #2 de `claude.md`.
- ✅ 10 de los 17 `except Exception as e` ahora también muestran `gr.Warning()`/toast visible al
  usuario, no solo log de servidor. Los 4 restantes se dejaron con logging solamente, a propósito:
  - `_export_single_clip` (corre dentro de `ThreadPoolExecutor` — el contexto de Gradio para
    `gr.Warning`/`gr.Error` no propaga de forma confiable a threads manuales, riesgoso sin poder
    probarlo en vivo con una exportación real fallida).
  - `_build_captions_text` (duplicaría el warning que ya dispara `_generate_clip_metadata` para
    el mismo fallo subyacente).
  - Thumbnail de galería (fallback visual ya se auto-resuelve, no amerita alarmar al usuario).
  - Precheck de video (ya tiene fallback de texto inline adecuado, evitar fatiga de toasts).

## 🎬 Feature F4: Compresión de pausas largas (2026-07-02)

Nueva feature del roadmap UX, implementada y probada con video real (`videos para editar/1 Hour
of IshowSpeed Funny Moments.mp4`, clip de 45s extraído con ffmpeg + `silencedetect` real).

- `video_editor.py`: funciones módulo-nivel `compute_keep_ranges()` y `remap_to_compressed()`
  (puro Python, sin ffmpeg — comprime huecos >`max_gap` (1.2s default) entre segmentos de diálogo
  consecutivos a solo `target_gap` (0.35s default), sin eliminarlos del todo para que no se sienta
  abrupto) + método `VideoEditor.compress_pauses()` (ffmpeg trim+concat vía `ffmpeg-python`,
  patrón: `input.video.filter('trim',...).filter('setpts','PTS-STARTPTS')` +
  `input.audio.filter('atrim',...).filter('asetpts','PTS-STARTPTS')` por cada rango a conservar,
  unidos con `ffmpeg.concat(*parts, v=1, a=1)`).
- `create_viral_clip()` acepta `compress_pauses`/`max_pause_gap`/`target_pause_gap`; si está
  activado, comprime el crop ANTES de quemar subtítulos y remapea los timestamps de los
  segmentos (así los subtítulos quedan sincronizados con el timeline comprimido).
- UI: checkbox "✂️ Comprimir pausas largas (F4)" en el panel de exportación, cableado
  `export_clips()` → `_export_single_clip()` → `create_viral_clip()`.
- **Validado con video real**: clip de prueba de 45s con pausas reales (detectadas con
  `ffmpeg silencedetect`) comprimido a 36.9s, cada segmento de diálogo remapeado con duración
  exacta preservada (sin drift), audio y video sincronizados en el archivo final (verificado con
  `ffprobe`, diferencia de duración audio/video < 0.02s).
- Diseño intencional: no elimina silencios del todo (deja `target_gap` de buffer) para que el
  corte no se sienta como un jump-cut brusco — más parecido a como lo hace Opus Clip real.

## 🔴 Bugs CRÍTICOS pre-existentes encontrados mientras se probaba F4 con video real (2026-07-02)

Ninguno de estos tres bugs tiene que ver con la feature de compresión de pausas — se
descubrieron porque F4 fue la primera vez que se probó el pipeline de exportación completo
contra un video real durante una sesión de Claude Code (las auditorías anteriores fueron de
código/UI, no de video generado). Son independientes entre sí pero los tres bloqueaban la
promesa central de la app.

1. ✅ **CRÍTICO — `crop_to_vertical_ffmpeg()` producía clips SIN AUDIO, siempre.**
   El código armaba el filtro de crop+scale sobre el stream de video (`ffmpeg.filter(stream,
   'crop', ...)`) pero nunca pasaba el stream de audio del input a `ffmpeg.output()` — el comando
   ffmpeg compilado no tenía ningún `-map` de audio. Esto significa que **todos los clips
   exportados por esta app, desde siempre, salían mudos** (confirmado con `ffprobe`: el output
   del crop solo tenía `codec_type=video`, cero streams de audio). Fix: separar explícitamente
   `input_node.video` (para los filtros de crop/scale) de `input_node.audio`, detectar si la
   fuente tiene audio (`ffmpeg.probe()`), y pasar ambos streams a `ffmpeg.output()` cuando
   corresponda. Verificado: el crop ahora tiene `codec_type=video` + `codec_type=audio`.
2. ✅ **CRÍTICO — quemado de subtítulos vía MoviePy roto por completo (API 1.x vs 2.x).**
   `requirements.txt` pineaba `moviepy==1.0.3` pero el venv real tiene `2.1.2` instalado (mismo
   patrón de drift que ya se vio con gradio). MoviePy 2.x renombró todos los métodos mutadores
   `set_*` a `with_*` (`set_position`→`with_position`, `set_start`→`with_start`,
   `set_end`→`with_end`). El código seguía usando la API vieja → `AttributeError` en cada llamada,
   cayendo siempre al fallback de ffmpeg drawtext. Fix: reemplazadas las 13 ocurrencias de
   `.set_position/.set_start/.set_end` por `.with_position/.with_start/.with_end` en
   `burn_subtitles_moviepy` y `burn_karaoke_subtitles`. `requirements.txt` actualizado a
   `moviepy==2.1.2` para que coincida con lo que realmente corre.
3. ✅ **CRÍTICO — el fallback de ffmpeg drawtext (Windows) también fallaba.**
   Con el bug #2 arreglado, apareció este: `fontfile='C:/Windows/Fonts/arialbd.ttf'` rompe el
   parser de filtros de ffmpeg porque `:` es el separador de opciones del filtro, y las rutas de
   Windows tienen `C:` sin escapar. Error real: `"Error parsing a filter description"` (el mensaje
   de error que se logueaba antes solo mostraba los últimos 500 caracteres del stderr, ocultando
   la causa real al principio del mensaje). Esto significa que en Windows, **ni el camino
   principal (MoviePy) ni el fallback (ffmpeg drawtext) funcionaban** — subtítulos rotos al 100%
   en esta plataforma. Fix: escapar `:` → `\:` en la ruta de la fuente antes de insertarla en el
   filtro (`font_path_escaped = font_path.replace(':', '\\:')`).

### ✅ Resuelto (2026-07-02, sesión siguiente) — MoviePy con múltiples subtítulos
Causa raíz encontrada: **`burn_subtitles_moviepy` y `burn_karaoke_subtitles` cerraban el clip
base (`video.close()`) INMEDIATAMENTE después de armar el `CompositeVideoClip`, antes de
`write_videofile()`**. En MoviePy 1.x esto aparentemente no rompía nada, pero en 2.x los frames
se leen de forma perezosa recién durante la escritura — el composite necesita `video` todavía
abierto en ese momento. Con 1 solo segmento a veces no fallaba (posible carrera/orden de reads),
pero con 7-10 segmentos siempre tiraba `'NoneType' object has no attribute 'get_frame'`.
Fix: se eliminó el `close()` prematuro en ambos métodos — el `finally` ya existente se encarga de
cerrar `video`/`final_video` después de que `write_videofile()` termina. De paso se encontró y
arregló otra rotura de la migración 1.x→2.x: `.fadein()/.fadeout()` en `burn_karaoke_subtitles`
ya no existen como métodos en MoviePy 2.x, ahora son efectos (`from moviepy.video.fx import
FadeIn, FadeOut`, aplicados vía `.with_effects([FadeIn(d), FadeOut(d)])`).
**Verificado con video real**: `burn_subtitles_moviepy` con 7 segmentos y `burn_karaoke_subtitles`
con 11 palabras ahora completan sin caer al fallback de ffmpeg ("Video con subtítulos guardado" /
"Video karaoke guardado", sin el warning previo). Pipeline completo (`create_viral_clip` con
`compress_pauses=True` + subtítulos) probado de punta a punta: crop con audio ✅, pausas
comprimidas ✅, subtítulos quemados directo con MoviePy (calidad superior, sin fallback) ✅,
audio+video sincronizados en el archivo final ✅.

### Próximos pasos sugeridos (loop de mejora continua)
- Cada vez que Claude Code trabaje en este proyecto y encuentre un bug/duda/decisión de
  producto, agregarlo a este archivo con fecha antes de cerrar la sesión.
- Re-ejecutar la auditoría de bugs pendiente (arriba) cuando haya presupuesto de sesión.
- Decidir sobre `GRADIO_SERVER_NAME` (arriba) y actualizar esta tabla cuando se resuelva.
- Investigar el issue de MoviePy `CompositeVideoClip` con múltiples subtítulos (arriba).
