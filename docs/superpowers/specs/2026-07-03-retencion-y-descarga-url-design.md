# Diseño: Mejoras de retención de video + descarga automática por URL

**Fecha:** 2026-07-03
**Estado:** Aprobado por usuario en chat, pendiente de revisión del spec escrito.

## Contexto

OpusClip Pro (clon de Opus Clip: Whisper → Gemini → clips 9:16) ya tiene una base
técnica sólida tras la auditoría de bugs del 2026-07-03 (21 bugs reales arreglados,
ver `memory.md`). El pipeline de video/IA funciona de punta a punta. Esta ronda no es
sobre bugs — es sobre hacer que los clips que salen de la app retengan más en TikTok/
Reels/Shorts/X, más una feature de conveniencia (descarga por link).

Decisiones ya tomadas por el usuario (ver hilo de brainstorming):
- **Sin música/SFX por ahora** — se deja fuera de este spec, queda en backlog.
- **Descarga por URL**: YouTube + TikTok + Instagram + X vía `yt-dlp`.
- **Orden de entrega**: Fase 1 (retención) antes que Fase 3 (UI/UX). Fase 2
  (descarga URL) se intercala porque es independiente y acotada.
- **QA visual automático**: sí, se agrega al pipeline de export.

## Alcance de este spec

Cubre **Fase 1 (retención)** y **Fase 2 (descarga por URL)** con detalle de
implementación. **Fase 3 (UI/UX)** se deja enunciada a nivel de alcance únicamente
— se especificará en un spec separado cuando toque esa fase, para no sobrecargar
este documento ni el plan de implementación que sigue.

---

## Fase 1 — Retención

### R1. Subtítulos karaoke con timing real de Whisper

**Problema real (a confirmar el detalle exacto durante implementación, el síntoma
está confirmado por lectura de código):** `create_viral_clip()` en `video_editor.py`,
cuando `subtitle_mode` es `karaoke/highlight/pop`, reparte el tiempo de cada
segmento uniformemente entre sus palabras (`word_duration = (seg['end'] - seg['start']) / len(seg['text'].split())`).
Esto es correcto solo si el segmento de entrada ya es una sola palabra (que es el
caso cuando el segundo pase de Whisper con word-timestamps ya corrió — modo
"Balance"). Pero en modo "Calidad" (que transcribe el video COMPLETO con
`word_timestamps=True` desde el principio) los datos de palabra real existen en
`transcription['word_segments']` pero **no se están recortando por clip** — el
código solo puebla `clip_word_map` vía un segundo pase redundante
(`transcribe_clip_words`) que solo se dispara en modo Balance. En modo Calidad esto
cae al fallback de segmentos por FRASE (`get_segments_with_text`), tirando a la
basura el timing real de palabra que la transcripción completa ya tiene.

**Cambio:**
1. Antes de transcribir de nuevo por clip, revisar si `transcription['word_segments']`
   ya tiene cobertura de palabra real para el rango del clip (esto pasa siempre en
   modo Calidad, y nunca en modo Fast). Si existe, recortar esos word_segments al
   rango `[clip.start, clip.end]` y usarlos directamente — cero llamadas extra a
   Whisper.
2. Si no existe (modo Fast, o Balance antes de su segundo pase), mantener el flujo
   actual tal cual (segundo pase específico del clip en Balance; fallback de frase
   en Fast, aceptable ya que ahí nunca se pidió timing por palabra).
3. Confirmar con un test dirigido (no solo lectura de código) que el modo Balance
   efectivamente preserva el timing real hoy — si ya funciona correctamente, R1 se
   reduce a arreglar solo el caso de modo Calidad.

**Criterio de aceptación:** exportar el mismo clip en modo Calidad y modo Balance
con subtítulos "Pop", extraer frames en 3-4 timestamps de palabra conocidos, y
confirmar que la palabra resaltada coincide con el audio en esos instantes (no
solo "se ve bien", sino verificado contra los timestamps reales de Whisper).

### R2. Nuevo estilo de subtítulo "🔥 Viral" (chunks de 2-4 palabras)

Quinto preset en `config.SUBTITLE_STYLES` (junto a Modern/TikTok/Minimal/Clásico).
Difiere de los estilos existentes no solo en tipografía sino en **agrupamiento**:
en vez de mostrar la frase completa o palabra por palabra sueltas, agrupa las
palabras en chunks de 2-4 (por ejemplo, dividiendo por ventana de tiempo objetivo
~0.6-0.9s por chunk en vez de por frase), con la palabra activa en color de acento
y un pequeño "pop" de escala al aparecer (reutilizando el patrón de animación ya
usado en `animation_mode="pop"`).

**Implementación:** nueva función en `video_editor.py`, `_group_words_into_chunks()`,
que toma `word_segments` (formato ya existente: `start/end/text/parent_id/word_index`)
y los reagrupa en chunks de 2-4 palabras conservando los timestamps reales de cada
palabra (no se inventan tiempos nuevos). El burn en sí reutiliza
`burn_karaoke_subtitles` con un nuevo `animation_mode="viral"` que dibuja el chunk
completo con la palabra activa resaltada, en vez de una palabra sola.

**Riesgo evaluado y descartado:** migrar el burn de subtítulos a filtros ASS/libass
de ffmpeg en vez de MoviePy TextClip sería más rápido de renderizar, pero
introduciría un segundo motor de subtítulos a mantener en paralelo al ya existente
(MoviePy + fallback drawtext). Para clips de 15-60s con NVENC el costo de render
actual es aceptable — se anota como optimización futura, no se hace ahora.

### R3. Zoom punch-ins activados por defecto

`apply_zoom_cues` ya fue arreglado en la auditoría de bugs (usaba `n` en vez de
`on` en la expresión zoompan, nunca se había aplicado). Cambio: el checkbox
"Zoom dinámico por energía (F2.3)" en la pantalla Exportar pasa de `value=False`
a `value=True` por defecto, con `zoom_factor` bajado a un valor más sutil para uso
por defecto (a calibrar durante implementación, probablemente 1.05-1.08 en vez del
1.08 actual usado en pruebas, más agresivo).

### R4. Barra de progreso quemada en el video

Nuevo método `add_progress_bar_overlay()` en `video_editor.py`: filtro ffmpeg que
dibuja una línea delgada (4-6px) en la parte inferior del frame, con ancho
proporcional a `t / duration_total`, color = acento de marca (`#7C5CFF` o el que
configure el usuario vía brand_color). Se integra como paso opcional en
`create_viral_clip` (parámetro `show_progress_bar: bool`), con checkbox nuevo en
Exportar, activado por defecto.

### R5. Corte final seco (sin cola de silencio)

En `create_viral_clip`, después de calcular `adjusted_segments` (y de aplicar F4
si corresponde), recortar `end_time` del clip al `end` del último segmento con
texto + un margen fijo pequeño (~0.3s), en vez de usar el `end_time` que vino de
Gemini/el snap a silencio. Evita que el clip termine en 1-2s de video mudo/estático
que invita a hacer scroll antes de que termine.

### R6. Keyword highlight en el hook

Extender el prompt de Gemini (`llm_analyzer.py`, ambos `_build_prompt` y
`analyze_with_frames`) para pedir un campo nuevo `hook_keywords: List[str]`
(1-2 palabras o una frase corta dentro del hook que deben destacarse). El
`ViralClip` dataclass y `ClipState` ganan el campo `hook_keywords`. En
`_build_hook_clip` (video_editor.py, ya existe desde el trabajo de hoy), el texto
del hook se renderiza con esas palabras en el color de acento y el resto en blanco
— requiere pasar el hook como texto enriquecido en vez de un `TextClip` plano, lo
que implica renderizar 2-3 `TextClip` superpuestos (uno por segmento de color) en
vez de uno solo, posicionados en línea. Se investiga durante implementación si
MoviePy permite esto de forma simple o si conviene resolverlo con markup HTML-like
vía `method='caption'` con colores por palabra (similar al highlight ya usado en
`burn_karaoke_subtitles`).

### R7. QA visual automático post-export

Nuevo paso en `export_clips` (app.py), después de exportar cada clip
exitosamente: extraer 3 frames (a 0.5s, mitad, y `duración - 0.5s`) usando
`generate_thumbnail` (ya arreglado hoy para respetar el aspect ratio real). Un
nuevo método `_qa_check_clip()` corre chequeos automáticos simples y determinísticos
(no otra llamada a Gemini — sería lento/caro por clip):
- ¿El frame inicial no es negro/vacío? (verificación de brillo promedio del frame)
- ¿Hay overlay del hook visible en el frame inicial, si `show_hook` estaba activo?
  (chequeo de presencia, no de contenido — buscar contraste alto en la zona donde
  se dibuja el hook)
- ¿El frame final no es negro? (detecta el problema que R5 debería prevenir)

Los 3 frames + resultados de estos checks se muestran en una nueva sección
"Control de Calidad" en la galería de exports (`output_gallery` ya existente, o una
nueva `gr.Gallery` al lado), con ✅/⚠️ por clip. No bloquea el export — es
informativo, para que el usuario vea el problema en la UI en vez de en TikTok.

---

## Fase 2 — Descarga automática por URL

### I1. Campo de URL en la pantalla Importar

Nuevo `gr.Textbox` "Pegá un link de YouTube/TikTok/Instagram/X" en el panel
"Fuente de Video", junto al `gr.File` existente (no lo reemplaza — conviven ambos
caminos: subir archivo o pegar link).

**Dependencia nueva:** `yt-dlp` (se agrega a `requirements.txt` con versión fija).

**Flujo:**
1. Usuario pega URL, click en un botón nuevo "⬇️ Descargar".
2. Nuevo método `download_from_url(url, progress)` en `app.py` (o un módulo nuevo
   `url_downloader.py` si el código crece — a decidir en el plan de implementación
   según cuánto código resulte):
   - Valida que la URL sea de un dominio soportado antes de invocar yt-dlp (evita
     pasar URLs arbitrarias a un proceso externo sin chequeo).
   - Descarga a `videos para editar/` (mismo directorio que usa el flujo manual)
     con `yt-dlp`, formato preferido `best[ext=mp4]` para evitar remux innecesario.
   - Progreso de descarga (yt-dlp expone hooks de progreso) mapeado al
     `gr.Progress` de Gradio, igual que el resto del pipeline.
   - Al terminar, auto-selecciona el archivo descargado en `video_input` (dispara
     el mismo `on_video_select` que ya corre para archivos subidos manualmente —
     cero duplicación de lógica de precheck).
3. Manejo de errores: video privado/eliminado/con restricción de edad/geo-bloqueado
   → mensaje claro vía `gr.Warning`, no un traceback crudo.

**Seguridad:** yt-dlp ejecuta la descarga en el mismo proceso (no shell externo con
concatenación de strings) — se usa la API de Python de yt-dlp
(`yt_dlp.YoutubeDL`), no `subprocess` con la URL interpolada directo en un comando,
para evitar cualquier vector de inyección de argumentos.

**Límite razonable:** duración máxima descargable configurable (default 2 horas,
mismo espíritu que `MAX_VIDEO_SIZE_MB` ya existente en `config.py`) para evitar que
alguien pegue un stream de 10 horas por error y sature disco/tiempo de proceso.

---

## Fase 3 — UI/UX (alcance enunciado, spec propio pendiente)

Se especificará en un documento separado cuando corresponda esa fase. Alcance
acordado en el chat: fix del HTML crudo visible en pantalla Editar (Gradio 6 dejó
de renderizar HTML dentro de `gr.Markdown`, hay que migrar esos paneles a
`gr.HTML`), preview del clip con subtítulos ya quemados antes de exportar,
thumbnails reales en las cards de clips, y selector de estilos con vista previa
visual en vez de solo texto.

---

## Fuera de alcance (backlog, no en este spec)

Música de fondo y SFX automáticos, B-roll automático, publicación directa a redes
sociales, voiceover con IA, soporte multi-usuario/multi-tenant. Se registran en
`memory.md` como ideas futuras, no se tocan en esta ronda.

## Testing / verificación

Cada ítem se verifica con render real (ffmpeg/MoviePy) y frames extraídos antes de
darse por completo, siguiendo el mismo estándar aplicado en la auditoría de bugs
de hoy — no alcanza con que el código "se vea correcto" en la lectura.
