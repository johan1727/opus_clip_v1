# Retención de video + descarga por URL — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hacer que los clips exportados retengan más viewers en TikTok/Reels/
Shorts/X (subtítulos con timing real, chunks estilo "Viral", zoom activado,
barra de progreso, corte final seco, keyword highlight en el hook, QA visual
automático) y agregar ingesta de video por URL (YouTube/TikTok/IG/X vía
`yt-dlp`).

**Architecture:** Todos los cambios de retención tocan el pipeline existente
`app.py` (orquestación) → `video_editor.py` (ffmpeg/MoviePy) →
`llm_analyzer.py`/`state_manager.py` (datos de Gemini). No se introduce
infraestructura nueva — se extienden funciones ya existentes siguiendo el
mismo patrón que el resto del código (MoviePy para composición de texto,
ffmpeg puro para filtros de video, fallback silencioso a "copiar sin
modificar" cuando algo opcional falla). La descarga por URL es un módulo
nuevo y aislado (`url_downloader.py`) que produce un archivo local y luego
reutiliza el flujo de importación ya existente sin tocarlo.

**Tech Stack:** Python 3.10, Gradio 6.13, MoviePy 2.1.2, ffmpeg-python 0.2.0,
google-generativeai 0.8.3, yt-dlp (nuevo).

## Global Constraints

- Sin música/SFX en este plan (decisión explícita del usuario) — no agregar
  dependencias de audio nuevas (pydub ya está como fallback opcional, no se
  toca).
- Este proyecto NO usa pytest ni tiene carpeta `tests/` — la convención real
  y ya establecida (`test_api.py` en la raíz, más los scripts usados durante
  la auditoría de bugs del 2026-07-03) es: scripts standalone en la raíz del
  repo, ejecutados directo con `venv/Scripts/python.exe archivo.py`,
  verificados con salida real (render de ffmpeg/MoviePy + extracción de
  frames cuando aplica), no con asserts de un framework. Cada tarea de este
  plan sigue esa misma convención.
- Todo cambio a un filtro de ffmpeg o a un parámetro de MoviePy se verifica
  con una ejecución REAL antes de darse por bueno — no alcanza con que el
  código "se vea bien" (estándar ya aplicado en la sesión de bugs de hoy,
  donde aparecieron 3 bugs de ffmpeg que solo se detectaron ejecutando).
- Video de prueba disponible en
  `videos para editar/1 Hour of IshowSpeed Funny Moments.mp4` (1h, real,
  landscape 16:9) y un clip corto ya recortado en
  `D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_clip.mp4`
  (45s, landscape 16:9, con audio) — usar el corto para iterar rápido, el
  largo solo si un test necesita duración real.
- Servidor de la app: para relanzarlo tras cambios, matar el proceso en el
  puerto 7860 (`netstat -ano | grep ":7860"` + `Stop-Process`) y volver a
  correr `venv/Scripts/python.exe app.py`, igual que se hizo toda la sesión
  de hoy.

---

### Task 1: Subtítulos karaoke con timing real de Whisper en modo Calidad

**Files:**
- Modify: `transcriber.py` (agregar 2 métodos nuevos después de
  `get_segments_with_text`, que termina en la línea 799)
- Modify: `app.py:359-367` (loop que arma `clips_data`)
- Test: `test_word_timing_fix.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de tareas previas (es la primera tarea).
- Produces: `Transcriber.has_real_word_timing(transcription: dict) -> bool` y
  `Transcriber.get_word_segments_for_clip(transcription: dict, start_time: float, end_time: float) -> List[Dict[str, Any]]`
  (cada dict con claves `id`, `start`, `end`, `text` — timestamps ABSOLUTOS
  del video, mismo formato que `get_segments_with_text`). Tareas posteriores
  no dependen de esto directamente.

- [ ] **Step 1: Escribir el script de verificación (falla primero)**

Crear `test_word_timing_fix.py` en la raíz del repo:

```python
"""Verifica que has_real_word_timing/get_word_segments_for_clip existen y
funcionan antes de integrarlos en app.py. Corre standalone, sin GPU ni red."""
import sys
sys.path.insert(0, ".")
from transcriber import Transcriber

t = Transcriber.__new__(Transcriber)  # no queremos cargar el modelo real

# Caso 1: transcripción SIN word-timestamps (modo Fast/Balance antes del 2do pase)
# _flatten_word_segments cae a 1 entrada por frase cuando no hay 'words' en el
# segmento de Whisper, así que word_segments == segments en longitud.
transcription_sin_palabras = {
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "hola mundo"},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "como estas"},
    ],
    "word_segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "hola mundo"},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "como estas"},
    ],
}
assert t.has_real_word_timing(transcription_sin_palabras) is False, \
    "no deberia detectar timing real cuando word_segments == segments en longitud"

# Caso 2: transcripción CON word-timestamps reales (modo Calidad) — más
# word_segments que segments de frase.
transcription_con_palabras = {
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "hola mundo"},
        {"id": 1, "start": 5.0, "end": 7.0, "text": "como estas hoy"},
    ],
    "word_segments": [
        {"id": 0, "start": 0.0, "end": 0.9, "text": "hola"},
        {"id": 1, "start": 0.9, "end": 2.0, "text": "mundo"},
        {"id": 2, "start": 5.0, "end": 5.6, "text": "como"},
        {"id": 3, "start": 5.6, "end": 6.3, "text": "estas"},
        {"id": 4, "start": 6.3, "end": 7.0, "text": "hoy"},
    ],
}
assert t.has_real_word_timing(transcription_con_palabras) is True, \
    "deberia detectar timing real cuando hay mas word_segments que segments"

# Recorte al rango de un clip (5.0s a 7.0s) debe traer solo esas 3 palabras
recortado = t.get_word_segments_for_clip(transcription_con_palabras, 5.0, 7.0)
assert len(recortado) == 3, f"esperaba 3 palabras, salieron {len(recortado)}"
assert recortado[0]["text"] == "como"
assert recortado[0]["start"] == 5.0
assert recortado[-1]["text"] == "hoy"
assert recortado[-1]["end"] == 7.0

print("TODOS LOS CHECKS PASARON")
```

- [ ] **Step 2: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_word_timing_fix.py`
Expected: `AttributeError: 'Transcriber' object has no attribute 'has_real_word_timing'`
(los métodos todavía no existen — esto confirma que el test realmente
ejercita el código nuevo, no un mock).

- [ ] **Step 3: Implementar los dos métodos en `transcriber.py`**

Insertar inmediatamente después del final de `get_segments_with_text`
(que termina en la línea 799, justo antes de `def generate_srt_content`):

```python
    def has_real_word_timing(self, transcription: Dict[str, Any]) -> bool:
        """
        True si transcription['word_segments'] tiene granularidad de PALABRA
        real (la transcripción completa corrió con word_timestamps=True, ej.
        modo Calidad), no solo un fallback de una entrada por frase. Se
        detecta comparando cantidades: si Whisper partió por palabra, hay
        más word_segments que segments de frase; si no, ambas listas miden
        lo mismo (_flatten_word_segments cae a 1:1 sin datos de palabra).
        """
        word_segments = transcription.get('word_segments', [])
        segments = transcription.get('segments', [])
        return len(segments) > 0 and len(word_segments) > len(segments)

    def get_word_segments_for_clip(
        self,
        transcription: Dict[str, Any],
        start_time: float,
        end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Recorta transcription['word_segments'] (timestamps reales por
        palabra) al rango absoluto [start_time, end_time] de un clip.
        Llamar solo cuando has_real_word_timing() devolvió True — si no,
        word_segments no tiene más granularidad que segments y no aporta
        nada sobre get_segments_with_text().

        Returns:
            Lista de dicts {id, start, end, text} con timestamps ABSOLUTOS
            del video (mismo formato que get_segments_with_text), una
            entrada por PALABRA.
        """
        word_segments = transcription.get('word_segments', [])
        filtered = []
        for w in word_segments:
            w_start = w['start']
            w_end = w['end']
            if w_end < start_time or w_start > end_time:
                continue
            filtered.append({
                'id': w.get('id', len(filtered)),
                'start': w_start,
                'end': w_end,
                'text': w['text'].strip(),
            })
        return filtered
```

- [ ] **Step 4: Correr el script y confirmar que pasa**

Run: `venv/Scripts/python.exe test_word_timing_fix.py`
Expected: `TODOS LOS CHECKS PASARON`

- [ ] **Step 5: Integrar en `app.py` — usar el timing real cuando exista**

En `app.py`, el loop que arma `clips_data` (líneas 359-367) hoy es:

```python
            clips_data = []
            total_tokens = 0
            for i, clip in enumerate(viral_clips):
                word_segs_for_clip = clip_word_map.get(i, [])
                if not word_segs_for_clip:
                    # Fallback: phrase-level segments from fast transcription
                    word_segs_for_clip = self.transcriber.get_segments_with_text(
                        transcription, clip.start, clip.end
                    )
```

Reemplazar por:

```python
            clips_data = []
            total_tokens = 0
            # Modo Calidad transcribe el video COMPLETO con word_timestamps=True
            # desde el principio — si eso pasó, hay timing real de palabra
            # disponible sin necesidad de re-transcribir por clip (eso es lo
            # que hace clip_word_map, pero solo corre en modo Balance).
            has_word_timing = self.transcriber.has_real_word_timing(transcription)
            for i, clip in enumerate(viral_clips):
                word_segs_for_clip = clip_word_map.get(i, [])
                if not word_segs_for_clip and has_word_timing:
                    word_segs_for_clip = self.transcriber.get_word_segments_for_clip(
                        transcription, clip.start, clip.end
                    )
                if not word_segs_for_clip:
                    # Fallback: phrase-level segments from fast transcription
                    word_segs_for_clip = self.transcriber.get_segments_with_text(
                        transcription, clip.start, clip.end
                    )
```

- [ ] **Step 6: Verificación end-to-end con Whisper real en modo Calidad**

Crear `test_quality_word_timing_e2e.py` en la raíz:

```python
"""Corre analyze_video en modo Calidad sobre el clip corto de prueba y
confirma que los segments guardados por clip son de UNA PALABRA cada uno
(prueba de que se usó get_word_segments_for_clip, no el fallback de frase)."""
import sys, os
sys.path.insert(0, ".")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from app import OpusClipPro

class FakeProgress:
    def __call__(self, *a, **kw):
        pass

VIDEO = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_clip.mp4"

app = OpusClipPro()
app.analyze_video(
    VIDEO, num_clips=3, min_duration=2, max_duration=45,
    model_size="base", progress=FakeProgress(), analysis_mode="quality"
)

assert app.current_state is not None, "el analisis no genero current_state"
assert len(app.current_state.clips) > 0, "no se identifico ningun clip (ajustar min_duration si hace falta)"

clip = app.current_state.clips[0]
avg_words_per_segment = sum(len(s['text'].split()) for s in clip.segments) / max(len(clip.segments), 1)
print(f"Clip 0: {len(clip.segments)} segments, promedio {avg_words_per_segment:.1f} palabras/segment")
assert avg_words_per_segment < 1.5, (
    f"promedio de {avg_words_per_segment:.1f} palabras/segment sugiere que se "
    "esta usando el fallback de FRASE, no el timing real por palabra"
)
print("OK: modo Calidad usa timing real por palabra")
```

Run: `venv/Scripts/python.exe test_quality_word_timing_e2e.py`
Expected: termina con `OK: modo Calidad usa timing real por palabra` (si no
hay clips con `min_duration=2`, bajar a `min_duration=1` — el objetivo es
solo obtener AL MENOS un clip real, el contenido no importa para este check).

- [ ] **Step 7: Borrar los scripts de test temporales y commitear**

```bash
rm test_word_timing_fix.py test_quality_word_timing_e2e.py
git add transcriber.py app.py
git commit -m "R1: usar timing real de Whisper por palabra en modo Calidad

Modo Calidad ya transcribe el video completo con word_timestamps=True,
pero el codigo de armado de clips solo miraba clip_word_map (poblado
unicamente en modo Balance via un segundo pase) y caia al fallback de
segmentos por frase, tirando el timing real a la basura."
```

---

### Task 2: Modo de animación "🔥 Viral" (chunks de 2-4 palabras)

**Files:**
- Modify: `video_editor.py` (nuevo método `_group_words_into_chunks`, insertado
  antes de `burn_karaoke_subtitles` en la línea 652; modificar
  `burn_karaoke_subtitles` en 3 puntos)
- Modify: `app.py:3120-3130` (`subtitle_mode_dropdown` choices)
- Test: `test_viral_chunks.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de Task 1.
- Produces: `VideoEditor._group_words_into_chunks(word_segments, max_words_per_chunk=4, target_chunk_duration=0.9) -> List[Dict[str, Any]]`.
  `burn_karaoke_subtitles(..., animation_mode="viral")` como valor válido nuevo.

**Nota de alcance:** el spec original describía "Viral" como un 5º *estilo*
visual. Durante esta implementación se determinó que encaja mejor como un
5º *modo de animación* (junto a static/karaoke/highlight/pop) porque lo que
lo distingue es el AGRUPAMIENTO de palabras en el tiempo, no la
tipografía/color — el usuario sigue eligiendo la tipografía/color con el
selector de "Estilo Visual" existente (Moderno/TikTok/Minimal/Clásico), y
además puede elegir "Viral" en el selector de animación para ese chunking.
Mismo resultado visual que describía el spec, mejor encaje en la
arquitectura ya existente (que ya separa "estilo" de "modo de animación").

- [ ] **Step 1: Escribir el script de verificación (falla primero)**

Crear `test_viral_chunks.py`:

```python
"""Verifica _group_words_into_chunks: agrupa 2-4 palabras conservando los
timestamps reales de cada palabra (no inventa tiempos nuevos)."""
import sys
sys.path.insert(0, ".")
from video_editor import VideoEditor

editor = VideoEditor.__new__(VideoEditor)  # no queremos verificar ffmpeg/GPU acá

word_segments = [
    {"start": 0.0, "end": 0.3, "text": "no", "parent_id": 0, "word_index": 0},
    {"start": 0.3, "end": 0.6, "text": "vas", "parent_id": 0, "word_index": 1},
    {"start": 0.6, "end": 0.9, "text": "a", "parent_id": 0, "word_index": 2},
    {"start": 0.9, "end": 1.3, "text": "creer", "parent_id": 0, "word_index": 3},
    {"start": 1.3, "end": 1.7, "text": "lo", "parent_id": 0, "word_index": 4},
    {"start": 1.7, "end": 2.1, "text": "que", "parent_id": 0, "word_index": 5},
    {"start": 2.1, "end": 2.6, "text": "paso", "parent_id": 0, "word_index": 6},
]

chunks = editor._group_words_into_chunks(word_segments, max_words_per_chunk=4, target_chunk_duration=0.9)

# No se pierde ninguna palabra ni se inventan timestamps
assert len(chunks) == len(word_segments), f"esperaba {len(word_segments)} palabras totales, salieron {len(chunks)}"
original_by_text = {(w["text"], w["start"], w["end"]) for w in word_segments}
chunk_by_text = {(w["text"], w["start"], w["end"]) for w in chunks}
assert original_by_text == chunk_by_text, "los timestamps de las palabras cambiaron durante el chunking"

# El primer chunk agrupa como maximo 4 palabras (tope max_words_per_chunk)
first_chunk_id = chunks[0]["parent_id"]
first_chunk_words = [w for w in chunks if w["parent_id"] == first_chunk_id]
assert 2 <= len(first_chunk_words) <= 4, f"el primer chunk tiene {len(first_chunk_words)} palabras, esperaba 2-4"

# Los chunks son secuenciales (parent_id 0, 1, 2...) y cada uno tiene
# word_index reiniciado en 0
parent_ids = sorted(set(w["parent_id"] for w in chunks))
assert parent_ids == list(range(len(parent_ids))), f"parent_ids no son secuenciales: {parent_ids}"

print(f"OK: {len(word_segments)} palabras agrupadas en {len(parent_ids)} chunks")
```

- [ ] **Step 2: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_viral_chunks.py`
Expected: `AttributeError: 'VideoEditor' object has no attribute '_group_words_into_chunks'`

- [ ] **Step 3: Implementar `_group_words_into_chunks`**

Insertar en `video_editor.py` inmediatamente antes de `def burn_karaoke_subtitles`
(línea 652):

```python
    def _group_words_into_chunks(
        self,
        word_segments: List[Dict[str, Any]],
        max_words_per_chunk: int = 4,
        target_chunk_duration: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Reagrupa word_segments (ya con timestamps reales) en chunks de 2-4
        palabras para el modo de animación "Viral" (estilo MrBeast/Hormozi/
        Submagic: pocas palabras grandes en pantalla, no la frase completa).
        No inventa timestamps — cada palabra conserva su start/end original;
        solo se reasignan parent_id/word_index para que cada chunk actúe
        como su propio "segmento padre" en burn_karaoke_subtitles.
        """
        if not word_segments:
            return []
        ordered = sorted(word_segments, key=lambda w: (w.get('parent_id', 0), w.get('word_index', 0)))
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for w in ordered:
            current.append(w)
            chunk_duration = current[-1]['end'] - current[0]['start']
            if len(current) >= max_words_per_chunk or chunk_duration >= target_chunk_duration:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        regrouped = []
        for chunk_id, words in enumerate(chunks):
            for word_index, w in enumerate(words):
                regrouped.append({
                    'start': w['start'],
                    'end': w['end'],
                    'text': w['text'],
                    'parent_id': chunk_id,
                    'word_index': word_index,
                })
        return regrouped
```

- [ ] **Step 4: Correr el script y confirmar que pasa**

Run: `venv/Scripts/python.exe test_viral_chunks.py`
Expected: `OK: 7 palabras agrupadas en N chunks` (sin AssertionError)

- [ ] **Step 5: Integrar el modo "viral" en `burn_karaoke_subtitles`**

En `video_editor.py`, dentro de `burn_karaoke_subtitles` (línea 652), hacer
3 cambios puntuales:

1. Justo después de `style = style or self.subtitle_style` (línea 678),
   agregar el chunking condicional:

```python
        style = style or self.subtitle_style

        if animation_mode == "viral":
            word_segments = self._group_words_into_chunks(word_segments)
```

2. En la línea 711, incluir "viral" entre los modos que atenúan la base
   (para que la palabra activa resalte contra el resto del chunk en gris):

```python
                base_color = 'gray' if animation_mode in ("karaoke", "highlight", "viral") else style.color
```

3. En la línea 753, incluir "viral" entre los modos con fontsize agrandado
   en la palabra activa (mismo "pop" visual que ya existe):

```python
                    highlight_fontsize = int(style.fontsize * 1.18) if animation_mode in ("pop", "viral") else style.fontsize
```

- [ ] **Step 6: Agregar "🔥 Viral" como opción en el dropdown de modo de subtítulos**

En `app.py`, `subtitle_mode_dropdown` (líneas 3120-3130) hoy es:

```python
                                                subtitle_mode_dropdown = gr.Dropdown(
                                                    choices=[
                                                        ("Estático", "static"),
                                                        ("Karaoke", "karaoke"),
                                                        ("Highlight", "highlight"),
                                                        ("Pop", "pop"),
                                                    ],
                                                    value="static",
                                                    label="",
                                                    show_label=False
                                                )
```

Reemplazar la lista `choices` por:

```python
                                                subtitle_mode_dropdown = gr.Dropdown(
                                                    choices=[
                                                        ("Estático", "static"),
                                                        ("Karaoke", "karaoke"),
                                                        ("Highlight", "highlight"),
                                                        ("Pop", "pop"),
                                                        ("🔥 Viral (chunks 2-4 palabras)", "viral"),
                                                    ],
                                                    value="static",
                                                    label="",
                                                    show_label=False
                                                )
```

`create_viral_clip` en `video_editor.py` ya rutea cualquier valor en
`("karaoke", "highlight", "pop")` hacia `burn_karaoke_subtitles` — hay que
agregar `"viral"` a esa tupla también. Ubicar la línea con
`grep -n 'if subtitle_mode in' video_editor.py` (está dentro del cuerpo de
`create_viral_clip`, después del bloque de compresión de pausas F4):

```python
            if subtitle_mode in ("karaoke", "highlight", "pop"):
```

y reemplazar por:

```python
            if subtitle_mode in ("karaoke", "highlight", "pop", "viral"):
```

- [ ] **Step 7: Verificación con render real**

Crear `test_viral_render.py`:

```python
"""Renderiza un clip real con animation_mode="viral" y confirma que el
video resultante existe y no cayo al fallback de ffmpeg drawtext (que no
soporta chunks, solo texto plano por segmento)."""
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO)
from video_editor import VideoEditor, SubtitleStyle

editor = VideoEditor()
src = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_cropped_fixed.mp4"
out = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\test_viral_render.mp4"

word_segments = [
    {"start": 0.0, "end": 0.3, "text": "no", "parent_id": 0, "word_index": 0},
    {"start": 0.3, "end": 0.6, "text": "vas", "parent_id": 0, "word_index": 1},
    {"start": 0.6, "end": 0.9, "text": "a", "parent_id": 0, "word_index": 2},
    {"start": 0.9, "end": 1.3, "text": "creer", "parent_id": 0, "word_index": 3},
    {"start": 1.3, "end": 1.7, "text": "lo", "parent_id": 0, "word_index": 4},
    {"start": 1.7, "end": 2.1, "text": "que", "parent_id": 0, "word_index": 5},
    {"start": 2.1, "end": 2.6, "text": "paso", "parent_id": 0, "word_index": 6},
]
style = SubtitleStyle(font="C:/Windows/Fonts/arialbd.ttf", fontsize=64, color="white",
                       stroke_color="black", stroke_width=3, position="center")

result = editor.burn_karaoke_subtitles(src, out, word_segments, style=style, animation_mode="viral")
print("RESULT:", result)
import os
assert os.path.exists(out), "el archivo de salida no se genero"
assert os.path.getsize(out) > 100_000, "el archivo de salida es sospechosamente chico"
print("OK: render viral generado")
```

Run: `venv/Scripts/python.exe test_viral_render.py`
Expected: termina con `OK: render viral generado` sin ningún
`WARNING - MoviePy karaoke falló` en el log (si aparece ese warning, el
render cayó al fallback y algo del chunking rompió MoviePy — diagnosticar
antes de seguir).

- [ ] **Step 8: Extraer un frame y confirmar visualmente el agrupamiento**

```bash
ffmpeg -y -ss 0.4 -i "D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\test_viral_render.mp4" -frames:v 1 -update 1 "D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\viral_frame_check.png"
```

Leer el PNG resultante con la herramienta Read de la sesión y confirmar
visualmente: se ve un grupo de 2-4 palabras juntas (no la frase de 7
palabras completa, no una sola palabra suelta), con una de ellas resaltada
más grande/brillante que el resto.

- [ ] **Step 9: Borrar scripts temporales y commitear**

```bash
rm test_viral_chunks.py test_viral_render.py
git add video_editor.py app.py
git commit -m "R2: modo de animacion Viral (chunks de 2-4 palabras)

Nuevo modo junto a static/karaoke/highlight/pop, reutilizando el mismo
render base+highlight que ya existe para karaoke — solo cambia como se
agrupan las palabras en el tiempo, no el resto del pipeline."
```

---

### Task 3: Zoom dinámico activado por defecto, intensidad más sutil

**Files:**
- Modify: `app.py:3211-3215` (checkbox `zoom_cues_checkbox`)
- Modify: `app.py` (llamada a `export_clips` en `export_btn.click`, para pasar
  un `zoom_factor` configurable — ver Step 3)
- Test: manual, sin script nuevo (cambio de un solo valor default + un
  parámetro ya existente, se verifica con el render que Task 2 ya probó
  que funciona)

**Interfaces:**
- Consumes: nada nuevo (usa `apply_zoom_cues`, ya arreglado en la auditoría
  de bugs de hoy).
- Produces: nada que otras tareas consuman.

- [ ] **Step 1: Cambiar el default del checkbox a activado**

En `app.py`, `zoom_cues_checkbox` (línea 3211):

```python
                                        zoom_cues_checkbox = gr.Checkbox(
                                            label="Zoom dinámico por energía (F2.3)",
                                            value=False,
                                            show_label=True
                                        )
```

Cambiar `value=False` a `value=True`:

```python
                                        zoom_cues_checkbox = gr.Checkbox(
                                            label="Zoom dinámico por energía (F2.3)",
                                            value=True,
                                            show_label=True
                                        )
```

- [ ] **Step 2: Bajar la intensidad default del zoom para verse sutil por defecto**

En `video_editor.py`, `apply_zoom_cues` tiene el default
`zoom_factor: float = 1.08`. Dejarlo en 1.08 (ya es razonablemente sutil —
confirmado visualmente durante la auditoría de bugs de hoy con
`zoom_factor=2.0` que se notaba fuerte). No cambiar el código de
`apply_zoom_cues`; el ajuste real es en cómo se invoca desde
`_export_single_clip` (`app.py`), que hoy no pasa `zoom_factor` explícito
así que ya usa el default 1.08 — no se requiere cambio de código acá, solo
confirmar que sigue así:

```bash
grep -n "apply_zoom_cues(current" app.py
```

Expected output: una sola línea, sin argumento `zoom_factor=` explícito
(usa el default de la función). Si aparece con un valor explícito distinto
de 1.08, ajustarlo a 1.08.

- [ ] **Step 3: Verificar en el servidor real que el checkbox arranca marcado**

Reiniciar el servidor (matar proceso en 7860, relanzar
`venv/Scripts/python.exe app.py`) y en el navegador (o con Playwright)
entrar a la pestaña Exportar — el checkbox "Zoom dinámico por energía
(F2.3)" debe aparecer YA marcado sin que el usuario lo toque.

- [ ] **Step 4: Commitear**

```bash
git add app.py
git commit -m "R3: activar zoom dinamico por defecto en exportacion

apply_zoom_cues ya fue arreglado en la auditoria de bugs de hoy (usaba
la variable 'n' en vez de 'on' en zoompan, nunca se habia aplicado). Con
el bug arreglado, se activa por defecto para que los clips tengan
movimiento constante sin que el usuario tenga que buscar el checkbox."
```

---

### Task 4: Barra de progreso quemada en el video

**Files:**
- Modify: `video_editor.py` (nuevo método `add_progress_bar_overlay`,
  insertado después de `apply_zoom_cues`, que termina en la línea 1477 según
  el estado actual del archivo — confirmar con `grep -n "def apply_audio_ducking" video_editor.py`
  antes de insertar, ya que Tasks 1-3 no tocan esta zona del archivo)
- Modify: `app.py` (nuevo checkbox en Exportar + threading del parámetro por
  `_export_single_clip`/`export_clips`/`export_btn.click`, siguiendo el
  mismo patrón que `compress_pauses`/`show_hook` ya usan)
- Test: `test_progress_bar.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: `VideoEditor.add_progress_bar_overlay(input_path: str, output_path: str, bar_color: str = "#7C5CFF", bar_height: int = 6) -> str`.
  Nuevo parámetro `show_progress_bar: bool = False` en `_export_single_clip`
  y `export_clips` (mismo patrón posicional que `show_hook`/`compress_pauses`).

**Verificado antes de escribir este plan:** el filtro
`drawbox=x=0:y=ih-H:w='iw*(t/DURATION)':h=H:color=0xRRGGBB@1.0:t=fill`
funciona correctamente con ffmpeg 8.0 — el parámetro `t=fill` (thickness)
no choca con la variable de tiempo `t` usada dentro de la expresión `w=`.
Confirmado con un render real de 3s y frames extraídos al inicio y al final
mostrando la barra creciendo de 0 a ancho completo.

- [ ] **Step 1: Escribir el script de verificación (falla primero)**

Crear `test_progress_bar.py`:

```python
"""Renderiza un clip corto con barra de progreso y confirma que crece con
el tiempo comparando el ancho de pixeles no-negros en 2 frames distintos."""
import sys, subprocess
sys.path.insert(0, ".")
from video_editor import VideoEditor

editor = VideoEditor()
src = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_cropped_fixed.mp4"
out = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\test_progress_bar_out.mp4"

result = editor.add_progress_bar_overlay(src, out, bar_color="#7C5CFF", bar_height=6)
print("RESULT:", result)

import os
assert os.path.exists(out), "el archivo de salida no se genero"

# Extraer frame temprano y tardio, medir cuantos pixeles de la ultima fila
# NO son negros (la barra ocupa la fila inferior)
def count_nonblack_pixels_in_bottom_row(video_path, timestamp):
    frame_path = video_path.replace(".mp4", f"_frame_{timestamp}.png")
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
        "-frames:v", "1", "-update", "1", frame_path
    ], capture_output=True, timeout=15)
    from PIL import Image
    img = Image.open(frame_path)
    w, h = img.size
    row = img.crop((0, h - 3, w, h - 2)).convert("RGB")
    pixels = list(row.getdata())
    nonblack = sum(1 for p in pixels if sum(p) > 30)
    os.unlink(frame_path)
    return nonblack, w

early_count, width = count_nonblack_pixels_in_bottom_row(out, 0.3)
late_count, _ = count_nonblack_pixels_in_bottom_row(out, 2.7)
print(f"Ancho total: {width}px | pixeles no-negros a 0.3s: {early_count} | a 2.7s: {late_count}")

assert late_count > early_count, (
    f"la barra no crecio: {early_count}px a 0.3s vs {late_count}px a 2.7s"
)
assert late_count > width * 0.7, f"a 2.7s (cerca del final) la barra deberia cubrir la mayoria del ancho, cubre {late_count}/{width}"
print("OK: barra de progreso crece con el tiempo")
```

- [ ] **Step 2: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_progress_bar.py`
Expected: `AttributeError: 'VideoEditor' object has no attribute 'add_progress_bar_overlay'`

- [ ] **Step 3: Implementar `add_progress_bar_overlay`**

Ubicar el final de `apply_audio_ducking` en `video_editor.py` con
`grep -n "def add_branding_overlay" video_editor.py` e insertar el método
nuevo inmediatamente ANTES de esa línea (entre `apply_audio_ducking` y
`add_branding_overlay`):

```python
    # ------------------------------------------------------------------
    # R4 — Barra de progreso quemada (retención: "ya casi termina")
    # ------------------------------------------------------------------

    def add_progress_bar_overlay(
        self,
        input_path: str,
        output_path: str,
        bar_color: str = "#7C5CFF",
        bar_height: int = 6,
    ) -> str:
        """
        Dibuja una barra de progreso delgada en el borde inferior del video,
        que crece de 0 a ancho completo a medida que avanza el clip. Trigger
        de retención: "ya casi termina, me quedo a ver el final".
        """
        try:
            info = self.get_video_info(input_path)
            duration = info.get('duration', 0)
            if duration <= 0:
                import shutil
                shutil.copy2(input_path, output_path)
                return output_path

            color_hex = bar_color.lstrip('#')
            ffmpeg_color = f"0x{color_hex}"
            vf = (
                f"drawbox=x=0:y=ih-{bar_height}:w='iw*(t/{duration:.3f})'"
                f":h={bar_height}:color={ffmpeg_color}@1.0:t=fill"
            )
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    vf=vf,
                    vcodec=self.config.codec,
                    acodec="copy",
                    preset=self.config.preset,
                    pix_fmt="yuv420p",
                    loglevel="warning",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"📊 Barra de progreso agregada ({duration:.1f}s)")
            return output_path
        except Exception as e:
            logger.warning(f"Barra de progreso falló ({e}), usando original")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
```

- [ ] **Step 4: Correr el script y confirmar que pasa**

Run: `venv/Scripts/python.exe test_progress_bar.py`
Expected: `OK: barra de progreso crece con el tiempo`

- [ ] **Step 5: Integrar como paso opcional en `_export_single_clip`**

En `app.py`, `_export_single_clip` (línea 928), agregar el parámetro
`show_progress_bar: bool = False` a la firma (después de `show_hook: bool = True,`):

```python
        show_hook: bool = True,
        show_progress_bar: bool = False,
    ) -> Tuple[int, str, bool]:
```

Y agregar el paso nuevo justo después del "Step 5: branding overlay"
(después de la línea que hoy es `logger.warning(f"Branding overlay omitido en clip {i+1}: {_be}")`
y antes de `# Move final result to output_file`):

```python
            # Step 6: progress bar overlay (R4)
            if show_progress_bar:
                try:
                    bar_out = str(temp_base) + "_bar.mp4"
                    current = self.editor.add_progress_bar_overlay(current, bar_out, bar_color=brand_color)
                except Exception as _pbe:
                    logger.warning(f"Barra de progreso omitida en clip {i+1}: {_pbe}")
```

Y agregar `"_bar.mp4"` a la lista de sufijos de limpieza de temporales
(línea con `for suffix in ["_base.mp4", "_zoomed.mp4", "_ducked.mp4", "_graded.mp4", "_branded.mp4"]:`):

```python
            for suffix in ["_base.mp4", "_zoomed.mp4", "_ducked.mp4", "_graded.mp4", "_branded.mp4", "_bar.mp4"]:
```

- [ ] **Step 6: Hacer lo mismo en `export_clips` y agregar el checkbox de UI**

En `app.py`, `export_clips` (línea 1208), agregar el mismo parámetro a la
firma (después de `show_hook: bool = True,`):

```python
        show_hook: bool = True,
        show_progress_bar: bool = False,
    ) -> Tuple[str, List, List[str], str]:
```

Hay 2 llamadas a `self._export_single_clip` dentro de `export_clips` que
pasan `show_hook` como último argumento posicional — una en el bloque
paralelo (`executor.submit`), otra en el bloque secuencial. La del bloque
paralelo hoy es:

```python
                        future = executor.submit(
                            self._export_single_clip,
                            (i, clip),
                            style_name,
                            track_faces,
                            subtitle_mode,
                            target_width,
                            target_height,
                            enable_mood_grade,
                            enable_ducking,
                            brand_name,
                            brand_color,
                            enable_zoom_cues,
                            compress_pauses,
                            show_hook,
                        )
```

Reemplazar el `show_hook,` final por:

```python
                            show_hook,
                            show_progress_bar,
                        )
```

Y la del bloque secuencial hoy es:

```python
                    idx, result, success = self._export_single_clip(
                        (i, clip_state), style_name, track_faces, subtitle_mode,
                        target_width, target_height,
                        enable_mood_grade, enable_ducking,
                        brand_name, brand_color,
                        enable_zoom_cues, compress_pauses,
                        show_hook,
                    )
```

Reemplazar el `show_hook,` final por:

```python
                        show_hook,
                        show_progress_bar,
                    )
```

Agregar el checkbox de UI: en `app.py`, justo después de donde está
`show_hook_checkbox = gr.Checkbox(...)` (línea 3144), agregar un checkbox
hermano dentro del mismo bloque `ai-tool`:

```python
                                            with gr.Row(elem_classes=["ai-tool"]):
                                                gr.HTML("""
                                                <div class="ai-tool-left">
                                                    <div class="ai-tool-icon">
                                                        <span class="material-symbols-outlined">linear_scale</span>
                                                    </div>
                                                    <div class="ai-tool-info">
                                                        <div class="ai-tool-name">Barra de Progreso</div>
                                                        <div class="ai-tool-desc">Línea que avanza con el video — retiene hasta el final</div>
                                                    </div>
                                                </div>
                                                """)
                                                progress_bar_checkbox = gr.Checkbox(
                                                    label="",
                                                    value=True,
                                                    show_label=False,
                                                )
```

Finalmente, el wiring de `export_btn.click` hoy es:

```python
            export_btn.click(
                fn=lambda style, sub_mode, face_track, platform, srt, vtt, brand, brand_color, mood_grade, ducking, zoom, pauses, hook, prog=gr.Progress(): self.export_clips(
                    style, prog, parallel=True, track_faces=face_track, subtitle_mode=sub_mode,
                    platform=platform, export_srt=srt, export_vtt=vtt,
                    brand_name=brand, brand_color=brand_color,
                    enable_mood_grade=mood_grade, enable_ducking=ducking,
                    enable_zoom_cues=zoom, compress_pauses=pauses, show_hook=hook
                ),
                inputs=[style_dropdown, subtitle_mode_dropdown, face_tracking_checkbox, platform_preset,
                        export_srt_checkbox, export_vtt_checkbox, brand_name_input, brand_color_input,
                        mood_grade_checkbox, audio_ducking_checkbox, zoom_cues_checkbox, compress_pauses_checkbox,
                        show_hook_checkbox],
                outputs=[export_status, output_gallery, output_files, captions_output]
            )
```

Reemplazar por (agrega `pbar` al lambda, `show_progress_bar=pbar` a la
llamada, y `progress_bar_checkbox` a `inputs`):

```python
            export_btn.click(
                fn=lambda style, sub_mode, face_track, platform, srt, vtt, brand, brand_color, mood_grade, ducking, zoom, pauses, hook, pbar, prog=gr.Progress(): self.export_clips(
                    style, prog, parallel=True, track_faces=face_track, subtitle_mode=sub_mode,
                    platform=platform, export_srt=srt, export_vtt=vtt,
                    brand_name=brand, brand_color=brand_color,
                    enable_mood_grade=mood_grade, enable_ducking=ducking,
                    enable_zoom_cues=zoom, compress_pauses=pauses, show_hook=hook,
                    show_progress_bar=pbar
                ),
                inputs=[style_dropdown, subtitle_mode_dropdown, face_tracking_checkbox, platform_preset,
                        export_srt_checkbox, export_vtt_checkbox, brand_name_input, brand_color_input,
                        mood_grade_checkbox, audio_ducking_checkbox, zoom_cues_checkbox, compress_pauses_checkbox,
                        show_hook_checkbox, progress_bar_checkbox],
                outputs=[export_status, output_gallery, output_files, captions_output]
            )
```

- [ ] **Step 7: Reiniciar el servidor y confirmar visualmente en la UI**

Reiniciar el servidor, entrar a Exportar, confirmar que aparece el nuevo
checkbox "Barra de Progreso" marcado por defecto, junto a los demás de
"Mejoras IA".

- [ ] **Step 8: Borrar script temporal y commitear**

```bash
rm test_progress_bar.py
git add video_editor.py app.py
git commit -m "R4: barra de progreso quemada en el video (retencion)

Filtro ffmpeg drawbox con ancho variable segun 't' (tiempo transcurrido),
verificado que 't=fill' (thickness) no choca con la variable de tiempo
usada dentro de la expresion de ancho."
```

---

### Task 5: Corte final seco (sin cola de silencio)

**Files:**
- Modify: `video_editor.py:1279-1293` (cálculo de `adjusted_segments` dentro
  de `create_viral_clip`)
- Test: `test_trim_silence.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: nada que otras tareas consuman (cambio interno de
  `create_viral_clip`, mismo comportamiento externo/firma).

- [ ] **Step 1: Escribir el script de verificación (falla primero)**

Crear `test_trim_silence.py`:

```python
"""Verifica que create_viral_clip recorta el final del clip al ultimo
segmento con texto + margen, en vez de usar el end_time completo cuando
hay cola de silencio. Usa un video de prueba de 45s y pide un clip de
0s a 10s, pero con segments que terminan en el segundo 6 — el video
resultante deberia durar ~6.3s, no 10s."""
import sys, subprocess, json
sys.path.insert(0, ".")
from video_editor import VideoEditor

editor = VideoEditor()
src = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_cropped_fixed.mp4"
out = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\test_trim_silence_out.mp4"

segments = [
    {"start": 0.0, "end": 2.0, "text": "primer segmento"},
    {"start": 3.0, "end": 6.0, "text": "ultimo segmento con texto"},
    # nada de texto entre 6.0s y 10.0s: cola de silencio
]

editor.create_viral_clip(
    src, out, start_time=0.0, end_time=10.0, segments=segments,
    add_subtitles=True, subtitle_mode="static", show_hook=False,
)

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", out],
    capture_output=True, text=True, timeout=15,
)
result_duration = float(probe.stdout.strip())
print(f"Duracion del clip resultante: {result_duration:.2f}s (pedido: 10.0s, ultimo texto termina en 6.0s)")

assert result_duration < 8.0, (
    f"el clip dura {result_duration:.2f}s, esperaba que se recorte cerca de "
    "6.3s (ultimo texto + margen), no que use el end_time completo de 10s"
)
assert result_duration > 6.0, (
    f"el clip dura {result_duration:.2f}s, es MENOS que el ultimo texto (6.0s) "
    "- se recorto de mas y probablemente corta la ultima palabra"
)
print("OK: el clip se recorto cerca del ultimo texto, no al end_time completo")
```

- [ ] **Step 2: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_trim_silence.py`
Expected: falla con un `AssertionError` de la primera aserción (duración
≈10s, no <8s) — esto confirma el comportamiento actual (usa el end_time
completo) antes de arreglarlo.

- [ ] **Step 3: Implementar el recorte en `create_viral_clip`**

En `video_editor.py`, dentro de `create_viral_clip`, el bloque completo
desde el `try:` hasta el final del cálculo de `adjusted_segments` hoy es
(líneas 1264-1293):

```python
        try:
            # Paso 1: Crop a 9:16 con FFmpeg (rápido)
            logger.info(f"Creando clip viral: {start_time:.1f}s - {end_time:.1f}s")

            self.crop_to_vertical_ffmpeg(
                str(input_video),
                str(temp_crop),
                start_time,
                end_time,
                target_width or self.config.width,
                target_height or self.config.height,
                track_faces=track_faces
            )

            # Paso 2: Ajustar timestamps de segmentos al clip recortado
            adjusted_segments = []
            for seg in segments:
                adj_start = seg['start'] - start_time
                adj_end = seg['end'] - start_time

                # Solo incluir segmentos dentro del clip
                if adj_end > 0 and adj_start < (end_time - start_time):
                    adj_start = max(0, adj_start)
                    adj_end = min(end_time - start_time, adj_end)

                    adjusted_segments.append({
                        'start': adj_start,
                        'end': adj_end,
                        'text': seg['text']
                    })
```

`crop_to_vertical_ffmpeg` corre PRIMERO, usando el `end_time` recibido —
por eso el recorte de cola de silencio (R5) tiene que decidirse ANTES de
esa llamada, no después: si se recalcula `end_time` después del crop, el
video ya fue cortado con el `end_time` viejo y el recorte no tiene efecto
real. Reemplazar el bloque completo de arriba por este (mismo `try:`,
orden invertido: primero se ajustan y recortan los segments, DESPUÉS se
cortea el video con el `end_time` ya definitivo):

```python
        try:
            # Paso 1: Ajustar timestamps de segmentos al clip pedido (todavía
            # sin cortar el video) — hace falta esto ANTES de cortar con
            # ffmpeg para poder recortar la cola de silencio (R5) sobre el
            # end_time real que se le va a pedir a ffmpeg.
            adjusted_segments = []
            for seg in segments:
                adj_start = seg['start'] - start_time
                adj_end = seg['end'] - start_time

                # Solo incluir segmentos dentro del clip
                if adj_end > 0 and adj_start < (end_time - start_time):
                    adj_start = max(0, adj_start)
                    adj_end = min(end_time - start_time, adj_end)

                    adjusted_segments.append({
                        'start': adj_start,
                        'end': adj_end,
                        'text': seg['text']
                    })

            # Paso 1.5 (R5): recortar la cola de silencio del final — un clip
            # que termina 2-3s después de la última palabra invita al scroll
            # antes de que termine; recortamos al último texto + margen fijo,
            # ANTES de cortar el video con ffmpeg para que el recorte sea real.
            TRAILING_SILENCE_MARGIN = 0.3
            clip_duration_requested = end_time - start_time
            if adjusted_segments:
                last_text_end = max(seg['end'] for seg in adjusted_segments)
                trimmed_duration = min(clip_duration_requested, last_text_end + TRAILING_SILENCE_MARGIN)
                if trimmed_duration < clip_duration_requested - 0.5:
                    logger.info(
                        f"✂️ Recorte de cola de silencio: {clip_duration_requested:.1f}s -> {trimmed_duration:.1f}s"
                    )
                    end_time = start_time + trimmed_duration
                    adjusted_segments = [
                        seg for seg in adjusted_segments if seg['start'] < trimmed_duration
                    ]
                    for seg in adjusted_segments:
                        seg['end'] = min(seg['end'], trimmed_duration)

            # Paso 2: Crop a 9:16 con FFmpeg (rápido) — usa el end_time ya
            # recortado si el paso 1.5 aplicó un recorte de cola de silencio.
            logger.info(f"Creando clip viral: {start_time:.1f}s - {end_time:.1f}s")

            self.crop_to_vertical_ffmpeg(
                str(input_video),
                str(temp_crop),
                start_time,
                end_time,
                target_width or self.config.width,
                target_height or self.config.height,
                track_faces=track_faces
            )
```

El resto de la función (Paso 2.5 en adelante: compresión de pausas F4,
burn de subtítulos, etc.) queda exactamente igual — sigue leyendo
`adjusted_segments` y `end_time` de estas mismas variables, que ahora
llegan ya recortadas cuando corresponde.

- [ ] **Step 4: Correr el script y confirmar que pasa**

Run: `venv/Scripts/python.exe test_trim_silence.py`
Expected: `OK: el clip se recorto cerca del ultimo texto, no al end_time completo`

- [ ] **Step 5: Verificar que F4 (compress_pauses) sigue funcionando junto con este cambio**

Correr el test de compresión de pausas ya usado en la sesión de auditoría
(recrear si se borró):

```python
import sys
sys.path.insert(0, ".")
from video_editor import VideoEditor, SubtitleStyle

editor = VideoEditor()
src = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_cropped_fixed.mp4"
out = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\test_trim_plus_f4.mp4"
segments = [
    {"start": 0.0, "end": 2.0, "text": "primer segmento"},
    {"start": 8.0, "end": 10.0, "text": "segundo tras pausa larga"},
]
editor.create_viral_clip(
    src, out, start_time=0.0, end_time=15.0, segments=segments,
    add_subtitles=True, subtitle_mode="static", show_hook=False,
    compress_pauses=True,
)
import os
assert os.path.exists(out), "no se genero el archivo con F4 + recorte de cola activos a la vez"
print("OK: F4 + recorte de cola conviven sin romperse")
```

Run: `venv/Scripts/python.exe test_f4_plus_trim.py`
Expected: `OK: F4 + recorte de cola conviven sin romperse`

- [ ] **Step 6: Borrar scripts temporales y commitear**

```bash
rm test_trim_silence.py test_f4_plus_trim.py
git add video_editor.py
git commit -m "R5: recortar cola de silencio al final del clip

Un clip que termina 2-3s despues de la ultima palabra invita al scroll
antes de que termine — se recorta al ultimo texto + 0.3s de margen en
vez de usar el end_time completo pedido."
```

---

### Task 6: Keyword highlight en el hook

**Files:**
- Modify: `llm_analyzer.py` (dataclass `ViralClip`, ambos prompts, ambos
  sitios de construcción de `ViralClip`)
- Modify: `state_manager.py` (dataclass `ClipState`, `create_project`,
  `save_state`, `load_state`)
- Modify: `app.py:928-976` (`_export_single_clip`, pasar `hook_keywords` a
  `create_viral_clip`)
- Modify: `video_editor.py:444-478` (`_build_hook_clip`)
- Test: `test_hook_keywords.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: campo nuevo `hook_keywords: str = ""` en `ViralClip` y
  `ClipState` (una sola frase corta a resaltar dentro del hook, no una
  lista — más simple de aplicar con mayúsculas que una lista de palabras
  sueltas que podrían no ser contiguas).

**Nota de alcance (resuelta durante esta planificación, no en ejecución):**
el spec original sugería resaltar la keyword en OTRO COLOR dentro del hook.
Se investigó la implementación con MoviePy y requeriría posicionar 2-3
`TextClip` de ancho variable uno al lado del otro (sin wrapping automático
de `method='caption'`), lo cual es fràgil cuando el hook es largo y
necesita wrap a 2 líneas. Se opta por una técnica más simple y robusta:
la keyword se renderiza en MAYÚSCULAS dentro del mismo `TextClip` (un solo
clip, mismo `method='caption'`, cero cambios de layout) — técnica de
énfasis igual de reconocible visualmente, sin el riesgo de layout roto.

- [ ] **Step 1: Escribir el script de verificación (falla primero)**

Crear `test_hook_keywords.py`:

```python
"""Verifica que ViralClip acepta hook_keywords y que _build_hook_clip
efectivamente pone en mayusculas la keyword dentro del texto del hook
antes de renderizarlo."""
import sys
sys.path.insert(0, ".")
from llm_analyzer import ViralClip
from video_editor import VideoEditor

clip = ViralClip(
    start=0.0, end=10.0, virality_score=8.5, reason="test",
    hook="no vas a creer lo que paso", hook_keywords="no vas a creer",
)
assert clip.hook_keywords == "no vas a creer"
assert "hook_keywords" in clip.to_dict()

editor = VideoEditor.__new__(VideoEditor)
resultado = editor._apply_keyword_emphasis("no vas a creer lo que paso", "no vas a creer")
assert resultado == "NO VAS A CREER lo que paso", f"resultado inesperado: {resultado!r}"

# Si la keyword no aparece literal en el hook (Gemini no siempre es exacto),
# no debe romper — debe devolver el hook sin cambios.
resultado_sin_match = editor._apply_keyword_emphasis("un hook cualquiera", "frase que no esta")
assert resultado_sin_match == "un hook cualquiera"

print("OK: hook_keywords + _apply_keyword_emphasis funcionan")
```

- [ ] **Step 2: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_hook_keywords.py`
Expected: `TypeError: __init__() got an unexpected keyword argument 'hook_keywords'`

- [ ] **Step 3: Agregar el campo a `ViralClip` en `llm_analyzer.py`**

Localizar la dataclass `ViralClip` (`grep -n "hashtags: List\[str\] = field" llm_analyzer.py`)
y agregar el campo nuevo justo después:

```python
    hashtags: List[str] = field(default_factory=list)
    hook_keywords: str = ""
```

En el método `to_dict()` de la misma clase, agregar la entrada
correspondiente junto a `'hashtags': self.hashtags,`:

```python
            'hashtags': self.hashtags,
            'hook_keywords': self.hook_keywords,
```

- [ ] **Step 4: Agregar `hook_keywords` al prompt de Gemini (ambos prompts)**

En `_build_prompt` (el prompt de solo-texto), buscar el bloque
`"hashtags": [...]` dentro del `### FORMATO DE RESPUESTA` (línea ~395-398)
y agregar la línea nueva justo después de `"hashtags"`:

```python
      "hashtags": ["#storytime", "#miedo", "#casaembrujada", "#viral", "#fyp"],
      "hook_keywords": "no vas a creer"
```

Y en las instrucciones finales del mismo prompt (buscar
`"- \`hashtags\`: 4-6 hashtags"`), agregar una línea nueva justo después:

```python
- `hook_keywords`: 2-4 palabras CONSECUTIVAS tomadas literalmente del texto
  de `hook` que deberían resaltarse visualmente (la parte más impactante).
  Deben ser una subcadena EXACTA de `hook`, no una paráfrasis.
```

El mismo agregado va también en `analyze_with_frames` (el prompt
multimodal), que tiene su propio bloque `### FORMATO JSON` y su propia
lista de instrucciones finales, independientes de `_build_prompt`. Esa
sección hoy es (líneas 738-753):

```python
      "reason": "explicación detallada considerando elementos visuales y expresiones faciales",
      "emotional_start": "calm|excited|frustrated|suspenseful",
      "tension_point": "descripción del momento de máxima emoción visible",
      "payoff": "descripción del payoff/resolución",
      "edit_recipe": "instrucciones específicas basadas en la expresión facial/gesto visible",
      "hashtags": ["#tema-especifico-1", "#tema-especifico-2", "#fyp", "#viral", "#parati"]
    }}
  ]
}}

- virality_score = hook_score*0.30 + engagement_score*0.25 + emotional_arc_score*0.20 + pacing_score*0.15 + value_score*0.10
- Scores 1.0-10.0
- Clips NO solapados (min 3s entre clips)
- Hooks genéricos = descartar
- hashtags: 4-6 hashtags sobre el TEMA real del clip (no sobre por qué es viral), minúsculas,
  sin espacios ni acentos, mezclando nicho + alcance genérico (#fyp, #viral, #parati)

Identifica los {num_clips} mejores clips:"""
```

Reemplazar por:

```python
      "reason": "explicación detallada considerando elementos visuales y expresiones faciales",
      "emotional_start": "calm|excited|frustrated|suspenseful",
      "tension_point": "descripción del momento de máxima emoción visible",
      "payoff": "descripción del payoff/resolución",
      "edit_recipe": "instrucciones específicas basadas en la expresión facial/gesto visible",
      "hashtags": ["#tema-especifico-1", "#tema-especifico-2", "#fyp", "#viral", "#parati"],
      "hook_keywords": "no vas a creer"
    }}
  ]
}}

- virality_score = hook_score*0.30 + engagement_score*0.25 + emotional_arc_score*0.20 + pacing_score*0.15 + value_score*0.10
- Scores 1.0-10.0
- Clips NO solapados (min 3s entre clips)
- Hooks genéricos = descartar
- hashtags: 4-6 hashtags sobre el TEMA real del clip (no sobre por qué es viral), minúsculas,
  sin espacios ni acentos, mezclando nicho + alcance genérico (#fyp, #viral, #parati)
- hook_keywords: 2-4 palabras CONSECUTIVAS tomadas literalmente del texto de
  `hook` que deberían resaltarse visualmente. Deben ser una subcadena EXACTA
  de `hook`, no una paráfrasis.

Identifica los {num_clips} mejores clips:"""
```

- [ ] **Step 5: Leer `hook_keywords` en ambos sitios de construcción de `ViralClip`**

En `analyze_transcription`, donde se construye cada `ViralClip` (buscar
`hashtags=_parse_hashtags(clip.get('hashtags')),`), agregar la línea:

```python
                            hashtags=_parse_hashtags(clip.get('hashtags')),
                            hook_keywords=str(clip.get('hook_keywords', '')),
```

La línea `hashtags=_parse_hashtags(clip.get('hashtags')),` aparece dos
veces en `llm_analyzer.py`: una en `analyze_transcription` (línea 591,
editada arriba) y otra idéntica en `analyze_with_frames` (línea 810).
Aplicar el mismo agregado (`hook_keywords=str(clip.get('hook_keywords', '')),`
justo después) en la segunda ocurrencia, línea 810 — es el mismo texto de
línea y el mismo agregado en ambos sitios, la única diferencia es la
función que los contiene.

- [ ] **Step 6: Propagar el campo por `state_manager.py`**

En `ClipState` (dataclass), agregar después de `hashtags: List[str] = None`:

```python
    hashtags: List[str] = None
    hook_keywords: str = ""
```

En `create_project`, donde se construye cada `ClipState` (buscar
`hashtags=list(clip_data.get('hashtags', []) or []),`), agregar:

```python
                hashtags=list(clip_data.get('hashtags', []) or []),
                hook_keywords=str(clip_data.get('hook_keywords', '')),
```

En `save_state`, dentro del dict que arma cada clip para el JSON (buscar
`'hashtags': c.hashtags,`), agregar:

```python
                    'hashtags': c.hashtags,
                    'hook_keywords': c.hook_keywords,
```

En `load_state`, donde se reconstruye cada `ClipState` desde el JSON
(buscar `hashtags=list(clip_data.get('hashtags', []) or []),`), agregar:

```python
                    hashtags=list(clip_data.get('hashtags', []) or []),
                    hook_keywords=str(clip_data.get('hook_keywords', '')),
```

También en `app.py`, dentro de `analyze_video`, donde se arma cada entrada
de `clips_data` con los campos del clip (buscar `'hashtags': clip.hashtags,`
dentro del loop que construye `clips_data.append({...})`), agregar:

```python
                    'hashtags': clip.hashtags,
                    'hook_keywords': clip.hook_keywords,
```

- [ ] **Step 7: Implementar `_apply_keyword_emphasis` y usarlo en `_build_hook_clip`**

En `video_editor.py`, agregar el método nuevo justo antes de `_build_hook_clip`
(línea 444):

```python
    @staticmethod
    def _apply_keyword_emphasis(hook_text: str, hook_keywords: str) -> str:
        """
        Pone en MAYÚSCULAS la subcadena `hook_keywords` dentro de `hook_text`,
        preservando el resto del texto tal cual. Si `hook_keywords` no
        aparece literal dentro de `hook_text` (Gemini no siempre cita exacto),
        devuelve `hook_text` sin cambios — no rompe el render por esto.
        """
        if not hook_keywords or not hook_keywords.strip():
            return hook_text
        keywords = hook_keywords.strip()
        idx = hook_text.lower().find(keywords.lower())
        if idx == -1:
            return hook_text
        return hook_text[:idx] + keywords.upper() + hook_text[idx + len(keywords):]
```

En `_build_hook_clip`, agregar el parámetro `hook_keywords: str = ""` a la
firma:

```python
    def _build_hook_clip(
        self,
        hook_text: str,
        style: SubtitleStyle,
        video_w: int,
        video_h: int,
        duration: float = 2.2,
        hook_keywords: str = "",
    ) -> List[Any]:
```

Y, dentro del cuerpo, justo después de
`if not hook_text or not hook_text.strip(): return []`, aplicar el énfasis:

```python
        if not hook_text or not hook_text.strip():
            return []
        hook_text = self._apply_keyword_emphasis(hook_text.strip(), hook_keywords)
```

- [ ] **Step 8: Enhebrar `hook_keywords` desde `create_viral_clip` hasta `_build_hook_clip`**

`create_viral_clip` ya recibe `hook_text` y lo pasa a
`burn_subtitles_moviepy`/`burn_karaoke_subtitles`, que a su vez llaman a
`_build_hook_clip`. Hay que agregar `hook_keywords` a las 3 firmas y
enhebrarlo hasta la llamada final.

En `burn_subtitles_moviepy`, la firma hoy es:

```python
    def burn_subtitles_moviepy(
        self,
        video_path: str,
        output_path: str,
        segments: List[Dict[str, Any]],
        style: Optional[SubtitleStyle] = None,
        hook_text: Optional[str] = None,
        hook_duration: float = 2.2,
    ) -> str:
```

Agregar `hook_keywords: str = "",` al final, antes del `) -> str:`:

```python
    def burn_subtitles_moviepy(
        self,
        video_path: str,
        output_path: str,
        segments: List[Dict[str, Any]],
        style: Optional[SubtitleStyle] = None,
        hook_text: Optional[str] = None,
        hook_duration: float = 2.2,
        hook_keywords: str = "",
    ) -> str:
```

En `burn_karaoke_subtitles`, la firma hoy es:

```python
    def burn_karaoke_subtitles(
        self,
        video_path: str,
        output_path: str,
        word_segments: List[Dict[str, Any]],
        style: Optional[SubtitleStyle] = None,
        animation_mode: str = "karaoke",
        hook_text: Optional[str] = None,
        hook_duration: float = 2.2,
    ) -> str:
```

Agregar el mismo `hook_keywords: str = "",` al final:

```python
    def burn_karaoke_subtitles(
        self,
        video_path: str,
        output_path: str,
        word_segments: List[Dict[str, Any]],
        style: Optional[SubtitleStyle] = None,
        animation_mode: str = "karaoke",
        hook_text: Optional[str] = None,
        hook_duration: float = 2.2,
        hook_keywords: str = "",
    ) -> str:
```

En ambas funciones, la línea que arma el overlay del hook hoy es idéntica:

```python
            subtitle_clips.extend(self._build_hook_clip(hook_text, style, video.w, video.h, hook_duration))
```

Reemplazar por (en las dos funciones):

```python
            subtitle_clips.extend(self._build_hook_clip(hook_text, style, video.w, video.h, hook_duration, hook_keywords))
```

En `create_viral_clip`, la firma hoy termina en:

```python
        hook_text: Optional[str] = None,
        show_hook: bool = True,
        hook_duration: float = 2.2,
    ) -> str:
```

Agregar `hook_keywords: str = "",` antes de `) -> str:`:

```python
        hook_text: Optional[str] = None,
        show_hook: bool = True,
        hook_duration: float = 2.2,
        hook_keywords: str = "",
    ) -> str:
```

Dentro de `create_viral_clip`, las 2 llamadas a los burners hoy son:

```python
                self.burn_karaoke_subtitles(
                    str(temp_crop),
                    str(output_path),
                    word_segments,
                    style=style,
                    animation_mode=subtitle_mode,
                    hook_text=hook_text if show_hook else None,
                    hook_duration=hook_duration,
                )
```

y

```python
                self.burn_subtitles_moviepy(
                    str(temp_crop),
                    str(output_path),
                    adjusted_segments,
                    style=style,
                    hook_text=hook_text if show_hook else None,
                    hook_duration=hook_duration,
                )
```

Agregar `hook_keywords=hook_keywords,` como última línea en ambas llamadas:

```python
                self.burn_karaoke_subtitles(
                    str(temp_crop),
                    str(output_path),
                    word_segments,
                    style=style,
                    animation_mode=subtitle_mode,
                    hook_text=hook_text if show_hook else None,
                    hook_duration=hook_duration,
                    hook_keywords=hook_keywords,
                )
```

```python
                self.burn_subtitles_moviepy(
                    str(temp_crop),
                    str(output_path),
                    adjusted_segments,
                    style=style,
                    hook_text=hook_text if show_hook else None,
                    hook_duration=hook_duration,
                    hook_keywords=hook_keywords,
                )
```

Finalmente, en `app.py`, `_export_single_clip`, la llamada a
`self.editor.create_viral_clip(...)` incluye hoy `hook_text=clip_state.hook,`
y `show_hook=show_hook,` como sus últimos dos argumentos con nombre.
Agregar un tercero justo después:

```python
                hook_text=clip_state.hook,
                show_hook=show_hook,
                hook_keywords=getattr(clip_state, 'hook_keywords', ''),
```

- [ ] **Step 9: Correr el script de verificación y confirmar que pasa**

Run: `venv/Scripts/python.exe test_hook_keywords.py`
Expected: `OK: hook_keywords + _apply_keyword_emphasis funcionan`

- [ ] **Step 10: Verificación con render real**

Crear `test_hook_keyword_render.py`:

```python
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO)
from video_editor import VideoEditor, SubtitleStyle

editor = VideoEditor()
src = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\silence_test\test_cropped_fixed.mp4"
out = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\test_hook_keyword_render.mp4"
segments = [{"start": 0.0, "end": 2.0, "text": "un segmento cualquiera"}]
style = SubtitleStyle(font="C:/Windows/Fonts/arialbd.ttf", fontsize=64, color="white",
                       stroke_color="black", stroke_width=3, position="bottom", margin_vertical=220)

editor.burn_subtitles_moviepy(
    src, out, segments, style=style,
    hook_text="no vas a creer lo que encontre",
    hook_keywords="no vas a creer",
)
print("RESULT:", out)
```

Run: `venv/Scripts/python.exe test_hook_keyword_render.py`, luego extraer un
frame a 0.5s con ffmpeg y leerlo con la herramienta Read — confirmar
visualmente que el texto muestra "NO VAS A CREER lo que encontre" (la
keyword en mayúsculas, el resto en minúsculas normales).

- [ ] **Step 11: Borrar scripts temporales y commitear**

```bash
rm test_hook_keywords.py test_hook_keyword_render.py
git add llm_analyzer.py state_manager.py app.py video_editor.py
git commit -m "R6: resaltar keyword del hook en mayusculas

Gemini ahora devuelve hook_keywords (subcadena exacta del hook a
enfatizar). En vez de renderizar en otro color (fragil con MoviePy y
wrapping automatico), se aplica mayusculas sobre esa subcadena dentro
del mismo TextClip -- mismo efecto de enfasis, cero riesgo de layout."
```

---

### Task 7: QA visual automático post-export

**Files:**
- Modify: `video_editor.py` (nuevo método `check_frame_brightness`, después
  de `generate_thumbnail`)
- Modify: `app.py` (nuevo método `_qa_check_clip`, integrado en `export_clips`;
  nueva sección de UI "Control de Calidad")
- Test: `test_qa_check.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de tareas previas (aunque se beneficia conceptualmente de
  R4/R5 — un clip con barra de progreso y sin cola de silencio pasa este
  chequeo más fácil, pero no hay dependencia de código).
- Produces: `VideoEditor.check_frame_brightness(frame_path: str) -> float`
  (brillo promedio 0-255). `OpusClipPro._qa_check_clip(output_path: str, has_hook: bool) -> Dict[str, Any]`
  con claves `frames: List[str]` (rutas), `checks: Dict[str, bool]`,
  `passed: bool`.

- [ ] **Step 1: Escribir el script de verificación (falla primero)**

Crear `test_qa_check.py`:

```python
"""Verifica check_frame_brightness sobre 2 imagenes sinteticas (una negra,
una blanca) generadas con PIL directamente -- no depende de un render de
video para este check unitario."""
import sys
sys.path.insert(0, ".")
from PIL import Image
from video_editor import VideoEditor

editor = VideoEditor.__new__(VideoEditor)

black_path = "test_qa_black.png"
white_path = "test_qa_white.png"
Image.new("RGB", (100, 100), (0, 0, 0)).save(black_path)
Image.new("RGB", (100, 100), (255, 255, 255)).save(white_path)

black_brightness = editor.check_frame_brightness(black_path)
white_brightness = editor.check_frame_brightness(white_path)
print(f"Negro: {black_brightness:.1f} | Blanco: {white_brightness:.1f}")

assert black_brightness < 10, f"un frame negro deberia dar brillo cercano a 0, dio {black_brightness}"
assert white_brightness > 240, f"un frame blanco deberia dar brillo cercano a 255, dio {white_brightness}"

import os
os.unlink(black_path)
os.unlink(white_path)
print("OK: check_frame_brightness distingue negro de blanco")
```

- [ ] **Step 2: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_qa_check.py`
Expected: `AttributeError: 'VideoEditor' object has no attribute 'check_frame_brightness'`

- [ ] **Step 3: Implementar `check_frame_brightness`**

Insertar en `video_editor.py` inmediatamente después de `generate_thumbnail`
(buscar el `return str(output_path)` que cierra esa función y el
`except ffmpeg.Error` que le sigue, insertar después de ese bloque, antes
de `def extract_keyframes`):

```python
    def check_frame_brightness(self, frame_path: str) -> float:
        """
        Brillo promedio (0-255) de una imagen — usado por el QA visual
        automático (R7) para detectar frames negros/vacíos al inicio o
        final de un clip exportado.
        """
        from PIL import Image
        img = Image.open(frame_path).convert("L")  # escala de grises
        pixels = list(img.getdata())
        return sum(pixels) / len(pixels) if pixels else 0.0
```

- [ ] **Step 4: Correr el script y confirmar que pasa**

Run: `venv/Scripts/python.exe test_qa_check.py`
Expected: `OK: check_frame_brightness distingue negro de blanco`

- [ ] **Step 5: Implementar `_qa_check_clip` en `app.py`**

Agregar el método nuevo en la clase `OpusClipPro`, inmediatamente después
de `generate_thumbnail`'s caller — buscar `def _generate_clip_metadata`
(está cerca de donde se genera metadata por clip) e insertar el método
nuevo justo antes:

```python
    def _qa_check_clip(self, output_path: str, has_hook: bool) -> Dict[str, Any]:
        """
        QA visual automático (R7): extrae 3 frames del clip ya exportado
        (inicio/medio/fin) y corre chequeos determinísticos simples — no
        otra llamada a Gemini (sería lento/caro por clip). Informativo, no
        bloquea el export.
        """
        result = {"frames": [], "checks": {}, "passed": True}
        try:
            probe_info = self.editor.get_video_info(output_path)
            duration = probe_info.get('duration', 0)
            if duration <= 0:
                return result

            timestamps = {
                "inicio": 0.3,
                "medio": duration / 2,
                "final": max(0.3, duration - 0.3),
            }
            frame_paths = {}
            for label, ts in timestamps.items():
                frame_path = str(config.TEMP_DIR / f"qa_{Path(output_path).stem}_{label}.jpg")
                self.editor.generate_thumbnail(output_path, frame_path, timestamp=ts, width=320)
                frame_paths[label] = frame_path
                result["frames"].append(frame_path)

            brightness = {
                label: self.editor.check_frame_brightness(path)
                for label, path in frame_paths.items()
            }

            # Check 1: el frame inicial no está negro/vacío
            result["checks"]["frame_inicial_no_negro"] = brightness["inicio"] > 15
            # Check 2: el frame final no está negro (R5 debería prevenir esto)
            result["checks"]["frame_final_no_negro"] = brightness["final"] > 15
            # Check 3 (informativo, solo si show_hook estaba activo): el
            # frame inicial no es prácticamente idéntico a un frame vacío —
            # mismo chequeo de brillo, umbral más permisivo porque el hook
            # puede tener fondo semi-transparente sobre un frame oscuro.
            if has_hook:
                result["checks"]["hook_probable_visible"] = brightness["inicio"] > 10

            result["passed"] = all(result["checks"].values())
        except Exception as e:
            logger.warning(f"QA visual omitido: {e}")
            result["passed"] = True  # no bloquear el export por un fallo del QA en sí
        return result
```

- [ ] **Step 6: Integrar en `export_clips` y devolver los resultados**

En `export_clips`, dentro del loop que ya genera thumbnails y sidecars por
clip (buscar `if export_srt:` dentro de ese loop), agregar justo antes:

```python
                    qa_result = self._qa_check_clip(output_path, has_hook=show_hook)
                    if not qa_result["passed"]:
                        logger.warning(f"⚠️ QA visual: clip {i+1} tiene problemas: {qa_result['checks']}")
```

Esto por ahora solo loguea — mostrarlo en la UI requiere una `gr.Gallery`
nueva, que se agrega en el Step 7 sin cambiar la firma de retorno de
`export_clips` (para no romper el wiring existente de
`outputs=[export_status, output_gallery, output_files, captions_output]`).
Para exponerlo en la UI sin cambiar esa firma, se agrega el detalle del QA
al `caption` de cada item de `output_gallery` (que ya es un string libre):
buscar la línea `caption = f"Clip {i+1} | ⭐ {sc:.1f} | ..."` dentro del
mismo loop y modificarla:

```python
                        qa_badge = "✅" if qa_result["passed"] else "⚠️"
                        caption = f"Clip {i+1} | ⭐ {sc:.1f} | {clip.start:.0f}s-{clip.end:.0f}s | {platform.upper()} | QA {qa_badge}"
```

- [ ] **Step 7: Verificación con export real**

Reutilizar el flujo de `test_direct_analysis.py` de la auditoría de hoy
(o recrearlo) para generar un `current_state` con clips reales, luego
llamar `app.export_clips(...)` directo y confirmar en el `status`/gallery
resultante que aparece el badge QA (`✅` o `⚠️`) en el caption de cada
clip exportado.

- [ ] **Step 8: Commitear**

```bash
git add video_editor.py app.py
git commit -m "R7: QA visual automatico post-export

Extrae 3 frames (inicio/medio/final) de cada clip exportado y corre
chequeos deterministicos de brillo -- detecta frames negros al inicio
o al final sin necesidad de otra llamada a Gemini por clip. Informativo,
no bloquea el export; se muestra como badge en la galeria de resultados."
```

---

### Task 8: Descarga automática por URL (yt-dlp)

**Files:**
- Create: `url_downloader.py`
- Modify: `requirements.txt` (agregar `yt-dlp`)
- Modify: `app.py` (nuevo método `download_from_url`, nuevo campo de URL +
  botón en la pantalla Importar, wiring)
- Test: `test_url_downloader.py` (nuevo, raíz del repo)

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: `url_downloader.is_supported_url(url: str) -> bool`,
  `url_downloader.download_video(url: str, output_dir: str, progress_callback=None, max_duration_s: int = 7200) -> str`
  (devuelve la ruta del archivo descargado, o lanza `ValueError`/`RuntimeError`
  con mensaje claro en caso de error).

- [ ] **Step 1: Agregar la dependencia**

En `requirements.txt`, agregar una línea nueva (versión fija, mismo
criterio que el resto del archivo):

```
yt-dlp==2024.12.13
```

Instalar en el venv del proyecto:

```bash
venv/Scripts/pip.exe install yt-dlp==2024.12.13
```

- [ ] **Step 2: Escribir el script de verificación de `is_supported_url` (falla primero)**

Crear `test_url_downloader.py`:

```python
"""Verifica is_supported_url sin hacer ninguna descarga real (no depende
de red para este check)."""
import sys
sys.path.insert(0, ".")
from url_downloader import is_supported_url

casos_validos = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.tiktok.com/@usuario/video/1234567890",
    "https://www.instagram.com/reel/ABC123/",
    "https://x.com/usuario/status/1234567890",
    "https://twitter.com/usuario/status/1234567890",
]
casos_invalidos = [
    "no es una url",
    "https://ejemplo-random-no-soportado.com/video.mp4",
    "",
    "ftp://youtube.com/video",
]

for url in casos_validos:
    assert is_supported_url(url), f"deberia aceptar: {url}"
for url in casos_invalidos:
    assert not is_supported_url(url), f"deberia rechazar: {url}"

print("OK: is_supported_url distingue dominios soportados de no soportados")
```

- [ ] **Step 3: Correr el script y confirmar que falla**

Run: `venv/Scripts/python.exe test_url_downloader.py`
Expected: `ModuleNotFoundError: No module named 'url_downloader'`

- [ ] **Step 4: Crear `url_downloader.py`**

```python
"""
Descarga de video por URL (YouTube, TikTok, Instagram, X/Twitter) vía yt-dlp.
Módulo aislado: no importa nada de app.py/video_editor.py, produce un
archivo local y devuelve su ruta — el resto del pipeline lo trata igual
que un archivo subido manualmente.
"""
import logging
import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import yt_dlp

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = (
    "youtube.com", "youtu.be",
    "tiktok.com",
    "instagram.com",
    "x.com", "twitter.com",
)


def is_supported_url(url: str) -> bool:
    """
    True si `url` es una URL http(s) válida de un dominio soportado.
    Chequeo defensivo antes de invocar yt-dlp — evita pasarle cualquier
    string arbitrario a un proceso de descarga externo.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.netloc or "").lower()
    host = re.sub(r"^www\.", "", host)
    return any(host == d or host.endswith(f".{d}") for d in SUPPORTED_DOMAINS)


def download_video(
    url: str,
    output_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_duration_s: int = 7200,
) -> str:
    """
    Descarga `url` a `output_dir` usando la API de Python de yt-dlp (no
    subprocess con la URL interpolada en un comando — evita cualquier
    vector de inyección de argumentos).

    Args:
        url: link del video (YouTube/TikTok/Instagram/X)
        output_dir: carpeta destino (se crea si no existe)
        progress_callback: (progress 0-1, mensaje) opcional
        max_duration_s: duración máxima descargable en segundos (default 2h)
            — evita que un stream/video excesivamente largo sature disco/tiempo

    Returns:
        Ruta absoluta del archivo descargado.

    Raises:
        ValueError: si la URL no es de un dominio soportado, o el video
            excede max_duration_s.
        RuntimeError: si yt-dlp falla (video privado, eliminado, geo-bloqueado, etc.)
    """
    if not is_supported_url(url):
        raise ValueError(
            f"URL no soportada: {url}\n"
            "Dominios soportados: YouTube, TikTok, Instagram, X/Twitter."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _progress_hook(d):
        if not progress_callback:
            return
        if d.get('status') == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                progress_callback(downloaded / total, f"⬇️ Descargando... {downloaded / total * 100:.0f}%")
        elif d.get('status') == 'finished':
            progress_callback(1.0, "✅ Descarga completa, procesando...")

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': str(output_path / '%(title).100s.%(ext)s'),
        'progress_hooks': [_progress_hook],
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration') or 0
            if duration > max_duration_s:
                raise ValueError(
                    f"El video dura {duration/60:.0f} min, excede el máximo "
                    f"permitido de {max_duration_s/60:.0f} min."
                )
            ydl.download([url])
            filename = ydl.prepare_filename(info)
            logger.info(f"Video descargado: {filename}")
            return filename
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"No se pudo descargar el video: {e}")
```

- [ ] **Step 5: Correr el script y confirmar que pasa**

Run: `venv/Scripts/python.exe test_url_downloader.py`
Expected: `OK: is_supported_url distingue dominios soportados de no soportados`

- [ ] **Step 6: Verificación de descarga real (requiere red)**

Crear `test_url_download_real.py`:

```python
"""Descarga un video corto y público real para verificar el flujo
completo. Usa un video corto conocido (menos de 1 minuto) para no
tardar. Requiere conexión a internet."""
import sys
sys.path.insert(0, ".")
from url_downloader import download_video

progress_log = []
def on_progress(p, msg):
    progress_log.append((p, msg))
    print(f"  {p*100:.0f}% - {msg}")

# Video de prueba corto y estable (clip publico conocido)
url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", ~19s

output_dir = r"D:\Temp\claude\d--TODO-opus-clip-v2\978d3860-9ade-41dc-8fb3-5d4ba426b620\scratchpad\url_download_test"
result_path = download_video(url, output_dir, progress_callback=on_progress, max_duration_s=120)

import os
assert os.path.exists(result_path), f"el archivo descargado no existe: {result_path}"
assert os.path.getsize(result_path) > 10_000, "el archivo descargado es sospechosamente chico"
assert len(progress_log) > 0, "no se recibio ningun callback de progreso"
print(f"OK: video descargado en {result_path} ({os.path.getsize(result_path)} bytes)")
```

Run: `venv/Scripts/python.exe test_url_download_real.py`
Expected: `OK: video descargado en ...` (si falla por red/geo-bloqueo, probar
con otra URL corta y pública conocida — no es un problema del código si
un video puntual falla, es un problema del video puntual).

- [ ] **Step 7: Verificación del límite de duración**

Agregar al mismo script `test_url_download_real.py`, al final:

```python
try:
    download_video(url, output_dir, max_duration_s=5)  # "Me at the zoo" dura ~19s
    assert False, "deberia haber lanzado ValueError por exceder max_duration_s"
except ValueError as e:
    print(f"OK: rechaza videos que exceden max_duration_s ({e})")
```

Run de nuevo: `venv/Scripts/python.exe test_url_download_real.py`
Expected: ambos `OK:` impresos, sin `AssertionError`.

- [ ] **Step 8: Integrar en `app.py` — método + UI**

Agregar el import al principio de `app.py` (junto a los demás imports de
módulos propios):

```python
from url_downloader import is_supported_url, download_video
```

Agregar un método nuevo en `OpusClipPro`, cerca de `validate_video`:

```python
    def download_from_url(self, url: str, progress: gr.Progress) -> Tuple[str, Optional[str]]:
        """
        Descarga un video desde una URL (YouTube/TikTok/Instagram/X) a
        `videos para editar/`, mismo directorio que usa el flujo manual.

        Returns:
            (mensaje_status, ruta_del_archivo_o_None)
        """
        if not url or not url.strip():
            return "❌ Pegá un link primero", None
        if not is_supported_url(url):
            return "❌ Link no soportado (YouTube, TikTok, Instagram o X)", None
        try:
            def _progress_cb(p, msg):
                progress(p, desc=msg)
            filepath = download_video(
                url.strip(), "videos para editar", progress_callback=_progress_cb
            )
            return f"✅ Descargado: {Path(filepath).name}", filepath
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Descarga por URL falló: {e}")
            gr.Warning(f"❌ No se pudo descargar: {e}")
            return f"❌ {e}", None
```

Agregar el campo de UI: en la pantalla Importar, dentro del panel "Fuente
de Video" (`app.py`, buscar `video_info = gr.Textbox(label="", value="No hay video seleccionado"`,
línea 2781), agregar justo después:

```python
                                        # Descarga por URL (YouTube/TikTok/IG/X)
                                        with gr.Row():
                                            url_input = gr.Textbox(
                                                label="",
                                                placeholder="O pegá un link de YouTube/TikTok/Instagram/X...",
                                                show_label=False,
                                                scale=4,
                                            )
                                            download_url_btn = gr.Button("⬇️ Descargar", elem_classes=["btn-secondary"], scale=1)
                                        url_download_status = gr.Textbox(label="", show_label=False, interactive=False, visible=False)
```

Wiring: buscar `video_input.change(` (ya existente, dispara
`on_video_select`) y agregar el evento nuevo justo antes:

```python
            def _on_download_url(url, progress=gr.Progress()):
                status, filepath = self.download_from_url(url, progress)
                if filepath:
                    return status, gr.update(visible=True), gr.update(value=[filepath])
                return status, gr.update(visible=True), gr.update()

            download_url_btn.click(
                fn=_on_download_url,
                inputs=[url_input],
                outputs=[url_download_status, url_download_status, video_input]
            )
```

Esto reutiliza `video_input` (el mismo `gr.File` de subida manual) como
destino — al setear su valor programáticamente dispara el mismo
`video_input.change` → `on_video_select` que ya corre para archivos
subidos a mano, sin duplicar lógica de precheck.

- [ ] **Step 9: Reiniciar el servidor y verificar en la UI real**

Reiniciar el servidor. Con Playwright (mismo patrón usado en toda la
sesión de hoy): entrar a Importar, pegar la URL de prueba
(`https://www.youtube.com/watch?v=jNQXAC9IVRw`), click en "⬇️ Descargar",
esperar y confirmar por screenshot que:
1. Aparece el mensaje de progreso mientras descarga.
2. Al terminar, el `video_input` muestra el archivo descargado (mismo
   comportamiento visual que subir un archivo a mano).
3. El precheck (`video_precheck`/`video_info`) se dispara solo, mostrando
   duración y ETA — prueba de que `on_video_select` corrió sin cambios.

- [ ] **Step 10: Borrar scripts y carpeta de test, commitear**

```bash
rm test_url_downloader.py test_url_download_real.py
rm -rf "D:/Temp/claude/d--TODO-opus-clip-v2/978d3860-9ade-41dc-8fb3-5d4ba426b620/scratchpad/url_download_test"
git add url_downloader.py requirements.txt app.py
git commit -m "I1: descarga automatica por URL (YouTube/TikTok/IG/X)

Nuevo modulo url_downloader.py aislado (usa la API de Python de yt-dlp,
no subprocess con la URL interpolada). Reutiliza el componente video_input
y su evento .change() existente -- cero logica de precheck duplicada."
```

---

## Orden de ejecución

Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8, en ese orden (mismo orden que el spec
aprobado). Cada task es independiente de las anteriores salvo por tocar
funciones vecinas en los mismos archivos (`video_editor.py`/`app.py`) — si
se ejecuta con subagentes en paralelo, Tasks 1, 2, 3, 6 y 8 no tienen
conflictos de líneas entre sí y podrían correr en paralelo; Tasks 4 y 5
tocan zonas cercanas de `create_viral_clip`/`_export_single_clip` y
conviene serializarlas con las demás para evitar conflictos de merge.
