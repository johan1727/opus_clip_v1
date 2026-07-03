"""
Módulo de transcripción con Whisper.
Detecta CUDA automáticamente y optimiza para GTX 1080 (8GB VRAM).
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path

import ffmpeg
import whisper
import torch

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Transcriber:
    """
    Clase para transcribir videos usando Whisper.
    Optimizada para GTX 1080 (8GB VRAM) - usa modelo 'base'.
    """
    
    CACHE_DIR = Path("temp/transcription_cache")
    CHUNK_DURATION_S = 600   # 10 min per chunk
    CHUNK_OVERLAP_S  = 15    # 15 s overlap to avoid word cuts

    def __init__(
        self, 
        model_size: str = "base",
        language: str = "es",
        device: Optional[str] = None,
        use_cache: bool = True,
        word_timestamps: bool = True
    ) -> None:
        """
        Inicializa el transcriptor Whisper.
        
        Args:
            model_size: Tamaño del modelo ('tiny', 'base', 'small', 'medium', 'large')
            language: Código de idioma ('es' para español)
            device: 'cuda', 'cpu', o None para auto-detectar
            use_cache: Guardar/leer transcripción cacheada por hash del video
            word_timestamps: Pedir timestamps por palabra a Whisper
        """
        self.model_size = model_size
        self.language = language
        self.device = self._detect_device(device)
        self.model: Optional[whisper.Whisper] = None
        self.use_cache = use_cache
        self.word_timestamps = word_timestamps
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Transcriber inicializado: modelo={model_size}, device={self.device}, idioma={language}, word_ts={word_timestamps}")
    
    def _detect_device(self, device: Optional[str]) -> str:
        """
        Detecta automáticamente si CUDA está disponible.
        GTX 1080 tiene 8GB VRAM, suficiente para modelo 'base' (~1GB).
        """
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"GPU detectada: {gpu_name} ({vram:.1f}GB VRAM)")
            logger.info(f"CUDA disponible, usando GPU para transcripción")
            return "cuda"
        else:
            logger.warning("CUDA no disponible, usando CPU (será más lento)")
            return "cpu"
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Obtiene la duración real del video usando FFprobe.
        
        Args:
            video_path: Ruta al video
            
        Returns:
            Duración en segundos
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            if video_stream and 'duration' in video_stream:
                return float(video_stream['duration'])
            # Fallback a format duration
            if 'format' in probe and 'duration' in probe['format']:
                return float(probe['format']['duration'])
            return 0.0
        except Exception as e:
            logger.warning(f"No se pudo obtener duración del video: {e}")
            return 0.0
    
    def load_model(self) -> whisper.Whisper:
        """
        Carga el modelo Whisper en memoria (GPU o CPU).
        
        Returns:
            Modelo Whisper cargado
        """
        if self.model is not None:
            logger.debug("Modelo ya cargado, reutilizando")
            return self.model
        
        logger.info(f"Cargando modelo Whisper '{self.model_size}' en {self.device}...")
        
        try:
            self.model = whisper.load_model(self.model_size).to(self.device)
            logger.info(f"Modelo cargado exitosamente")
            return self.model
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo Whisper: {e}")
    
    def unload_model(self) -> None:
        """
        Libera el modelo de memoria y limpia caché GPU.
        Útil para liberar VRAM después de transcripción.
        """
        if self.model is not None:
            logger.info("Liberando modelo Whisper de memoria...")
            del self.model
            self.model = None
            
            # Limpiar caché CUDA si está disponible
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("🧹 Caché GPU limpiada")
            
            logger.info("✅ Modelo descargado")
    
    def extract_audio(
        self,
        video_path: str,
        output_audio_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Extrae audio de un video usando FFmpeg.
        
        Args:
            video_path: Ruta al video
            output_audio_path: Ruta de salida (opcional)
            progress_callback: Función opcional(progress: float, message: str)
            
        Returns:
            Ruta al archivo de audio extraído
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        # Generar ruta de salida temporal si no se proporciona — antes se escribía junto
        # al video fuente, lo que falla en carpetas de solo lectura y ensucia la carpeta
        # del usuario con archivos .mp3 temporales.
        if output_audio_path is None:
            temp_dir = Path("temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_audio_path = temp_dir / f"{video_path.stem}_audio.mp3"
        else:
            output_audio_path = Path(output_audio_path)
        
        if progress_callback:
            progress_callback(0.0, "🔊 Extrayendo audio del video...")
        
        logger.info(f"Extrayendo audio de: {video_path.name}")
        
        try:
            # Extraer audio: MP3, 16kHz, mono (óptimo para Whisper)
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_audio_path),
                    vn=None,  # No video
                    acodec='libmp3lame',
                    ar=16000,  # 16kHz (requerido por Whisper)
                    ac=1,  # Mono
                    q=2,  # Calidad VBR
                    loglevel='warning'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Verificar tamaño
            audio_size = output_audio_path.stat().st_size / (1024**2)  # MB
            video_size = video_path.stat().st_size / (1024**2)  # MB
            reduction = (1 - audio_size / video_size) * 100 if video_size > 0 else 0
            
            if progress_callback:
                progress_callback(1.0, f"✅ Audio extraído: {audio_size:.1f}MB ({reduction:.0f}% más pequeño)")
            
            logger.info(f"Audio extraído: {output_audio_path} ({audio_size:.1f}MB, {reduction:.0f}% reducción)")
            
            return str(output_audio_path)
            
        except ffmpeg.Error as e:
            logger.error(f"Error extrayendo audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"No se pudo extraer audio: {e}")
    
    # ------------------------------------------------------------------
    # Cache helpers (F1.3)
    # ------------------------------------------------------------------

    def _video_hash(self, video_path: Path) -> str:
        """SHA-256 of the first 1 MB + file size — fast fingerprint."""
        h = hashlib.sha256()
        h.update(str(video_path.stat().st_size).encode())
        with open(video_path, 'rb') as f:
            h.update(f.read(1024 * 1024))
        return h.hexdigest()[:16]

    def _cache_path(self, video_path: Path, suffix: str = "") -> Path:
        key = f"{video_path.stem}_{self._video_hash(video_path)}_{self.model_size}{suffix}.json"
        return self.CACHE_DIR / key

    def _load_cache(self, video_path: Path, suffix: str = "") -> Optional[Dict[str, Any]]:
        cp = self._cache_path(video_path, suffix)
        if cp.exists():
            try:
                data = json.loads(cp.read_text(encoding='utf-8'))
                logger.info(f"⚡ Transcripción cargada desde caché: {cp.name}")
                return data
            except Exception as e:
                logger.warning(f"Caché corrupto, ignorando: {e}")
        return None

    def _save_cache(self, video_path: Path, data: Dict[str, Any], suffix: str = "") -> None:
        cp = self._cache_path(video_path, suffix)
        try:
            cp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            logger.info(f"💾 Transcripción guardada en caché: {cp.name}")
        except Exception as e:
            logger.warning(f"No se pudo guardar caché: {e}")

    # ------------------------------------------------------------------
    # Word-level segment flattening (F1.2)
    # ------------------------------------------------------------------

    def _flatten_word_segments(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Converts word-level timestamps from Whisper into per-word segments
        compatible with the rest of the pipeline. Falls back to phrase-level
        segments if word data is not present.
        """
        flat = []
        seg_id = 0
        for seg in result.get('segments', []):
            words = seg.get('words', [])
            if words:
                for w in words:
                    flat.append({
                        'id': seg_id,
                        'start': float(w.get('start', seg['start'])),
                        'end': float(w.get('end', seg['end'])),
                        'text': w.get('word', '').strip(),
                        'phrase_start': seg['start'],
                        'phrase_end': seg['end'],
                        'phrase_text': seg['text'].strip(),
                    })
                    seg_id += 1
            else:
                flat.append({
                    'id': seg_id,
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip(),
                    'phrase_start': seg['start'],
                    'phrase_end': seg['end'],
                    'phrase_text': seg['text'].strip(),
                })
                seg_id += 1
        return flat

    # ------------------------------------------------------------------
    # Silence boundary snap (F1.4)
    # ------------------------------------------------------------------

    def snap_to_silence(self, segments: List[Dict[str, Any]], t: float,
                        window: float = 1.5) -> float:
        """
        Moves timestamp t to the nearest inter-segment gap within +-window seconds.
        Returns t unchanged if no gap is found.
        """
        best_t = t
        best_dist = float('inf')
        for i in range(len(segments) - 1):
            gap_t = (segments[i]['end'] + segments[i + 1]['start']) / 2
            dist = abs(gap_t - t)
            if dist < best_dist and dist <= window:
                best_dist = dist
                best_t = gap_t
        return best_t

    # ------------------------------------------------------------------
    # Main transcribe (F1.2 + F1.3)
    # ------------------------------------------------------------------

    def estimate_analysis_time(self, video_path: str, model_size: str = "base") -> Dict[str, Any]:
        """
        Returns estimated total analysis time and detected video info for UX precheck.
        Speed factors (rough, GTX1080 GPU): tiny~1x, base~5x, small~12x realtime.
        """
        dur = self._get_video_duration(str(video_path))
        speed = {'tiny': 18, 'base': 7, 'small': 3, 'medium': 1.5, 'large': 0.8}
        rt = speed.get(model_size, 7)
        transcribe_est = max(10, dur / rt)
        gemini_est = 15 + (dur / 60) * 3   # ~3s per minute of content
        total_est = transcribe_est + gemini_est
        cached = self.use_cache and self._load_cache(Path(video_path)) is not None
        return {
            'duration_s': dur,
            'duration_min': dur / 60,
            'estimated_total_s': total_est,
            'cached': cached,
            'chunked': dur > 2700,   # >45 min
            'model': model_size,
        }

    def transcribe(
        self, 
        video_path: str,
        verbose: bool = False,
        extract_audio_first: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        word_timestamps: Optional[bool] = None,
        cache_key_suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Transcribe un video completo. Usa caché si está disponible.

        Args:
            video_path: Ruta al archivo de video
            verbose: Mostrar progreso detallado
            extract_audio_first: Extraer audio antes de transcribir (más rápido)
            progress_callback: Función opcional(progress: float, message: str)
            word_timestamps: Override del flag de la instancia (None = usar default)
            cache_key_suffix: Sufijo extra para diferenciar caché (ej. "_fast")
            
        Returns:
            Dict con: 'text', 'segments', 'word_segments', 'language', 'duration'
            
        Raises:
            FileNotFoundError: Si el video no existe
            RuntimeError: Si falla la transcripción
        """
        use_word_ts = self.word_timestamps if word_timestamps is None else word_timestamps
        video_path = Path(video_path)
        audio_path = None
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        if not video_path.is_file():
            raise ValueError(f"La ruta no es un archivo: {video_path}")

        # --- Cache hit? ---
        cache_path_override = None
        if cache_key_suffix and self.use_cache:
            key = f"{video_path.stem}_{self._video_hash(video_path)}_{self.model_size}{cache_key_suffix}.json"
            cache_path_override = self.CACHE_DIR / key
            if cache_path_override.exists():
                try:
                    data = json.loads(cache_path_override.read_text(encoding='utf-8'))
                    if progress_callback:
                        progress_callback(1.0, "⚡ Transcripción lista (caché)")
                    return data
                except Exception:
                    pass
        elif self.use_cache:
            cached = self._load_cache(video_path)
            if cached is not None:
                if progress_callback:
                    progress_callback(1.0, "⚡ Transcripción lista (caché)")
                return cached
        
        model = self.load_model()
        logger.info(f"Iniciando transcripción de: {video_path.name} [word_ts={use_word_ts}]")
        
        file_to_transcribe = str(video_path)
        temp_audio_created = False
        
        if extract_audio_first:
            try:
                audio_path = self.extract_audio(
                    str(video_path),
                    progress_callback=progress_callback
                )
                file_to_transcribe = audio_path
                temp_audio_created = True
            except Exception as e:
                logger.warning(f"No se pudo extraer audio: {e}. Usando video directamente.")
                if progress_callback:
                    progress_callback(0.0, "⚠️ Usando video directamente (sin extraer audio)")
        
        try:
            if progress_callback:
                progress_callback(0.1, "🎤 Transcribiendo con Whisper...")
            
            result = model.transcribe(
                file_to_transcribe,
                language=self.language,
                verbose=verbose,
                fp16=(self.device == "cuda"),
                word_timestamps=use_word_ts,
            )
            
            if progress_callback:
                progress_callback(0.35, "📝 Procesando resultado...")
            
            actual_duration = self._get_video_duration(str(video_path))
            if actual_duration == 0 and result['segments']:
                actual_duration = result['segments'][-1]['end']
            
            word_segments = self._flatten_word_segments(result)
            
            transcription_data = {
                'text': result['text'],
                'segments': result['segments'],
                'word_segments': word_segments,
                'language': result.get('language', self.language),
                'duration': actual_duration
            }
            
            logger.info(
                f"Transcripción completada: "
                f"{len(transcription_data['segments'])} segmentos, "
                f"{len(word_segments)} palabras, "
                f"{transcription_data['duration']:.1f}s"
            )
            
            if self.use_cache:
                if cache_path_override:
                    cache_path_override.write_text(json.dumps(transcription_data, ensure_ascii=False), encoding='utf-8')
                else:
                    self._save_cache(video_path, transcription_data)
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Error en transcripción: {e}")
            raise RuntimeError(f"Fallo la transcripción: {e}")
        finally:
            if temp_audio_created and file_to_transcribe != str(video_path):
                try:
                    Path(file_to_transcribe).unlink(missing_ok=True)
                    logger.info(f"Archivo temporal eliminado: {file_to_transcribe}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar archivo temporal: {e}")

    # ------------------------------------------------------------------
    # Chunked transcription for long videos (F1.1)
    # ------------------------------------------------------------------

    def transcribe_clip_words(
        self,
        video_path: str,
        clips: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Second-pass: transcribes only the clip windows with word_timestamps=True.
        Much faster than full-video word transcription for 30min+ videos.

        Args:
            video_path: Ruta al video original
            clips: List of {'id': int, 'start': float, 'end': float}
            progress_callback: (progress 0-1, msg)

        Returns:
            Dict mapping clip_id -> list of word-level segments
        """
        model = self.load_model()
        result_map: Dict[int, List[Dict[str, Any]]] = {}
        total = len(clips)

        for i, clip in enumerate(clips):
            clip_id = clip.get('id', i)
            start = clip['start']
            end = clip['end']
            dur = end - start

            if progress_callback:
                progress_callback(i / total, f"🔍 Word timestamps clip {i+1}/{total}...")

            tmp_audio = Path("temp") / f"clip_words_{clip_id}.mp3"
            tmp_audio.parent.mkdir(parents=True, exist_ok=True)
            try:
                (
                    ffmpeg
                    .input(str(video_path), ss=start, t=dur)
                    .output(
                        str(tmp_audio),
                        vn=None, acodec='libmp3lame',
                        ar=16000, ac=1, q=4,
                        loglevel='warning'
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                chunk_result = model.transcribe(
                    str(tmp_audio),
                    language=self.language,
                    fp16=(self.device == "cuda"),
                    word_timestamps=True,
                )
                # Re-offset timestamps back to absolute video time
                adjusted = []
                for seg in chunk_result.get('segments', []):
                    words = seg.get('words', [])
                    for w in words:
                        adjusted.append({
                            'id': len(adjusted),
                            'start': w['start'] + start,
                            'end': w['end'] + start,
                            'text': w.get('word', '').strip(),
                            'phrase_start': seg['start'] + start,
                            'phrase_end': seg['end'] + start,
                            'phrase_text': seg['text'].strip(),
                        })
                    if not words:
                        adjusted.append({
                            'id': len(adjusted),
                            'start': seg['start'] + start,
                            'end': seg['end'] + start,
                            'text': seg['text'].strip(),
                            'phrase_start': seg['start'] + start,
                            'phrase_end': seg['end'] + start,
                            'phrase_text': seg['text'].strip(),
                        })
                result_map[clip_id] = adjusted
                logger.info(f"Word timestamps clip {clip_id}: {len(adjusted)} palabras")
            except Exception as e:
                logger.warning(f"Word timestamps fallaron para clip {clip_id}: {e}")
                result_map[clip_id] = []
            finally:
                tmp_audio.unlink(missing_ok=True)

        if progress_callback:
            progress_callback(1.0, f"✅ Word timestamps listos ({total} clips)")
        return result_map

    def transcribe_chunked(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        chunk_duration: int = None,
        overlap: int = None,
        word_timestamps: Optional[bool] = None,
        cache_key_suffix: str = "",
    ) -> Dict[str, Any]:
        """
        Transcribe videos largos (1h+) dividiendo el audio en chunks.
        Evita OOM en GPU y permite seguimiento de progreso granular.

        Args:
            video_path: Ruta al video
            progress_callback: (progress 0-1, mensaje)
            chunk_duration: Duración de cada chunk en segundos (default 600)
            overlap: Overlap entre chunks en segundos (default 15)
            cache_key_suffix: Sufijo extra para diferenciar caché (ej. "_fast") — sin esto,
                una transcripción rápida (sin word-timestamps) y una de calidad (con
                word-timestamps) del MISMO video compartían la misma entrada de caché, y
                un run de "Calidad" podía recibir en silencio los datos degradados del
                run "Rápido" anterior.

        Returns:
            Dict con: 'text', 'segments', 'word_segments', 'language', 'duration'
        """
        chunk_dur = chunk_duration or self.CHUNK_DURATION_S
        ovlp = overlap or self.CHUNK_OVERLAP_S
        use_word_ts = self.word_timestamps if word_timestamps is None else word_timestamps
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        # --- Cache hit? ---
        if self.use_cache:
            cached = self._load_cache(video_path, cache_key_suffix)
            if cached is not None:
                if progress_callback:
                    progress_callback(1.0, "⚡ Transcripción lista (caché)")
                return cached

        total_duration = self._get_video_duration(str(video_path))
        if total_duration == 0:
            logger.warning("No se pudo obtener duración; usando transcribe() normal")
            return self.transcribe(str(video_path), progress_callback=progress_callback,
                                    word_timestamps=word_timestamps, cache_key_suffix=cache_key_suffix)

        # Para videos cortos, usar el flujo normal
        if total_duration <= chunk_dur * 1.5:
            return self.transcribe(str(video_path), progress_callback=progress_callback,
                                    word_timestamps=word_timestamps, cache_key_suffix=cache_key_suffix)

        model = self.load_model()
        logger.info(f"📦 Transcripción chunked: {total_duration:.0f}s en chunks de {chunk_dur}s (overlap {ovlp}s)")

        # Calcular chunks: [start, end] con overlap
        starts = []
        t = 0.0
        while t < total_duration:
            starts.append(t)
            t += chunk_dur
        n_chunks = len(starts)

        all_segments: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        chunk_audio_dir = Path("temp") / f"chunks_{video_path.stem}"
        chunk_audio_dir.mkdir(parents=True, exist_ok=True)

        # S3: Incremental chunk cache directory
        chunk_cache_dir = self.CACHE_DIR / f"incremental_{video_path.stem}_{self._video_hash(video_path)}"
        chunk_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            for idx, start in enumerate(starts):
                end = min(start + chunk_dur + ovlp, total_duration)
                chunk_label = f"Chunk {idx+1}/{n_chunks} ({start:.0f}s-{end:.0f}s)"

                # S3: Check incremental chunk cache
                chunk_cache_file = chunk_cache_dir / f"chunk_{idx:03d}.json"
                if chunk_cache_file.exists():
                    try:
                        cached_chunk = json.loads(chunk_cache_file.read_text(encoding='utf-8'))
                        all_segments.extend(cached_chunk.get('segments', []))
                        full_text_parts.append(cached_chunk.get('text', ''))
                        if progress_callback:
                            progress_callback(idx / n_chunks * 0.9,
                                              f"⚡ {chunk_label} (caché)")
                        continue
                    except Exception:
                        pass

                if progress_callback:
                    progress_callback(idx / n_chunks * 0.9,
                                      f"🎤 {chunk_label}...")

                # Extraer chunk de audio con FFmpeg
                chunk_audio = chunk_audio_dir / f"chunk_{idx:03d}.mp3"
                try:
                    (
                        ffmpeg
                        .input(str(video_path), ss=start, t=(end - start))
                        .output(
                            str(chunk_audio),
                            vn=None, acodec='libmp3lame',
                            ar=16000, ac=1, q=2,
                            loglevel='warning'
                        )
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logger.error(f"Error extrayendo audio del chunk {idx}: {e}")
                    continue

                # Transcribir el chunk
                try:
                    chunk_result = model.transcribe(
                        str(chunk_audio),
                        language=self.language,
                        fp16=(self.device == "cuda"),
                        word_timestamps=use_word_ts,
                    )
                except Exception as e:
                    logger.error(f"Error transcribiendo chunk {idx}: {e}")
                    continue
                finally:
                    chunk_audio.unlink(missing_ok=True)

                # Ajustar timestamps al tiempo global y filtrar zona de overlap
                # Solo tomamos hasta chunk_dur (excluimos los ultimos 'ovlp' segundos
                # excepto para el ultimo chunk)
                cutoff = (chunk_dur if idx < n_chunks - 1 else (end - start))
                for seg in chunk_result.get('segments', []):
                    adj_start = seg['start'] + start
                    adj_end = seg['end'] + start
                    # Saltar si está completamente en la zona de overlap
                    if seg['start'] >= cutoff:
                        continue
                    adj_seg = dict(seg)
                    adj_seg['start'] = adj_start
                    adj_seg['end'] = min(adj_end, start + cutoff)
                    # Ajustar words si existen
                    if 'words' in adj_seg:
                        adj_words = []
                        for w in adj_seg['words']:
                            ws = w['start'] + start
                            we = w['end'] + start
                            if w['start'] < cutoff:
                                adj_words.append({**w, 'start': ws, 'end': we})
                        adj_seg['words'] = adj_words
                    all_segments.append(adj_seg)
                full_text_parts.append(chunk_result.get('text', ''))

                # S3: Save chunk to incremental cache
                if chunk_result.get('segments'):
                    try:
                        chunk_cache_file.write_text(
                            json.dumps({'segments': all_segments[-(len(chunk_result['segments'])):],
                                        'text': chunk_result.get('text', '')},
                                       ensure_ascii=False),
                            encoding='utf-8'
                        )
                    except Exception as _ce:
                        logger.debug(f"No se pudo guardar caché de chunk: {_ce}")

        finally:
            # Limpiar directorio de chunks
            import shutil
            try:
                shutil.rmtree(chunk_audio_dir, ignore_errors=True)
            except Exception:
                pass

        # Renumerar segmentos
        for i, seg in enumerate(all_segments):
            seg['id'] = i

        word_segments = self._flatten_word_segments({'segments': all_segments})
        full_text = ' '.join(full_text_parts).strip()
        language = 'es'
        if all_segments:
            # best guess from first chunk result
            language = self.language

        transcription_data = {
            'text': full_text,
            'segments': all_segments,
            'word_segments': word_segments,
            'language': language,
            'duration': total_duration,
        }

        logger.info(
            f"✅ Transcripción chunked completada: "
            f"{len(all_segments)} segmentos, {len(word_segments)} palabras, "
            f"{total_duration:.1f}s"
        )

        if self.use_cache:
            self._save_cache(video_path, transcription_data, cache_key_suffix)

        if progress_callback:
            progress_callback(1.0, f"✅ Transcripción lista ({len(all_segments)} segmentos)")

        return transcription_data
    
    def get_segments_with_text(
        self, 
        transcription: Dict[str, Any],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrae segmentos de texto en un rango de tiempo.
        
        Args:
            transcription: Resultado de transcribe()
            start_time: Tiempo de inicio (segundos), None = desde inicio
            end_time: Tiempo de fin (segundos), None = hasta el final
            
        Returns:
            Lista de segmentos filtrados con: id, start, end, text
        """
        segments = transcription.get('segments', [])
        
        filtered = []
        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']
            
            # Filtrar por tiempo
            if start_time is not None and seg_end < start_time:
                continue
            if end_time is not None and seg_start > end_time:
                continue
            
            filtered.append({
                'id': seg.get('id', len(filtered)),
                'start': seg_start,
                'end': seg_end,
                'text': seg['text'].strip()
            })
        
        return filtered
    
    def generate_srt_content(
        self, 
        transcription: Dict[str, Any]
    ) -> str:
        """
        Genera contenido SRT (subtítulos) desde la transcripción.
        
        Args:
            transcription: Resultado de transcribe()
            
        Returns:
            String con formato SRT
        """
        segments = transcription.get('segments', [])
        
        def format_time(seconds: float) -> str:
            """Convierte segundos a formato SRT HH:MM:SS,mmm"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_lines = []
        for seg in segments:
            srt_lines.append(str(seg['id'] + 1))  # SRT empieza en 1
            srt_lines.append(
                f"{format_time(seg['start'])} --> {format_time(seg['end'])}"
            )
            srt_lines.append(seg['text'].strip())
            srt_lines.append("")  # Línea vacía entre entradas
        
        return "\n".join(srt_lines)
    
    def save_srt(
        self, 
        transcription: Dict[str, Any], 
        output_path: str
    ) -> str:
        """
        Guarda la transcripción como archivo SRT.
        
        Args:
            transcription: Resultado de transcribe()
            output_path: Ruta donde guardar el archivo .srt
            
        Returns:
            Ruta del archivo guardado
        """
        srt_content = self.generate_srt_content(transcription)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.info(f"Subtítulos SRT guardados en: {output_path}")
        return str(output_path)


def quick_transcribe(
    video_path: str,
    output_srt: Optional[str] = None,
    model_size: str = "base"
) -> Dict[str, Any]:
    """
    Función de conveniencia para transcribir un video rápidamente.
    
    Args:
        video_path: Ruta al video
        output_srt: Ruta opcional para guardar subtítulos
        model_size: Tamaño del modelo Whisper
        
    Returns:
        Dict con la transcripción completa
    """
    transcriber = Transcriber(model_size=model_size)
    result = transcriber.transcribe(video_path)
    
    if output_srt:
        transcriber.save_srt(result, output_srt)
    
    return result


if __name__ == "__main__":
    # Test del módulo
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python transcriber.py <ruta_video> [ruta_srt_salida]")
        print("Ejemplo: python transcriber.py video.mp4 subtitulos.srt")
        sys.exit(1)
    
    video = sys.argv[1]
    srt_out = sys.argv[2] if len(sys.argv) > 2 else "output.srt"
    
    print("🎙️  Probando transcriptor...")
    print(f"   Video: {video}")
    print(f"   SRT: {srt_out}")
    
    try:
        result = quick_transcribe(video, srt_out, model_size="base")
        print(f"\n✅ Transcripción exitosa!")
        print(f"   Duración: {result['duration']:.1f}s")
        print(f"   Segmentos: {len(result['segments'])}")
        print(f"   Primeros 200 caracteres: {result['text'][:200]}...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
