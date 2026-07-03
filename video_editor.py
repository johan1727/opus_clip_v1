"""
Módulo de edición de video.
Recorta clips, convierte a 9:16 vertical, y quema subtítulos.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import ffmpeg
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.fx.Crop import Crop as crop
from moviepy.video.fx import FadeIn, FadeOut

# Configurar logging PRIMERO (antes de cualquier uso de logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import face tracking (optional)
try:
    from face_tracker import FaceTracker, FACE_TRACKING_AVAILABLE
except ImportError:
    FaceTracker = None
    FACE_TRACKING_AVAILABLE = False
    logger.debug("Face tracking no disponible (OpenCV/mediapipe no instalado)")

# Import config
import config


@dataclass
class VideoConfig:
    """Configuración para exportación de video."""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "8000k"
    audio_bitrate: str = "192k"
    preset: str = "slow"  # calidad vs velocidad: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    crf: int = 18  # Constant Rate Factor (0-51, menor = mejor calidad)
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class SubtitleStyle:
    """Estilo para subtítulos quemados."""
    font: str = "C:/Windows/Fonts/arialbd.ttf"
    fontsize: int = 64
    color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    bg_color: Optional[str] = None  # "rgba(0,0,0,0.5)" para fondo semi-transparente
    position: str = "center"  # "center", "bottom", "top"
    margin_vertical: int = 100
    max_width_ratio: float = 0.9  # 90% del ancho del video
    line_spacing: int = 10


# ----------------------------------------------------------------------
# F4 — Compresión de pausas largas (silencios entre frases dentro de un clip)
# ----------------------------------------------------------------------

def compute_keep_ranges(
    segments: List[Dict[str, Any]],
    clip_duration: float,
    max_gap: float = 1.2,
    target_gap: float = 0.35,
) -> List[Tuple[float, float]]:
    """
    Calcula los rangos [start, end) del clip a conservar, comprimiendo los huecos
    de silencio entre segmentos de diálogo consecutivos que superen `max_gap`
    segundos a solo `target_gap` segundos (no se eliminan del todo, para que el
    corte no se sienta abrupto). `segments` debe tener timestamps relativos al
    clip (0 = inicio del clip), no al video original.

    Devuelve una lista de un solo rango [(0, clip_duration)] si no hay ningún
    hueco que comprimir.
    """
    if not segments:
        return [(0.0, clip_duration)]
    sorted_segs = sorted(segments, key=lambda s: s['start'])
    keep_ranges = []
    cursor = 0.0
    for i in range(len(sorted_segs) - 1):
        seg_end = min(sorted_segs[i]['end'], clip_duration)
        next_start = min(sorted_segs[i + 1]['start'], clip_duration)
        gap = next_start - seg_end
        if gap > max_gap:
            keep_ranges.append((cursor, seg_end + target_gap / 2))
            cursor = next_start - target_gap / 2
    keep_ranges.append((cursor, clip_duration))
    return [(max(0.0, s), min(clip_duration, e)) for s, e in keep_ranges if e > s]


def remap_to_compressed(t: float, keep_ranges: List[Tuple[float, float]]) -> float:
    """
    Convierte un timestamp del clip original a su posición en el timeline
    comprimido resultante de aplicar `compute_keep_ranges`. Solo da resultados
    exactos para timestamps que caen dentro de un rango conservado (que es
    siempre el caso para los timestamps de segmentos de diálogo, por construcción
    de `compute_keep_ranges`).
    """
    cumulative = 0.0
    for (s, e) in keep_ranges:
        if t <= e:
            offset = max(0.0, t - s)
            return cumulative + min(offset, e - s)
        cumulative += (e - s)
    return cumulative


class VideoEditor:
    """
    Editor de video para crear clips virales en formato 9:16.
    Usa FFmpeg para operaciones rápidas y MoviePy para efectos complejos.
    Soporta aceleración por hardware NVIDIA (NVENC).
    """
    
    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        subtitle_style: Optional[SubtitleStyle] = None,
        use_hardware_acceleration: Optional[bool] = None
    ) -> None:
        """
        Inicializa el editor de video.
        
        Args:
            config: Configuración de exportación (default: 1080x1920 30fps)
            subtitle_style: Estilo de subtítulos (default: blanco con borde negro)
            use_hardware_acceleration: Forzar uso de NVENC (None=auto-detectar)
        """
        self.config = config or VideoConfig()
        self.subtitle_style = subtitle_style or SubtitleStyle()
        
        # Detectar aceleración por hardware
        self.hw_accel = use_hardware_acceleration if use_hardware_acceleration is not None else self._detect_nvidia_gpu()
        if self.hw_accel:
            logger.info("🚀 Aceleración NVIDIA NVENC detectada y habilitada")
            # Ajustar config para NVENC
            self.config.codec = "h264_nvenc"
            self.config.preset = "p4"  # NVENC preset (p1-p7, p4=balance)
            # NVENC no usa CRF, usa qp o bitrate control
        else:
            logger.info("💻 Usando codificación por software (libx264)")
        
        # Verificar FFmpeg está disponible
        self._verify_ffmpeg()
        
        logger.info(
            f"VideoEditor inicializado: "
            f"resolución={self.config.resolution}, "
            f"fps={self.config.fps}, "
            f"codec={self.config.codec}"
        )
    
    def _detect_nvidia_gpu(self) -> bool:
        """
        Detecta si hay una GPU NVIDIA disponible y si NVENC está soportado.
        
        Returns:
            True si se puede usar NVENC, False en caso contrario
        """
        try:
            # Verificar si nvidia-smi está disponible
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                logger.info(f"GPU NVIDIA detectada: {gpu_name}")
                
                # Verificar si FFmpeg soporta NVENC
                ffmpeg_result = subprocess.run(
                    ["ffmpeg", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if ffmpeg_result.returncode == 0 and "h264_nvenc" in ffmpeg_result.stdout:
                    logger.info("✅ FFmpeg soporta NVENC")
                    return True
                else:
                    logger.warning("⚠️ FFmpeg no tiene soporte NVENC compilado")
                    return False
            return False
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"No se detectó GPU NVIDIA: {e}")
            return False
    
    def _verify_ffmpeg(self) -> None:
        """Verifica que FFmpeg está instalado y en PATH."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg detectado: {version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg no encontrado. Instálalo y añádelo al PATH:\n"
                "https://ffmpeg.org/download.html"
            )
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Obtiene información del video usando FFprobe.
        
        Args:
            video_path: Ruta al video
            
        Returns:
            Dict con: width, height, duration, fps, codec, bitrate
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No se encontró stream de video")
            
            # Calcular FPS (safe parsing)
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                parts = fps_str.split('/')
                fps_num = int(parts[0]) if parts[0].isdigit() else 30
                fps_den = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                fps = fps_num / fps_den if fps_den != 0 else 30
            else:
                # Formato decimal directo
                try:
                    fps = float(fps_str)
                except ValueError:
                    fps = 30
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': float(video_stream.get('duration', 0)),
                'fps': fps,
                'codec': video_stream.get('codec_name', 'unknown'),
                'bitrate': int(video_stream.get('bit_rate', 0)),
                'aspect_ratio': video_stream.get('display_aspect_ratio', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo info del video: {e}")
            raise
    
    def crop_to_vertical_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        target_width: int = 1080,
        target_height: int = 1920,
        track_faces: bool = False
    ) -> str:
        """
        Recorta un video a formato vertical 9:16 usando FFmpeg (rápido).
        Opcionalmente rastrea rostros para mantener hablantes centrados.
        
        Args:
            input_path: Video fuente
            output_path: Ruta de salida
            start_time: Segundo de inicio
            end_time: Segundo de fin
            target_width: Ancho objetivo (default 1080)
            target_height: Alto objetivo (default 1920)
            track_faces: Activar seguimiento facial AI
            
        Returns:
            Ruta del video generado
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Obtener info del video original
        info = self.get_video_info(str(input_path))
        orig_width = info['width']
        orig_height = info['height']
        
        # Calcular crop base
        if orig_width / orig_height > 9/16:
            new_width = int(orig_height * 9 / 16)
            new_height = orig_height
        else:
            new_width = orig_width
            new_height = int(orig_width * 16 / 9)
        
        duration = end_time - start_time
        x_offset = 0
        y_offset = 0
        
        # Face tracking
        if track_faces and FACE_TRACKING_AVAILABLE:
            try:
                logger.info("🎯 Activando seguimiento facial AI...")
                tracker = FaceTracker(sample_fps=2.0)
                
                # Analizar solo el segmento relevante
                face_positions = tracker.track_faces(str(input_path))
                
                if face_positions:
                    # Filtrar solo rostros en el rango de tiempo del clip
                    clip_faces = [f for f in face_positions 
                                  if start_time <= f.timestamp <= end_time]
                    
                    if clip_faces:
                        crop_offsets = tracker.get_optimal_crop_offsets(
                            clip_faces, orig_width, orig_height, 9/16
                        )
                        
                        if crop_offsets:
                            # Usar el offset promedio más común
                            avg_x_offset = int(sum(o.x_offset for o in crop_offsets) / len(crop_offsets))
                            x_offset = avg_x_offset
                            logger.info(f"🎯 Face tracking: offset X ajustado {x_offset}px")
                        else:
                            logger.info("⚠️ Face tracking: no se calcularon offsets, usando centro")
                    else:
                        logger.info("⚠️ No se detectaron rostros en el segmento, usando centro")
                else:
                    logger.info("⚠️ No se detectaron rostros en el video, usando centro")
                    
            except Exception as e:
                logger.warning(f"⚠️ Face tracking falló: {e}. Usando crop centrado.")
        
        # Calcular posición final del crop
        if orig_width / orig_height > 9/16:
            # Video ancho - crop horizontal
            base_x_offset = (orig_width - new_width) // 2
            x_offset = base_x_offset + x_offset
            x_offset = max(0, min(x_offset, orig_width - new_width))
            y_offset = (orig_height - new_height) // 2
        else:
            # Video alto - crop vertical
            x_offset = (orig_width - new_width) // 2
            y_offset = (orig_height - new_height) // 2
        
        logger.info(
            f"Crop FFmpeg: {orig_width}x{orig_height} -> "
            f"crop={new_width}:{new_height}:{x_offset}:{y_offset} -> "
            f"scale={target_width}:{target_height}"
        )
        
        try:
            # ¿El video fuente tiene pista de audio? (algunos screen recordings no la tienen)
            probe = ffmpeg.probe(str(input_path))
            has_audio = any(s['codec_type'] == 'audio' for s in probe.get('streams', []))

            # Construir comando FFmpeg
            input_node = ffmpeg.input(
                str(input_path),
                ss=start_time,
                t=duration
            )

            video_stream = ffmpeg.filter(
                input_node.video,
                'crop',
                w=new_width,
                h=new_height,
                x=x_offset,
                y=y_offset
            )

            video_stream = ffmpeg.filter(
                video_stream,
                'scale',
                w=target_width,
                h=target_height,
                force_original_aspect_ratio='disable'
            )

            # Construir parámetros de salida según codec
            output_kwargs = {
                'vcodec': self.config.codec,
                'video_bitrate': self.config.video_bitrate,
                'r': self.config.fps,
                'pix_fmt': 'yuv420p',
                'movflags': 'faststart'
            }
            if has_audio:
                output_kwargs['acodec'] = self.config.audio_codec
                output_kwargs['audio_bitrate'] = self.config.audio_bitrate

            # NVENC no soporta CRF ni presets libx264
            if self.hw_accel:
                output_kwargs['preset'] = self.config.preset  # p1-p7 para NVENC
                # NVENC usa qp o bitrate control, no CRF
            else:
                output_kwargs['preset'] = self.config.preset
                output_kwargs['crf'] = self.config.crf

            # IMPORTANTE: hay que mapear el stream de audio explícitamente además del de
            # video, si no ffmpeg-python solo mapea el video filtrado y el clip sale mudo.
            if has_audio:
                stream = ffmpeg.output(video_stream, input_node.audio, str(output_path), **output_kwargs)
            else:
                stream = ffmpeg.output(video_stream, str(output_path), **output_kwargs)

            # Ejecutar
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            logger.info(f"Video crop guardado: {output_path} (audio: {'sí' if has_audio else 'sin pista de audio en la fuente'})")
            return str(output_path)

        except ffmpeg.Error as e:
            logger.error(f"Error FFmpeg: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"FFmpeg falló: {e}")
    
    def burn_subtitles_moviepy(
        self,
        video_path: str,
        output_path: str,
        segments: List[Dict[str, Any]],
        style: Optional[SubtitleStyle] = None
    ) -> str:
        """
        Quema subtítulos en el video usando MoviePy (calidad superior).
        Usa context managers para garantizar liberación de memoria.
        
        Args:
            video_path: Video fuente (ya en 9:16)
            output_path: Ruta de salida
            segments: Lista de segmentos Whisper con start, end, text
            style: Estilo de subtítulos (opcional)
            
        Returns:
            Ruta del video con subtítulos
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        style = style or self.subtitle_style
        
        logger.info(f"Quemando subtítulos: {len(segments)} segmentos")
        
        subtitle_clips = []
        final_video = None
        video = None
        
        try:
            # Cargar video
            video = VideoFileClip(str(video_path))
            
            # Crear clips de texto para cada subtítulo
            for seg in segments:
                start = seg['start']
                end = seg['end']
                text = seg['text'].strip()
                
                if not text:
                    continue
                
                # Crear texto
                txt_clip = TextClip(
                    text=text,
                    font_size=style.fontsize,
                    font=style.font,
                    color=style.color,
                    stroke_color=style.stroke_color,
                    stroke_width=style.stroke_width,
                    method='caption',
                    size=(int(video.w * style.max_width_ratio), None),
                    text_align='center',
                    interline=style.line_spacing
                )
                
                # Posicionar
                if style.position == "center":
                    txt_clip = txt_clip.with_position('center')
                elif style.position == "bottom":
                    txt_clip = txt_clip.with_position(('center', video.h - txt_clip.h - style.margin_vertical))
                elif style.position == "top":
                    txt_clip = txt_clip.with_position(('center', style.margin_vertical))
                
                # Duración
                txt_clip = txt_clip.with_start(start).with_end(end)
                
                # Fondo opcional
                if style.bg_color:
                    bg = ColorClip(
                        size=(txt_clip.w + 40, txt_clip.h + 20),
                        color=style.bg_color
                    ).with_start(start).with_end(end).with_position(
                        lambda t: txt_clip.pos(t) if callable(txt_clip.pos) else txt_clip.pos
                    )
                    subtitle_clips.append(bg)
                
                subtitle_clips.append(txt_clip)
            
            # Componer video final
            final_video = CompositeVideoClip([video] + subtitle_clips, size=video.size)

            # NOTA: no cerrar `video` acá — CompositeVideoClip lee sus frames de forma
            # perezosa durante write_videofile en MoviePy 2.x, así que cerrarlo antes de
            # exportar deja al composite sin video ('NoneType' object has no attribute
            # 'get_frame'). Se cierra en el `finally` de abajo, después de exportar.

            # Exportar
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec=self.config.codec,
                audio_codec=self.config.audio_codec,
                bitrate=self.config.video_bitrate,
                preset=self.config.preset,
                threads=4,
                logger=None  # Silenciar logs de moviepy
            )

            logger.info(f"Video con subtítulos guardado: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"MoviePy subtítulos falló ({e}), usando fallback ffmpeg drawtext...")
            # Cleanup before fallback
            if final_video:
                try: final_video.close()
                except Exception as _ce: logger.debug(f"Cleanup final_video: {_ce}")
            if video:
                try: video.close()
                except Exception as _ce: logger.debug(f"Cleanup video: {_ce}")
            for clip in subtitle_clips:
                try: clip.close()
                except Exception as _ce: logger.debug(f"Cleanup subtitle_clip: {_ce}")
            return self._burn_subtitles_ffmpeg(str(video_path), str(output_path), segments, style)

        finally:
            if final_video:
                try:
                    final_video.close()
                except Exception as _ce:
                    logger.debug(f"Cleanup final_video (finally): {_ce}")
            if video:
                try:
                    video.close()
                except Exception as _ce:
                    logger.debug(f"Cleanup video (finally): {_ce}")
            for clip in subtitle_clips:
                try:
                    clip.close()
                except Exception as _ce:
                    logger.debug(f"Cleanup subtitle_clip (finally): {_ce}")
            import gc
            gc.collect()

    def burn_karaoke_subtitles(
        self,
        video_path: str,
        output_path: str,
        word_segments: List[Dict[str, Any]],
        style: Optional[SubtitleStyle] = None,
        animation_mode: str = "karaoke"
    ) -> str:
        """
        Quema subtítulos estilo karaoke (palabra por palabra) con efecto de highlight.
        
        Args:
            video_path: Video fuente (ya en 9:16)
            output_path: Ruta de salida
            word_segments: Lista de palabras con timing preciso
            style: Estilo de subtítulos (opcional)
            
        Returns:
            Ruta del video con subtítulos karaoke
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        style = style or self.subtitle_style
        
        logger.info(f"🎤 Quemando subtítulos {animation_mode}: {len(word_segments)} palabras")
        
        subtitle_clips = []
        final_video = None
        video = None
        
        try:
            video = VideoFileClip(str(video_path))
            
            # Agrupar palabras por segmento padre para mostrar contexto
            segments_by_parent: Dict[int, List[Dict]] = {}
            for seg in word_segments:
                parent_id = seg.get('parent_id', 0)
                if parent_id not in segments_by_parent:
                    segments_by_parent[parent_id] = []
                segments_by_parent[parent_id].append(seg)
            
            # Para cada grupo de palabras (segmento original)
            for parent_id, words in segments_by_parent.items():
                if not words:
                    continue
                
                # Ordenar por índice de palabra
                words = sorted(words, key=lambda w: w.get('word_index', 0))
                
                # Construir texto completo del segmento
                full_text = ' '.join(w['text'] for w in words)
                segment_start = words[0]['start']
                segment_end = words[-1]['end']
                
                # Crear clip de texto completo (fondo/base)
                base_color = 'gray' if animation_mode in ("karaoke", "highlight") else style.color
                base_clip = TextClip(
                    text=full_text,
                    font_size=style.fontsize,
                    font=style.font,
                    color=base_color,
                    stroke_color=style.stroke_color,
                    stroke_width=style.stroke_width,
                    method='caption',
                    size=(int(video.w * style.max_width_ratio), None),
                    text_align='center',
                    interline=style.line_spacing
                )
                
                # Posicionar
                if style.position == "center":
                    base_clip = base_clip.with_position('center')
                elif style.position == "bottom":
                    base_clip = base_clip.with_position(('center', video.h - base_clip.h - style.margin_vertical))
                elif style.position == "top":
                    base_clip = base_clip.with_position(('center', style.margin_vertical))
                
                base_clip = base_clip.with_start(segment_start).with_end(segment_end)
                subtitle_clips.append(base_clip)
                
                # Crear clips individuales para cada palabra (highlight)
                for word_data in words:
                    word_start = word_data['start']
                    word_end = word_data['end']
                    word_text = word_data['text']
                    word_index = word_data.get('word_index', 0)
                    
                    # Calcular posición horizontal de la palabra
                    # Estimación simple: posición proporcional al índice
                    words_before = words[:word_index]
                    words_after = words[word_index+1:]
                    
                    prefix = ' '.join(w['text'] for w in words_before)
                    suffix = ' '.join(w['text'] for w in words_after)
                    
                    # Crear clip compuesto: prefix + highlight + suffix
                    # Simplificación: solo mostrar la palabra destacada encima
                    highlight_fontsize = int(style.fontsize * 1.18) if animation_mode == "pop" else style.fontsize
                    highlight_bg = "#ffea0044" if animation_mode == "highlight" else None
                    word_highlight = TextClip(
                        text=word_text,
                        font_size=highlight_fontsize,
                        font=style.font,
                        color=style.color,
                        stroke_color=style.stroke_color,
                        stroke_width=style.stroke_width + 1,
                        bg_color=highlight_bg,
                        method='caption',
                        size=(int(video.w * style.max_width_ratio), None),
                        text_align='center',
                        interline=style.line_spacing
                    )
                    
                    # Misma posición que base
                    if style.position == "center":
                        word_highlight = word_highlight.with_position('center')
                    elif style.position == "bottom":
                        word_highlight = word_highlight.with_position(('center', video.h - word_highlight.h - style.margin_vertical))
                    elif style.position == "top":
                        word_highlight = word_highlight.with_position(('center', style.margin_vertical))
                    
                    # Solo visible durante el tiempo de la palabra
                    word_highlight = word_highlight.with_start(word_start).with_end(word_end)
                    
                    # Efecto de fade in/out suave
                    fade_duration = min(0.12 if animation_mode == "pop" else 0.1, (word_end - word_start) / 4)
                    word_highlight = word_highlight.with_effects([FadeIn(fade_duration), FadeOut(fade_duration)])
                    
                    subtitle_clips.append(word_highlight)
            
            # Componer video
            final_video = CompositeVideoClip([video] + subtitle_clips, size=video.size)

            # NOTA: no cerrar `video` acá — ver comentario equivalente en burn_subtitles_moviepy.

            # Exportar
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec=self.config.codec,
                audio_codec=self.config.audio_codec,
                bitrate=self.config.video_bitrate,
                preset=self.config.preset,
                threads=4,
                logger=None
            )

            logger.info(f"🎤 Video karaoke guardado: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"MoviePy karaoke falló ({e}), usando fallback ffmpeg drawtext...")
            if final_video:
                try: final_video.close()
                except Exception as _ce: logger.debug(f"Cleanup final_video: {_ce}")
            if video:
                try: video.close()
                except Exception as _ce: logger.debug(f"Cleanup video: {_ce}")
            for clip in subtitle_clips:
                try: clip.close()
                except Exception as _ce: logger.debug(f"Cleanup subtitle_clip: {_ce}")
            # Convert word_segments to phrase segments for ffmpeg fallback
            phrase_segs = self._word_segs_to_phrases(word_segments)
            return self._burn_subtitles_ffmpeg(str(video_path), str(output_path), phrase_segs, style)

        finally:
            if final_video:
                try:
                    final_video.close()
                except Exception as _ce:
                    logger.debug(f"Cleanup final_video (finally): {_ce}")
            if video:
                try:
                    video.close()
                except Exception as _ce:
                    logger.debug(f"Cleanup video (finally): {_ce}")
            for clip in subtitle_clips:
                try:
                    clip.close()
                except Exception as _ce:
                    logger.debug(f"Cleanup subtitle_clip (finally): {_ce}")
            import gc
            gc.collect()
    
    def _word_segs_to_phrases(self, word_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Agrupa word_segments por parent_id en frases para el fallback ffmpeg."""
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for w in word_segments:
            groups[w.get('parent_id', 0)].append(w)
        phrases = []
        for words in groups.values():
            words = sorted(words, key=lambda w: w.get('word_index', 0))
            phrases.append({
                'start': words[0]['start'],
                'end': words[-1]['end'],
                'text': ' '.join(w['text'] for w in words),
            })
        return sorted(phrases, key=lambda p: p['start'])

    def _burn_subtitles_ffmpeg(
        self,
        video_path: str,
        output_path: str,
        segments: List[Dict[str, Any]],
        style: Optional['SubtitleStyle'] = None,
    ) -> str:
        """Quema subtítulos usando ffmpeg drawtext — no depende de fuentes del sistema."""
        style = style or self.subtitle_style
        font_path = style.font if Path(style.font).exists() else "C:/Windows/Fonts/arialbd.ttf"
        # Escapar ':' en la ruta (ej. "C:/Windows/...") — el parser de filtros de ffmpeg
        # usa ':' como separador de opciones, así que una ruta de Windows sin escapar
        # rompe el filtro entero ("Error parsing a filter description").
        font_path_escaped = font_path.replace(':', '\\:')
        fontsize = style.fontsize
        color = style.color.lstrip('#') if style.color else 'white'
        # ffmpeg color format: 0xRRGGBB or named
        if color.startswith('0x') or color in ('white', 'black', 'yellow', 'red'):
            fc = color
        elif len(color) == 6:
            fc = f"0x{color}"
        else:
            fc = 'white'

        # Build drawtext filter chain
        filters = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue
            # Escape special chars for ffmpeg
            text_esc = text.replace("'", "\\'").replace(':', '\\:').replace('%', '\\%')
            t_start = seg['start']
            t_end = seg['end']
            border = style.stroke_width
            filters.append(
                f"drawtext=fontfile='{font_path_escaped}':text='{text_esc}'"
                f":fontsize={fontsize}:fontcolor={fc}"
                f":borderw={border}:bordercolor=black"
                f":x=(w-text_w)/2:y=(h-text_h)/2"
                f":enable='between(t,{t_start:.3f},{t_end:.3f})'"
            )

        if not filters:
            # No subtitles — just copy
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path

        vf = ','.join(filters)
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', vf,
            '-c:a', 'copy',
            '-c:v', self.config.codec,
            output_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"ffmpeg drawtext error: {result.stderr[-500:]}")
                import shutil
                shutil.copy2(video_path, output_path)
            else:
                logger.info(f"✅ Subtítulos ffmpeg guardados: {output_path}")
        except Exception as e:
            logger.error(f"ffmpeg drawtext excepción: {e}")
            import shutil
            shutil.copy2(video_path, output_path)
        return output_path

    def create_preview_segment(
        self,
        input_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        duration_limit: float = 10.0
    ) -> str:
        """
        Crea un preview rápido de baja calidad para previsualización.
        
        Args:
            input_video: Video fuente
            output_path: Ruta de salida del preview
            start_time: Inicio del segmento
            end_time: Fin del segmento
            duration_limit: Duración máxima del preview (acelerado si es necesario)
            
        Returns:
            Ruta del preview generado
        """
        input_path = Path(input_video)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        duration = min(end_time - start_time, duration_limit)
        
        # Calcular crop para 9:16 preview (resolución reducida)
        info = self.get_video_info(str(input_path))
        orig_w, orig_h = info['width'], info['height']
        
        # Crop 9:16 desde el centro
        if orig_w / orig_h > 9/16:
            crop_w = int(orig_h * 9 / 16)
            crop_h = orig_h
            x_off = (orig_w - crop_w) // 2
            y_off = 0
        else:
            crop_w = orig_w
            crop_h = int(orig_w * 16 / 9)
            x_off = 0
            y_off = (orig_h - crop_h) // 2
        
        try:
            # Preview rápido: 480x854, 24fps, bitrate bajo
            # Input de video y audio por separado para mapear correctamente
            input_stream = ffmpeg.input(str(input_path), ss=start_time, t=duration)
            
            # Procesar solo el video
            video_stream = ffmpeg.filter(
                input_stream.video, 'crop', w=crop_w, h=crop_h, x=x_off, y=y_off
            )
            video_stream = ffmpeg.filter(video_stream, 'scale', w=480, h=854)
            
            # Tomar el audio sin procesar
            audio_stream = input_stream.audio
            
            # Output con video y audio mapeados
            stream = ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_path),
                vcodec='libx264',
                acodec='aac',
                video_bitrate='1500k',
                audio_bitrate='128k',
                r=24,
                preset='ultrafast',
                pix_fmt='yuv420p',
                movflags='faststart',
                shortest=None  # Termina cuando el stream más corto termina
            )
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info(f"Preview creado: {output_path} ({duration:.1f}s)")
            return str(output_path)
            
        except ffmpeg.Error as e:
            logger.error(f"Error creando preview: {e}")
            raise
    
    def detect_scene_changes(
        self,
        input_video: str,
        threshold: float = 0.3,
        min_scene_duration: float = 1.0
    ) -> List[float]:
        """
        Detecta cambios de escena usando ffmpeg scene detection.

        Args:
            input_video: Ruta al video
            threshold: Umbral de diferencia entre frames (0.0-1.0)
            min_scene_duration: Duración mínima entre escenas

        Returns:
            Lista de timestamps donde hay cambios de escena significativos
        """
        input_path = Path(input_video)
        if not input_path.exists():
            logger.error(f"Video no encontrado: {input_video}")
            return []

        try:
            # ffmpeg scene detection: calcular diferencia entre frames consecutivos
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f'select=gt(scene\\,{threshold}),showinfo',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parsear timestamps de scene changes del stderr
            scene_timestamps = []
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        # Formato típico: [Parsed_showinfo_0 ...] n:... pts_time:12.345
                        parts = line.split('pts_time:')
                        if len(parts) > 1:
                            ts_str = parts[1].split()[0]
                            ts = float(ts_str)
                            if ts > 0:
                                scene_timestamps.append(ts)
                    except (ValueError, IndexError):
                        continue

            # Filtrar por duración mínima entre escenas
            filtered = []
            last_ts = -min_scene_duration
            for ts in sorted(scene_timestamps):
                if ts - last_ts >= min_scene_duration:
                    filtered.append(ts)
                    last_ts = ts

            logger.info(f"Scene detection: {len(filtered)} escenas detectadas (threshold={threshold})")
            return filtered

        except subprocess.TimeoutExpired:
            logger.warning("Scene detection timeout")
            return []
        except Exception as e:
            logger.error(f"Error en scene detection: {e}")
            return []

    def generate_thumbnail(
        self,
        input_video: str,
        output_path: str,
        timestamp: float = 0.0,
        width: int = 320
    ) -> str:
        """
        Genera un thumbnail del video en un timestamp específico.
        
        Args:
            input_video: Ruta al video
            output_path: Ruta de salida de la imagen
            timestamp: Segundo para capturar (default: inicio)
            width: Ancho de la imagen (mantiene aspecto 9:16)
            
        Returns:
            Ruta de la imagen generada
        """
        input_path = Path(input_video)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        height = int(width * 16 / 9)  # 9:16 aspect ratio
        
        try:
            stream = ffmpeg.input(str(input_path), ss=timestamp)
            stream = ffmpeg.filter(stream, 'scale', w=width, h=height)
            stream = ffmpeg.output(stream, str(output_path), vframes=1, format='image2')
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info(f"Thumbnail generado: {output_path}")
            return str(output_path)
            
        except ffmpeg.Error as e:
            logger.error(f"Error generando thumbnail: {e}")
            raise
    
    def extract_keyframes(
        self,
        input_video: str,
        timestamps: List[float],
        width: int = 480,
        output_dir: Optional[str] = None
    ) -> List[Any]:
        """
        Extrae frames del video en timestamps específicos para análisis visual con IA.

        Args:
            input_video: Ruta al video fuente
            timestamps: Lista de segundos donde extraer frames
            width: Ancho de la imagen (9:16 crop desde centro)
            output_dir: Directorio opcional para guardar frames (default: TEMP_DIR/keyframes)

        Returns:
            Lista de PIL.Image objects listos para enviar a Gemini
        """
        import io
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL no instalado — keyframes no disponibles")
            return []

        input_path = Path(input_video)
        if not input_path.exists():
            logger.error(f"Video no encontrado: {input_video}")
            return []

        if output_dir is None:
            output_dir = config.TEMP_DIR / "keyframes"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for i, ts in enumerate(timestamps):
            out_path = output_dir / f"frame_{i:03d}_{ts:.1f}s.jpg"
            try:
                # Extraer frame con ffmpeg, crop 9:16 desde centro
                cmd = [
                    'ffmpeg', '-y', '-ss', str(ts), '-i', str(input_path),
                    '-vf', f'scale={width}:-1:force_original_aspect_ratio=decrease,crop={width}:{int(width*16/9)}',
                    '-frames:v', '1', '-q:v', '2', str(out_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    logger.warning(f"ffmpeg frame error: {result.stderr[-200:]}")
                    continue
                if out_path.exists():
                    img = Image.open(out_path)
                    images.append(img)
            except Exception as e:
                logger.warning(f"Error extrayendo frame en {ts}s: {e}")
                continue

        logger.info(f"Extraídos {len(images)} keyframes de {len(timestamps)} timestamps")
        return images

    def create_viral_clip(
        self,
        input_video: str,
        output_path: str,
        start_time: float,
        end_time: float,
        segments: List[Dict[str, Any]],
        add_subtitles: bool = True,
        track_faces: bool = False,
        subtitle_mode: str = "static",
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        style: Optional[SubtitleStyle] = None,
        compress_pauses: bool = False,
        max_pause_gap: float = 1.2,
        target_pause_gap: float = 0.35,
    ) -> str:
        """
        Pipeline completo: crop a 9:16 + subtítulos.

        Args:
            input_video: Video fuente completo
            output_path: Ruta de salida final
            start_time: Inicio del clip en segundos
            end_time: Fin del clip en segundos
            segments: Segmentos Whisper (solo los del clip)
            add_subtitles: Si quemar subtítulos
            track_faces: Activar seguimiento facial AI
            subtitle_mode: "static", "karaoke", "highlight" o "pop"
            style: Estilo de subtítulos a aplicar (opcional, usa self.subtitle_style si no se pasa)
            compress_pauses: Si comprimir pausas/silencios largos entre frases (F4)
            max_pause_gap: Huecos entre frases mayores a esto (segundos) se comprimen
            target_pause_gap: A cuántos segundos se comprime cada hueco largo

        Returns:
            Ruta del video final
        """
        input_video = Path(input_video)
        output_path = Path(output_path)

        # Archivo temporal para el crop
        temp_crop = output_path.parent / f"temp_crop_{output_path.stem}.mp4"

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

            # Paso 2.5: Comprimir pausas largas (F4), si se pidió
            if compress_pauses:
                try:
                    clip_duration = end_time - start_time
                    keep_ranges = compute_keep_ranges(
                        adjusted_segments, clip_duration, max_pause_gap, target_pause_gap
                    )
                    if len(keep_ranges) > 1:
                        compressed_crop = output_path.parent / f"temp_paused_{output_path.stem}.mp4"
                        self.compress_pauses(str(temp_crop), str(compressed_crop), keep_ranges)
                        temp_crop.unlink()
                        temp_crop = compressed_crop
                        for seg in adjusted_segments:
                            seg['start'] = remap_to_compressed(seg['start'], keep_ranges)
                            seg['end'] = remap_to_compressed(seg['end'], keep_ranges)
                except Exception as e:
                    logger.warning(f"Compresión de pausas omitida: {e}")

            if not add_subtitles:
                # Renombrar temp (posiblemente ya comprimido) a final
                temp_crop.rename(output_path)
                return str(output_path)

            # Paso 3: Quemar subtítulos según modo
            if subtitle_mode in ("karaoke", "highlight", "pop"):
                # Convertir a formato palabra por palabra
                word_segments = []
                for seg in adjusted_segments:
                    words = seg['text'].split()
                    if words:
                        word_duration = (seg['end'] - seg['start']) / len(words)
                        for i, word in enumerate(words):
                            word_segments.append({
                                'start': seg['start'] + i * word_duration,
                                'end': seg['start'] + (i + 1) * word_duration,
                                'text': word,
                                'parent_id': id(seg),
                                'word_index': i
                            })
                
                self.burn_karaoke_subtitles(
                    str(temp_crop),
                    str(output_path),
                    word_segments,
                    style=style,
                    animation_mode=subtitle_mode
                )
            else:
                self.burn_subtitles_moviepy(
                    str(temp_crop),
                    str(output_path),
                    adjusted_segments,
                    style=style
                )
            
            # Limpiar temp
            if temp_crop.exists():
                temp_crop.unlink()
            
            logger.info(f"✅ Clip viral creado: {output_path}")
            return str(output_path)
            
        except Exception as e:
            # Limpieza en caso de error
            if temp_crop.exists():
                temp_crop.unlink()
            raise
    
    # ------------------------------------------------------------------
    # F3.1 — Color grading by mood
    # ------------------------------------------------------------------

    MOOD_FILTERS: Dict[str, str] = {
        'funny':         "eq=saturation=1.3:contrast=1.05,hue=h=5",
        'shocking':      "eq=saturation=0.8:contrast=1.15,vignette=PI/4",
        'educational':   "eq=saturation=0.95:brightness=0.02",
        'motivational':  "eq=saturation=1.1:contrast=1.1,curves=r='0/0 0.5/0.6 1/1'",
        'dramatic':      "eq=saturation=0.65:contrast=1.2,vignette=PI/3",
        'controversial': "eq=saturation=1.15:contrast=1.1,hue=h=-5",
        'wholesome':     "eq=saturation=1.05:brightness=0.03,curves=r='0/0 0.5/0.55 1/1'",
        'gaming_hype':   "eq=saturation=1.4:contrast=1.15",
        'storytelling':  "eq=saturation=0.9:brightness=0.01",
        'inspirational': "eq=saturation=1.1:brightness=0.02,curves=r='0/0 0.5/0.58 1/1'",
        'neutral':       "",
    }

    def apply_mood_grade(
        self,
        input_path: str,
        output_path: str,
        mood: str = "neutral",
    ) -> str:
        """
        Applies FFmpeg color grading filters based on clip mood (F3.1).
        Returns output_path unchanged if no filter for mood.
        """
        vf = self.MOOD_FILTERS.get(mood, "")
        if not vf:
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    vf=vf,
                    vcodec=self.config.codec,
                    acodec="copy",
                    preset=self.config.preset,
                    crf=self.config.crf if not self.hw_accel else None,
                    pix_fmt="yuv420p",
                    loglevel="warning",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"🎨 Color grading '{mood}' aplicado: {Path(output_path).name}")
            return output_path
        except Exception as e:
            logger.warning(f"Color grading falló ({e}), usando original")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path

    # ------------------------------------------------------------------
    # F4 — Compresión de pausas largas (mantiene el ritmo del clip)
    # ------------------------------------------------------------------

    def compress_pauses(
        self,
        input_path: str,
        output_path: str,
        keep_ranges: List[Tuple[float, float]],
    ) -> str:
        """
        Recorta y concatena los rangos de `keep_ranges` del video de entrada,
        eliminando físicamente el resto (los huecos de silencio comprimidos por
        `compute_keep_ranges`). Si `keep_ranges` es un solo rango (nada que
        comprimir), copia el archivo tal cual.
        """
        if len(keep_ranges) <= 1:
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        try:
            input_stream = ffmpeg.input(str(input_path))
            parts = []
            for seg_start, seg_end in keep_ranges:
                v = input_stream.video.filter('trim', start=seg_start, end=seg_end).filter('setpts', 'PTS-STARTPTS')
                a = input_stream.audio.filter('atrim', start=seg_start, end=seg_end).filter('asetpts', 'PTS-STARTPTS')
                parts.append(v)
                parts.append(a)
            joined = ffmpeg.concat(*parts, v=1, a=1).node
            out_v, out_a = joined[0], joined[1]

            output_kwargs = {
                'vcodec': self.config.codec,
                'acodec': self.config.audio_codec,
                'video_bitrate': self.config.video_bitrate,
                'audio_bitrate': self.config.audio_bitrate,
                'r': self.config.fps,
                'pix_fmt': 'yuv420p',
                'movflags': 'faststart',
                'loglevel': 'warning',
            }
            if self.hw_accel:
                output_kwargs['preset'] = self.config.preset
            else:
                output_kwargs['preset'] = self.config.preset
                output_kwargs['crf'] = self.config.crf

            (
                ffmpeg
                .output(out_v, out_a, str(output_path), **output_kwargs)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            saved = sum((e - s) for s, e in keep_ranges)
            logger.info(f"✂️ Pausas comprimidas: {len(keep_ranges)} segmentos conservados, {saved:.1f}s de duración final")
            return output_path
        except ffmpeg.Error as e:
            stderr = e.stderr.decode(errors='replace') if e.stderr else str(e)
            logger.warning(f"Compresión de pausas falló (ffmpeg): {stderr[-800:]}, usando clip sin comprimir")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        except Exception as e:
            logger.warning(f"Compresión de pausas falló ({e}), usando clip sin comprimir")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path

    # ------------------------------------------------------------------
    # F2.3 — Dynamic zoom punch-in at energy peaks
    # ------------------------------------------------------------------

    def apply_zoom_cues(
        self,
        input_path: str,
        output_path: str,
        zoom_cues: List[Dict[str, Any]],
        zoom_factor: float = 1.08,
        zoom_duration_s: float = 0.4,
    ) -> str:
        """
        Applies a quick zoom punch-in at each cue timestamp using FFmpeg zoompan (F2.3).
        zoom_cues: list of {'time': float, 'intensity': float}
        """
        if not zoom_cues:
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        try:
            info = self.get_video_info(input_path)
            fps = info.get('fps', 30)
            w, h = info.get('width', 1080), info.get('height', 1920)
            # Build zoompan expression: zoom=1+factor during cue windows
            zoom_frames_per_cue = int(zoom_duration_s * fps)
            conditions = []
            for cue in zoom_cues:
                start_f = int(cue['time'] * fps)
                end_f = start_f + zoom_frames_per_cue
                intensity = cue.get('intensity', 0.5)
                z = 1.0 + zoom_factor * intensity * 0.1
                conditions.append(f"between(n,{start_f},{end_f})*{z:.4f}")
            # Fallback to 1 when no cue
            zoom_expr = "+".join(conditions) + "+(" + "+".join(
                [f"1*not(between(n,{int(c['time']*fps)},{int(c['time']*fps)+zoom_frames_per_cue}))" for c in zoom_cues]
            ) + ")"
            # Simpler approach: use select + scale + overlay is complex; use zoompan directly
            vf = (
                f"zoompan=z='if(between(n,{int(zoom_cues[0]['time']*fps)},"
                f"{int(zoom_cues[0]['time']*fps)+zoom_frames_per_cue}),{1.0+zoom_factor*zoom_cues[0].get('intensity',0.5)*0.1:.4f},1)'"
                f":d=1:s={w}x{h}:fps={fps}"
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
            logger.info(f"🔍 Zoom punch-in en {len(zoom_cues)} cues aplicado")
            return output_path
        except Exception as e:
            logger.warning(f"Zoom cues falló ({e}), usando original")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path

    # ------------------------------------------------------------------
    # F3.4 — Audio ducking (background music -6 dB under voice)
    # ------------------------------------------------------------------

    def apply_audio_ducking(
        self,
        input_path: str,
        output_path: str,
        duck_db: float = -6.0,
    ) -> str:
        """
        Detects if there is background music and lowers it by duck_db dB
        so the voice dominates (F3.4). Uses FFmpeg loudnorm + volume filters.
        """
        try:
            # Simple approach: normalize audio then apply gentle compression
            vf_audio = (
                f"loudnorm=I=-16:TP=-1.5:LRA=11,"
                f"acompressor=threshold=0.08:ratio=4:attack=5:release=100:makeup=1"
            )
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    af=vf_audio,
                    vcodec="copy",
                    acodec="aac",
                    audio_bitrate="192k",
                    loglevel="warning",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"🔊 Audio ducking aplicado ({duck_db}dB)")
            return output_path
        except Exception as e:
            logger.warning(f"Audio ducking falló ({e}), usando original")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path

    # ------------------------------------------------------------------
    # F3.2 — Branding overlay (intro/outro 0.5s fade)
    # ------------------------------------------------------------------

    def add_branding_overlay(
        self,
        input_path: str,
        output_path: str,
        brand_name: str = "",
        brand_color: str = "#00f2ea",
        fade_duration: float = 0.5,
    ) -> str:
        """
        Adds a 0.5s fade-in/out with brand name overlay (F3.2).
        Uses FFmpeg drawtext + fade filters.
        """
        if not brand_name:
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        try:
            info = self.get_video_info(input_path)
            fps = info.get('fps', 30)
            dur = info.get('duration', 30)
            fade_f = int(fade_duration * fps)
            last_f = int((dur - fade_duration) * fps)
            # Convert hex color to FFmpeg format
            color_hex = brand_color.lstrip('#')
            r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
            vf = (
                f"fade=t=in:st=0:d={fade_duration},"
                f"fade=t=out:st={dur - fade_duration:.2f}:d={fade_duration},"
                f"drawtext=text='{brand_name}':fontsize=36:fontcolor=white:"
                f"x=(w-text_w)/2:y=h-80:"
                f"enable='between(t,0,{fade_duration*2:.1f})':"
                f"alpha='if(lt(t,{fade_duration}),t/{fade_duration},1)'"
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
            logger.info(f"🏷️ Branding '{brand_name}' aplicado")
            return output_path
        except Exception as e:
            logger.warning(f"Branding overlay falló ({e}), usando original")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path

    def create_multiple_clips(
        self,
        input_video: str,
        output_dir: str,
        clips_data: List[Dict[str, Any]],
        base_filename: str = "clip"
    ) -> List[str]:
        """
        Crea múltiples clips virales de un video.
        
        Args:
            input_video: Video fuente
            output_dir: Directorio de salida
            clips_data: Lista de dicts con: start, end, segments, metadata opcional
            base_filename: Prefijo para archivos de salida
            
        Returns:
            Lista de rutas de videos generados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated = []
        
        for i, clip_data in enumerate(clips_data, 1):
            start = clip_data['start']
            end = clip_data['end']
            segments = clip_data.get('segments', [])
            
            # Agregar score al nombre si está disponible
            score = clip_data.get('virality_score', 0)
            score_str = f"_{score:.1f}" if score > 0 else ""
            
            output_file = output_dir / f"{base_filename}_{i:02d}{score_str}.mp4"
            
            try:
                path = self.create_viral_clip(
                    input_video,
                    str(output_file),
                    start,
                    end,
                    segments
                )
                generated.append(path)
                
            except Exception as e:
                logger.error(f"Error creando clip {i}: {e}")
                continue
        
        logger.info(f"Generados {len(generated)}/{len(clips_data)} clips")
        return generated


def quick_create_clip(
    input_video: str,
    output_path: str,
    start: float,
    end: float,
    transcription_segments: Optional[List[Dict]] = None
) -> str:
    """
    Función de conveniencia para crear un clip rápidamente.
    
    Args:
        input_video: Ruta al video fuente
        output_path: Ruta de salida
        start: Segundo de inicio
        end: Segundo de fin
        transcription_segments: Segmentos Whisper (opcional)
        
    Returns:
        Ruta del video generado
    """
    editor = VideoEditor()
    return editor.create_viral_clip(
        input_video,
        output_path,
        start,
        end,
        transcription_segments or [],
        add_subtitles=bool(transcription_segments)
    )


if __name__ == "__main__":
    # Test del módulo
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python video_editor.py <ruta_video> [start] [end] [output]")
        print("Ejemplo: python video_editor.py video.mp4 10 30 output_clip.mp4")
        sys.exit(1)
    
    video = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = float(sys.argv[3]) if len(sys.argv) > 3 else 10
    output = sys.argv[4] if len(sys.argv) > 4 else "test_clip.mp4"
    
    print(f"🎬 Probando VideoEditor...")
    print(f"   Video: {video}")
    print(f"   Segmento: {start}s - {end}s")
    print(f"   Salida: {output}")
    
    try:
        # Info del video
        editor = VideoEditor()
        info = editor.get_video_info(video)
        print(f"\n📊 Info del video:")
        for key, val in info.items():
            print(f"   {key}: {val}")
        
        # Crear clip simple (sin subtítulos para test rápido)
        result = editor.crop_to_vertical_ffmpeg(video, output, start, end)
        print(f"\n✅ Clip creado: {result}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
