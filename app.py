"""
OpusClip Pro - UI Profesional mejorada
Diseño moderno tipo OpusClip/CapCut con tema oscuro elegante
"""

import logging
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr

from config import config, SUPPORTED_VIDEO_FORMATS, ERRORS, SUCCESS, SUBTITLE_STYLES
from transcriber import Transcriber
from llm_analyzer import GeminiAnalyzer, ViralClip
from video_editor import VideoEditor, FACE_TRACKING_AVAILABLE, SubtitleStyle, compute_keep_ranges, remap_to_compressed
from state_manager import StateManager, ProjectState, ClipState
from subtitle_editor import SubtitleEditor, create_editor_from_transcription, PREDEFINED_STYLES
from audio_analyzer import AudioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpusClipPro:
    """Aplicación OpusClip con UI profesional."""
    
    def __init__(self):
        self.transcriber: Optional[Transcriber] = None
        self.analyzer: Optional[GeminiAnalyzer] = None
        self.editor: Optional[VideoEditor] = None
        self.audio_analyzer = AudioAnalyzer()
        self.state_manager = StateManager()
        self.cancel_requested: bool = False
        
        self.current_video: Optional[str] = None
        self.current_state: Optional[ProjectState] = None
        self.current_preview: Optional[str] = None
        self.subtitle_editors: Dict[int, SubtitleEditor] = {}
        
    def _init_components(self, model_size: Optional[str] = None):
        """Inicializa componentes lazy-loading."""
        # Recrear el Transcriber si el llamador pidió explícitamente un modelo Whisper
        # distinto al ya cargado — antes se creaba una sola vez y quedaba fijo para
        # siempre, así que cambiar el dropdown o usar los presets (Rápido/Calidad) no
        # tenía ningún efecto real. Si no se pide un modelo específico (model_size=None,
        # como en generate_preview) se respeta el que ya esté cargado, sin recargar.
        if model_size is not None and self.transcriber is not None and self.transcriber.model_size != model_size:
            logger.info(f"Cambiando modelo Whisper de '{self.transcriber.model_size}' a '{model_size}'")
            self.transcriber.unload_model()
            self.transcriber = None

        if self.transcriber is None:
            desired_model = model_size or config.WHISPER_MODEL
            logger.info(f"Inicializando Whisper '{desired_model}'...")
            self.transcriber = Transcriber(
                model_size=desired_model,
                language=config.WHISPER_LANGUAGE
            )
            self.transcriber.load_model()

        if self.analyzer is None:
            logger.info("Inicializando Gemini...")
            self.analyzer = GeminiAnalyzer(
                model_name=config.GEMINI_MODEL,
                temperature=config.GEMINI_TEMPERATURE,
                max_output_tokens=config.GEMINI_MAX_TOKENS
            )
            
        if self.editor is None:
            logger.info("Inicializando VideoEditor...")
            self.editor = VideoEditor()
    
    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """Valida que el video sea válido."""
        if not video_path:
            return False, "❌ Select a video file"
            
        path = Path(video_path)
        
        if not path.exists():
            return False, ERRORS['VIDEO_NOT_FOUND']
        
        if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            return False, ERRORS['UNSUPPORTED_FORMAT']
        
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 2048:  # 2GB
            return False, ERRORS['VIDEO_TOO_LARGE']
        
        return True, "✅ Video válido"
    
    def cancel_analysis(self) -> str:
        """Flags the current analysis to stop after the current chunk."""
        self.cancel_requested = True
        return "⏹️ Cancelando análisis..."

    def analyze_video(
        self,
        video_path: str,
        num_clips: int,
        min_duration: int,
        max_duration: int,
        model_size: str,
        progress: gr.Progress,
        custom_prompt: str = "",
        analysis_mode: str = "balance",
    ) -> Tuple[str, str, str, gr.update, gr.update, str]:
        """
        Step 1-2: Transcription + viral clip analysis.
        Modes:
          - 'fast':    tiny/base, no word-timestamps, compact prompt.
          - 'balance': base, fast pass first then word-ts only on clips.
          - 'quality': base/small, full word-timestamps everywhere.
        """
        self.cancel_requested = False
        is_valid, msg = self.validate_video(video_path)
        if not is_valid:
            return msg, "", "", gr.update(visible=False), gr.update(), "0 tokens"
        
        if min_duration >= max_duration:
            return "❌ La duración mínima debe ser menor que la máxima", "", "", gr.update(visible=False), gr.update(), "0 tokens"
        
        self.current_video = video_path
        
        try:
            self._init_components(model_size=model_size)
            
            # --- ETA precheck ---
            try:
                eta_info = self.transcriber.estimate_analysis_time(video_path, model_size)
                dur_min = eta_info['duration_min']
                est_s   = eta_info['estimated_total_s']
                eta_str = f"{int(est_s//60)}m {int(est_s%60)}s" if est_s >= 60 else f"{est_s:.0f}s"
                cache_note = " (caché ⚡)" if eta_info['cached'] else ""
                progress(0.04, desc=f"⏱️ Video: {dur_min:.1f} min | ETA: ~{eta_str}{cache_note}")
            except Exception:
                dur_min = 0
                progress(0.04, desc="🎤 Preparando...")

            progress(0.05, desc=f"🎤 Cargando modelo Whisper '{model_size}'...")

            # --- Determine transcription strategy ---
            # Balance: word_timestamps=False for full video, then second-pass on clips only
            # Quality: word_timestamps=True full video
            # Fast:    word_timestamps=False, no second pass
            fast_pass   = analysis_mode in ('fast', 'balance')
            second_pass = analysis_mode == 'balance'

            # Chunked threshold: >45 min in Balance/Fast, >15 min in Quality
            chunked_threshold = 2700 if fast_pass else 900

            vid_info = self.editor.get_video_info(video_path) if self.editor else {}
            vid_duration_probe = vid_info.get('duration', 0)

            progress(0.08, desc="🎤 Transcribiendo con Whisper (pasada rápida)..." if fast_pass else "🎤 Transcribiendo con Whisper...")

            def _prog_transcribe(p, m):
                if self.cancel_requested:
                    raise InterruptedError("⏹️ Cancelado por usuario")
                progress(0.08 + p * 0.27, desc=m)

            if vid_duration_probe > chunked_threshold:
                progress(0.08, desc=f"📦 Video largo ({vid_duration_probe/60:.0f} min) — transcripción en chunks...")
                transcription = self.transcriber.transcribe_chunked(
                    video_path,
                    progress_callback=_prog_transcribe,
                    word_timestamps=False if fast_pass else None,
                    cache_key_suffix="_fast" if fast_pass else "",
                )
            else:
                transcription = self.transcriber.transcribe(
                    video_path,
                    progress_callback=_prog_transcribe,
                    word_timestamps=False if fast_pass else None,
                    cache_key_suffix="_fast" if fast_pass else "",
                )

            if self.cancel_requested:
                return "⏹️ Análisis cancelado", "", "", gr.update(visible=False), gr.update(), "0 tokens"

            duration = transcription['duration']
            num_segments = len(transcription['segments'])

            progress(0.36, desc="🧠 Liberando VRAM...")
            if hasattr(self, 'transcriber') and self.transcriber:
                self.transcriber.unload_model()
                logger.info("🧹 VRAM liberado")

            time.sleep(0.05)

            # --- Audio Energy + Scene Detection Analysis ---
            progress(0.37, desc="🔊 Analizando energía de audio...")
            engagement_data = None
            scene_changes = None
            audio_segments = None
            try:
                from audio_analyzer_enhanced import AudioAnalyzerEnhanced
                audio_analyzer = AudioAnalyzerEnhanced()
                audio_segments = audio_analyzer.analyze_audio_energy(video_path, window_secs=1.0)
                if audio_segments:
                    logger.info(f"✅ Audio energy: {len(audio_segments)} segments analizados")
            except Exception as e:
                logger.warning(f"Audio analysis error: {e}")
                gr.Warning(f"⚠️ Análisis de energía de audio omitido: {e}")

            progress(0.38, desc="🎬 Detectando cambios de escena...")
            try:
                if self.editor:
                    scene_changes = self.editor.detect_scene_changes(video_path, threshold=0.3)
                    if scene_changes:
                        logger.info(f"✅ Scene detection: {len(scene_changes)} cambios detectados")
            except Exception as e:
                logger.warning(f"Scene detection error: {e}")
                gr.Warning(f"⚠️ Detección de escenas omitida: {e}")

            # Calculate engagement scores combining audio + scenes + transcription
            if audio_segments and not (analysis_mode == 'fast' or duration > 1800):
                try:
                    progress(0.39, desc="📊 Calculando scores de engagement...")
                    engagement_data = audio_analyzer.calculate_engagement_scores(
                        audio_segments, scene_changes or [], transcription.get('segments', [])
                    )
                    top_scores = sorted(engagement_data, key=lambda x: x.combined_score, reverse=True)[:5]
                    for e in top_scores:
                        logger.info(f"   Peak at {e.timestamp:.0f}s: combined={e.combined_score:.1f}")
                except Exception as e:
                    logger.warning(f"Engagement calculation error: {e}")
                    gr.Warning(f"⚠️ Cálculo de engagement omitido: {e}")

            progress(0.40, desc="🧠 Analizando momentos virales con Gemini...")

            # Extract keyframes for visual analysis
            progress(0.41, desc="🎬 Extrayendo frames clave...")
            frames = []
            frame_timestamps: List[float] = []
            try:
                if self.editor and duration > 0:
                    # Frames distribuidos uniformemente en TODA la duración — antes se
                    # tomaban cada 3s desde el inicio con tope de 15, lo que en videos
                    # de más de 45s dejaba a Gemini "ciego" para el resto del video.
                    num_frames = 15
                    step = duration / num_frames
                    frame_timestamps = [min(i * step, max(0.0, duration - 0.5)) for i in range(num_frames)]
                    frames = self.editor.extract_keyframes(
                        video_path,
                        timestamps=frame_timestamps,
                        width=720
                    )
                    progress(0.42, desc=f"🎬 {len(frames)} frames extraídos")
            except Exception as e:
                logger.warning(f"No se pudieron extraer frames: {e}")
                gr.Warning(f"⚠️ No se pudieron extraer frames clave, análisis solo con transcripción: {e}")
                frames = []
                frame_timestamps = []

            # Gemini analysis with all data
            if frames:
                progress(0.43, desc="🧠 Analizando con Gemini multimodal...")
                viral_clips = self.analyzer.analyze_with_frames(
                    transcription,
                    frames=frames,
                    num_clips=num_clips,
                    clip_duration_range=(min_duration, max_duration),
                    progress_callback=lambda p, m: progress(0.43 + p * 0.32, desc=m),
                    custom_prompt=custom_prompt,
                    frame_timestamps=frame_timestamps,
                )
            else:
                progress(0.43, desc="🧠 Analizando con Gemini (datos enriquecidos)...")
                viral_clips = self.analyzer.analyze_transcription(
                    transcription,
                    num_clips=num_clips,
                    clip_duration_range=(min_duration, max_duration),
                    progress_callback=lambda p, m: progress(0.43 + p * 0.32, desc=m),
                    custom_prompt=custom_prompt,
                    compact=(analysis_mode == 'fast' or duration > 1800),
                    engagement_data=engagement_data,
                    scene_changes=scene_changes,
                )
            
            if self.cancel_requested:
                return "⏹️ Análisis cancelado", "", "", gr.update(visible=False), gr.update(), "0 tokens"

            if not viral_clips:
                return "⚠️ No se identificaron clips virales", "", "", gr.update(visible=False), gr.update(), "0 tokens"

            all_segs = transcription.get('segments', [])

            # Snap boundaries to silence AND scene changes for more natural cuts
            def snap_to_nearest(val, candidates, max_dist=2.0):
                """Snap value to nearest candidate within max_dist."""
                # `best_dist` arranca en max_dist (no en 0) — antes arrancaba comparando
                # contra la distancia de `val` a sí mismo (siempre 0), así que ningún
                # candidato podía ganar nunca y la función era un no-op silencioso.
                best = val
                best_dist = max_dist
                for c in candidates:
                    dist = abs(c - val)
                    if dist < best_dist:
                        best_dist = dist
                        best = c
                return best

            for clip in viral_clips:
                # Snap to silence (transcription-based)
                snapped_start = self.transcriber.snap_to_silence(all_segs, clip.start)
                snapped_end   = self.transcriber.snap_to_silence(all_segs, clip.end)

                # Snap to scene changes (±2 seconds)
                if scene_changes:
                    snapped_start = snap_to_nearest(snapped_start, scene_changes, max_dist=2.0)
                    snapped_end = snap_to_nearest(snapped_end, scene_changes, max_dist=2.0)

                # Ensure minimum duration and no negative values
                if snapped_end <= snapped_start:
                    snapped_end = snapped_start + min_duration
                if snapped_end - snapped_start >= min_duration:
                    clip.start = max(0.0, snapped_start)
                    clip.end = min(duration, snapped_end)
                else:
                    # Fallback: keep original but ensure valid
                    clip.start = max(0.0, float(clip.start))
                    clip.end = min(duration, max(clip.start + min_duration, float(clip.end)))

            # --- Second pass: word-level timestamps only on final clips ---
            clip_word_map: Dict[int, List] = {}
            if second_pass:
                progress(0.76, desc="🔍 Timestamps por palabra (solo clips finales)...")
                self._init_components(model_size=model_size)  # reload Whisper
                clips_for_second_pass = [
                    {'id': i, 'start': c.start, 'end': c.end}
                    for i, c in enumerate(viral_clips)
                ]
                try:
                    clip_word_map = self.transcriber.transcribe_clip_words(
                        video_path,
                        clips_for_second_pass,
                        progress_callback=lambda p, m: progress(0.76 + p * 0.12, desc=m),
                    )
                    self.transcriber.unload_model()
                except Exception as _wp:
                    logger.warning(f"Second pass omitido: {_wp}")

            # G1: score segments by audio energy (non-blocking)
            try:
                scored_segs = self.audio_analyzer.score_segments(video_path, all_segs)
                top_energy = sorted(scored_segs, key=lambda s: s.get('audio_energy_score', 0), reverse=True)[:5]
                logger.info(f"⚡ Top 5 segmentos por energía: {[(round(s['start'],1), round(s.get('audio_energy_score',0),2)) for s in top_energy]}")
            except Exception as _ae:
                logger.debug(f"Audio energy scoring omitido: {_ae}")

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
                clips_data.append({
                    'start': clip.start,
                    'end': clip.end,
                    'virality_score': clip.virality_score,
                    'hook_score': clip.hook_score,
                    'pacing_score': clip.pacing_score,
                    'engagement_score': clip.engagement_score,
                    'flow_score': clip.flow_score,
                    'value_score': clip.value_score,
                    'trend_score': clip.trend_score,
                    'mood': clip.mood,
                    'hook_type': clip.hook_type,
                    'ideal_platform': clip.ideal_platform,
                    'edit_recipe': clip.edit_recipe,
                    'reason': clip.reason,
                    'hook': clip.hook,
                    'segments': word_segs_for_clip,
                })
                total_tokens += len(clip.reason + clip.hook) // 4 + 100
            
            self.current_state = self.state_manager.create_project(video_path, transcription, clips_data)
            
            self.subtitle_editors = {}
            for clip_state in self.current_state.clips:
                editor = create_editor_from_transcription(clip_state.segments, style_name="modern")
                self.subtitle_editors[clip_state.id] = editor
            
            progress(1.0, desc=f"✅ ¡Análisis completado! {len(viral_clips)} clips identificados")
            
            clips_summary = self._build_clips_summary()
            analysis_summary = self._build_analysis_summary(viral_clips, duration, num_segments)
            token_info = f"~{total_tokens} tokens usados"
            
            clip_choices = [(f"🎬 Clip {i+1} (Puntaje: {c.virality_score:.1f})", c.id) for i, c in enumerate(self.current_state.clips)]
            
            return (
                f"✅ {len(viral_clips)} clips identificados",
                clips_summary,
                analysis_summary,
                gr.update(visible=True),
                gr.update(choices=clip_choices, value=0),
                token_info
            )
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            gr.Warning(f"❌ Falló el análisis: {e}")
            return f"❌ Error: {str(e)}", "", "", gr.update(visible=False), gr.update(choices=[]), "Error"

    def analyze_video_batch(
        self,
        video_paths: List[str],
        num_clips: int,
        min_duration: int,
        max_duration: int,
        model_size: str,
        progress: gr.Progress,
        custom_prompt: str = "",
        analysis_mode: str = "balance",
    ) -> Tuple[str, str, str, gr.update, gr.update, str]:
        """
        Analiza varios videos en secuencia (no en paralelo — Whisper compite por VRAM y
        el pool de Gemini se satura si se lanzan varias llamadas a la vez). Cada video se
        guarda como un proyecto separado vía `analyze_video()` (reutiliza toda la lógica
        existente); al terminar, el estado activo queda en el último video procesado y el
        resto queda disponible en "Proyectos Recientes" para cargarlos individualmente.

        Returns: mismo shape que `analyze_video()` (status, clips_summary, analysis_summary,
        timeline_visible, clip_choices, token_info) para no duplicar el cableado de UI.
        """
        total = len(video_paths)
        if total == 1:
            return self.analyze_video(
                video_paths[0], num_clips, min_duration, max_duration, model_size,
                progress, custom_prompt, analysis_mode
            )

        results: List[str] = []
        last_result = None
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).name

            def _batch_progress(p: float, desc: str = "", _i=i) -> None:
                progress((_i + p) / total, desc=f"[Video {_i+1}/{total}] {desc}")

            try:
                last_result = self.analyze_video(
                    video_path, num_clips, min_duration, max_duration, model_size,
                    _batch_progress, custom_prompt, analysis_mode
                )
                # analyze_video() atrapa sus propias excepciones y devuelve una tupla de
                # error (no las relanza), así que el except de acá abajo casi nunca
                # disparaba — un video fallido se reportaba igual como "✅" con el conteo
                # de clips del proyecto ANTERIOR. Hay que leer el status devuelto.
                status_msg = last_result[0] if last_result else ""
                if status_msg.startswith("❌") or status_msg.startswith("⚠️"):
                    logger.error(f"'{video_name}' falló en el lote: {status_msg}")
                    gr.Warning(f"❌ Falló '{video_name}' en el lote: {status_msg}")
                    results.append(f"❌ {video_name}: {status_msg}")
                else:
                    n_clips = len(self.current_state.clips) if self.current_state else 0
                    results.append(f"✅ {video_name}: {n_clips} clips")
            except Exception as e:
                logger.error(f"Error procesando '{video_name}' en el lote: {e}", exc_info=True)
                gr.Warning(f"❌ Falló '{video_name}' en el lote: {e}")
                results.append(f"❌ {video_name}: {e}")

        success_count = sum(1 for r in results if r.startswith("✅"))
        summary = "\n".join(results)
        if success_count > 0:
            progress(1.0, desc=f"✅ Lote completo: {success_count}/{total} videos exitosos")
            status = f"🎉 Lote completo ({success_count}/{total} exitosos):\n{summary}"
        else:
            progress(1.0, desc=f"❌ Lote falló: 0/{total} videos exitosos")
            status = f"❌ Lote falló (0/{total} exitosos):\n{summary}"

        if success_count > 0 and last_result and self.current_state:
            _, clips_sum, analysis_sum, timeline_vis, clip_dropdown, tokens = last_result
            return status, clips_sum, analysis_sum, timeline_vis, clip_dropdown, tokens
        return status, "", "", gr.update(visible=False), gr.update(choices=[]), "0 tokens"

    def _score_color(self, score: float) -> str:
        """Returns color based on score 0-10."""
        if score >= 8.5:
            return "#7C5CFF"  # cyan - excellent
        elif score >= 7.0:
            return "#4ecdc4"  # teal - good
        elif score >= 5.0:
            return "#feca57"  # yellow - average
        else:
            return "#ff6b6b"  # red - low

    def _build_clips_summary(self) -> str:
        """Builds visual summary of clips with score badges and factor bars."""
        if not self.current_state:
            return ""
        
        MOOD_EMOJI = {
            'funny': '😂', 'shocking': '😱', 'educational': '📚',
            'motivational': '💪', 'dramatic': '🎭', 'controversial': '🔥',
            'wholesome': '🥰', 'gaming_hype': '🎮', 'storytelling': '📖',
            'inspirational': '✨', 'neutral': '🎬'
        }
        PLATFORM_EMOJI = {
            'tiktok': '🎵', 'reels': '📸', 'shorts': '▶️',
            'linkedin': '💼', 'twitter': '🐦', 'landscape': '🖥️'
        }
        parts = []
        for i, clip in enumerate(self.current_state.clips, 1):
            status_icon = "✅" if clip.selected else "⏸️"
            hook_preview = clip.hook[:80] + "..." if len(clip.hook) > 80 else clip.hook
            reason_preview = clip.reason[:110] + "..." if len(clip.reason) > 110 else clip.reason
            score_color = self._score_color(clip.virality_score)
            h_color = self._score_color(clip.hook_score)
            p_color = self._score_color(clip.pacing_score)
            e_color = self._score_color(clip.engagement_score)
            f_color = self._score_color(getattr(clip, 'flow_score', 0))
            v_color = self._score_color(getattr(clip, 'value_score', 0))
            t_color = self._score_color(getattr(clip, 'trend_score', 5))
            rank_badge = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            mood = getattr(clip, 'mood', 'neutral')
            hook_type = getattr(clip, 'hook_type', 'unknown')
            ideal_platform = getattr(clip, 'ideal_platform', 'tiktok')
            mood_icon = MOOD_EMOJI.get(mood, '🎬')
            plat_icon = PLATFORM_EMOJI.get(ideal_platform, '📱')
            flow_s = getattr(clip, 'flow_score', 0)
            value_s = getattr(clip, 'value_score', 0)
            trend_s = getattr(clip, 'trend_score', 5)
            
            parts.append(f"""
            <div style="background: linear-gradient(135deg, #1e2d40 0%, #16213e 100%); 
                        padding: 16px; border-radius: 14px; margin-bottom: 14px; 
                        border-left: 4px solid {score_color}; color: white;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.1em;">{rank_badge}</span>
                        <h4 style="margin: 0; color: #fff; font-size: 1em;">Clip {i} {status_icon}</h4>
                    </div>
                    <div style="text-align: center; background: {score_color}22; border: 3px solid {score_color};
                                border-radius: 50%; width: 72px; height: 72px; display: flex;
                                flex-direction: column; align-items: center; justify-content: center;
                                box-shadow: 0 0 16px {score_color}55;">
                        <span style="color: {score_color}; font-weight: 900; font-size: 1.6em; line-height: 1;">{clip.virality_score:.1f}</span>
                        <span style="color: {score_color}aa; font-size: 0.58em; line-height: 1.4; letter-spacing: 0.04em; text-transform: uppercase;">Virality</span>
                    </div>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px;">
                    <span style="background:#ffffff15; border-radius:20px; padding:2px 9px; font-size:0.72em;">{mood_icon} {mood}</span>
                    <span style="background:#ffffff15; border-radius:20px; padding:2px 9px; font-size:0.72em;">🪝 {hook_type.replace('_',' ')}</span>
                    <span style="background:#ffffff15; border-radius:20px; padding:2px 9px; font-size:0.72em;">{plat_icon} {ideal_platform}</span>
                </div>
                <p style="margin: 0 0 8px 0; color: #a8d8ea; font-size: 0.88em;">
                    🎯 <em>{hook_preview}</em>
                </p>
                <p style="margin: 0 0 8px 0; color: #8fa3ab; font-size: 0.78em;" title="Por qué Gemini le dio este puntaje">
                    💡 {reason_preview}
                </p>
                <p style="margin: 0 0 8px 0; color: #d4e5ed; font-size: 0.82em;">
                    ⏱️ {clip.start:.1f}s → {clip.end:.1f}s &nbsp;·&nbsp; {clip.duration:.1f}s
                </p>
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 5px; margin-top: 8px;">
                    {self._score_bar('Hook', clip.hook_score, h_color)}
                    {self._score_bar('Ritmo', clip.pacing_score, p_color)}
                    {self._score_bar('Engage', clip.engagement_score, e_color)}
                    {self._score_bar('Flow', flow_s, f_color)}
                    {self._score_bar('Valor', value_s, v_color)}
                    {self._score_bar('Trend', trend_s, t_color)}
                </div>
            </div>
            """)
        return "".join(parts)
    
    def _score_bar(self, label: str, score: float, color: str) -> str:
        """Renders a compact score bar cell."""
        return f"""<div>
            <div style="font-size:0.62em;color:#888;margin-bottom:2px;">{label}</div>
            <div style="background:#222;border-radius:3px;height:5px;">
                <div style="width:{score*10:.0f}%;background:{color};height:100%;border-radius:3px;"></div>
            </div>
            <div style="font-size:0.7em;color:{color};margin-top:1px;">{score:.1f}</div>
        </div>"""

    def _build_analysis_summary(self, clips: List[ViralClip], duration: float, segments: int) -> str:
        """Analysis summary."""
        sorted_clips = sorted(clips, key=lambda x: x.virality_score, reverse=True)
        
        ranking = []
        for i, clip in enumerate(sorted_clips[:5], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐"
            bar_width = int(clip.virality_score * 10)
            ranking.append(f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span>{emoji} Clip {i}</span>
                    <div style="flex: 1; background: #333; height: 8px; border-radius: 4px;">
                        <div style="width: {bar_width}%; background: linear-gradient(90deg, #e94560, #ff6b6b); 
                                    height: 100%; border-radius: 4px;"></div>
                    </div>
                    <span style="font-weight: bold; color: #e94560;">{clip.virality_score:.1f}</span>
                </div>
            </div>
            """)
        
        return f"""
        <div style="background: #1a1a2e; padding: 20px; border-radius: 12px; color: white;">
            <h3 style="margin-top: 0; color: #e94560;">📊 Estadísticas de Video</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="text-align: center; background: #16213e; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 1.5em; color: #e94560;">{duration:.0f}s</div>
                    <div style="font-size: 0.85em; color: #a0a0a0;">Duración</div>
                </div>
                <div style="text-align: center; background: #16213e; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 1.5em; color: #4ecdc4;">{segments}</div>
                    <div style="font-size: 0.85em; color: #a0a0a0;">Segmentos</div>
                </div>
                <div style="text-align: center; background: #16213e; padding: 15px; border-radius: 8px;">
                    <div style="font-size: 1.5em; color: #feca57;">{len(clips)}</div>
                    <div style="font-size: 0.85em; color: #a0a0a0;">Clips</div>
                </div>
            </div>
            
            <h4 style="color: #4ecdc4;">🏆 Top {min(5, len(clips))} por Viralidad</h4>
            {''.join(ranking)}
        </div>
        """

    def _build_projects_dashboard(self, query: str = "") -> str:
        """Builds recent projects dashboard from saved states."""
        projects = self.state_manager.list_saved_projects()
        if query:
            projects = [p for p in projects if query.lower() in p.lower()]
        if not projects:
            return """
            <div style="background: #16213e; border-radius: 12px; padding: 18px; color: #b9cac8;">
                No hay proyectos guardados todavía.
            </div>
            """
        cards = []
        previous_state = self.state_manager.current_state
        for project_name in projects[:8]:
            state = self.state_manager.load_state(project_name)
            if not state:
                continue
            best_score = max((c.virality_score for c in state.clips), default=0)
            best_color = self._score_color(best_score)
            cards.append(f"""
            <div style="background: linear-gradient(135deg, #1e2d40, #16213e); border: 1px solid rgba(255,255,255,0.08);
                        border-radius: 12px; padding: 14px; color: #e4e1e6;">
                <div style="display: flex; justify-content: space-between; gap: 10px; align-items: center;">
                    <div style="font-weight: 700; font-size: 0.92em; overflow: hidden; text-overflow: ellipsis;">{project_name}</div>
                    <div style="color: {best_color}; font-weight: 900;">{best_score:.1f}</div>
                </div>
                <div style="font-size: 0.78em; color: #b9cac8; margin-top: 8px;">
                    {len(state.clips)} clips · {state.total_duration:.0f}s · {state.updated_at[:19].replace('T', ' ')}
                </div>
            </div>
            """)
        self.state_manager.current_state = previous_state
        return f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px;">
            {''.join(cards)}
        </div>
        """

    def refresh_project_list(self, query: str = "") -> Tuple[str, gr.update]:
        """Refreshes recent project cards and dropdown choices."""
        projects = self.state_manager.list_saved_projects()
        if query:
            projects = [p for p in projects if query.lower() in p.lower()]
        return self._build_projects_dashboard(query), gr.update(choices=projects, value=projects[0] if projects else None)

    def load_saved_project(
        self, project_name: str
    ) -> Tuple[str, str, str, gr.update, str, gr.update, gr.update, gr.update]:
        """Loads a saved project into the editor and switches to Edit tab."""
        _no_tab = gr.update(), gr.update(), gr.update()
        if not project_name:
            return ("⚠️ Selecciona un proyecto guardado", "", "", gr.update(choices=[]), "No hay video seleccionado",
                    gr.update(), gr.update(), gr.update())
        state = self.state_manager.load_state(project_name)
        if not state:
            return ("❌ No se pudo cargar el proyecto", "", "", gr.update(choices=[]), "No hay video seleccionado",
                    gr.update(), gr.update(), gr.update())
        self.current_state = state
        self.current_video = state.video_path
        self.subtitle_editors = {}
        for clip_state in self.current_state.clips:
            editor = create_editor_from_transcription(clip_state.segments, style_name="modern")
            # Re-aplicar las ediciones de subtítulos guardadas — sin esto, el editor se
            # reconstruía solo desde los segments originales y las ediciones del usuario
            # (guardadas en clip_state.subtitle_edits) desaparecían en silencio al recargar
            # un proyecto. Las claves llegan como str tras el round-trip por JSON.
            for seg_id, edited_text in (clip_state.subtitle_edits or {}).items():
                try:
                    editor.edit_entry(int(seg_id), edited_text)
                except (ValueError, TypeError) as e:
                    logger.warning(f"No se pudo re-aplicar edición de subtítulo {seg_id}: {e}")
            self.subtitle_editors[clip_state.id] = editor
        clips_summary = self._build_clips_summary()
        analysis_summary = self._build_analysis_summary(self.current_state.clips, self.current_state.total_duration, len(self.current_state.transcription.get('segments', [])))
        clip_choices = [(f"🎬 Clip {i+1} (Puntaje: {c.virality_score:.1f})", c.id) for i, c in enumerate(self.current_state.clips)]
        return (
            f"✅ Proyecto cargado: {project_name}", clips_summary, analysis_summary,
            gr.update(choices=clip_choices, value=0), f"Video: {Path(state.video_path).name}",
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        )
    
    def update_clip_time(self, clip_id: int, new_start: float, new_end: float) -> str:
        if not self.current_state:
            return "❌ No hay proyecto activo"
        success = self.state_manager.update_clip(clip_id, start=new_start, end=new_end)
        if success:
            self.current_state = self.state_manager.current_state
            # Reportar los valores REALES guardados, no los que tipeó el usuario —
            # state_manager.update_clip() ajusta en silencio si start >= end o si se
            # pasan del límite del video, y el mensaje confirmaba los valores tipeados
            # aunque no fueran los que terminaron guardados.
            clip = next((c for c in self.current_state.clips if c.id == clip_id), None)
            if clip:
                return f"✅ Clip {clip_id + 1} actualizado: {clip.start:.1f}s - {clip.end:.1f}s"
            return f"✅ Clip {clip_id + 1} actualizado"
        return "❌ Error"
    
    def toggle_clip_selection(self, clip_id: int) -> Tuple[str, str, str, str]:
        """Toggle selection for a single clip and return updated UI elements."""
        if not self.current_state:
            return "", "", "0 clips seleccionados", "→ Exportar clips (0)"
        clip = next((c for c in self.current_state.clips if c.id == clip_id), None)
        if clip is None:
            return "❌ Clip no encontrado", "", "0 clips seleccionados", "→ Exportar clips (0)"
        new_state = not clip.selected
        success = self.state_manager.update_clip(clip_id, selected=new_state)
        if success:
            self.current_state = self.state_manager.current_state
            status = "✅ seleccionado" if new_state else "⏸️ omitido"
            count, total = self._get_selected_count()
            count_txt = f"{count} clip{'s' if count != 1 else ''} seleccionado{'s' if count != 1 else ''}"
            btn_txt = f"→ Exportar clips ({count})"
            return f"Clip {clip_id + 1} {status}", self._build_clips_summary(), count_txt, btn_txt
        return "❌ Falló toggle", "", "0 clips seleccionados", "→ Exportar clips (0)"

    def _get_selected_count(self) -> Tuple[int, int]:
        """Returns (selected_count, total_count) of clips."""
        if not self.current_state:
            return 0, 0
        selected = sum(1 for c in self.current_state.clips if c.selected)
        return selected, len(self.current_state.clips)

    def select_all_clips(self) -> Tuple[str, str, str, str, str]:
        """Select all clips and return updated UI elements."""
        if not self.current_state:
            return "", "", "0 clips seleccionados", "→ Exportar clips (0)", "0/0 clips"
        for clip in self.current_state.clips:
            self.state_manager.update_clip(clip.id, selected=True)
        self.current_state = self.state_manager.current_state
        count, total = self._get_selected_count()
        count_txt = f"{count} clip{'s' if count != 1 else ''} seleccionado{'s' if count != 1 else ''}"
        btn_txt = f"→ Exportar clips ({count})"
        counter_txt = f"{count}/{total} clips"
        return "✅ Todos los clips seleccionados", self._build_clips_summary(), count_txt, btn_txt, counter_txt

    def deselect_all_clips(self) -> Tuple[str, str, str, str, str]:
        """Deselect all clips and return updated UI elements."""
        if not self.current_state:
            return "", "", "0 clips seleccionados", "→ Exportar clips (0)", "0/0 clips"
        for clip in self.current_state.clips:
            self.state_manager.update_clip(clip.id, selected=False)
        self.current_state = self.state_manager.current_state
        count, total = self._get_selected_count()
        count_txt = f"{count} clips seleccionados"
        btn_txt = f"→ Exportar clips ({count})"
        counter_txt = f"{count}/{total} clips"
        return "⏸️ Todos los clips deseleccionados", self._build_clips_summary(), count_txt, btn_txt, counter_txt

    def generate_preview(self, clip_id: int) -> Optional[str]:
        if not self.current_state or not self.current_video:
            logger.warning("generate_preview: no hay estado o video activo")
            return None
        if not Path(self.current_video).exists():
            logger.error(f"generate_preview: video no encontrado en {self.current_video}")
            return None
        clip = None
        for c in self.current_state.clips:
            if c.id == clip_id:
                clip = c
                break
        if not clip:
            logger.warning(f"generate_preview: clip_id {clip_id} no encontrado")
            return None
        if self.editor is None:
            self._init_components()
        preview_path = config.TEMP_DIR / f"preview_clip_{clip_id}.mp4"
        try:
            self.editor.create_preview_segment(
                self.current_video, str(preview_path), clip.start, clip.end, duration_limit=15.0
            )
            if preview_path.exists():
                return str(preview_path)
            logger.error("generate_preview: ffmpeg no generó el archivo")
            return None
        except Exception as e:
            logger.error(f"generate_preview error: {e}", exc_info=True)
            gr.Warning(f"⚠️ No se pudo generar la vista previa: {e}")
            return None
    
    def get_subtitle_data(self, clip_id: int) -> List[List[Any]]:
        if clip_id not in self.subtitle_editors:
            # Return empty row with correct column count to prevent Gradio 5 errors
            return [["—", "—", "—", "No hay subtítulos", "—", "—"]]
        editor = self.subtitle_editors[clip_id]
        data = editor.get_entries_for_dataframe()
        if not data:
            return [["—", "—", "—", "No hay subtítulos", "—", "—"]]
        return [[e['ID'], e['Inicio'], e['Fin'], e['Texto Original'], e['Texto Editado'], e['Editado']] for e in data]

    def update_subtitles_from_dataframe(self, clip_id: int, df_data: List[List[Any]]) -> Tuple[str, List[List[Any]]]:
        """
        Aplica ediciones hechas directamente en la grilla de subtítulos — antes la tabla
        tenía `interactive=True` pero ningún evento leía sus cambios; solo el flujo
        separado "Fila ID + Nuevo Texto + Guardar" funcionaba, así que editar la celda
        obvia (columna "Editado") se perdía en silencio.
        """
        if clip_id not in self.subtitle_editors or not df_data:
            return "❌ Editor no encontrado", self.get_subtitle_data(clip_id)
        editor = self.subtitle_editors[clip_id]
        updated = 0
        for row in df_data:
            if not row or len(row) < 5:
                continue
            try:
                row_id = int(row[0])
            except (ValueError, TypeError):
                continue
            new_text = row[4]
            if new_text is None:
                continue
            new_text = str(new_text).strip()
            if not new_text:
                continue
            entry = next((e for e in editor.entries if e.id == row_id), None)
            if entry and new_text != entry.display_text:
                if editor.edit_entry(row_id, new_text):
                    self.state_manager.update_subtitle_edit(clip_id, row_id, new_text)
                    updated += 1
        if updated:
            self._save_all_subtitle_edits()
        return f"✅ {updated} subtítulo(s) actualizado(s) desde la tabla", self.get_subtitle_data(clip_id)

    def _save_all_subtitle_edits(self) -> None:
        if not self.current_state:
            return
        for clip_id, editor in self.subtitle_editors.items():
            for entry in editor.entries:
                if entry.id in editor.custom_edits:
                    self.state_manager.update_subtitle_edit(clip_id, entry.id, entry.edited_text if entry.edited_text is not None else entry.original_text)
        logger.debug("💾 Auto-save: subtitles saved")
    
    def update_subtitle(self, clip_id: int, row_id: int, new_text: str) -> str:
        if clip_id not in self.subtitle_editors:
            return "❌ Editor no encontrado"
        editor = self.subtitle_editors[clip_id]
        success = editor.edit_entry(row_id, new_text)
        if success:
            self.state_manager.update_subtitle_edit(clip_id, row_id, new_text)
            self._save_all_subtitle_edits()
            return f"✅ Subtítulo {row_id} actualizado y guardado"
        return "❌ Error"
    
    def apply_auto_corrections(self, clip_id: int) -> Tuple[str, List[List[Any]]]:
        """Applies auto-corrections to subtitles of the selected clip."""
        if clip_id not in self.subtitle_editors:
            return "❌ Editor no encontrado", []
        editor = self.subtitle_editors[clip_id]
        count = editor.apply_auto_corrections()
        self._save_all_subtitle_edits()
        data = self.get_subtitle_data(clip_id)
        return f"✅ {count} subtítulos auto-correctos guardados", data
    
    def remove_filler_words(self, clip_id: int) -> Tuple[str, List[List[Any]]]:
        """Removes filler words from the selected clip."""
        if clip_id not in self.subtitle_editors:
            return "❌ Editor no encontrado", []
        editor = self.subtitle_editors[clip_id]
        count = editor.remove_filler_words()
        self._save_all_subtitle_edits()
        data = self.get_subtitle_data(clip_id)
        return f"✅ {count} subtítulos limpiados y guardados", data
    
    def add_emojis(self, clip_id: int) -> Tuple[str, List[List[Any]]]:
        """Adds emojis to the selected clip."""
        if clip_id not in self.subtitle_editors:
            return "❌ Editor no encontrado", []
        editor = self.subtitle_editors[clip_id]
        count = editor.add_emojis(max_emojis_per_entry=2)
        self._save_all_subtitle_edits()
        data = self.get_subtitle_data(clip_id)
        return f"✅ {count} subtítulos mejorados con emojis", data
    
    # Margen inferior (px) para subtítulos "bottom" — deja espacio libre para la UI nativa de
    # TikTok/Reels/Shorts (caption, botones de interacción) que tapa esa zona al publicar.
    SAFE_ZONE_MARGIN_BOTTOM = 220

    def _subtitle_style_from_name(self, style_name: str) -> SubtitleStyle:
        """Convierte el nombre de estilo elegido en la UI (modern/tiktok/minimal/classic) a un
        SubtitleStyle real para el renderizado — antes este mapeo no existía y el picker de
        estilo visual no tenía ningún efecto en el video exportado."""
        preset = SUBTITLE_STYLES.get(style_name, SUBTITLE_STYLES["modern"])
        margin_vertical = self.SAFE_ZONE_MARGIN_BOTTOM if preset.get("position") == "bottom" else 100
        return SubtitleStyle(
            font=preset.get("font", "C:/Windows/Fonts/arialbd.ttf"),
            fontsize=preset.get("fontsize", 64),
            color=preset.get("color", "white"),
            stroke_color=preset.get("stroke_color", "black"),
            stroke_width=preset.get("stroke_width", 3),
            bg_color=preset.get("bg_color"),
            position=preset.get("position", "center"),
            margin_vertical=margin_vertical,
        )

    def _export_single_clip(
        self,
        args: Tuple[int, Any],
        style_name: str = "modern",
        track_faces: bool = False,
        subtitle_mode: str = "static",
        target_width: int = 1080,
        target_height: int = 1920,
        enable_mood_grade: bool = True,
        enable_ducking: bool = True,
        brand_name: str = "",
        brand_color: str = "#7C5CFF",
        enable_zoom_cues: bool = False,
        compress_pauses: bool = False,
        show_hook: bool = True,
    ) -> Tuple[int, str, bool]:
        """
        Exporta un solo clip con post-procesamiento opcional:
        - Zoom dinámico por energía de audio (G4/F2.3)
        - Audio ducking (F3.4)
        - Color grading por mood (F3.1)
        - Branding overlay (F3.2)

        Args:
            args: (index, clip_state)
        Returns:
            (index, output_path, success)
        """
        i, clip_state = args
        try:
            score_str = f"_{clip_state.virality_score:.1f}"
            output_file = config.OUTPUT_DIR / f"viral_clip_{i+1:02d}{score_str}.mp4"
            temp_base = config.OUTPUT_DIR / f"_tmp_clip_{i+1:02d}"

            editor = self.subtitle_editors.get(clip_state.id)
            segments = editor.get_segments_for_export() if editor else clip_state.segments

            # Step 1: base clip (crop + subtitles)
            base_out = str(temp_base) + "_base.mp4"
            self.editor.create_viral_clip(
                self.current_video, base_out, clip_state.start, clip_state.end,
                segments, add_subtitles=True, track_faces=track_faces,
                subtitle_mode=subtitle_mode,
                target_width=target_width, target_height=target_height,
                style=self._subtitle_style_from_name(style_name),
                compress_pauses=compress_pauses,
                hook_text=clip_state.hook,
                show_hook=show_hook,
            )
            current = base_out

            # Step 2: zoom punch-in at audio energy peaks (G4/F2.3)
            if enable_zoom_cues and self.current_video:
                try:
                    clip_segs = [s for s in (clip_state.segments or [])
                                 if s.get('start', 0) < clip_state.end
                                 and s.get('end', 0) > clip_state.start]
                    zoom_cues = self.audio_analyzer.get_zoom_cues(
                        current, clip_segs, max_cues=4
                    )
                    if zoom_cues:
                        zoomed = str(temp_base) + "_zoomed.mp4"
                        current = self.editor.apply_zoom_cues(current, zoomed, zoom_cues)
                except Exception as _ze:
                    logger.debug(f"Zoom cues omitido: {_ze}")

            # Step 3: audio ducking (F3.4)
            if enable_ducking:
                try:
                    ducked = str(temp_base) + "_ducked.mp4"
                    current = self.editor.apply_audio_ducking(current, ducked)
                except Exception as _de:
                    logger.warning(f"Audio ducking omitido en clip {i+1}: {_de}")

            # Step 4: color grading by mood (F3.1)
            mood = getattr(clip_state, 'mood', 'neutral')
            if enable_mood_grade and mood and mood != 'neutral':
                try:
                    graded = str(temp_base) + "_graded.mp4"
                    current = self.editor.apply_mood_grade(current, graded, mood)
                except Exception as _ge:
                    logger.warning(f"Color grading omitido en clip {i+1}: {_ge}")

            # Step 5: branding overlay (F3.2)
            if brand_name:
                try:
                    branded = str(temp_base) + "_branded.mp4"
                    current = self.editor.add_branding_overlay(
                        current, branded, brand_name=brand_name, brand_color=brand_color
                    )
                except Exception as _be:
                    logger.warning(f"Branding overlay omitido en clip {i+1}: {_be}")

            # Move final result to output_file
            import shutil as _sh
            _sh.move(current, str(output_file))

            # Clean up any remaining temps
            for suffix in ["_base.mp4", "_zoomed.mp4", "_ducked.mp4", "_graded.mp4", "_branded.mp4"]:
                p = Path(str(temp_base) + suffix)
                if p.exists() and str(p) != str(output_file):
                    try:
                        p.unlink()
                    except Exception:
                        pass

            return (i, str(output_file), True)
        except Exception as e:
            logger.error(f"Error exportando clip {i}: {e}")
            return (i, str(e), False)
    
    def _relative_segments(self, clip_state, segments: List[Dict[str, Any]], compress_pauses: bool = False) -> List[Dict[str, Any]]:
        """
        Convierte segmentos (timestamps absolutos del video) a relativos al inicio del clip,
        y si F4 (compress_pauses) estuvo activo en el export, remapea con el MISMO cálculo de
        `compute_keep_ranges`/`remap_to_compressed` que usó `create_viral_clip` para comprimir
        el video — si no, los .srt/.vtt quedan corridos respecto al video real porque el video
        se acortó pero los sidecars seguían con los timestamps sin comprimir.
        """
        offset = clip_state.start
        relative = []
        for seg in segments:
            start_s = max(0.0, seg.get('start', 0.0) - offset)
            end_s = max(start_s + 0.1, seg.get('end', seg.get('start', 0.0) + 1.0) - offset)
            relative.append({'start': start_s, 'end': end_s, 'text': seg.get('text', '')})

        if compress_pauses:
            try:
                keep_ranges = compute_keep_ranges(relative, clip_state.duration)
                if len(keep_ranges) > 1:
                    for seg in relative:
                        seg['start'] = remap_to_compressed(seg['start'], keep_ranges)
                        seg['end'] = remap_to_compressed(seg['end'], keep_ranges)
            except Exception as e:
                logger.warning(f"No se pudo remapear subtítulos comprimidos (F4): {e}")

        return relative

    def _generate_srt(self, clip_state, index: int, compress_pauses: bool = False) -> Optional[str]:
        """Genera un archivo SRT para un clip dado (timestamps relativos al inicio del clip)."""
        try:
            editor = self.subtitle_editors.get(clip_state.id)
            segments = editor.get_segments_for_export() if editor else clip_state.segments
            if not segments:
                return None
            srt_path = config.OUTPUT_DIR / f"viral_clip_{index+1:02d}_{clip_state.virality_score:.1f}.srt"
            relative = self._relative_segments(clip_state, segments, compress_pauses)
            lines = []
            entry_num = 0
            for seg in relative:
                start_s, end_s = seg['start'], seg['end']
                text = seg['text'].strip()
                if not text:
                    continue
                entry_num += 1
                def _fmt(s):
                    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
                    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')
                lines.append(f"{entry_num}\n{_fmt(start_s)} --> {_fmt(end_s)}\n{text}\n")
            srt_path.write_text('\n'.join(lines), encoding='utf-8')
            return str(srt_path)
        except Exception as e:
            logger.warning(f"No se pudo generar SRT para clip {index+1}: {e}")
            gr.Warning(f"⚠️ No se pudo generar el SRT del clip {index+1}: {e}")
            return None

    def _generate_vtt(self, clip_state, index: int, compress_pauses: bool = False) -> Optional[str]:
        """Genera un archivo VTT para un clip dado (timestamps relativos al inicio del clip)."""
        try:
            editor = self.subtitle_editors.get(clip_state.id)
            segments = editor.get_segments_for_export() if editor else clip_state.segments
            if not segments:
                return None
            vtt_path = config.OUTPUT_DIR / f"viral_clip_{index+1:02d}_{clip_state.virality_score:.1f}.vtt"
            relative = self._relative_segments(clip_state, segments, compress_pauses)
            lines = ["WEBVTT\n"]
            for seg in relative:
                start_s, end_s = seg['start'], seg['end']
                text = seg['text'].strip()
                if not text:
                    continue
                def _fmt(s):
                    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
                    return f"{h:02d}:{m:02d}:{sec:06.3f}"
                lines.append(f"{_fmt(start_s)} --> {_fmt(end_s)}\n{text}\n")
            vtt_path.write_text('\n'.join(lines), encoding='utf-8')
            return str(vtt_path)
        except Exception as e:
            logger.warning(f"No se pudo generar VTT para clip {index+1}: {e}")
            gr.Warning(f"⚠️ No se pudo generar el VTT del clip {index+1}: {e}")
            return None

    def _build_social_metadata(
        self, clip_state, index: int, platform: str, brand_name: str = ""
    ) -> Dict[str, Any]:
        """
        Construye título, caption, hashtags y CTA para un clip.

        Los hashtags vienen directo de Gemini (`clip_state.hashtags`, sobre el tema real del
        clip). Si no hay ninguno — proyectos guardados antes de este cambio, o si Gemini no
        los devolvió — se recurre a extraerlos de `hook`/`reason` como antes (menos preciso,
        ya que ese texto explica *por qué* el clip es viral, no de qué trata).
        """
        gemini_hashtags = [h for h in (getattr(clip_state, 'hashtags', None) or []) if h]
        if gemini_hashtags:
            hashtags = gemini_hashtags
        else:
            keywords = []
            text = f"{clip_state.hook} {clip_state.reason}".lower()
            for word in text.replace(",", " ").replace(".", " ").split():
                clean = word.strip("#:;!?¡¿()[]{}\"'").lower()
                if len(clean) > 4 and clean not in keywords:
                    keywords.append(clean)
                if len(keywords) >= 8:
                    break
            hashtags = [f"#{w}" for w in keywords[:5]]
        title = clip_state.hook.strip()[:80] or f"Clip viral {index+1}"
        description = f"{clip_state.reason.strip()[:180]}\n\n{' '.join(hashtags)}".strip()
        return {
            "clip": index + 1,
            "platform": platform,
            "brand": brand_name,
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "cta": "Sígueme para más clips como este.",
            "mood": getattr(clip_state, 'mood', 'neutral'),
            "hook_type": getattr(clip_state, 'hook_type', 'unknown'),
            "ideal_platform": getattr(clip_state, 'ideal_platform', platform),
            "edit_recipe": getattr(clip_state, 'edit_recipe', ''),
            "score": {
                "virality": clip_state.virality_score,
                "hook": clip_state.hook_score,
                "pacing": clip_state.pacing_score,
                "engagement": clip_state.engagement_score,
                "flow": getattr(clip_state, 'flow_score', 0.0),
                "value": getattr(clip_state, 'value_score', 0.0),
                "trend": getattr(clip_state, 'trend_score', 5.0),
            },
            "timing": {
                "start": clip_state.start,
                "end": clip_state.end,
                "duration": clip_state.duration
            }
        }

    def _generate_clip_metadata(self, clip_state, index: int, platform: str, brand_name: str = "") -> Optional[str]:
        """Genera metadata JSON por clip con score, hook y textos sociales básicos."""
        try:
            metadata = self._build_social_metadata(clip_state, index, platform, brand_name)
            meta_path = config.OUTPUT_DIR / f"viral_clip_{index+1:02d}_{clip_state.virality_score:.1f}.metadata.json"
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(meta_path)
        except Exception as e:
            logger.warning(f"No se pudo generar metadata para clip {index+1}: {e}")
            gr.Warning(f"⚠️ No se pudo generar metadata/caption del clip {index+1}: {e}")
            return None

    def _build_captions_text(
        self, clips_with_paths: List[Tuple[Any, Optional[str]]], platform: str, brand_name: str = ""
    ) -> str:
        """Arma un bloque de texto plano con título+caption+hashtags+CTA por clip, listo para copiar."""
        blocks = []
        for i, (clip_state, output_path) in enumerate(clips_with_paths):
            if not output_path:
                continue
            try:
                meta = self._build_social_metadata(clip_state, i, platform, brand_name)
                blocks.append(
                    f"— Clip {meta['clip']} (⭐ {meta['score']['virality']:.1f}) —\n"
                    f"{meta['title']}\n\n"
                    f"{meta['description']}\n\n"
                    f"{meta['cta']}"
                )
            except Exception as e:
                logger.warning(f"No se pudo armar caption para clip {i+1}: {e}")
        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def export_clips(
        self,
        style_name: str,
        progress: gr.Progress,
        parallel: bool = True,
        track_faces: bool = False,
        subtitle_mode: str = "static",
        platform: str = "tiktok",
        export_srt: bool = True,
        export_vtt: bool = True,
        brand_name: str = "",
        brand_color: str = "#7C5CFF",
        enable_mood_grade: bool = True,
        enable_ducking: bool = True,
        enable_zoom_cues: bool = False,
        compress_pauses: bool = False,
        show_hook: bool = True,
    ) -> Tuple[str, List, List[str], str]:
        """
        Exporta clips seleccionados, soporta procesamiento paralelo.

        Args:
            style_name: Nombre del estilo de subtítulos
            progress: Callback de progreso Gradio
            parallel: Usar paralelización (default: True)
            track_faces: Activar seguimiento facial AI
            subtitle_mode: "static" o "karaoke"

        Returns:
            (status_message, gallery_data, output_file_paths, captions_text)
        """
        if not self.current_state or not self.current_video:
            return "❌ No hay proyecto activo. Por favor analiza un video primero en la pestaña Importar.", [], [], ""
        # Sync state_manager with current in-memory state
        self.state_manager.current_state = self.current_state
        selected_clips = [c for c in self.current_state.clips if c.selected]
        if not selected_clips:
            return "⚠️ No hay clips seleccionados. Por favor selecciona clips en la pestaña Editar.", [], [], ""
        
        # Map platform to real output resolution
        platform_resolution = {
            "tiktok": (1080, 1920), "reels": (1080, 1920), "shorts": (1080, 1920),
            "linkedin": (1080, 1080), "twitter": (1080, 1080),
            "landscape": (1920, 1080)
        }
        target_width, target_height = platform_resolution.get(platform, (1080, 1920))
        
        total = len(selected_clips)
        output_files = [None] * total  # Pre-allocate list
        
        try:
            if parallel and total > 1:
                # Exportación paralela para múltiples clips
                progress(0.0, desc=f"🚀 Exportando {total} clips en paralelo (NVENC)...")
                logger.info(f"Iniciando exportación paralela de {total} clips (face_track={track_faces}, karaoke={subtitle_mode})")
                
                with ThreadPoolExecutor(max_workers=min(4, total)) as executor:
                    future_to_index = {}
                    for i, clip in enumerate(selected_clips):
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
                        future_to_index[future] = i
                    
                    # Procesar resultados a medida que completan
                    completed = 0
                    for future in as_completed(future_to_index):
                        idx, result, success = future.result()
                        completed += 1
                        
                        if success:
                            output_files[idx] = result
                            logger.info(f"✅ Clip {idx+1}/{total} exportado")
                        else:
                            logger.error(f"❌ Clip {idx+1}/{total} falló: {result}")
                        
                        progress(
                            completed / total, 
                            desc=f"🎬 Exportados {completed}/{total} clips..."
                        )
            else:
                # Exportación secuencial para un solo clip
                for i, clip_state in enumerate(selected_clips):
                    progress((i / total), desc=f"🎬 Exportando {i+1}/{total}...")
                    idx, result, success = self._export_single_clip(
                        (i, clip_state), style_name, track_faces, subtitle_mode,
                        target_width, target_height,
                        enable_mood_grade, enable_ducking,
                        brand_name, brand_color,
                        enable_zoom_cues, compress_pauses,
                        show_hook,
                    )
                    if success:
                        output_files[idx] = result
                    else:
                        logger.error(f"Error exportando clip {i}: {result}")
            
            # Filtrar None (fallidos)
            successful_exports = [f for f in output_files if f is not None]
            
            # Generar thumbnails para la galería + SRT opcional
            gallery_data = []
            all_outputs = list(successful_exports)
            for i, (clip, output_path) in enumerate(zip(selected_clips, output_files)):
                if output_path and Path(output_path).exists():
                    try:
                        # Crear thumbnail del frame inicial del clip
                        thumb_path = config.OUTPUT_DIR / f"thumb_{i+1:02d}.jpg"
                        self.editor.generate_thumbnail(
                            output_path, 
                            str(thumb_path), 
                            timestamp=0.5  # Frame a 0.5s para evitar fade negro
                        )
                        sc = clip.virality_score
                        caption = f"Clip {i+1} | ⭐ {sc:.1f} | {clip.start:.0f}s-{clip.end:.0f}s | {platform.upper()}"
                        gallery_data.append((str(thumb_path), caption))
                    except Exception as e:
                        logger.warning(f"No se pudo generar thumbnail para clip {i+1}: {e}")
                        caption = f"Clip {i+1} | Puntaje: {clip.virality_score:.1f}"
                        gallery_data.append((output_path, caption))
                    
                    # Generar SRT si se solicitó
                    if export_srt:
                        srt_file = self._generate_srt(clip, i, compress_pauses=compress_pauses)
                        if srt_file:
                            all_outputs.append(srt_file)
                            logger.info(f"SRT generado: {srt_file}")
                    if export_vtt:
                        vtt_file = self._generate_vtt(clip, i, compress_pauses=compress_pauses)
                        if vtt_file:
                            all_outputs.append(vtt_file)
                            logger.info(f"VTT generado: {vtt_file}")
                    meta_file = self._generate_clip_metadata(clip, i, platform, brand_name)
                    if meta_file:
                        all_outputs.append(meta_file)
                        logger.info(f"Metadata generada: {meta_file}")
            
            progress(1.0, desc=f"✅ {len(successful_exports)}/{total} clips exportados!")

            captions_text = self._build_captions_text(
                list(zip(selected_clips, output_files)), platform, brand_name
            )

            platform_label = platform.upper()
            if len(successful_exports) == total:
                sidecars = sum(1 for f in all_outputs if f.endswith(('.srt', '.vtt')))
                sidecar_note = f" + {sidecars} subtítulos" if sidecars else ""
                return f"🎉 {len(successful_exports)} clips exportados para {platform_label} ({target_width}x{target_height}){sidecar_note}", gallery_data, all_outputs, captions_text
            else:
                return f"⚠️ {len(successful_exports)}/{total} clips exportados para {platform_label} ({target_width}x{target_height})", gallery_data, all_outputs, captions_text

        except Exception as e:
            logger.error(f"Error en exportación: {e}", exc_info=True)
            gr.Warning(f"❌ Falló la exportación: {e}")
            successful = [f for f in output_files if f is not None]
            return f"❌ Error: {str(e)}", [], successful, ""
    
    def create_ui(self) -> gr.Blocks:
        """UI profesional tipo OpusClip."""
        
        custom_css = """
        /* Stitch Design System - Complete */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
        
        :root {
            /* Core Colors */
            --background: #131316;
            --surface: #1f1f22;
            --surface-dim: #131316;
            --surface-bright: #39393c;
            --surface-container: #1f1f22;
            --surface-container-low: #1b1b1e;
            --surface-container-lowest: #0e0e11;
            --surface-container-high: #2a2a2d;
            --surface-container-highest: #353438;
            --surface-variant: #353438;
            
            /* Primary (Cyan) */
            --primary: #cffffb;
            --primary-container: #7C5CFF;
            --primary-fixed: #29fcf3;
            --primary-fixed-dim: #00ddd6;
            --on-primary: #003735;
            --on-primary-container: #006a66;
            --on-primary-fixed: #00201e;
            --on-primary-fixed-variant: #00504d;
            
            /* Secondary (Fuchsia) */
            --secondary: #ffb2b7;
            --secondary-container: #ff516a;
            --secondary-fixed: #ffdadb;
            --secondary-fixed-dim: #ffb2b7;
            --on-secondary: #67001b;
            --on-secondary-container: #5b0017;
            --on-secondary-fixed: #40000d;
            --on-secondary-fixed-variant: #92002a;
            
            /* Tertiary (Purple) */
            --tertiary: #fdf3ff;
            --tertiary-container: #e9d0ff;
            --tertiary-fixed: #efdbff;
            --tertiary-fixed-dim: #dcb8ff;
            --on-tertiary: #480081;
            --on-tertiary-container: #5B3FD9;
            --on-tertiary-fixed: #2c0051;
            --on-tertiary-fixed-variant: #6700b5;
            
            /* Utility */
            --on-surface: #e4e1e6;
            --on-surface-variant: #b9cac8;
            --on-background: #e4e1e6;
            --outline: #849492;
            --outline-variant: #3a4a48;
            
            /* Error */
            --error: #ffb4ab;
            --error-container: #93000a;
            --on-error: #690005;
            --on-error-container: #ffdad6;
            
            /* Inverse */
            --inverse-surface: #e4e1e6;
            --inverse-on-surface: #303033;
            --inverse-primary: #006a66;
            --surface-tint: #00ddd6;
            
            /* Effects */
            --glass-bg: rgba(31, 31, 34, 0.6);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-border-light: rgba(255, 255, 255, 0.15);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
            --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.3);
            --glow-cyan: 0 0 20px rgba(124, 92, 255, 0.3);
            --glow-purple: 0 0 20px rgba(91, 63, 217, 0.3);
        }
        
        /* Base */
        .gradio-container {
            background: var(--background) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--on-surface) !important;
            max-width: 100% !important;
            padding: 0 !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        /* Layout Container */
        .stitch-layout {
            display: flex;
            min-height: 100vh;
        }
        
        /* Sidebar Navigation */
        .stitch-sidebar {
            width: 256px;
            min-height: 100vh;
            background: rgba(24, 24, 27, 0.4);
            backdrop-filter: blur(32px);
            -webkit-backdrop-filter: blur(32px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            padding: 16px 0;
            position: fixed;
            left: 0;
            top: 0;
            z-index: 100;
        }
        
        .stitch-sidebar-header {
            padding: 0 24px 32px;
            margin-bottom: 16px;
        }
        
        .stitch-sidebar-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        
        .stitch-logo-icon {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--primary-container), var(--on-tertiary-container));
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 15px rgba(124, 92, 255, 0.3);
        }
        
        .stitch-logo-icon span {
            color: white;
            font-size: 20px;
        }
        
        .stitch-sidebar-title {
            color: var(--on-surface);
            font-weight: 600;
            font-size: 14px;
        }
        
        .stitch-sidebar-version {
            color: var(--primary-container);
            font-size: 12px;
        }
        
        .stitch-nav-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 0 12px;
        }
        
        .stitch-nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: 8px;
            color: var(--on-surface-variant);
            font-weight: 500;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .stitch-nav-item:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--on-surface);
        }
        
        .stitch-nav-item.active {
            background: linear-gradient(90deg, rgba(124, 92, 255, 0.1), transparent);
            color: var(--primary-container);
            border-right: 2px solid var(--primary-container);
            box-shadow: 0 0 15px rgba(124, 92, 255, 0.1);
        }
        
        .stitch-nav-footer {
            margin-top: auto;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding: 16px 12px 0;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        /* Main Content Area */
        .stitch-main {
            flex: 1;
            margin-left: 256px;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        
        /* Top Bar */
        .stitch-topbar {
            height: 64px;
            background: rgba(9, 9, 11, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 24px;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        
        .stitch-brand {
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #7C5CFF, #5B3FD9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stitch-topbar-right {
            display: flex;
            align-items: center;
            gap: 24px;
        }
        
        .stitch-search {
            position: relative;
            width: 384px;
        }
        
        .stitch-search input {
            width: 100%;
            background: rgba(31, 31, 34, 0.5);
            border: 1px solid var(--outline-variant);
            border-radius: 9999px;
            padding: 8px 16px 8px 40px;
            color: var(--on-surface);
            font-size: 14px;
            outline: none;
            transition: all 0.2s;
        }
        
        .stitch-search input:focus {
            border-color: var(--primary-container);
            background: rgba(31, 31, 34, 0.8);
        }
        
        .stitch-search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--on-surface-variant);
        }
        
        .stitch-tokens {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.05);
            padding: 6px 16px;
            border-radius: 9999px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--primary-container);
            font-size: 14px;
            font-weight: 600;
        }
        
        .stitch-icon-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--on-surface-variant);
            transition: all 0.2s;
            cursor: pointer;
        }
        
        .stitch-icon-btn:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--on-surface);
        }
        
        /* Canvas / Content */
        .stitch-canvas {
            flex: 1;
            padding: 32px;
            background: radial-gradient(ellipse at top right, rgba(42, 42, 45, 0.3), transparent 40%);
            overflow-y: auto;
        }
        
        /* Glass Panels */
        .glass-panel {
            background: rgba(31, 31, 34, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            border-left: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
        }
        
        .glass-panel-sm {
            background: rgba(31, 31, 34, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
        }
        
        /* Panel Header */
        .panel-header {
            padding: 16px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .panel-header-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--on-surface);
            margin-bottom: 4px;
        }
        
        .panel-header-accent {
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, var(--primary-container), transparent);
            margin-top: 8px;
        }
        
        /* Buttons */
        .btn-primary {
            background: linear-gradient(90deg, #7C5CFF, #5B3FD9);
            border: none;
            color: white;
            font-weight: 600;
            font-size: 14px;
            padding: 14px 28px;
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(124, 92, 255, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary:hover {
            box-shadow: 0 0 30px rgba(124, 92, 255, 0.4);
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: var(--surface-container-high);
            border: 1px solid var(--outline-variant);
            color: var(--on-surface);
            font-weight: 600;
            font-size: 14px;
            padding: 12px 20px;
            border-radius: 8px;
            transition: all 0.2s ease;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-secondary:hover {
            background: var(--surface-container-highest);
            border-color: var(--primary-container);
        }
        
        .btn-ghost {
            background: transparent;
            border: none;
            color: var(--on-surface-variant);
            font-weight: 500;
            padding: 8px 12px;
            border-radius: 6px;
            transition: all 0.2s;
            cursor: pointer;
        }
        
        .btn-ghost:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--on-surface);
        }
        
        /* Form Elements */
        .stitch-input {
            background: var(--surface-container-lowest);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--on-surface);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            outline: none;
            transition: all 0.2s;
            width: 100%;
        }
        
        .stitch-input:focus {
            border-color: var(--primary-container);
            box-shadow: 0 0 0 2px rgba(124, 92, 255, 0.2);
        }
        
        .stitch-select {
            background: var(--surface-container-highest);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 4px;
            display: flex;
        }
        
        .stitch-select-option {
            flex: 1;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            color: var(--on-surface-variant);
        }
        
        .stitch-select-option:hover {
            color: var(--on-surface);
        }
        
        .stitch-select-option.active {
            background: rgba(124, 92, 255, 0.2);
            color: var(--primary-container);
            border: 1px solid rgba(124, 92, 255, 0.3);
            box-shadow: 0 0 10px rgba(124, 92, 255, 0.1);
        }
        
        /* Sliders */
        .stitch-slider-container {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .stitch-slider {
            flex: 1;
            -webkit-appearance: none;
            height: 4px;
            background: var(--surface-variant);
            border-radius: 2px;
            outline: none;
        }
        
        .stitch-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: white;
            border: 2px solid var(--primary-container);
            box-shadow: 0 0 10px rgba(124, 92, 255, 0.5);
            cursor: pointer;
        }
        
        /* Upload Zone */
        .upload-zone {
            border: 2px dashed var(--outline-variant);
            border-radius: 16px;
            background: rgba(14, 14, 17, 0.4);
            padding: 48px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-zone:hover {
            border-color: rgba(124, 92, 255, 0.5);
            background: rgba(14, 14, 17, 0.6);
        }
        
        .upload-zone-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 16px;
            background: var(--surface-container);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .upload-zone-icon span {
            color: var(--primary-container);
            font-size: 28px;
        }

        /* Hide Gradio file input default elements */
        .upload-zone .file-upload-container {
            background: transparent !important;
            border: none !important;
        }
        .upload-zone .file-upload-icon {
            display: none !important;
        }
        .upload-zone .file-upload-text {
            display: none !important;
        }
        .upload-zone label {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .upload-zone .icon svg,
        .upload-zone .icon img {
            display: none !important;
        }
        .upload-zone .wrap > .icon,
        .upload-zone .wrap > center,
        .upload-zone .wrap svg[height="60"],
        .upload-zone .wrap .h4 {
            display: none !important;
        }
        /* Hide all default file upload text */
        .upload-zone .wrap *:not(input):not(.svelte-*):not([class*="label"]) {
            color: transparent !important;
        }

        /* Clip Cards */
        .clip-card {
            background: rgba(31, 31, 34, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .clip-card:hover {
            border-color: rgba(124, 92, 255, 0.3);
            box-shadow: 0 0 20px rgba(124, 92, 255, 0.15);
        }
        
        .clip-card-active {
            border-left: 2px solid var(--primary-container);
        }
        
        .clip-thumbnail {
            height: 160px;
            position: relative;
            background: var(--surface-container-lowest);
        }
        
        .clip-thumbnail img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.8;
        }
        
        .clip-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(to top, var(--background), transparent);
        }
        
        .clip-score {
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(19, 19, 22, 0.8);
            backdrop-filter: blur(8px);
            padding: 6px 12px;
            border-radius: 9999px;
            border: 1px solid rgba(124, 92, 255, 0.3);
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 0 15px rgba(124, 92, 255, 0.2);
        }
        
        .clip-score-icon {
            color: var(--primary-container);
            font-size: 14px;
        }
        
        .clip-score-value {
            color: var(--primary-container);
            font-weight: 700;
            font-size: 13px;
        }
        
        .clip-duration {
            position: absolute;
            bottom: 12px;
            right: 12px;
            background: rgba(0, 0, 0, 0.6);
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-size: 12px;
        }
        
        .clip-info {
            padding: 16px;
        }
        
        .clip-title {
            font-size: 15px;
            font-weight: 600;
            color: var(--on-surface);
            line-height: 1.4;
            margin-bottom: 8px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        
        .clip-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--on-surface-variant);
            font-size: 12px;
        }
        
        /* Timeline */
        .timeline {
            height: 64px;
            background: var(--surface-container-low);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .timeline-waveform {
            position: absolute;
            inset: 0;
            opacity: 0.2;
            display: flex;
            align-items: flex-end;
            gap: 1px;
            padding: 0 8px 4px;
        }
        
        .timeline-bar {
            flex: 1;
            background: white;
            border-radius: 1px;
        }
        
        .timeline-segment {
            position: absolute;
            top: 4px;
            bottom: 4px;
            border-radius: 4px;
            border: 1px solid;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .timeline-segment:hover {
            opacity: 0.8;
        }
        
        .timeline-segment-primary {
            background: rgba(124, 92, 255, 0.2);
            border-color: rgba(124, 92, 255, 0.5);
        }
        
        .timeline-segment-secondary {
            background: rgba(255, 81, 106, 0.2);
            border-color: rgba(255, 81, 106, 0.5);
        }
        
        .timeline-segment-tertiary {
            background: rgba(220, 184, 255, 0.2);
            border-color: rgba(220, 184, 255, 0.5);
        }
        
        /* Style Selector Cards */
        .style-card {
            background: rgba(42, 42, 45, 0.5);
            border: 1px solid var(--outline-variant);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .style-card:hover {
            background: var(--surface-variant);
            border-color: var(--outline);
        }
        
        .style-card-active {
            background: rgba(124, 92, 255, 0.1);
            border-color: rgba(124, 92, 255, 0.5);
            box-shadow: 0 0 15px rgba(124, 92, 255, 0.1);
        }
        
        .style-card-active::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(to bottom, rgba(124, 92, 255, 0.1), transparent);
        }
        
        .style-card-icon {
            font-size: 32px;
            color: var(--on-surface-variant);
            transition: all 0.3s;
        }
        
        .style-card:hover .style-card-icon,
        .style-card-active .style-card-icon {
            color: var(--primary-container);
            transform: scale(1.1);
        }
        
        .style-card-label {
            font-size: 13px;
            font-weight: 600;
            color: var(--on-surface-variant);
            transition: all 0.3s;
        }
        
        .style-card-active .style-card-label {
            color: var(--primary-container);
        }
        
        /* Toggle Switch */
        .toggle-switch {
            width: 44px;
            height: 24px;
            background: var(--surface-variant);
            border-radius: 12px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .toggle-switch-active {
            background: rgba(124, 92, 255, 0.3);
            border-color: rgba(124, 92, 255, 0.5);
            box-shadow: 0 0 10px rgba(124, 92, 255, 0.3);
        }
        
        .toggle-switch-knob {
            width: 18px;
            height: 18px;
            background: var(--outline);
            border-radius: 50%;
            position: absolute;
            top: 2px;
            left: 2px;
            transition: all 0.3s;
        }
        
        .toggle-switch-active .toggle-switch-knob {
            background: var(--primary-container);
            left: 22px;
            box-shadow: 0 0 10px rgba(124, 92, 255, 0.8);
        }
        
        /* AI Tool Row */
        .ai-tool {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: rgba(42, 42, 45, 0.3);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.2s;
        }
        
        .ai-tool:hover {
            border-color: rgba(255, 255, 255, 0.1);
        }
        
        .ai-tool-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .ai-tool-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: var(--surface-container-highest);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--on-surface-variant);
        }
        
        .ai-tool-active .ai-tool-icon {
            background: rgba(124, 92, 255, 0.1);
            color: var(--primary-container);
        }
        
        .ai-tool-info {
            display: flex;
            flex-direction: column;
        }
        
        .ai-tool-name {
            font-size: 14px;
            font-weight: 600;
            color: var(--on-surface);
        }
        
        .ai-tool-desc {
            font-size: 12px;
            color: var(--on-surface-variant);
        }
        
        .ai-tool-active .ai-tool-name {
            color: var(--primary-container);
        }
        
        .ai-tool-active .ai-tool-desc {
            color: rgba(124, 92, 255, 0.7);
        }
        
        /* Progress Bar */
        .progress-container {
            margin-bottom: 16px;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 8px;
        }
        
        .progress-label {
            color: var(--primary-container);
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .progress-value {
            color: var(--on-surface-variant);
        }
        
        .progress-track {
            height: 6px;
            background: var(--surface-container-highest);
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--primary-container), var(--tertiary-container));
            background-size: 200% 100%;
            transition: width 0.3s ease;
        }
        
        /* Subtitle Editor */
        .subtitle-row {
            display: flex;
            gap: 12px;
            padding: 12px;
            border-radius: 8px;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        
        .subtitle-row:hover {
            background: var(--surface-container);
        }
        
        .subtitle-row-active {
            background: var(--surface-container);
            border-color: rgba(124, 92, 255, 0.3);
            position: relative;
        }
        
        .subtitle-row-active::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: var(--primary-container);
            border-radius: 8px 0 0 8px;
            box-shadow: 0 0 8px rgba(124, 92, 255, 0.5);
        }
        
        .subtitle-timestamp {
            min-width: 50px;
            text-align: right;
            font-size: 11px;
            color: var(--on-surface-variant);
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding-top: 4px;
        }
        
        .subtitle-row-active .subtitle-timestamp {
            color: var(--primary-container);
        }
        
        .subtitle-text {
            flex: 1;
        }
        
        .subtitle-text textarea {
            width: 100%;
            background: transparent;
            border: 1px solid transparent;
            color: var(--on-surface);
            font-size: 14px;
            line-height: 1.5;
            resize: none;
            padding: 4px 8px;
            border-radius: 4px;
            outline: none;
        }
        
        .subtitle-text textarea:hover {
            border-color: var(--outline-variant);
        }
        
        .subtitle-text textarea:focus {
            background: var(--surface-container-high);
            border-color: var(--primary-container);
        }
        
        .subtitle-row-active .subtitle-text textarea {
            background: var(--surface-container-high);
            border-color: var(--primary-container);
            box-shadow: 0 0 10px rgba(124, 92, 255, 0.1);
        }
        
        /* Gallery Grid */
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 16px;
        }
        
        .gallery-item {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            background: var(--surface-container);
            border: 1px solid rgba(255, 255, 255, 0.05);
            aspect-ratio: 9/16;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .gallery-item:hover {
            transform: scale(1.02);
        }
        
        .gallery-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.8;
            transition: all 0.3s;
        }
        
        .gallery-item:hover img {
            opacity: 1;
        }
        
        .gallery-item-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(to top, rgba(0, 0, 0, 0.9), transparent);
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 16px;
            opacity: 0;
            transition: all 0.3s;
        }
        
        .gallery-item:hover .gallery-item-overlay {
            opacity: 1;
        }
        
        .gallery-item-title {
            font-size: 13px;
            font-weight: 600;
            color: white;
            margin-bottom: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .gallery-item-actions {
            display: flex;
            gap: 8px;
        }
        
        .gallery-item-btn {
            flex: 1;
            padding: 6px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .gallery-item-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Utility */
        .text-gradient {
            background: linear-gradient(90deg, #7C5CFF, #5B3FD9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .glow-cyan {
            box-shadow: 0 0 20px rgba(124, 92, 255, 0.3);
        }
        
        .glow-purple {
            box-shadow: 0 0 20px rgba(91, 63, 217, 0.3);
        }
        
        /* Animation */
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(124, 92, 255, 0.3); }
            50% { box-shadow: 0 0 30px rgba(124, 92, 255, 0.5); }
        }
        
        .animate-pulse-glow {
            animation: pulse-glow 2s ease-in-out infinite;
        }
        
        /* Material Icons */
        .material-symbols-outlined {
            font-family: 'Material Symbols Outlined';
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }
        
        .material-symbols-outlined.filled {
            font-variation-settings: 'FILL' 1;
        }
        
        /* Nav Buttons - Clean, clickable */
        .nav-btn {
            background: transparent !important;
            border: none !important;
            border-right: 2px solid transparent !important;
            color: #b9cac8 !important;
            text-align: left !important;
            padding: 12px 16px !important;
            margin: 4px 12px !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            transition: all 0.2s !important;
            box-shadow: none !important;
            cursor: pointer !important;
            width: calc(100% - 24px) !important;
            display: block !important;
        }
        .nav-btn:hover {
            background: rgba(255,255,255,0.05) !important;
            color: #e4e1e6 !important;
        }
        .nav-btn:active {
            background: rgba(124, 92, 255, 0.1) !important;
        }
        .nav-btn-active {
            background: linear-gradient(90deg, rgba(124, 92, 255, 0.15), transparent) !important;
            color: #7C5CFF !important;
            border-right: 2px solid #7C5CFF !important;
        }

        /* Analysis presets */
        .preset-row {
            display: grid !important;
            grid-template-columns: repeat(3, 1fr) !important;
            gap: 8px;
        }
        .preset-btn {
            min-width: 0 !important;
            width: 100% !important;
            background: var(--surface-container-highest) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: var(--on-surface-variant) !important;
            font-weight: 600 !important;
            font-size: 12px !important;
            padding: 10px 6px !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
            white-space: nowrap !important;
        }
        .preset-btn:hover {
            border-color: rgba(124, 92, 255, 0.4) !important;
            color: var(--on-surface) !important;
        }
        
        /* Nav container */
        .nav-container {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 8px;
        }
        .nav-container > div {
            margin: 0 !important;
            width: 100% !important;
        }
        
        /* Hide file input label */
        .clean-file-input label {
            font-size: 16px !important;
            font-weight: 500 !important;
            color: #e4e1e6 !important;
        }
        .clean-file-input .wrap {
            border: 2px dashed rgba(255,255,255,0.2) !important;
            border-radius: 12px !important;
            background: rgba(31, 31, 34, 0.5) !important;
            min-height: 180px !important;
        }
        .clean-file-input .wrap:hover {
            border-color: #7C5CFF !important;
            background: rgba(124, 92, 255, 0.05) !important;
        }
        .clean-file-input .file-preview {
            display: none !important;
        }
        
        /* Clean video player */
        .clean-video {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        .clean-video video {
            border-radius: 12px !important;
        }
        
        /* Style Card Buttons */
        .style-card-btn {
            flex: 1 !important;
            background: rgba(31, 31, 34, 0.6) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 12px !important;
            padding: 16px 8px !important;
            color: #b9cac8 !important;
            font-weight: 500 !important;
            transition: all 0.2s !important;
        }
        .style-card-btn:hover {
            border-color: #7C5CFF !important;
            background: rgba(124, 92, 255, 0.1) !important;
        }
        .style-card-active {
            background: rgba(124, 92, 255, 0.15) !important;
            border-color: #7C5CFF !important;
            color: #7C5CFF !important;
        }
        
        /* Clip cards container */
        .clip-cards-container {
            min-height: 160px;
            max-height: 65vh;
            overflow-y: auto;
            padding-right: 8px;
        }
        .clip-cards-container::-webkit-scrollbar {
            width: 6px;
        }
        .clip-cards-container::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        .clip-cards-container::-webkit-scrollbar-thumb {
            background: rgba(124, 92, 255, 0.4);
            border-radius: 3px;
        }
        .clip-card-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
            background: rgba(31, 31, 34, 0.4);
            border-radius: 12px;
            border: 1px dashed rgba(255,255,255,0.1);
        }
        
        /* Hide Gradio's default file upload inner text completely */
        .clean-file-input .upload-text,
        .clean-file-input .or-text,
        .clean-file-input .secondary-text,
        .clean-file-input .file-upload-label .secondary-text {
            display: none !important;
        }
        .clean-file-input .center {
            display: none !important;
        }
        /* Target the specific Gradio upload container */
        .clean-file-input > .label-wrap + div,
        .clean-file-input .wrap .file-upload-label {
            min-height: 120px !important;
        }
        .clean-file-input .file-upload-label {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }
        /* Ocultar el footer de Gradio ("Usar via API" / "Construido con Gradio") para mantener
           la ilusión de producto propio en vez de una demo de Gradio */
        footer {
            display: none !important;
        }
        """
        
        self._css = custom_css
        with gr.Blocks(title="OpusClip Pro") as app:
            # === STITCH LAYOUT: Sidebar + Main Content ===
            with gr.Row(elem_classes=["stitch-layout"]):
                # Sidebar Navigation
                with gr.Column(scale=2, elem_classes=["stitch-sidebar"]):
                    # Logo Header
                    gr.HTML("""
                    <div class="stitch-sidebar-header">
                        <div class="stitch-sidebar-logo">
                            <div class="stitch-logo-icon">
                                <span class="material-symbols-outlined filled">movie_filter</span>
                            </div>
                            <div>
                                <div class="stitch-sidebar-title">OpusClip Pro</div>
                                <div class="stitch-sidebar-version">V2.4 Powered by AI</div>
                            </div>
                        </div>
                    </div>
                    """)
                    
                    # Nav buttons (functional)
                    with gr.Column(elem_classes=["nav-container"]):
                        nav_import_btn = gr.Button("Importar", elem_classes=["nav-btn", "nav-btn-active"])
                        nav_edit_btn = gr.Button("Editar", elem_classes=["nav-btn"])
                        nav_export_btn = gr.Button("Exportar", elem_classes=["nav-btn"])

                # Main Content Area
                with gr.Column(scale=10, elem_classes=["stitch-main"]):
                    # Top Bar
                    with gr.Row(elem_classes=["stitch-topbar"]):
                        gr.HTML("""
                        <div class="stitch-brand">OpusClip Pro</div>
                        <div class="stitch-topbar-right">
                            <div class="stitch-search">
                                <span class="material-symbols-outlined stitch-search-icon">search</span>
                                <input type="text" placeholder="Buscar proyectos..." />
                            </div>
                            <div class="stitch-tokens">
                                <span class="material-symbols-outlined" style="font-size: 16px;">toll</span>
                                <span>1,200 Tokens</span>
                            </div>
                            <div class="stitch-icon-btn">
                                <span class="material-symbols-outlined">notifications</span>
                            </div>
                            <div class="stitch-icon-btn">
                                <span class="material-symbols-outlined">account_circle</span>
                            </div>
                        </div>
                        """)
                    
                    # Canvas Content
                    with gr.Column(elem_classes=["stitch-canvas"]):
                        # === WIZARD STEP BAR ===
                        wizard_bar = gr.HTML("""
                        <div style="display:flex;align-items:center;gap:0;margin-bottom:24px;background:rgba(22,33,62,0.6);border-radius:12px;padding:16px 24px;border:1px solid rgba(255,255,255,0.07);">
                          <div id="wstep1" style="display:flex;align-items:center;gap:8px;flex:1;">
                            <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#7C5CFF,#5B3FD9);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:#fff;">1</div>
                            <span style="font-size:14px;font-weight:600;color:#7C5CFF;">Subir &amp; Analizar</span>
                          </div>
                          <div style="flex:0.3;height:2px;background:rgba(255,255,255,0.1);"></div>
                          <div id="wstep2" style="display:flex;align-items:center;gap:8px;flex:1;opacity:0.4;">
                            <div style="width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:#b9cac8;">2</div>
                            <span style="font-size:14px;font-weight:500;color:#b9cac8;">Seleccionar Clips</span>
                          </div>
                          <div style="flex:0.3;height:2px;background:rgba(255,255,255,0.1);"></div>
                          <div id="wstep3" style="display:flex;align-items:center;gap:8px;flex:1;opacity:0.4;">
                            <div style="width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:#b9cac8;">3</div>
                            <span style="font-size:14px;font-weight:500;color:#b9cac8;">Exportar</span>
                          </div>
                        </div>
                        """)
                        # === TAB 1: IMPORT ===
                        with gr.Column(visible=True) as tab_import:
                            # Page Title
                            gr.HTML("""
                            <div style="margin-bottom: 24px;">
                                <h1 style="font-size: 32px; font-weight: 700; color: #e4e1e6; margin-bottom: 8px;">
                                    Importar Video
                                </h1>
                                <p style="font-size: 16px; color: #b9cac8; margin-bottom: 20px;">
                                    Sube tu video fuente y dejá que la IA encuentre los mejores clips.
                                </p>
                                <div style="display:flex;gap:12px;flex-wrap:wrap;">
                                    <div class="glass-panel-sm" style="flex:1;min-width:200px;padding:14px 18px;display:flex;align-items:center;gap:10px;">
                                        <span style="font-size:20px;line-height:1;">1️⃣</span>
                                        <span style="font-size:13px;color:#b9cac8;">Subí tu video</span>
                                    </div>
                                    <div class="glass-panel-sm" style="flex:1;min-width:200px;padding:14px 18px;display:flex;align-items:center;gap:10px;">
                                        <span style="font-size:20px;line-height:1;">2️⃣</span>
                                        <span style="font-size:13px;color:#b9cac8;">La IA encuentra los mejores momentos</span>
                                    </div>
                                    <div class="glass-panel-sm" style="flex:1;min-width:200px;padding:14px 18px;display:flex;align-items:center;gap:10px;">
                                        <span style="font-size:20px;line-height:1;">3️⃣</span>
                                        <span style="font-size:13px;color:#b9cac8;">Exportá listo para TikTok / Reels / Shorts</span>
                                    </div>
                                </div>
                            </div>
                            """)
                            
                            with gr.Row():
                                # Left: Video Upload
                                with gr.Column(scale=8):
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""
                                        <div class="panel-header">
                                            <div class="panel-header-title">Fuente de Video</div>
                                            <div class="panel-header-accent"></div>
                                        </div>
                                        """)
                                        
                                        with gr.Column(elem_classes=["upload-zone"]):
                                            # File upload with clean styling
                                            video_input = gr.File(
                                                label="Arrastra video(s) o hacé clic para buscar",
                                                file_types=["video"],
                                                file_count="multiple",
                                                type="filepath"
                                            )
                                        
                                        # Video info display
                                        video_info = gr.Textbox(label="", value="No hay video seleccionado", show_label=False, interactive=False)
                                        
                                        # Recent projects dashboard
                                        with gr.Column(elem_classes=["glass-panel-sm"]):
                                            gr.HTML("""<div style="font-size: 14px; font-weight: 700; color: #e4e1e6; margin-bottom: 8px;">📁 Proyectos Recientes</div>""")
                                            project_search_input = gr.Textbox(
                                                label="",
                                                placeholder="Buscar proyecto guardado...",
                                                show_label=False
                                            )
                                            with gr.Row():
                                                saved_project_dropdown = gr.Dropdown(
                                                    choices=self.state_manager.list_saved_projects(),
                                                    label="",
                                                    show_label=False
                                                )
                                                refresh_projects_btn = gr.Button("🔄", elem_classes=["btn-secondary"], size="sm")
                                                load_project_btn = gr.Button("Cargar", elem_classes=["btn-primary"], size="sm")
                                            recent_projects_html = gr.HTML(self._build_projects_dashboard())
                                
                                # Right: Analysis Settings
                                with gr.Column(scale=4):
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""
                                        <div class="panel-header">
                                            <div class="panel-header-title">Ajustes de Análisis</div>
                                            <div class="panel-header-accent"></div>
                                        </div>
                                        """)
                                        
                                        with gr.Column(elem_classes=["glass-panel-sm"], scale=1):
                                            # Quick presets (default experience for non-technical users)
                                            gr.HTML("""<label style="font-size: 14px; font-weight: 600; color: #e4e1e6; margin-bottom: 8px; display: block;">Elegí un estilo de análisis</label>""")
                                            with gr.Row(elem_classes=["preset-row"]):
                                                preset_fast_btn = gr.Button("💨 Rápido", elem_classes=["preset-btn"], size="sm")
                                                preset_balance_btn = gr.Button("⚖️ Balanceado ★", elem_classes=["preset-btn"], size="sm")
                                                preset_quality_btn = gr.Button("🎯 Calidad", elem_classes=["preset-btn"], size="sm")
                                            gr.HTML("""<p style="font-size: 12px; color: #b9cac8; margin-top: 8px;">★ Balanceado funciona bien para la mayoría de los videos. Para ajustar todo a mano, abrí "Ajustes avanzados".</p>""")

                                            with gr.Accordion("⚙️ Ajustes avanzados (opcional)", open=False):
                                                # Target Clip Count
                                                gr.HTML("""<label style="font-size: 14px; font-weight: 600; color: #e4e1e6; margin-bottom: 8px; display: block;">Cantidad de Clips Objetivo</label>""")
                                                num_clips = gr.Slider(1, 30, value=15, step=1, label="", show_label=False)

                                                # Duration Range
                                                gr.HTML("""
                                                <div style="margin-top: 16px; margin-bottom: 8px;">
                                                    <label style="font-size: 14px; font-weight: 600; color: #e4e1e6;">Rango de Duración de Clips</label>
                                                </div>
                                                """)
                                                with gr.Row():
                                                    min_duration = gr.Slider(5, 180, value=15, step=5, label="", show_label=False)
                                                    max_duration = gr.Slider(15, 300, value=60, step=5, label="", show_label=False)

                                                # Whisper Model
                                                gr.HTML("""<label style="font-size: 14px; font-weight: 600; color: #e4e1e6; margin: 16px 0 8px; display: block;">Modelo Whisper IA</label>""")
                                                model_size_dropdown = gr.Dropdown(
                                                    choices=[
                                                        ("Tiny (más rápido)", "tiny"),
                                                        ("Base (balanceado)", "base"),
                                                        ("Small (preciso)", "small"),
                                                    ],
                                                    value="base",
                                                    label="",
                                                    show_label=False
                                                )
                                                gr.HTML("""<p style="font-size: 12px; color: #b9cac8; margin-top: 8px;">Base proporciona el mejor balance de velocidad y precisión.</p>""")

                                                # Analysis mode
                                                gr.HTML("""<label style="font-size: 14px; font-weight: 600; color: #e4e1e6; margin: 16px 0 8px; display: block;">⚡ Modo de Análisis</label>""")
                                                analysis_mode_dropdown = gr.Dropdown(
                                                    choices=[
                                                        ("💨 Rápido (sin timestamps por palabra)", "fast"),
                                                        ("⚖️ Balance (2 pasadas, recomendado)", "balance"),
                                                        ("🎯 Calidad (timestamps completos)", "quality"),
                                                    ],
                                                    value="balance",
                                                    label="",
                                                    show_label=False
                                                )

                                            # Precheck info (filled on video select)
                                            video_precheck = gr.HTML("")

                                            # Custom prompt (natural language)
                                            gr.HTML("""<label style="font-size: 14px; font-weight: 600; color: #e4e1e6; margin: 16px 0 6px; display: block;">💭 Instrucción adicional (opcional)</label>""")
                                            custom_prompt_input = gr.Textbox(
                                                label="",
                                                placeholder='Ej: "encuentra los momentos más graciosos" o "clips de gaming"',
                                                show_label=False,
                                                lines=2
                                            )

                                            # Analyze + Cancel buttons
                                            gr.HTML("""<div style="margin-top: 20px;">""")
                                            with gr.Row():
                                                analyze_btn = gr.Button(
                                                    "✨ Analizar con IA",
                                                    variant="primary",
                                                    elem_classes=["btn-primary"],
                                                    scale=3
                                                )
                                                cancel_btn = gr.Button(
                                                    "⏹️ Cancelar",
                                                    elem_classes=["btn-secondary"],
                                                    scale=1,
                                                    interactive=False
                                                )
                                            gr.HTML("""</div>""")
                                            clear_cache_btn = gr.Button(
                                                "🗑️ Limpiar caché de transcripción",
                                                elem_classes=["btn-secondary"],
                                                size="sm"
                                            )

                                            with gr.Row():
                                                analysis_status = gr.Textbox(label="", value="Listo", show_label=False)
                                                token_usage = gr.Textbox(label="", value="~45 tokens", show_label=False)

                        # === TAB 2: EDIT ===
                        with gr.Column(visible=False) as tab_editor_content:
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.HTML("""
                                    <div style="margin-bottom: 8px;">
                                        <h1 style="font-size: 28px; font-weight: 700; color: #e4e1e6; margin-bottom: 4px;">Seleccionar Clips</h1>
                                        <p style="font-size: 14px; color: #b9cac8;">La IA detectó segmentos de alto potencial. Selecciona los que querés exportar.</p>
                                    </div>
                                    """)
                                with gr.Column(scale=0, min_width=320):
                                    with gr.Row():
                                        select_all_btn = gr.Button("☑️ Todos", elem_classes=["btn-secondary"], size="sm")
                                        deselect_all_btn = gr.Button("⬜ Ninguno", elem_classes=["btn-secondary"], size="sm")
                                    selected_count_txt = gr.Textbox(
                                        label="",
                                        value="0 clips seleccionados",
                                        show_label=False,
                                        interactive=False,
                                        container=False
                                    )
                                    go_export_btn = gr.Button(
                                        "→ Exportar clips (0)",
                                        variant="primary",
                                        elem_classes=["btn-primary"],
                                        scale=0,
                                        min_width=260,
                                    )
                            
                            with gr.Row():
                                # Left: Clip Cards + Timeline
                                with gr.Column(scale=8):
                                    # Selection counter bar
                                    with gr.Row(elem_classes=["glass-panel-sm"], visible=False) as selection_bar:
                                        gr.HTML("""<span style="font-size: 13px; color: #b9cac8;">📊 Selección:</span>""")
                                        selection_counter = gr.Textbox(
                                            label="",
                                            value="0/0 clips",
                                            show_label=False,
                                            interactive=False,
                                            container=False
                                        )
                                    # Clip Cards Container (filled dynamically)
                                    timeline_clips = gr.HTML("""
                                    <div class="clip-cards-container" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
                                        <div class="clip-card-placeholder">
                                            <span class="material-symbols-outlined" style="font-size: 48px; color: #b9cac8;">movie_filter</span>
                                            <p style="color: #b9cac8; margin-top: 12px;">Analiza un video para ver clips detectados</p>
                                        </div>
                                    </div>
                                    """)
                                    
                                    # Timeline
                                    with gr.Column(elem_classes=["glass-panel-sm"]):
                                        gr.HTML("""
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                            <span style="font-size: 14px; font-weight: 600; color: #e4e1e6;">Línea de Tiempo</span>
                                            <span style="font-size: 12px; color: #b9cac8;">00:00:00 / 01:45:20</span>
                                        </div>
                                        """)
                                        # Timeline visualization HTML
                                        gr.HTML("""
                                        <div class="timeline">
                                            <div class="timeline-waveform">
                                                <!-- Simulated waveform bars -->
                                                <div class="timeline-bar" style="height: 12px;"></div>
                                                <div class="timeline-bar" style="height: 24px;"></div>
                                                <div class="timeline-bar" style="height: 32px;"></div>
                                                <div class="timeline-bar" style="height: 16px;"></div>
                                                <div class="timeline-bar" style="height: 40px;"></div>
                                                <div class="timeline-bar" style="height: 48px;"></div>
                                                <div class="timeline-bar" style="height: 20px;"></div>
                                                <div class="timeline-bar" style="height: 8px;"></div>
                                            </div>
                                        </div>
                                        """)
                                        
                                        # Clip Controls
                                        with gr.Row():
                                            selected_clip_id = gr.Dropdown(label="Seleccionar Clip", choices=[])
                                            with gr.Row():
                                                clip_start = gr.Number(label="Inicio (s)", value=0)
                                                clip_end = gr.Number(label="Fin (s)", value=0)
                                            with gr.Row():
                                                update_time_btn = gr.Button("💾 Guardar", elem_classes=["btn-secondary"])
                                                select_clip_btn = gr.Button("✅ Seleccionar", elem_classes=["btn-secondary"])
                                                generate_preview_btn = gr.Button("▶️ Vista Previa", elem_classes=["btn-primary"])
                                            
                                            clip_action_status = gr.Textbox(label="Acción", interactive=False, show_label=False)
                                    
                                    # Analysis Result
                                    analysis_result = gr.Markdown("*El análisis de IA aparecerá aquí...*")
                                
                                # Right: AI Tools + Subtitle Editor
                                with gr.Column(scale=4):
                                    # AI Enhancements
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""
                                        <div class="panel-header" style="display: flex; align-items: center; gap: 8px;">
                                            <span class="material-symbols-outlined" style="color: #dcb8ff;">auto_awesome</span>
                                            <div class="panel-header-title" style="margin: 0;">Mejoras IA</div>
                                            <div style="flex: 1; height: 1px; background: linear-gradient(90deg, #dcb8ff, transparent); margin-left: 8px;"></div>
                                        </div>
                                        """)
                                        
                                        with gr.Column():
                                            # Auto-correct
                                            auto_correct_btn = gr.Button("✨ Auto-corregir Subtítulos", elem_classes=["btn-secondary"])
                                            # Remove fillers
                                            remove_filler_btn = gr.Button("🗑️ Eliminar Muletillas y Pausas", elem_classes=["btn-secondary"])
                                            # Add emojis
                                            add_emoji_btn = gr.Button("😊 Auto-Emojis", elem_classes=["btn-secondary"])
                                    
                                    # Subtitle Editor
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""
                                        <div class="panel-header" style="display: flex; justify-content: space-between; align-items: center;">
                                            <div style="display: flex; align-items: center; gap: 8px;">
                                                <span class="material-symbols-outlined">subtitles</span>
                                                <div class="panel-header-title" style="margin: 0;">Subtítulos</div>
                                            </div>
                                            <button class="btn-ghost" style="color: #7C5CFF;">Estilos</button>
                                        </div>
                                        """)
                                        
                                        subtitle_df = gr.Dataframe(
                                            headers=["ID", "Inicio", "Fin", "Original", "Editado", "✓"],
                                            label="",
                                            show_label=False,
                                            wrap=True,
                                            interactive=True,
                                            type="array",
                                        )
                                        
                                        with gr.Row():
                                            refresh_subs_btn = gr.Button("🔄 Recargar", elem_classes=["btn-secondary"], size="sm")
                                            subtitle_edit_row = gr.Number(label="Fila ID", value=0)
                                            subtitle_new_text = gr.Textbox(label="Nuevo Texto")
                                        update_subtitle_btn = gr.Button("💾 Guardar Edición", elem_classes=["btn-secondary"])
                                        subtitle_status = gr.Textbox(label="", show_label=False)
                                    
                                    # Video Preview
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""<div class="panel-header"><div class="panel-header-title">Vista Previa 9:16</div></div>""")
                                        preview_video = gr.Video(label="", height=400, elem_classes=["clean-video"])
                                        clip_info = gr.Markdown("Selecciona un clip para ver detalles...")
                        
                        # === TAB 3: EXPORT ===
                        with gr.Column(visible=False) as tab_export_content:
                            with gr.Row():
                                back_to_editor_btn = gr.Button(
                                    "← Volver a clips",
                                    elem_classes=["btn-secondary"],
                                    scale=0,
                                    min_width=160,
                                )
                                gr.HTML("""
                                <div style="margin-bottom: 8px;">
                                    <h1 style="font-size: 28px; font-weight: 700; color: #e4e1e6; margin-bottom: 4px;">Exportar Proyecto</h1>
                                    <p style="font-size: 14px; color: #b9cac8;">Configurá el estilo y exportá tus clips para redes sociales.</p>
                                </div>
                                """)
                            
                            with gr.Row():
                                # Left: Settings
                                with gr.Column(scale=8):
                                    # Visual Style Selector
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""<div class="panel-header"><div class="panel-header-title">Estilo Visual</div></div>""")
                                        with gr.Row():
                                            style_dropdown = gr.Dropdown(
                                                choices=list(SUBTITLE_STYLES.keys()),
                                                value="modern",
                                                label="",
                                                show_label=False
                                            )
                                        # Style cards as clickable buttons
                                        with gr.Row():
                                            style_modern_btn = gr.Button("✨ Moderno", elem_classes=["style-card-btn", "style-card-active"])
                                            style_tiktok_btn = gr.Button("🎵 TikTok", elem_classes=["style-card-btn"])
                                            style_minimal_btn = gr.Button("⬜ Minimal", elem_classes=["style-card-btn"])
                                            style_classic_btn = gr.Button("🎬 Clásico", elem_classes=["style-card-btn"])
                                    
                                    # AI Features Toggle
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""<div class="panel-header"><div class="panel-header-title">Mejoras IA</div></div>""")
                                        
                                        with gr.Column():
                                            with gr.Row(elem_classes=["ai-tool"]):
                                                gr.HTML("""
                                                <div class="ai-tool-left">
                                                    <div class="ai-tool-icon">
                                                        <span class="material-symbols-outlined">face_retouching_natural</span>
                                                    </div>
                                                    <div class="ai-tool-info">
                                                        <div class="ai-tool-name">Seguimiento Facial IA</div>
                                                        <div class="ai-tool-desc">Mantiene al sujeto principal centrado automáticamente</div>
                                                    </div>
                                                </div>
                                                """)
                                                face_tracking_checkbox = gr.Checkbox(
                                                    label="",
                                                    value=False,
                                                    show_label=False,
                                                    interactive=FACE_TRACKING_AVAILABLE,
                                                    info="" if FACE_TRACKING_AVAILABLE else "⚠️ No disponible (OpenCV no instalado)"
                                                )
                                            
                                            with gr.Row(elem_classes=["ai-tool"]):
                                                gr.HTML("""
                                                <div class="ai-tool-left">
                                                    <div class="ai-tool-icon">
                                                        <span class="material-symbols-outlined">closed_caption</span>
                                                    </div>
                                                    <div class="ai-tool-info">
                                                        <div class="ai-tool-name">Subtítulos Automáticos</div>
                                                        <div class="ai-tool-desc">Genera subtítulos dinámicos</div>
                                                    </div>
                                                </div>
                                                """)
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

                                            with gr.Row(elem_classes=["ai-tool"]):
                                                gr.HTML("""
                                                <div class="ai-tool-left">
                                                    <div class="ai-tool-icon">
                                                        <span class="material-symbols-outlined">tag</span>
                                                    </div>
                                                    <div class="ai-tool-info">
                                                        <div class="ai-tool-name">Hook al Inicio</div>
                                                        <div class="ai-tool-desc">Quema la frase gancho de la IA en pantalla los primeros segundos, con el mismo estilo visual elegido arriba</div>
                                                    </div>
                                                </div>
                                                """)
                                                show_hook_checkbox = gr.Checkbox(
                                                    label="",
                                                    value=True,
                                                    show_label=False,
                                                )

                                # Right: Export Action
                                with gr.Column(scale=4):
                                    with gr.Column(elem_classes=["glass-panel"]):
                                        gr.HTML("""
                                        <div class="panel-header" style="display: flex; align-items: center; gap: 8px;">
                                            <span class="material-symbols-outlined" style="color: #7C5CFF;">rocket_launch</span>
                                            <div class="panel-header-title" style="margin: 0;">Listo para Renderizar</div>
                                        </div>
                                        """)
                                        
                                        # Export Info
                                        gr.HTML("""
                                        <div style="background: rgba(53, 52, 56, 0.4); padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 16px;">
                                            </div>
                                        </div>
                                        """)
                                        
                                        # Platform preset
                                        gr.HTML("""<label style="font-size: 13px; font-weight: 600; color: #e4e1e6; margin-bottom: 6px; display: block;">📱 Plataforma destino</label>""")
                                        platform_preset = gr.Dropdown(
                                            choices=[
                                                ("TikTok (9:16)", "tiktok"),
                                                ("Instagram Reels (9:16)", "reels"),
                                                ("YouTube Shorts (9:16)", "shorts"),
                                                ("LinkedIn (1:1)", "linkedin"),
                                                ("Twitter/X (1:1)", "twitter"),
                                                ("Landscape (16:9)", "landscape"),
                                            ],
                                            value="tiktok",
                                            label="",
                                            show_label=False
                                        )
                                        gr.HTML("""<div style="margin-top: 12px;">""")
                                        
                                        # Export SRT option
                                        export_srt_checkbox = gr.Checkbox(
                                            label="Exportar subtítulos (.srt)",
                                            value=True,
                                            show_label=True
                                        )
                                        export_vtt_checkbox = gr.Checkbox(
                                            label="Exportar subtítulos web (.vtt)",
                                            value=True,
                                            show_label=True
                                        )
                                        gr.HTML("""</div>""")

                                        # Post-processing options
                                        gr.HTML("""<div style="margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                                            <div style="font-size: 13px; font-weight: 700; color: #e4e1e6; margin-bottom: 6px;">✨ Post-procesamiento</div>
                                        </div>""")
                                        mood_grade_checkbox = gr.Checkbox(
                                            label="Color grading por mood (F3.1)",
                                            value=True,
                                            show_label=True
                                        )
                                        audio_ducking_checkbox = gr.Checkbox(
                                            label="Audio ducking automático (F3.4)",
                                            value=True,
                                            show_label=True
                                        )
                                        zoom_cues_checkbox = gr.Checkbox(
                                            label="Zoom dinámico por energía (F2.3)",
                                            value=True,
                                            show_label=True
                                        )
                                        compress_pauses_checkbox = gr.Checkbox(
                                            label="✂️ Comprimir pausas largas (F4)",
                                            value=False,
                                            show_label=True,
                                            info="Acorta silencios de +1.2s entre frases para mantener el ritmo"
                                        )

                                        # Brand kit básico
                                        gr.HTML("""<div style="margin-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px;">
                                            <div style="font-size: 13px; font-weight: 700; color: #e4e1e6; margin-bottom: 8px;">🎨 Brand Kit</div>
                                        </div>""")
                                        brand_name_input = gr.Textbox(
                                            label="",
                                            placeholder="Nombre de marca / canal",
                                            show_label=False
                                        )
                                        brand_color_input = gr.Textbox(
                                            label="",
                                            value="#7C5CFF",
                                            placeholder="Color primario (#7C5CFF)",
                                            show_label=False
                                        )
                                        
                                        # Progress
                                        export_status = gr.Textbox(label="", value="Listo para exportar", show_label=False)
                                        
                                        # Export Button
                                        export_btn = gr.Button(
                                            "🚀 Exportar Video",
                                            variant="primary",
                                            elem_classes=["btn-primary"]
                                        )
                            
                            # Export Gallery
                            with gr.Column(elem_classes=["glass-panel"], scale=1):
                                gr.HTML("""
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                    <div class="panel-header-title" style="margin: 0;">Exports Recientes</div>
                                    <button class="btn-ghost" style="color: #7C5CFF; display: flex; align-items: center; gap: 4px;">
                                        Ver Todos <span class="material-symbols-outlined" style="font-size: 16px;">arrow_forward</span>
                                    </button>
                                </div>
                                """)
                                output_gallery = gr.Gallery(label="", columns=4, rows=1, height=300)
                                output_files = gr.File(label="Descargar archivos", file_count="multiple")
                                captions_output = gr.Textbox(
                                    label="📋 Captions listos para publicar (título + descripción + hashtags por clip)",
                                    lines=10,
                                    max_lines=24,
                                    interactive=False,
                                    buttons=["copy"],
                                    placeholder="Acá vas a ver el título, caption y hashtags de cada clip después de exportar, listos para copiar y pegar en TikTok/IG/YouTube."
                                )

                                # Social Sharing
                                gr.HTML("""
                                <div style="margin-top: 20px;">
                                    <div style="font-size: 14px; font-weight: 600; color: #e4e1e6; margin-bottom: 12px;">Compartir</div>
                                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                                        <a href="https://www.facebook.com/sharer/sharer.php?u=opusclip.pro" target="_blank" rel="noopener" 
                                           style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; background: #1877f2; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 13px;">
                                            <span class="material-symbols-outlined" style="font-size: 16px;">facebook</span> Facebook
                                        </a>
                                        <a href="https://twitter.com/intent/tweet?text=¡Mira este clip de IA!&url=opusclip.pro" target="_blank" rel="noopener"
                                           style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; background: #1da1f2; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 13px;">
                                            <span class="material-symbols-outlined" style="font-size: 16px;">share</span> Twitter
                                        </a>
                                        <a href="https://www.linkedin.com/sharing/share-offsite/?url=opusclip.pro" target="_blank" rel="noopener"
                                           style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; background: #0a66c2; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 13px;">
                                            <span class="material-symbols-outlined" style="font-size: 16px;">work</span> LinkedIn
                                        </a>
                                    </div>
                                </div>
                                """)
            
            # === EVENTOS ===
            
            def on_video_select(video_paths, model_sz):
                """Precheck: show duration + ETA estimate when video(s) are selected."""
                if not video_paths:
                    return "", "No hay video seleccionado"
                if len(video_paths) > 1:
                    names = ", ".join(Path(v).name for v in video_paths[:3])
                    more = f" y {len(video_paths) - 3} más" if len(video_paths) > 3 else ""
                    html = f"""<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px 14px;margin:8px 0;font-size:13px;color:#b9cac8;">
                        <b style="color:#e4e1e6;">📦 {len(video_paths)} videos seleccionados</b><br>
                        {names}{more} — se analizan uno por uno, cada uno queda como proyecto separado.
                    </div>"""
                    return html, f"📦 {len(video_paths)} videos en cola"
                video_path = video_paths[0]
                try:
                    self._init_components(model_size=model_sz)
                    info = self.transcriber.estimate_analysis_time(video_path, model_sz)
                    dur = info['duration_min']
                    est = info['estimated_total_s']
                    eta_str = f"{int(est//60)}m {int(est%60)}s" if est >= 60 else f"{est:.0f}s"
                    mode_est = int(est * 0.6)  # balance ~40% faster
                    mode_str = f"{int(mode_est//60)}m {int(mode_est%60)}s" if mode_est >= 60 else f"{mode_est:.0f}s"
                    cache_badge = ' <span style="background:#7C5CFF22;color:#7C5CFF;padding:2px 7px;border-radius:10px;font-size:11px;">⚡ caché</span>' if info['cached'] else ''
                    chunked_badge = ' <span style="background:#f59e0b22;color:#f59e0b;padding:2px 7px;border-radius:10px;font-size:11px;">chunks</span>' if info['chunked'] else ''
                    html = f"""<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px 14px;margin:8px 0;font-size:13px;color:#b9cac8;">
                        <b style="color:#e4e1e6;">⏱️ Video: {dur:.1f} min</b>{cache_badge}{chunked_badge}<br>
                        ETA modo Balance: <b style="color:#7C5CFF;">~{mode_str}</b> &nbsp;|
                        ETA Calidad: ~{eta_str}
                    </div>"""
                    vi_text = f"⏱️ {dur:.1f} min | ETA ~{mode_str} (balance)"
                    return html, vi_text
                except Exception as e:
                    logger.warning(f"Precheck de video falló: {e}")
                    return "", f"Video seleccionado: {Path(video_path).name if video_path else ''}"

            video_input.change(
                fn=on_video_select,
                inputs=[video_input, model_size_dropdown],
                outputs=[video_precheck, video_info]
            )

            def _apply_analysis_preset(n_clips: int, mn: int, mx: int, model: str, mode: str):
                return (
                    gr.update(value=n_clips), gr.update(value=mn), gr.update(value=mx),
                    gr.update(value=model), gr.update(value=mode)
                )

            preset_outputs = [num_clips, min_duration, max_duration, model_size_dropdown, analysis_mode_dropdown]
            preset_fast_btn.click(fn=lambda: _apply_analysis_preset(8, 15, 45, "tiny", "fast"), inputs=[], outputs=preset_outputs)
            preset_balance_btn.click(fn=lambda: _apply_analysis_preset(15, 15, 60, "base", "balance"), inputs=[], outputs=preset_outputs)
            preset_quality_btn.click(fn=lambda: _apply_analysis_preset(10, 20, 90, "small", "quality"), inputs=[], outputs=preset_outputs)

            def on_analyze(video, n_clips, min_dur, max_dur, model_sz, custom_p, mode, progress=gr.Progress()):
                if not video:
                    return (
                        "❌ Selecciona al menos un video", "", "",
                        gr.update(visible=False), gr.update(choices=[]), "0 tokens",
                        gr.update(interactive=True), gr.update(interactive=False),
                        "0 clips seleccionados", "→ Exportar clips (0)", "0/0 clips", gr.update(visible=False)
                    )
                status, clips_sum, analysis_sum, timeline_vis, clip_dropdown, tokens = self.analyze_video_batch(
                    video, n_clips, min_dur, max_dur, model_sz, progress,
                    custom_prompt=custom_p, analysis_mode=mode
                )
                # Calculate initial selection count (all clips selected by default)
                if self.current_state:
                    count, total = self._get_selected_count()
                    count_txt = f"{count} clip{'s' if count != 1 else ''} seleccionado{'s' if count != 1 else ''}"
                    btn_txt = f"→ Exportar clips ({count})"
                    counter_txt = f"{count}/{total} clips"
                    bar_vis = gr.update(visible=True)
                else:
                    count_txt, btn_txt, counter_txt, bar_vis = "0 clips seleccionados", "→ Exportar clips (0)", "0/0 clips", gr.update(visible=False)
                return (
                    status, clips_sum, analysis_sum,
                    gr.update(visible=True), clip_dropdown, tokens,
                    gr.update(interactive=True), gr.update(interactive=False),
                    count_txt, btn_txt, counter_txt, bar_vis
                )

            analyze_btn.click(
                fn=lambda: (gr.update(interactive=False), gr.update(interactive=True)),
                inputs=[],
                outputs=[analyze_btn, cancel_btn]
            ).then(
                fn=on_analyze,
                inputs=[video_input, num_clips, min_duration, max_duration,
                        model_size_dropdown, custom_prompt_input, analysis_mode_dropdown],
                outputs=[analysis_status, timeline_clips, analysis_result,
                         tab_editor_content, selected_clip_id, token_usage,
                         analyze_btn, cancel_btn,
                         selected_count_txt, go_export_btn, selection_counter, selection_bar]
            ).then(
                fn=lambda status: switch_tab("edit") if ("clips identificados" in status or "Lote completo" in status) else (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
                inputs=[analysis_status],
                outputs=[tab_import, tab_editor_content, tab_export_content]
            ).then(
                fn=lambda: self.refresh_project_list(""),
                inputs=[],
                outputs=[recent_projects_html, saved_project_dropdown]
            )

            cancel_btn.click(
                fn=self.cancel_analysis,
                inputs=[],
                outputs=[analysis_status]
            ).then(
                fn=lambda: (gr.update(interactive=True), gr.update(interactive=False)),
                inputs=[],
                outputs=[analyze_btn, cancel_btn]
            )

            def _clear_transcription_cache():
                import shutil as _sh
                cache_dir = Path("temp/transcription_cache")
                if cache_dir.exists():
                    count = len(list(cache_dir.glob("*.json")))
                    _sh.rmtree(cache_dir, ignore_errors=True)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    return f"🗑️ Caché limpiado ({count} archivos eliminados)"
                return "✅ Caché ya estaba vacío"

            clear_cache_btn.click(
                fn=_clear_transcription_cache,
                inputs=[],
                outputs=[analysis_status]
            )

            refresh_projects_btn.click(
                fn=lambda q: self.refresh_project_list(q),
                inputs=[project_search_input],
                outputs=[recent_projects_html, saved_project_dropdown]
            )
            
            project_search_input.change(
                fn=lambda q: self.refresh_project_list(q),
                inputs=[project_search_input],
                outputs=[recent_projects_html, saved_project_dropdown]
            )
            
            def _on_load_project(project_name):
                """Wrapper to handle project loading with selection counters."""
                results = self.load_saved_project(project_name)
                if self.current_state:
                    count, total = self._get_selected_count()
                    count_txt = f"{count} clip{'s' if count != 1 else ''} seleccionado{'s' if count != 1 else ''}"
                    btn_txt = f"→ Exportar clips ({count})"
                    counter_txt = f"{count}/{total} clips"
                    return (*results, count_txt, btn_txt, counter_txt, gr.update(visible=True))
                return (*results, "0 clips seleccionados", "→ Exportar clips (0)", "0/0 clips", gr.update(visible=False))

            load_project_btn.click(
                fn=_on_load_project,
                inputs=[saved_project_dropdown],
                outputs=[analysis_status, timeline_clips, analysis_result, selected_clip_id, video_info,
                         tab_import, tab_editor_content, tab_export_content,
                         selected_count_txt, go_export_btn, selection_counter, selection_bar]
            )
            
            MOOD_EMOJI_SEL = {
                'funny': '😂', 'shocking': '😱', 'educational': '📚',
                'motivational': '💪', 'dramatic': '🎭', 'controversial': '🔥',
                'wholesome': '🥰', 'gaming_hype': '🎮', 'storytelling': '📖',
                'inspirational': '✨', 'neutral': '🎬'
            }
            PLAT_EMOJI_SEL = {
                'tiktok': '🎵', 'reels': '📸', 'shorts': '▶️',
                'linkedin': '💼', 'twitter': '🐦', 'landscape': '🖥️'
            }

            def on_select_clip(clip_id):
                if not self.current_state:
                    return "No hay proyecto activo", [], 0, 0
                clip = next((c for c in self.current_state.clips if c.id == clip_id), None)
                if clip is None:
                    return "Clip no encontrado", [], 0, 0
                sc = self._score_color(clip.virality_score)
                hc = self._score_color(clip.hook_score)
                pc = self._score_color(clip.pacing_score)
                ec = self._score_color(clip.engagement_score)
                fc = self._score_color(getattr(clip, 'flow_score', 0))
                vc = self._score_color(getattr(clip, 'value_score', 0))
                tc = self._score_color(getattr(clip, 'trend_score', 5))
                mood = getattr(clip, 'mood', 'neutral')
                hook_type = getattr(clip, 'hook_type', 'unknown')
                ideal_platform = getattr(clip, 'ideal_platform', 'tiktok')
                edit_recipe = getattr(clip, 'edit_recipe', '')
                mood_icon = MOOD_EMOJI_SEL.get(mood, '🎬')
                plat_icon = PLAT_EMOJI_SEL.get(ideal_platform, '📱')
                flow_s = getattr(clip, 'flow_score', 0)
                value_s = getattr(clip, 'value_score', 0)
                trend_s = getattr(clip, 'trend_score', 5)
                recipe_html = (
                    f'<div style="background:#0d1b2a;border-left:3px solid #7C5CFF;border-radius:6px;'
                    f'padding:8px 10px;margin-top:10px;font-size:0.8em;color:#a8d8ea;">'
                    f'🎨 <strong>Receta de edición:</strong> {edit_recipe}</div>'
                ) if edit_recipe else ''
                info = f"""
                <div style="background: #16213e; padding: 15px; border-radius: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #e4e1e6;">🎬 Clip {clip_id + 1}</h4>
                        <div style="text-align: center; background: {sc}22; border: 3px solid {sc};
                                    border-radius: 50%; width: 72px; height: 72px; display: flex;
                                    flex-direction: column; align-items: center; justify-content: center;
                                    box-shadow: 0 0 16px {sc}55;">
                            <span style="color: {sc}; font-weight: 900; font-size: 1.6em; line-height: 1;">{clip.virality_score:.1f}</span>
                            <span style="color: {sc}aa; font-size: 0.58em; letter-spacing: 0.04em; text-transform: uppercase;">Virality</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px;">
                        <span style="background:#ffffff15;border-radius:20px;padding:2px 9px;font-size:0.72em;">{mood_icon} {mood}</span>
                        <span style="background:#ffffff15;border-radius:20px;padding:2px 9px;font-size:0.72em;">🪝 {hook_type.replace('_',' ')}</span>
                        <span style="background:#ffffff15;border-radius:20px;padding:2px 9px;font-size:0.72em;">{plat_icon} {ideal_platform}</span>
                        <span style="background:#ffffff15;border-radius:20px;padding:2px 9px;font-size:0.72em;">{"✅ Sel." if clip.selected else "⏸️ Omit."}</span>
                    </div>
                    <p style="font-size: 0.83em; color: #a8d8ea; margin: 0 0 6px 0;"><strong>⏱️</strong> {clip.start:.1f}s → {clip.end:.1f}s &nbsp;·&nbsp; {clip.duration:.1f}s</p>
                    <p style="font-size: 0.83em; color: #d4e5ed; margin: 0 0 6px 0;"><strong>🎯 Hook:</strong> {clip.hook[:120]}</p>
                    <p style="font-size: 0.78em; color: #b8e0f0; font-style: italic; margin: 0 0 8px 0;">{clip.reason[:180]}</p>
                    <div style="display: grid; grid-template-columns: repeat(6,1fr); gap: 6px; margin-top: 8px;">
                        {self._score_bar('Hook', clip.hook_score, hc)}
                        {self._score_bar('Ritmo', clip.pacing_score, pc)}
                        {self._score_bar('Engage', clip.engagement_score, ec)}
                        {self._score_bar('Flow', flow_s, fc)}
                        {self._score_bar('Valor', value_s, vc)}
                        {self._score_bar('Trend', trend_s, tc)}
                    </div>
                    {recipe_html}
                </div>
                """
                sub_data = self.get_subtitle_data(clip_id)
                return info, sub_data, clip.start, clip.end
            
            selected_clip_id.change(
                fn=on_select_clip,
                inputs=[selected_clip_id],
                outputs=[clip_info, subtitle_df, clip_start, clip_end]
            )
            
            update_time_btn.click(
                fn=lambda cid, start, end: self.update_clip_time(cid, start, end),
                inputs=[selected_clip_id, clip_start, clip_end],
                outputs=[clip_action_status]
            )
            
            select_clip_btn.click(
                fn=lambda cid: self.toggle_clip_selection(cid),
                inputs=[selected_clip_id],
                outputs=[clip_action_status, timeline_clips, selected_count_txt, go_export_btn]
            )

            select_all_btn.click(
                fn=self.select_all_clips,
                inputs=[],
                outputs=[clip_action_status, timeline_clips, selected_count_txt, go_export_btn, selection_counter]
            )

            deselect_all_btn.click(
                fn=self.deselect_all_clips,
                inputs=[],
                outputs=[clip_action_status, timeline_clips, selected_count_txt, go_export_btn, selection_counter]
            )
            
            generate_preview_btn.click(
                fn=lambda cid: self.generate_preview(cid),
                inputs=[selected_clip_id],
                outputs=[preview_video]
            )
            
            subtitle_df.input(
                fn=lambda df_data, cid: self.update_subtitles_from_dataframe(cid, df_data),
                inputs=[subtitle_df, selected_clip_id],
                outputs=[subtitle_status, subtitle_df]
            )

            refresh_subs_btn.click(
                fn=lambda cid: self.get_subtitle_data(cid),
                inputs=[selected_clip_id],
                outputs=[subtitle_df]
            )
            
            auto_correct_btn.click(
                fn=lambda cid: self.apply_auto_corrections(cid),
                inputs=[selected_clip_id],
                outputs=[subtitle_status, subtitle_df]
            )
            
            update_subtitle_btn.click(
                fn=lambda cid, rid, txt: self.update_subtitle(cid, int(rid), txt),
                inputs=[selected_clip_id, subtitle_edit_row, subtitle_new_text],
                outputs=[subtitle_status]
            )
            
            remove_filler_btn.click(
                fn=lambda cid: self.remove_filler_words(cid),
                inputs=[selected_clip_id],
                outputs=[subtitle_status, subtitle_df]
            )
            
            add_emoji_btn.click(
                fn=lambda cid: self.add_emojis(cid),
                inputs=[selected_clip_id],
                outputs=[subtitle_status, subtitle_df]
            )
            
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
            
            # === STYLE BUTTON EVENTS ===
            def set_style(style_name):
                return style_name
            
            style_modern_btn.click(fn=lambda: set_style("modern"), outputs=[style_dropdown])
            style_tiktok_btn.click(fn=lambda: set_style("tiktok"), outputs=[style_dropdown])
            style_minimal_btn.click(fn=lambda: set_style("minimal"), outputs=[style_dropdown])
            style_classic_btn.click(fn=lambda: set_style("classic"), outputs=[style_dropdown])
            
            # === NAVIGATION EVENTS ===
            def switch_tab(tab_name):
                """Switch between tabs."""
                if tab_name == "import":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                elif tab_name == "edit":
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                else:  # export
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            
            nav_import_btn.click(
                fn=lambda: switch_tab("import"),
                outputs=[tab_import, tab_editor_content, tab_export_content]
            )
            
            nav_edit_btn.click(
                fn=lambda: switch_tab("edit"),
                outputs=[tab_import, tab_editor_content, tab_export_content]
            )
            
            nav_export_btn.click(
                fn=lambda: switch_tab("export"),
                outputs=[tab_import, tab_editor_content, tab_export_content]
            )

            # Wizard navigation buttons
            go_export_btn.click(
                fn=lambda: switch_tab("export"),
                outputs=[tab_import, tab_editor_content, tab_export_content]
            )

            back_to_editor_btn.click(
                fn=lambda: switch_tab("edit"),
                outputs=[tab_import, tab_editor_content, tab_export_content]
            )

        return app


def main():
    logger.info("Iniciando OpusClip Pro...")
    app = OpusClipPro()
    ui = app.create_ui()
    
    ui.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        show_error=True,
        css=app._css
    )


if __name__ == "__main__":
    main()
