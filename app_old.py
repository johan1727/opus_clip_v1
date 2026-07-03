"""
Aplicación principal - UI profesional tipo OpusClip/CapCut.
Panel de 3 secciones: Timeline | Preview | Editor de Subtítulos
"""

import logging
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

import gradio as gr

from config import config, SUPPORTED_VIDEO_FORMATS, ERRORS, SUCCESS, SUBTITLE_STYLES
from transcriber import Transcriber
from llm_analyzer import GeminiAnalyzer, ViralClip
from video_editor import VideoEditor
from state_manager import StateManager, ProjectState, ClipState
from subtitle_editor import SubtitleEditor, create_editor_from_transcription, PREDEFINED_STYLES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpusClipPro:
    """Aplicación OpusClip con panel profesional de edición."""
    
    def __init__(self):
        self.transcriber: Optional[Transcriber] = None
        self.analyzer: Optional[GeminiAnalyzer] = None
        self.editor: Optional[VideoEditor] = None
        self.state_manager = StateManager()
        
        # Estado actual
        self.current_video: Optional[str] = None
        self.current_state: Optional[ProjectState] = None
        self.current_preview: Optional[str] = None
        self.subtitle_editors: Dict[int, SubtitleEditor] = {}  # clip_id -> editor
        
    def _init_components(self):
        """Inicializa componentes lazy-loading."""
        if self.transcriber is None:
            logger.info("Inicializando Whisper...")
            self.transcriber = Transcriber(
                model_size=config.WHISPER_MODEL,
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
            return False, "Selecciona un archivo de video"
            
        path = Path(video_path)
        
        if not path.exists():
            return False, ERRORS['VIDEO_NOT_FOUND']
        
        if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            return False, ERRORS['UNSUPPORTED_FORMAT']
        
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 2048:  # 2GB
            return False, ERRORS['VIDEO_TOO_LARGE']
        
        return True, "✅ Video válido"
    
    def analyze_video(
        self,
        video_path: str,
        num_clips: int,
        min_duration: int,
        max_duration: int,
        progress: gr.Progress
    ) -> Tuple[str, str, str, gr.update, gr.update]:
        """
        Paso 1-2: Transcripción y análisis de clips virales.
        
        Returns:
            (status, resumen clips, resumen análisis, update timeline, update video info)
        """
        is_valid, msg = self.validate_video(video_path)
        if not is_valid:
            return msg, "", "", gr.update(visible=False), gr.update()
        
        self.current_video = video_path
        
        try:
            self._init_components()
            
            # Transcripción
            progress(0.1, desc="🎙️ Transcribiendo con Whisper...")
            transcription = self.transcriber.transcribe(video_path)
            
            duration = transcription['duration']
            num_segments = len(transcription['segments'])
            
            progress(0.4, desc="🧠 Analizando con Gemini...")
            
            # Análisis
            viral_clips = self.analyzer.analyze_transcription(
                transcription,
                num_clips=num_clips,
                clip_duration_range=(min_duration, max_duration)
            )
            
            if not viral_clips:
                return "⚠️ No se identificaron clips", "", "", gr.update(visible=False), gr.update()
            
            # Preparar datos para estado
            clips_data = []
            for clip in viral_clips:
                segments = self.transcriber.get_segments_with_text(
                    transcription, clip.start, clip.end
                )
                clips_data.append({
                    'start': clip.start,
                    'end': clip.end,
                    'virality_score': clip.virality_score,
                    'reason': clip.reason,
                    'hook': clip.hook,
                    'segments': segments
                })
            
            # Crear estado del proyecto
            self.current_state = self.state_manager.create_project(
                video_path, transcription, clips_data
            )
            
            # Crear editores de subtítulos para cada clip
            self.subtitle_editors = {}
            for clip_state in self.current_state.clips:
                editor = create_editor_from_transcription(
                    clip_state.segments, style_name="modern"
                )
                self.subtitle_editors[clip_state.id] = editor
            
            progress(1.0, desc="✅ Análisis completado!")
            
            # Construir resúmenes
            clips_summary = self._build_clips_summary()
            analysis_summary = self._build_analysis_summary(viral_clips, duration, num_segments)
            
            return (
                f"✅ {len(viral_clips)} clips identificados",
                clips_summary,
                analysis_summary,
                gr.update(visible=True),
                gr.update(value=f"📹 {Path(video_path).name} | {duration:.0f}s | {num_segments} segmentos")
            )
            
        except Exception as e:
            logger.error(f"Error en análisis: {e}", exc_info=True)
            return f"❌ Error: {str(e)}", "", "", gr.update(visible=False), gr.update()
    
    def _build_clips_summary(self) -> str:
        """Construye resumen visual de los clips para mostrar en timeline."""
        if not self.current_state:
            return ""
        
        parts = []
        for i, clip in enumerate(self.current_state.clips, 1):
            hook_preview = clip.hook[:50] + "..." if len(clip.hook) > 50 else clip.hook
            parts.append(
                f"### 🎬 Clip {i} | Score: {clip.virality_score:.1f}/10\n"
                f"**🎯 Hook:** {hook_preview}\n"
                f"**⏱️ Tiempo:** {clip.start:.1f}s - {clip.end:.1f}s ({clip.duration:.1f}s)\n"
                f"**💡 Razón:** {clip.reason[:80]}...\n"
                f"---"
            )
        return "\n".join(parts)
    
    def _build_analysis_summary(self, clips: List[ViralClip], duration: float, segments: int) -> str:
        """Construye resumen del análisis de Gemini."""
        parts = [
            f"### 📊 Análisis del Video\n",
            f"• **Duración total:** {duration:.0f} segundos\n",
            f"• **Segmentos detectados:** {segments}\n",
            f"• **Clips virales identificados:** {len(clips)}\n\n",
            f"### 🏆 Ranking de Viralidad\n"
        ]
        
        sorted_clips = sorted(clips, key=lambda x: x.virality_score, reverse=True)
        for i, clip in enumerate(sorted_clips, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐"
            parts.append(
                f"{emoji} **Clip #{i}** - Score: `{clip.virality_score:.1f}`\n"
                f"   💬 {clip.reason[:60]}...\n"
            )
        
        return "\n".join(parts)
    
    def update_clip_time(self, clip_id: int, new_start: float, new_end: float) -> str:
        """Actualiza los tiempos de un clip."""
        if not self.current_state:
            return "❌ No hay proyecto activo"
        
        success = self.state_manager.update_clip(clip_id, start=new_start, end=new_end)
        
        if success:
            self.current_state = self.state_manager.current_state
            return f"✅ Clip {clip_id + 1} actualizado: {new_start:.1f}s - {new_end:.1f}s"
        return "❌ Error actualizando clip"
    
    def toggle_clip_selection(self, clip_id: int, selected: bool) -> str:
        """Activa/desactiva un clip para exportación."""
        if not self.current_state:
            return ""
        
        success = self.state_manager.update_clip(clip_id, selected=selected)
        
        if success:
            self.current_state = self.state_manager.current_state
            status = "✅ seleccionado" if selected else "⏸️ omitido"
            return f"Clip {clip_id + 1} {status}"
        return "Error"
    
    def generate_preview(self, clip_id: int) -> Optional[str]:
        """Genera un preview del clip seleccionado."""
        if not self.current_state or not self.current_video:
            return None
        
        clip = None
        for c in self.current_state.clips:
            if c.id == clip_id:
                clip = c
                break
        
        if not clip:
            return None
        
        preview_path = config.TEMP_DIR / f"preview_clip_{clip_id}.mp4"
        
        try:
            self.editor.create_preview_segment(
                self.current_video,
                str(preview_path),
                clip.start,
                clip.end,
                duration_limit=15.0
            )
            return str(preview_path)
        except Exception as e:
            logger.error(f"Error generando preview: {e}")
            return None
    
    def get_subtitle_editor_data(self, clip_id: int) -> List[Dict[str, Any]]:
        """Obtiene datos del editor de subtítulos para DataFrame."""
        if clip_id not in self.subtitle_editors:
            return []
        
        editor = self.subtitle_editors[clip_id]
        return editor.get_entries_for_dataframe()
    
    def update_subtitle(self, clip_id: int, row_index: int, new_text: str) -> str:
        """Actualiza un subtítulo específico."""
        if clip_id not in self.subtitle_editors:
            return "❌ Editor no encontrado"
        
        editor = self.subtitle_editors[clip_id]
        entry_id = row_index  # Asumiendo que coincide
        
        success = editor.edit_entry(entry_id, new_text)
        
        if success:
            # Guardar en estado
            self.state_manager.update_subtitle_edit(clip_id, entry_id, new_text)
            return f"✅ Subtítulo {entry_id} actualizado"
        return "❌ Error al actualizar"
    
    def export_clips(self, style_name: str, progress: gr.Progress) -> Tuple[str, List[str]]:
        """Exporta los clips seleccionados."""
        if not self.current_state or not self.current_video:
            return "❌ No hay proyecto activo", []
        
        selected_clips = self.state_manager.get_selected_clips()
        
        if not selected_clips:
            return "⚠️ No hay clips seleccionados para exportar", []
        
        output_files = []
        total = len(selected_clips)
        
        try:
            for i, clip_state in enumerate(selected_clips):
                progress((i / total), desc=f"🎬 Exportando clip {i+1}/{total}...")
                
                # Preparar nombre de archivo
                score_str = f"_{clip_state.virality_score:.1f}"
                output_file = config.OUTPUT_DIR / f"viral_clip_{i+1:02d}{score_str}.mp4"
                
                # Obtener segmentos editados
                editor = self.subtitle_editors.get(clip_state.id)
                if editor:
                    segments = editor.get_segments_for_export()
                else:
                    segments = clip_state.segments
                
                # Crear clip
                self.editor.create_viral_clip(
                    self.current_video,
                    str(output_file),
                    clip_state.start,
                    clip_state.end,
                    segments,
                    add_subtitles=True
                )
                
                output_files.append(str(output_file))
                logger.info(f"Clip exportado: {output_file}")
            
            progress(1.0, desc="✅ Exportación completada!")
            return f"🎉 {len(output_files)} clips exportados", output_files
            
        except Exception as e:
            logger.error(f"Error en exportación: {e}", exc_info=True)
            return f"❌ Error: {str(e)}", output_files
    
    def create_ui(self) -> gr.Blocks:
        """Crea la interfaz profesional tipo OpusClip."""
        
        custom_css = """
        .opus-container { max-width: 1400px; margin: 0 auto; }
        .opus-header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .opus-title { 
            color: white;
            font-size: 2.2em;
            font-weight: bold;
            text-align: center;
            margin: 0;
        }
        .opus-subtitle { 
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-top: 8px;
        }
        """
        
        with gr.Blocks(css=custom_css, title="OpusClip Pro", theme=gr.themes.Soft()) as app:
            
            # Header
            gr.HTML("""
            <div class="opus-header">
                <h1 class="opus-title">🎬 OpusClip Pro</h1>
                <p class="opus-subtitle">
                    Editor profesional de clips virales con IA | 
                    Transcripción automática + Análisis Gemini + Exportación 9:16
                </p>
            </div>
            """)
            
            # === PASO 1: IMPORTAR Y ANALIZAR ===
            with gr.Tab("1️⃣ Importar & Analizar"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Importar Video")
                        
                        video_input = gr.File(
                            label="Seleccionar video (máx. 2GB)",
                            file_types=["video"],
                            type="filepath"
                        )
                        
                        video_info = gr.Textbox(
                            label="Información",
                            value="Ningún video seleccionado",
                            interactive=False
                        )
                        
                        with gr.Accordion("⚙️ Configuración", open=True):
                            num_clips = gr.Slider(1, 10, value=3, step=1, label="Clips a detectar")
                            min_duration = gr.Slider(5, 120, value=15, step=5, label="Duración mínima")
                            max_duration = gr.Slider(15, 300, value=60, step=5, label="Duración máxima")
                        
                        analyze_btn = gr.Button("🔮 Analizar Video con IA", variant="primary")
                        analysis_status = gr.Textbox(label="Estado", value="Esperando...")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Análisis de Gemini")
                        analysis_result = gr.Markdown("*El análisis aparecerá aquí...*")
            
            # === PASO 2: EDITOR ===
            with gr.Tab("2️⃣ Editor de Clips") as tab_editor:
                with gr.Row():
                    # Timeline
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎬 Timeline")
                        timeline_clips = gr.Markdown("*Los clips aparecerán aquí...*")
                        
                        selected_clip_id = gr.Dropdown(label="Clip a editar", choices=[])
                        clip_start = gr.Number(label="Inicio (s)", value=0)
                        clip_end = gr.Number(label="Fin (s)", value=0)
                        
                        update_time_btn = gr.Button("💾 Guardar cambios")
                        select_clip_btn = gr.Button("✅ Seleccionar/Omitir")
                        clip_action_status = gr.Textbox(label="Estado")
                    
                    # Preview
                    with gr.Column(scale=2):
                        gr.Markdown("### 👁️ Preview")
                        preview_video = gr.Video(label="Preview 9:16", height=480)
                        generate_preview_btn = gr.Button("▶️ Generar Preview")
                        clip_info = gr.Markdown("Selecciona un clip...")
                    
                    # Subtítulos
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Subtítulos")
                        subtitle_df = gr.Dataframe(
                            headers=["ID", "Inicio", "Fin", "Original", "Editado", "✓"],
                            label="Subtítulos"
                        )
                        refresh_subs_btn = gr.Button("🔄 Recargar")
                        auto_correct_btn = gr.Button("✨ Auto-corregir")
                        
                        subtitle_edit_row = gr.Number(label="ID a editar", value=0)
                        subtitle_new_text = gr.Textbox(label="Nuevo texto")
                        update_subtitle_btn = gr.Button("💾 Guardar subtítulo")
                        subtitle_status = gr.Textbox(label="Estado")
            
            # === PASO 3: EXPORTAR ===
            with gr.Tab("3️⃣ Exportar"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎨 Estilo")
                        style_dropdown = gr.Dropdown(
                            choices=list(SUBTITLE_STYLES.keys()),
                            value="modern",
                            label="Estilo de subtítulos"
                        )
                        export_btn = gr.Button("🚀 Exportar Clips", variant="primary")
                        export_status = gr.Textbox(label="Estado")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📹 Clips Exportados")
                        output_gallery = gr.Gallery(label="Vista previa", columns=3)
                        output_files = gr.File(label="Descargar clips", file_count="multiple")
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>OpusClip Pro v2.0 | Whisper + Gemini AI | 9:16 para TikTok/Reels/Shorts</p>
            </div>
            """)
            
            # === EVENTOS ===
            
            def on_analyze(video, n_clips, min_dur, max_dur, progress=gr.Progress()):
                if not video:
                    return "❌ Selecciona un video", "", "", gr.update(visible=False), gr.update()
                
                status, clips_sum, analysis_sum, timeline_vis, video_inf = self.analyze_video(
                    video, n_clips, min_dur, max_dur, progress
                )
                
                clip_choices = [(f"Clip {i+1}", i) for i in range(len(self.current_state.clips))] if self.current_state else []
                
                return (
                    status, clips_sum, analysis_sum, gr.update(visible=True),
                    video_inf, gr.update(choices=clip_choices, value=0 if clip_choices else None)
                )
            
            analyze_btn.click(
                fn=on_analyze,
                inputs=[video_input, num_clips, min_duration, max_duration],
                outputs=[analysis_status, timeline_clips, analysis_result, tab_editor, video_info, selected_clip_id]
            )
            
            def on_load_clip_info(clip_id):
                if not self.current_state:
                    return "No hay proyecto", [], 0, 0
                
                clip = self.current_state.clips[clip_id]
                info = f"**Clip {clip_id + 1}**\n- {clip.start:.1f}s - {clip.end:.1f}s\n- Score: {clip.virality_score:.1f}"
                sub_data = self.get_subtitle_editor_data(clip_id)
                return info, sub_data, clip.start, clip.end
            
            selected_clip_id.change(
                fn=on_load_clip_info,
                inputs=[selected_clip_id],
                outputs=[clip_info, subtitle_df, clip_start, clip_end]
            )
            
            update_time_btn.click(
                fn=lambda cid, start, end: self.update_clip_time(cid, start, end),
                inputs=[selected_clip_id, clip_start, clip_end],
                outputs=[clip_action_status]
            )
            
            generate_preview_btn.click(
                fn=lambda cid: self.generate_preview(cid),
                inputs=[selected_clip_id],
                outputs=[preview_video]
            )
            
            refresh_subs_btn.click(
                fn=lambda cid: self.get_subtitle_editor_data(cid),
                inputs=[selected_clip_id],
                outputs=[subtitle_df]
            )
            
            update_subtitle_btn.click(
                fn=lambda cid, rid, txt: self.update_subtitle(cid, int(rid), txt),
                inputs=[selected_clip_id, subtitle_edit_row, subtitle_new_text],
                outputs=[subtitle_status]
            )
            
            export_btn.click(
                fn=lambda style, prog=gr.Progress(): self.export_clips(style, prog),
                inputs=[style_dropdown],
                outputs=[export_status, output_gallery, output_files]
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
        show_error=True
    )


if __name__ == "__main__":
    main()
