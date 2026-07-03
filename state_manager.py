"""
Gestor de estado para la aplicación OpusClip.
Guarda y carga el estado de edición entre sesiones.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClipState:
    """Estado de un clip individual (Virality Engine v2)."""
    id: int
    start: float
    end: float
    virality_score: float
    reason: str
    hook: str
    selected: bool = True
    hook_score: float = 0.0
    pacing_score: float = 0.0
    engagement_score: float = 0.0
    flow_score: float = 0.0
    value_score: float = 0.0
    trend_score: float = 0.0
    mood: str = "neutral"
    hook_type: str = "unknown"
    ideal_platform: str = "tiktok"
    edit_recipe: str = ""
    hashtags: List[str] = None
    segments: List[Dict[str, Any]] = None
    subtitle_edits: Dict[int, str] = None  # segment_id -> texto editado

    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.segments is None:
            self.segments = []
        if self.subtitle_edits is None:
            self.subtitle_edits = {}
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProjectState:
    """Estado completo de un proyecto."""
    video_path: str
    project_name: str
    created_at: str
    updated_at: str
    
    # Datos de transcripción
    transcription: Dict[str, Any]
    full_text: str
    
    # Clips detectados y editados
    clips: List[ClipState]
    
    # Configuración de exportación
    output_config: Dict[str, Any]
    subtitle_style: Dict[str, Any]
    
    # Metadatos
    total_duration: float
    language: str


class StateManager:
    """Gestiona el guardado y carga de estados de edición."""
    
    def __init__(self, state_dir: str = "temp/states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.current_state: Optional[ProjectState] = None
        self._save_lock = threading.Lock()
    
    def create_project(
        self,
        video_path: str,
        transcription: Dict[str, Any],
        clips_data: List[Dict[str, Any]]
    ) -> ProjectState:
        """Crea un nuevo estado de proyecto desde los datos iniciales."""
        
        now = datetime.now().isoformat()
        video_name = Path(video_path).stem
        
        # Convertir clips a ClipState
        clips = []
        for i, clip_data in enumerate(clips_data):
            def _f(val, default=0.0):
                try:
                    return float(val) if val is not None else default
                except (ValueError, TypeError):
                    return default
            clip_state = ClipState(
                id=i,
                start=_f(clip_data.get('start'), 0.0),
                end=_f(clip_data.get('end'), 0.0),
                virality_score=_f(clip_data.get('virality_score'), 0.0),
                reason=str(clip_data.get('reason', '')),
                hook=str(clip_data.get('hook', '')),
                selected=True,
                hook_score=_f(clip_data.get('hook_score'), _f(clip_data.get('virality_score'), 0.0)),
                pacing_score=_f(clip_data.get('pacing_score'), _f(clip_data.get('virality_score'), 0.0)),
                engagement_score=_f(clip_data.get('engagement_score'), _f(clip_data.get('virality_score'), 0.0)),
                flow_score=_f(clip_data.get('flow_score'), _f(clip_data.get('virality_score'), 0.0)),
                value_score=_f(clip_data.get('value_score'), _f(clip_data.get('virality_score'), 0.0)),
                trend_score=_f(clip_data.get('trend_score'), 5.0),
                mood=str(clip_data.get('mood', 'neutral')),
                hook_type=str(clip_data.get('hook_type', 'unknown')),
                ideal_platform=str(clip_data.get('ideal_platform', 'tiktok')),
                edit_recipe=str(clip_data.get('edit_recipe', '')),
                hashtags=list(clip_data.get('hashtags', []) or []),
                segments=clip_data.get('segments', []),
                subtitle_edits={}
            )
            clips.append(clip_state)
        
        state = ProjectState(
            video_path=video_path,
            project_name=f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=now,
            updated_at=now,
            transcription=transcription,
            full_text=transcription.get('text', ''),
            clips=clips,
            output_config={},
            subtitle_style={},
            total_duration=transcription.get('duration', 0),
            language=transcription.get('language', 'es')
        )
        
        self.current_state = state
        logger.info(f"Proyecto creado: {state.project_name}")
        return state
    
    def save_state(self, state: Optional[ProjectState] = None) -> str:
        """Guarda el estado actual a archivo JSON."""
        
        state = state or self.current_state
        if not state:
            raise ValueError("No hay estado para guardar")
        
        # Actualizar timestamp
        state.updated_at = datetime.now().isoformat()
        
        # Convertir a dict
        state_dict = {
            'video_path': state.video_path,
            'project_name': state.project_name,
            'created_at': state.created_at,
            'updated_at': state.updated_at,
            'transcription': state.transcription,
            'full_text': state.full_text,
            'clips': [
                {
                    'id': c.id,
                    'start': c.start,
                    'end': c.end,
                    'virality_score': c.virality_score,
                    'hook_score': c.hook_score,
                    'pacing_score': c.pacing_score,
                    'engagement_score': c.engagement_score,
                    'flow_score': c.flow_score,
                    'value_score': c.value_score,
                    'trend_score': c.trend_score,
                    'mood': c.mood,
                    'hook_type': c.hook_type,
                    'ideal_platform': c.ideal_platform,
                    'edit_recipe': c.edit_recipe,
                    'hashtags': c.hashtags,
                    'reason': c.reason,
                    'hook': c.hook,
                    'selected': c.selected,
                    'segments': c.segments,
                    'subtitle_edits': c.subtitle_edits
                }
                for c in state.clips
            ],
            'output_config': state.output_config,
            'subtitle_style': state.subtitle_style,
            'total_duration': state.total_duration,
            'language': state.language
        }
        
        # Guardar archivo de forma atómica (escribe a un temp y reemplaza)
        # para evitar que dos callbacks concurrentes dejen el JSON truncado/corrupto.
        state_file = self.state_dir / f"{state.project_name}.json"
        tmp_file = state_file.with_suffix(".json.tmp")
        with self._save_lock:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, state_file)

        logger.info(f"Estado guardado: {state_file}")
        return str(state_file)
    
    def load_state(self, project_name: str) -> Optional[ProjectState]:
        """Carga un estado desde archivo."""
        
        state_file = self.state_dir / f"{project_name}.json"
        
        if not state_file.exists():
            logger.warning(f"Archivo de estado no encontrado: {state_file}")
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruir ClipState objects
            def _f_load(val, default=0.0):
                try:
                    return float(val) if val is not None else default
                except (ValueError, TypeError):
                    return default
            clips = []
            for clip_data in data.get('clips', []):
                vs = _f_load(clip_data.get('virality_score'), 0.0)
                clip = ClipState(
                    id=int(clip_data.get('id', 0)),
                    start=_f_load(clip_data.get('start'), 0.0),
                    end=_f_load(clip_data.get('end'), 0.0),
                    virality_score=vs,
                    hook_score=_f_load(clip_data.get('hook_score'), vs),
                    pacing_score=_f_load(clip_data.get('pacing_score'), vs),
                    engagement_score=_f_load(clip_data.get('engagement_score'), vs),
                    flow_score=_f_load(clip_data.get('flow_score'), vs),
                    value_score=_f_load(clip_data.get('value_score'), vs),
                    trend_score=_f_load(clip_data.get('trend_score'), 5.0),
                    mood=str(clip_data.get('mood', 'neutral')),
                    hook_type=str(clip_data.get('hook_type', 'unknown')),
                    ideal_platform=str(clip_data.get('ideal_platform', 'tiktok')),
                    edit_recipe=str(clip_data.get('edit_recipe', '')),
                    hashtags=list(clip_data.get('hashtags', []) or []),
                    reason=str(clip_data.get('reason', '')),
                    hook=str(clip_data.get('hook', '')),
                    selected=bool(clip_data.get('selected', True)),
                    segments=clip_data.get('segments', []),
                    subtitle_edits=clip_data.get('subtitle_edits', {})
                )
                clips.append(clip)
            
            state = ProjectState(
                video_path=data['video_path'],
                project_name=data['project_name'],
                created_at=data['created_at'],
                updated_at=data['updated_at'],
                transcription=data['transcription'],
                full_text=data['full_text'],
                clips=clips,
                output_config=data.get('output_config', {}),
                subtitle_style=data.get('subtitle_style', {}),
                total_duration=data['total_duration'],
                language=data['language']
            )
            
            self.current_state = state
            logger.info(f"Estado cargado: {project_name}")
            return state
            
        except Exception as e:
            logger.error(f"Error cargando estado: {e}")
            return None
    
    def list_saved_projects(self) -> List[str]:
        """Lista todos los proyectos guardados."""
        projects = []
        for f in self.state_dir.glob("*.json"):
            projects.append(f.stem)
        return sorted(projects, reverse=True)
    
    def update_clip(
        self,
        clip_id: int,
        start: Optional[float] = None,
        end: Optional[float] = None,
        selected: Optional[bool] = None
    ) -> bool:
        """Actualiza un clip en el estado actual."""
        
        if not self.current_state:
            return False
        
        for clip in self.current_state.clips:
            if clip.id == clip_id:
                if start is not None:
                    clip.start = max(0, start)
                if end is not None:
                    clip.end = min(self.current_state.total_duration, end)
                if selected is not None:
                    clip.selected = selected
                
                # Validación: asegurar que start < end
                if clip.start >= clip.end:
                    # Ajustar automáticamente: mínimo 1 segundo de duración
                    clip.end = min(clip.start + 1.0, self.current_state.total_duration)
                    if clip.start >= clip.end:
                        clip.start = max(0, clip.end - 1.0)
                
                self.save_state()
                return True
        
        return False
    
    def update_subtitle_edit(
        self,
        clip_id: int,
        segment_id: int,
        new_text: str
    ) -> bool:
        """Guarda una edición de subtítulo."""
        
        if not self.current_state:
            return False
        
        for clip in self.current_state.clips:
            if clip.id == clip_id:
                clip.subtitle_edits[segment_id] = new_text
                self.save_state()
                return True
        
        return False
    
    def get_selected_clips(self) -> List[ClipState]:
        """Obtiene solo los clips seleccionados."""
        if not self.current_state:
            return []
        return [c for c in self.current_state.clips if c.selected]
    
    def reorder_clips(self, new_order: List[int]) -> bool:
        """Reordena los clips según la lista de IDs."""
        
        if not self.current_state:
            return False
        
        clips_dict = {c.id: c for c in self.current_state.clips}
        new_clips = []
        
        for clip_id in new_order:
            if clip_id in clips_dict:
                new_clips.append(clips_dict[clip_id])
        
        # Agregar clips que no estaban en el orden (mantener al final)
        for clip in self.current_state.clips:
            if clip.id not in new_order:
                new_clips.append(clip)
        
        self.current_state.clips = new_clips
        self.save_state()
        return True
