"""
Configuración centralizada de la aplicación.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Configuración principal de la aplicación."""
    
    # Directorios
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("outputs"))
    TEMP_DIR: Path = field(default_factory=lambda: Path("temp"))
    
    # Whisper Config
    WHISPER_MODEL: str = "base"  # Optimizado para GTX 1080 (8GB VRAM)
    WHISPER_LANGUAGE: str = "es"  # Español
    WHISPER_DEVICE: str = "auto"  # 'auto', 'cuda', 'cpu'
    
    # Gemini Config (gemini-2.5-flash - modelo más reciente)
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_TOKENS: int = 16000
    
    # Video Config (9:16 vertical)
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920
    VIDEO_FPS: int = 30
    VIDEO_BITRATE: str = "8000k"
    VIDEO_PRESET: str = "slow"  # calidad vs velocidad
    VIDEO_CRF: int = 18  # calidad (0-51, menor = mejor)
    
    # Subtitle Config
    SUBTITLE_FONT: str = "C:/Windows/Fonts/arialbd.ttf"
    SUBTITLE_FONTSIZE: int = 64
    SUBTITLE_COLOR: str = "white"
    SUBTITLE_STROKE_COLOR: str = "black"
    SUBTITLE_STROKE_WIDTH: int = 3
    SUBTITLE_POSITION: str = "center"  # center, bottom, top
    
    # Clip Config
    DEFAULT_NUM_CLIPS: int = 3
    CLIP_MIN_DURATION: int = 15  # segundos
    CLIP_MAX_DURATION: int = 60  # segundos
    
    # UI Config
    GRADIO_SHARE: bool = False
    GRADIO_SERVER_NAME: str = "0.0.0.0"
    GRADIO_SERVER_PORT: int = 7860
    
    def __post_init__(self):
        """Crea directorios si no existen."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def video_resolution(self) -> str:
        return f"{self.VIDEO_WIDTH}x{self.VIDEO_HEIGHT}"
    
    @property
    def aspect_ratio(self) -> str:
        return "9:16"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte config a diccionario."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }


# Instancia global de configuración
config = AppConfig()


# Estilos de subtítulos predefinidos
SUBTITLE_STYLES = {
    "modern": {
        "name": "Moderno",
        "font": "C:/Windows/Fonts/arialbd.ttf",
        "fontsize": 64,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 3,
        "bg_color": None,
        "position": "center"
    },
    "tiktok": {
        "name": "Estilo TikTok",
        "font": "C:/Windows/Fonts/impact.ttf",
        "fontsize": 72,
        "color": "#00f2ea",
        "stroke_color": "black",
        "stroke_width": 4,
        "bg_color": None,
        "position": "center"
    },
    "minimal": {
        "name": "Minimal",
        "font": "C:/Windows/Fonts/calibrib.ttf",
        "fontsize": 56,
        "color": "white",
        "stroke_color": None,
        "stroke_width": 0,
        "bg_color": "rgba(0,0,0,0.6)",
        "position": "bottom"
    },
    "classic": {
        "name": "Clásico",
        "font": "C:/Windows/Fonts/arialbd.ttf",
        "fontsize": 48,
        "color": "yellow",
        "stroke_color": "black",
        "stroke_width": 2,
        "bg_color": None,
        "position": "bottom"
    }
}

# Constantes
SUPPORTED_VIDEO_FORMATS = [
    '.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv'
]

MAX_VIDEO_SIZE_MB = 2048  # 2GB límite por video

# Mensajes de error
ERRORS = {
    'VIDEO_NOT_FOUND': "El archivo de video no existe.",
    'VIDEO_TOO_LARGE': f"El video excede el límite de {MAX_VIDEO_SIZE_MB}MB.",
    'UNSUPPORTED_FORMAT': f"Formato no soportado. Use: {', '.join(SUPPORTED_VIDEO_FORMATS)}",
    'FFMPEG_NOT_FOUND': "FFmpeg no está instalado o no está en el PATH.",
    'TRANSCRIPTION_FAILED': "Falló la transcripción del video.",
    'ANALYSIS_FAILED': "Falló el análisis con Gemini.",
    'CLIP_CREATION_FAILED': "No se pudieron crear los clips.",
    'NO_CLIPS_FOUND': "No se identificaron clips virales en el video.",
}

# Mensajes de éxito
SUCCESS = {
    'TRANSCRIPTION_COMPLETE': "✅ Transcripción completada exitosamente.",
    'ANALYSIS_COMPLETE': "✅ Análisis completado. Se identificaron {n} clips.",
    'CLIP_CREATED': "✅ Clip {i}/{n} creado: {path}",
    'ALL_CLIPS_CREATED': "🎉 Todos los clips ({n}) fueron creados exitosamente.",
}
