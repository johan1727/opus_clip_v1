"""
Editor de subtítulos - Corrección y personalización de texto transcrito.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """Entrada individual de subtítulo editable."""
    id: int
    start: float
    end: float
    original_text: str
    edited_text: Optional[str] = None
    
    @property
    def display_text(self) -> str:
        """Texto a mostrar (editado o original)."""
        return self.edited_text if self.edited_text else self.original_text
    
    @property
    def was_edited(self) -> bool:
        return self.edited_text is not None and self.edited_text != self.original_text
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SubtitleStyle:
    """Estilo visual de subtítulos."""
    name: str
    font: str
    fontsize: int
    color: str
    stroke_color: str
    stroke_width: int
    bg_color: Optional[str]
    position: str  # "center", "bottom", "top"
    

# Estilos predefinidos
PREDEFINED_STYLES = {
    "modern": SubtitleStyle(
        name="Modern",
        font="Arial-Bold",
        fontsize=64,
        color="white",
        stroke_color="black",
        stroke_width=3,
        bg_color=None,
        position="center"
    ),
    "tiktok": SubtitleStyle(
        name="TikTok Style",
        font="Impact",
        fontsize=72,
        color="#00f2ea",  # Cyan TikTok
        stroke_color="black",
        stroke_width=4,
        bg_color=None,
        position="center"
    ),
    "minimal": SubtitleStyle(
        name="Minimal",
        font="Helvetica",
        fontsize=56,
        color="white",
        stroke_color=None,
        stroke_width=0,
        bg_color="rgba(0,0,0,0.6)",
        position="bottom"
    ),
    "classic": SubtitleStyle(
        name="Classic",
        font="Arial",
        fontsize=48,
        color="yellow",
        stroke_color="black",
        stroke_width=2,
        bg_color=None,
        position="bottom"
    )
}


class SubtitleEditor:
    """Editor de subtítulos con correcciones automáticas y manuales."""
    
    # Filler words comunes en español e inglés
    FILLER_WORDS = [
        # Español
        'eh', 'um', 'ah', 'oh', 'pues', 'bueno', 'sabes', 'como', 'o sea',
        'entonces', 'vale', 'ya sabes', 'digamos', 'tipo', 'asi que',
        # Inglés
        'uh', 'um', 'uhm', 'like', 'you know', 'i mean', 'so', 'well',
        'actually', 'basically', 'literally', 'right', 'okay', 'ok'
    ]
    
    # Mapeo simple de palabras clave a emojis
    EMOJI_MAP = {
        # Emociones
        'feliz': '😊', 'contento': '😊', 'alegre': '😄', 'triste': '😢',
        'enojado': '😠', 'furioso': '🤬', 'enfadado': '😤',
        'amor': '❤️', 'corazon': '❤️', 'amo': '😍', 'odio': '😡',
        'miedo': '😱', 'susto': '😰', 'sorpresa': '😮', 'wow': '🤩',
        'risa': '😂', 'gracioso': '😂', 'jaja': '😂', 'jajaja': '🤣',
        'llorar': '😭', 'tears': '😭', 'pena': '😔',
        
        # Objetos comunes
        'fuego': '🔥', 'hot': '🔥', 'caliente': '🔥', 'tendencia': '🔥',
        'dinero': '💰', 'rico': '💰', 'pobre': '💸', 'casa': '🏠',
        'coche': '🚗', 'carro': '🚗', 'auto': '🚗', 'comida': '🍕',
        'telefono': '📱', 'celular': '📱', 'computadora': '💻',
        
        # Conceptos
        'idea': '💡', 'tip': '💡', 'truco': '🎩', 'magia': '✨',
        'exito': '🏆', 'ganar': '🏆', 'victoria': '✌️', 'logro': '🎯',
        'importante': '⚠️', 'atencion': '⚠️', 'cuidado': '⚠️',
        'nuevo': '✨', 'mejor': '⬆️', 'peor': '⬇️', 'primero': '🥇',
        'gratis': '🆓', 'free': '🆓', '100%': '💯', 'perfecto': '💯',
        
        # Tiempo
        'ahora': '⏰', 'tiempo': '⏱️', 'rapido': '⚡', 'lento': '🐌',
        'hoy': '📅', 'mañana': '🔮', 'pronto': '⌛',
        
        # Reacciones
        'bien': '👍', 'mal': '👎', 'ok': '👌', 'yes': '✅', 'no': '❌',
        'por favor': '🙏', 'gracias': '🙏', 'de nada': '😇',
        'bravo': '👏', 'aplauso': '👏', 'genial': '🤙', 'cool': '😎',
        
        # Numeros
        '1': '1️⃣', '2': '2️⃣', '3': '3️⃣', 'primero': '1️⃣', 'segundo': '2️⃣', 'tercero': '3️⃣',
        'numero uno': '1️⃣', 'top': '🏆', 'numero 1': '1️⃣'
    }
    
    def __init__(self):
        self.entries: List[SubtitleEntry] = []
        self.style: SubtitleStyle = PREDEFINED_STYLES["modern"]
        self.custom_edits: Dict[int, str] = {}  # id -> texto editado
        self.emoji_enabled: bool = False
        self.filler_removal_enabled: bool = False
    
    def load_segments(self, segments: List[Dict[str, Any]]) -> None:
        """Carga segmentos de Whisper como entradas editables."""
        
        self.entries = []
        for seg in segments:
            entry = SubtitleEntry(
                id=seg.get('id', len(self.entries)),
                start=seg.get('start', 0.0),
                end=seg.get('end', 0.0),
                original_text=seg.get('text', '').strip(),
                edited_text=None
            )
            self.entries.append(entry)
        
        logger.info(f"Cargados {len(self.entries)} subtítulos")
    
    def get_entries_for_dataframe(self) -> List[Dict[str, Any]]:
        """Convierte entradas a formato para Gradio DataFrame."""
        
        data = []
        for entry in self.entries:
            data.append({
                'ID': entry.id,
                'Inicio': f"{entry.start:.2f}s",
                'Fin': f"{entry.end:.2f}s",
                'Texto Original': entry.original_text,
                'Texto Editado': entry.edited_text or '',
                'Editado': '✓' if entry.was_edited else ''
            })
        return data
    
    def edit_entry(self, entry_id: int, new_text: str) -> bool:
        """Edita una entrada específica."""
        
        for entry in self.entries:
            if entry.id == entry_id:
                # Limpiar texto
                cleaned = self._clean_text(new_text)
                
                # Solo guardar si es diferente al original
                if cleaned != entry.original_text:
                    entry.edited_text = cleaned
                    self.custom_edits[entry_id] = cleaned
                else:
                    entry.edited_text = None
                    self.custom_edits.pop(entry_id, None)
                
                return True
        
        return False
    
    def reset_entry(self, entry_id: int) -> bool:
        """Resetea una entrada a su texto original."""
        
        for entry in self.entries:
            if entry.id == entry_id:
                entry.edited_text = None
                self.custom_edits.pop(entry_id, None)
                return True
        return False
    
    def reset_all(self) -> None:
        """Resetea todas las entradas."""
        
        for entry in self.entries:
            entry.edited_text = None
        self.custom_edits.clear()
        logger.info("Todos los subtítulos reseteados")
    
    def get_segments_for_karaoke(self) -> List[Dict[str, Any]]:
        """
        Convierte segmentos a formato karaoke con timing por palabra.
        
        Returns:
            Lista de segmentos donde cada uno tiene una sola palabra
            con timing preciso calculado.
        """
        word_segments = []
        
        for entry in self.entries:
            text = entry.display_text.strip()
            if not text:
                continue
            
            words = text.split()
            if not words:
                continue
            
            # Calcular duración por palabra
            total_duration = entry.end - entry.start
            word_duration = total_duration / len(words)
            
            for i, word in enumerate(words):
                word_start = entry.start + (i * word_duration)
                word_end = word_start + word_duration
                
                word_segments.append({
                    'id': f"{entry.id}_{i}",
                    'start': word_start,
                    'end': word_end,
                    'text': word,
                    'parent_id': entry.id,
                    'word_index': i,
                    'total_words': len(words)
                })
        
        return word_segments
    
    def apply_auto_corrections(self) -> int:
        """Aplica correcciones automáticas a todos los subtítulos.
        
        Returns:
            Número de correcciones aplicadas
        """
        
        corrections = 0
        
        for entry in self.entries:
            original = entry.edited_text or entry.original_text
            corrected = self._auto_correct(original)
            
            if corrected != original:
                entry.edited_text = corrected
                self.custom_edits[entry.id] = corrected
                corrections += 1
        
        logger.info(f"Correcciones automáticas aplicadas: {corrections}")
        return corrections
    
    def _clean_text(self, text: str) -> str:
        """Limpia texto de subtítulo."""
        
        # Quitar espacios extra
        text = ' '.join(text.split())
        
        # Capitalizar primera letra
        if text:
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def _auto_correct(self, text: str) -> str:
        """Aplica correcciones automáticas comunes."""
        
        # Espacios antes de puntuación
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Mayúsculas después de punto
        text = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), text)
        
        # Eliminar dobles espacios
        text = ' '.join(text.split())
        
        # Capitalizar inicio
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Quitar puntos suspensivos excesivos
        text = re.sub(r'\.{4,}', '...', text)
        
        return text.strip()
    
    def merge_short_entries(self, min_duration: float = 1.0) -> int:
        """Une subtítulos muy cortos con el siguiente."""
        
        if len(self.entries) < 2:
            return 0
        
        merged = 0
        new_entries = []
        i = 0
        
        while i < len(self.entries):
            entry = self.entries[i]
            
            if entry.duration < min_duration and i < len(self.entries) - 1:
                # Unir con siguiente
                next_entry = self.entries[i + 1]
                
                merged_text = f"{entry.display_text} {next_entry.display_text}"
                merged_entry = SubtitleEntry(
                    id=len(new_entries),
                    start=entry.start,
                    end=next_entry.end,
                    original_text=merged_text,
                    edited_text=None
                )
                new_entries.append(merged_entry)
                merged += 1
                i += 2  # Saltar ambos
            else:
                # Re-asignar ID
                entry.id = len(new_entries)
                new_entries.append(entry)
                i += 1
        
        self.entries = new_entries
        logger.info(f"Subtítulos unidos: {merged}")
        return merged
    
    def split_long_entries(self, max_chars: int = 40) -> int:
        """Divide subtítulos muy largos en múltiples líneas."""
        
        split_count = 0
        new_entries = []
        
        for entry in self.entries:
            text = entry.display_text
            
            if len(text) <= max_chars:
                entry.id = len(new_entries)
                new_entries.append(entry)
                continue
            
            # Dividir en oraciones o palabras
            parts = self._split_text(text, max_chars)
            
            duration_per_part = entry.duration / len(parts)
            
            for j, part in enumerate(parts):
                new_entry = SubtitleEntry(
                    id=len(new_entries),
                    start=entry.start + (j * duration_per_part),
                    end=entry.start + ((j + 1) * duration_per_part),
                    original_text=part,
                    edited_text=None
                )
                new_entries.append(new_entry)
            
            split_count += len(parts) - 1
        
        self.entries = new_entries
        logger.info(f"Subtítulos divididos: {split_count}")
        return split_count
    
    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """Divide texto en partes de máximo max_chars."""
        
        parts = []
        words = text.split()
        current = ""
        
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".strip()
            else:
                if current:
                    parts.append(current)
                current = word
        
        if current:
            parts.append(current)
        
        return parts if parts else [text]
    
    def get_segments_for_export(self) -> List[Dict[str, Any]]:
        """Devuelve segmentos en formato para exportar (video_editor)."""
        
        return [
            {
                'id': e.id,
                'start': e.start,
                'end': e.end,
                'text': e.display_text
            }
            for e in self.entries
        ]
    
    def set_style(self, style_name: str) -> bool:
        """Establece estilo predefinido."""
        
        if style_name in PREDEFINED_STYLES:
            self.style = PREDEFINED_STYLES[style_name]
            logger.info(f"Estilo aplicado: {style_name}")
            return True
        return False
    
    def customize_style(
        self,
        fontsize: Optional[int] = None,
        color: Optional[str] = None,
        position: Optional[str] = None
    ) -> None:
        """Personaliza el estilo actual."""
        
        if fontsize:
            self.style.fontsize = fontsize
        if color:
            self.style.color = color
        if position:
            self.style.position = position
    
    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del editor."""
        
        total = len(self.entries)
        edited = sum(1 for e in self.entries if e.was_edited)
        total_duration = sum(e.duration for e in self.entries)
        
        return {
            'total_entries': total,
            'edited_entries': edited,
            'total_duration': total_duration,
            'avg_duration': total_duration / total if total > 0 else 0,
            'style': self.style.name
        }
    
    # ============================================================
    # NUEVAS FEATURES: Filler Word Removal & Auto-Emoji
    # ============================================================
    
    def remove_filler_words(self) -> int:
        """
        Elimina automáticamente palabras de relleno (filler words).
        
        Returns:
            Número de subtítulos modificados
        """
        removed_count = 0
        
        for entry in self.entries:
            original = entry.display_text
            text = original.lower()
            
            # Remover filler words
            for filler in self.FILLER_WORDS:
                # Remover al inicio, medio o final
                text = re.sub(rf'\b{re.escape(filler)}\b', '', text, flags=re.IGNORECASE)
            
            # Limpiar espacios extra
            cleaned = ' '.join(text.split())
            
            # Capitalizar primera letra si no está vacío
            if cleaned and len(cleaned) > 0:
                cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
            
            if cleaned != original and cleaned.strip():
                entry.edited_text = cleaned
                self.custom_edits[entry.id] = cleaned
                removed_count += 1
        
        self.filler_removal_enabled = True
        logger.info(f"Filler words removidos en {removed_count} subtítulos")
        return removed_count
    
    def add_emojis(self, max_emojis_per_entry: int = 2) -> int:
        """
        Agrega emojis automáticamente basado en palabras clave.
        
        Args:
            max_emojis_per_entry: Máximo de emojis por subtítulo
            
        Returns:
            Número de subtítulos con emojis agregados
        """
        emoji_count = 0
        
        for entry in self.entries:
            original = entry.display_text
            text_lower = original.lower()
            
            # Encontrar emojis relevantes
            emojis_to_add = []
            for keyword, emoji in self.EMOJI_MAP.items():
                if keyword in text_lower and emoji not in emojis_to_add:
                    emojis_to_add.append(emoji)
                    if len(emojis_to_add) >= max_emojis_per_entry:
                        break
            
            if emojis_to_add:
                # Agregar emojis al inicio
                new_text = ''.join(emojis_to_add) + ' ' + original
                entry.edited_text = new_text
                self.custom_edits[entry.id] = new_text
                emoji_count += 1
        
        self.emoji_enabled = True
        logger.info(f"Emojis agregados a {emoji_count} subtítulos")
        return emoji_count
    
    def toggle_emoji(self, enabled: bool) -> None:
        """Activa/desactiva emojis."""
        self.emoji_enabled = enabled
        logger.info(f"Auto-emoji: {enabled}")
    
    def toggle_filler_removal(self, enabled: bool) -> None:
        """Activa/desactiva filler word removal."""
        self.filler_removal_enabled = enabled
        logger.info(f"Filler removal: {enabled}")


def create_editor_from_transcription(
    segments: List[Dict[str, Any]],
    style_name: str = "modern"
) -> SubtitleEditor:
    """Crea un editor configurado desde segmentos de transcripción."""
    
    editor = SubtitleEditor()
    editor.load_segments(segments)
    editor.set_style(style_name)
    
    # Auto-correcciones iniciales
    editor.apply_auto_corrections()
    
    return editor
