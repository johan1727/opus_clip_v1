"""
Módulo de análisis con LLM (Gemini).
Recibe transcripción y devuelve timestamps de momentos virales.
"""

import logging
import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Load API key from environment variable (set it in a local .env file, see .env.example)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY no está configurada. Copia .env.example a .env y agrega tu key."
    )


@dataclass
class ViralClip:
    """Representa un clip viral identificado por el LLM (Virality Engine v2)."""
    start: float
    end: float
    virality_score: float
    reason: str
    hook: str
    hook_score: float = 0.0
    pacing_score: float = 0.0
    engagement_score: float = 0.0
    flow_score: float = 0.0
    value_score: float = 0.0
    trend_score: float = 0.0
    emotional_arc_score: float = 0.0
    mood: str = "neutral"
    hook_type: str = "unknown"
    ideal_platform: str = "tiktok"
    edit_recipe: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start': self.start,
            'end': self.end,
            'virality_score': self.virality_score,
            'reason': self.reason,
            'hook': self.hook,
            'hook_score': self.hook_score,
            'pacing_score': self.pacing_score,
            'engagement_score': self.engagement_score,
            'flow_score': self.flow_score,
            'value_score': self.value_score,
            'trend_score': self.trend_score,
            'emotional_arc_score': self.emotional_arc_score,
            'mood': self.mood,
            'hook_type': self.hook_type,
            'ideal_platform': self.ideal_platform,
            'edit_recipe': self.edit_recipe,
        }


class GeminiAnalyzer:
    """
    Analizador de contenido viral usando Gemini (Google AI Studio).
    Identifica los mejores momentos de un video para clips cortos.
    """
    
    DEFAULT_MODEL = "gemini-2.5-flash"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_output_tokens: int = 4096
    ) -> None:
        """
        Inicializa el analizador Gemini.
        
        Args:
            api_key: API key de Gemini (usa env var GEMINI_API_KEY si es None)
            model_name: Modelo a usar ('gemini-2.5-flash' recomendado)
            temperature: Creatividad (0.0 = determinista, 1.0 = creativo)
            max_output_tokens: Máximo de tokens en respuesta
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model: Optional[genai.GenerativeModel] = None
        
        # Validate API key exists
        if not self.api_key or self.api_key == "":
            raise ValueError(
                "❌ GEMINI_API_KEY no está configurada.\n\n"
                "Para configurar en Windows PowerShell:\n"
                "  $env:GEMINI_API_KEY='tu-api-key-aquí'\n\n"
                "O en CMD:\n"
                "  set GEMINI_API_KEY=tu-api-key-aquí\n\n"
                "O permanentemente en Variables de Entorno del Sistema.\n\n"
                "Obtén tu API key gratuita en: https://aistudio.google.com/app/apikey"
            )
        
        self._configure_api()
        logger.info(f"GeminiAnalyzer inicializado: modelo={model_name}")
    
    def _configure_api(self) -> None:
        """Configura la API de Gemini con la key proporcionada."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("API Gemini configurada exitosamente")
        except Exception as e:
            logger.error(f"Error configurando Gemini API: {e}")
            raise RuntimeError(f"No se pudo configurar Gemini: {e}")
    
    def _build_prompt(
        self,
        transcription: Dict[str, Any],
        num_clips: int = 3,
        clip_duration_range: tuple = (15, 60),
        custom_prompt: str = "",
        compact: bool = False,
        engagement_data: Optional[List[Any]] = None,
        scene_changes: Optional[List[float]] = None
    ) -> str:
        """
        Construye el prompt para el LLM.
        
        Args:
            transcription: Dict con 'text' y 'segments'
            num_clips: Número de clips a identificar
            clip_duration_range: (min_segundos, max_segundos)
            engagement_data: Lista de EngagementData con scores de audio/escenas
            scene_changes: Timestamps de cambios de escena
            
        Returns:
            String con el prompt completo
        """
        full_text = transcription.get('text', '')
        segments = transcription.get('segments', [])
        duration = transcription.get('duration', 0)
        
        # Construir representación de la transcripción con timestamps
        # In compact mode: merge segments into ~20s windows to reduce tokens
        transcript_with_time = []
        if compact and len(segments) > 120:
            window_secs = 20.0
            window_start = segments[0]['start'] if segments else 0
            window_texts = []
            for seg in segments:
                if seg['start'] - window_start > window_secs:
                    if window_texts:
                        transcript_with_time.append(
                            f"[{window_start:.0f}s] {' '.join(window_texts)}"
                        )
                    window_start = seg['start']
                    window_texts = [seg['text'].strip()]
                else:
                    window_texts.append(seg['text'].strip())
            if window_texts:
                transcript_with_time.append(
                    f"[{window_start:.0f}s] {' '.join(window_texts)}"
                )
        else:
            for seg in segments:
                start = seg['start']
                end = seg['end']
                text = seg['text'].strip()
                transcript_with_time.append(f"[{start:.1f}s - {end:.1f}s] {text}")
        
        min_dur, max_dur = clip_duration_range
        
        custom_instruction = f"\n\n### INSTRUCCIÓN ADICIONAL DEL USUARIO:\n{custom_prompt}" if custom_prompt.strip() else ""
        
        # Build few-shot block before f-string so it interpolates correctly
        if not compact:
            few_shot_block = """### EJEMPLOS DE CALIBRACIÓN (few-shot) - TALKING HEAD / VLOG:
Estos son clips REALES de alta viralidad de contenido de personas hablando:

```json
{{
  "clips": [
    {{"start": 14.0, "end": 42.0, "virality_score": 9.3, "hook_score": 9.5, "pacing_score": 8.5,
      "engagement_score": 9.2, "flow_score": 8.8, "value_score": 8.5, "emotional_arc_score": 9.4,
      "mood": "shocking", "hook_type": "facial_reveal", "ideal_platform": "tiktok",
      "reason": "Ojos desorbitados de sorpresa en segundo 2 + payoff escalofriante con twist final. Emotional arc perfecto.",
      "hook": "No vas a creer lo que me encontré en mi armario.", "edit_recipe": "Zoom punch-in a cara de sorpresa seg 2 + freeze 0.5s + captions grandes del hook + SFX 'wait for it'"}},
    {{"start": 67.0, "end": 98.0, "virality_score": 8.7, "hook_score": 8.8, "pacing_score": 9.0,
      "engagement_score": 8.5, "flow_score": 9.1, "value_score": 9.0, "emotional_arc_score": 8.6,
      "mood": "controversial", "hook_type": "controversial_take", "ideal_platform": "tiktok",
      "reason": "Hot take polémico que genera comentarios. Setup calmado → intensidad creciente → pausa dramática → reveal.",
      "hook": "Odio decirlo, pero los padres millennials están arruinando a sus hijos.", "edit_recipe": "Texto del statement en ROJO grande seg 1 + zoom lento + pausa 1s + 'COMENTA si estás de acuerdo' CTA"}},
    {{"start": 145.0, "end": 178.0, "virality_score": 9.0, "hook_score": 9.2, "pacing_score": 8.3,
      "engagement_score": 9.1, "flow_score": 8.5, "value_score": 8.2, "emotional_arc_score": 9.3,
      "mood": "inspirational", "hook_type": "intimate_moment", "ideal_platform": "reels",
      "reason": "Vulnerabilidad genuina. Calmado → vulnerabilidad emocional → mensaje poderoso. Closure perfecto.",
      "hook": "El día que perdí todo, aprendí lo más importante.", "edit_recipe": "Soft lighting filter + captions serif elegantes + fade in lento + close-up cara emocional seg 3"}}
  ]
}}
```"""
        else:
            few_shot_block = ""  # skip in compact mode

        # Build engagement + scene data sections
        engagement_section = ""
        if engagement_data and not compact:
            top_moments = sorted(engagement_data, key=lambda x: x.combined_score, reverse=True)[:15]
            lines = [f"  [{e.timestamp:.0f}s] energy={e.audio_energy_score:.1f} speech={e.speech_density_score:.1f} scene={e.scene_change_score:.1f} combined={e.combined_score:.1f}" for e in top_moments]
            engagement_section = f"\n### DATOS DE AUDIO Y ENGAGEMENT:\n{chr(10).join(lines)}\nMomentos con 'combined' > 7.0 son de alto interés.\n"
        scene_section = ""
        if scene_changes and not compact:
            scene_times = ", ".join([f"{t:.1f}s" for t in scene_changes[:20]])
            scene_section = f"\n### CAMBIOS DE ESCENA (cortes de plano):\n{scene_times}\nUsa estos timestamps como puntos de corte naturales.\n"

        prompt = f"""Eres un experto en virality engineering para TikTok, Instagram Reels y YouTube Shorts.
Tu objetivo: identificar los {num_clips} MEJORES clips virales y generar una receta de edición por clip.{custom_instruction}

### VIDEO COMPLETO
**Duración total:** {duration:.1f} segundos

### TRANSCRIPCIÓN CON TIMESTAMPS:
{chr(10).join(transcript_with_time)}
{engagement_section}{scene_section}
### CRITERIOS DE VIRALIDAD - TALKING HEAD (Virality Engine v3):
1. **Hook VISUAL en segundo 1-2** — La expresión facial o gesto inicial DEBE interrumpir el scroll. No basta con el audio.
2. **Emotional Arc completo** — Estado inicial → tensión/emoción creciente → payoff o resolución. Arco débil = score bajo.
3. **Retention prediction** — ¿Qué mantiene la atención hasta segundo 8? "Wait for it" moments, suspense, delayed payoff.
4. **Valor emocional genuino** — Reacciones auténticas (no forzadas), vulnerabilidad real, humor natural.
5. **Autocontenido** — Comprensible sin ver el video completo. Story arc cerrado en 15-60s.
6. **Duración óptima** — Entre {min_dur} y {max_dur} segundos. Más corto = más intenso, más largo = más storytelling.
7. **Quotable / Comentable** — Frases que generen "yo también", polémica, o compartir.
8. **Cortes naturales** — Inicia/termina en cambios de escena, silencios, o pausas dramáticas.
9. **Facial expressions matter** — Prioriza momentos con expresiones faciales impactantes.

### TIPOS DE HOOK - TALKING HEAD (elige el más apropiado):
- facial_reveal: expresión facial impactante (sorpresa, shock, emoción) - para talking head
- story_hook: "Cuando yo tenía..." story opening con stakes - para storytelling
- reaction_setup: setup → pausa → reacción (comedy/tension) - para reacciones
- controversial_take: opinión fuerte que genera comentarios - para debate
- relatable_confession: "yo también hago esto" moment - para conexión
- wait_for_it: delayed payoff, "no vas a creer lo que pasó" - para suspense
- transformation_reveal: before/after, cambio visible - para reveals
- intimate_moment: vulnerabilidad genuina, emoción real - para inspiración
- pattern_interrupt: rompe expectativas visualmente
- bold_statement: afirmación polémica con cara seria/intensa
- curiosity_gap: información incompleta + expresión de intriga
- social_proof: "mira lo que todos están diciendo" + reacción

### MOODS DISPONIBLES:
funny | shocking | educational | motivational | dramatic | controversial | wholesome | gaming_hype | storytelling | inspirational

### RECETAS DE EDICIÓN - TALKING HEAD (edit_recipe por hook_type):
- facial_reveal → Zoom punch-in a cara en frame de reacción + freeze 0.5s + captions grandes del hook
- story_hook → B-roll del lugar/persona mencionada durante narración + captions con la frase clave
- reaction_setup → Split screen: setup arriba, reacción abajo + SFX en el payoff + captions de impacto
- controversial_take → Text overlay con opinión en ROJO + zoom lento + "COMENTA" CTA
- relatable_confession → Emoji reactions flotando + caption "¿A quién más le pasa?" + soft zoom
- wait_for_it → Cuenta regresiva overlay 3-2-1 + suspense cue + payoff con zoom rápido + freeze
- transformation_reveal → Before/after split + text overlay del cambio + zoom out dramático en reveal
- intimate_moment → Soft lighting filter + captions elegantes serif + fade in/out lento + close-up
- pattern_interrupt → Jump cut brusco + color invertido 0.3s + texto grande del hook inmediato
- bold_statement → Texto del statement en BLANCO sobre NEGRO 1s + zoom in + SFX impacto
- curiosity_gap → Texto "Esto es lo que nadie te cuenta..." + delay 2s + reveal con captions pop
- social_proof → Contador de views/likes flotante + "Mira lo que dicen" + testimonials rápidos
- value_first → Captions bullet list animada + número grande del beneficio + color brand

### PLATAFORMAS:
tiktok | reels | shorts | linkedin | twitter | landscape

### FORMATO DE RESPUESTA (JSON estricto, sin markdown):
{{
  "clips": [
    {{
      "start": 12.5,
      "end": 35.0,
      "virality_score": 8.7,
      "hook_score": 9.0,
      "engagement_score": 8.6,
      "emotional_arc_score": 8.8,
      "pacing_score": 8.5,
      "flow_score": 8.3,
      "value_score": 7.5,
      "mood": "shocking",
      "hook_type": "facial_reveal",
      "ideal_platform": "tiktok",
      "hook": "NO VAS A CREER lo que pasó cuando abrí la puerta",
      "reason": "Expresión facial de shock en segundo 2 + buildup de suspense + payoff inesperado. Emotional arc: calm→suspense→shock.",
      "emotional_start": "calm",
      "tension_point": "Segundo 15: ojos desorbitados, pausa dramática de 1s antes del reveal",
      "payoff": "Segundo 22: revelación escalofriante con reacción final",
      "edit_recipe": "Zoom punch-in a cara de sorpresa seg 2 + freeze 0.5s + captions grandes del hook"
    }}
  ]
}}

{few_shot_block}

### INSTRUCCIONES FINALES:
- Devuelve SOLO el JSON, sin texto ni markdown adicional
- Todos los scores de 1.0 a 10.0
- virality_score = hook_score*0.30 + engagement_score*0.25 + emotional_arc_score*0.20 + pacing_score*0.15 + value_score*0.10
- Ordena clips de mayor a menor virality_score
- Los clips NO deben solaparse (mínimo 3s entre clips)
- Cada clip debe ser autocontenido con Emotional Arc completo
- Hooks genéricos ("bienvenidos", "hola", "empecemos") = descartar, score bajo
- Cada clip DEBE tener un hook visual en los primeros 1-3 segundos
- emotional_arc_score bajo (< 6.0) = clip probablemente aburrido o monótono
- Idealmente inicia/termina en cambios de escena o silencios naturales

Identifica ahora los {num_clips} mejores clips:"""
        
        return prompt
    
    def _extract_json_str(self, response_text: str) -> str:
        """Extrae el bloque JSON de la respuesta cruda de Gemini."""
        # 1. JSON en bloque de código markdown
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if m:
            return m.group(1)
        # 2. JSON raw con clave 'clips'
        m = re.search(r'(\{[\s\S]*"clips"[\s\S]*\})', response_text)
        if m:
            return m.group(1)
        # 3. Devolver tal cual y dejar que el parser lo intente
        return response_text

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extrae el JSON de la respuesta del LLM.
        Usa json_repair como fallback para JSON malformado (trailing commas,
        comentarios, truncamiento) que Gemini produce ocasionalmente.
        """
        json_str = self._extract_json_str(response_text)

        # Intento 1: json.loads estricto
        data = None
        parse_error = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            parse_error = e
            logger.warning(f"json.loads falló ({e}), intentando json_repair...")

        # Intento 2: json_repair (maneja trailing commas, comentarios, truncamiento)
        if data is None and _HAS_JSON_REPAIR:
            try:
                repaired = repair_json(json_str, return_objects=True)
                if isinstance(repaired, dict):
                    data = repaired
                    logger.info("JSON reparado con json_repair ✅")
            except Exception as re_err:
                logger.warning(f"json_repair también falló: {re_err}")

        # Intento 3: extraer clips individuales con regex como último recurso
        if data is None:
            clips_raw = re.findall(
                r'\{[^{}]*"start"\s*:\s*[\d.]+[^{}]*"end"\s*:\s*[\d.]+[^{}]*\}',
                json_str, re.DOTALL
            )
            if clips_raw:
                parsed_clips = []
                for c in clips_raw:
                    try:
                        parsed_clips.append(json.loads(c))
                    except Exception:
                        pass
                if parsed_clips:
                    data = {'clips': parsed_clips}
                    logger.warning(f"Extracción de emergencia: {len(parsed_clips)} clips rescatados")

        if data is None:
            logger.error(f"Respuesta recibida: {response_text[:600]}...")
            raise ValueError(f"La respuesta no es JSON válido: {parse_error}")

        # Validar estructura
        if 'clips' not in data:
            raise ValueError("La respuesta no contiene la clave 'clips'")
        if not isinstance(data['clips'], list):
            raise ValueError("'clips' debe ser una lista")

        required_fields = ['start', 'end', 'virality_score']
        for i, clip in enumerate(data['clips']):
            for field in required_fields:
                if field not in clip:
                    raise ValueError(f"Clip {i} falta campo '{field}'")
            clip['start'] = float(clip['start'])
            clip['end'] = float(clip['end'])
            clip['virality_score'] = float(clip['virality_score'])
            clip.setdefault('reason', 'Momento viral identificado')
            clip.setdefault('hook', 'Hook impactante')

        return data
    
    def analyze_transcription(
        self,
        transcription: Dict[str, Any],
        num_clips: int = 3,
        clip_duration_range: tuple = (15, 60),
        retry_attempts: int = 3,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        custom_prompt: str = "",
        compact: bool = False,
        engagement_data: Optional[List[Any]] = None,
        scene_changes: Optional[List[float]] = None
    ) -> List[ViralClip]:
        """
        Analiza una transcripción y devuelve los clips virales identificados.
        
        Args:
            transcription: Dict con 'text', 'segments', 'duration'
            num_clips: Cuántos clips identificar (default 3)
            clip_duration_range: (min_seg, max_seg) duración de clips
            retry_attempts: Intentos en caso de error
            progress_callback: Función opcional(progress: float, message: str)
            engagement_data: Datos de audio engagement (opcional)
            scene_changes: Timestamps de cambios de escena (opcional)
            
        Returns:
            Lista de objetos ViralClip ordenados por score
            
        Raises:
            RuntimeError: Si no se puede obtener respuesta válida
        """
        if self.model is None:
            raise RuntimeError("Modelo no inicializado")
        
        def update_progress(p: float, msg: str):
            if progress_callback:
                progress_callback(p, msg)
        
        prompt = self._build_prompt(transcription, num_clips, clip_duration_range, custom_prompt, compact=compact, engagement_data=engagement_data, scene_changes=scene_changes)
        
        last_error = None
        for attempt in range(retry_attempts):
            try:
                update_progress(0.1 * attempt, f" Analizando con Gemini (intento {attempt + 1}/{retry_attempts})...")
                logger.info(f"Enviando análisis a Gemini (intento {attempt + 1}/{retry_attempts})...")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json"
                    )
                )
                
                response_text = response.text
                logger.debug(f"Respuesta recibida: {response_text[:200]}...")
                
                update_progress(0.8, " Procesando respuesta...")
                
                # Parsear respuesta
                data = self._parse_response(response_text)
                
                # Convertir a objetos ViralClip v3 - castear a float para evitar errores
                clips = []
                for clip in data['clips']:
                    try:
                        clips.append(ViralClip(
                            start=float(clip['start']),
                            end=float(clip['end']),
                            virality_score=float(clip['virality_score']),
                            reason=str(clip.get('reason', '')),
                            hook=str(clip.get('hook', '')),
                            hook_score=float(clip.get('hook_score', clip.get('virality_score', 0))),
                            pacing_score=float(clip.get('pacing_score', clip.get('virality_score', 0))),
                            engagement_score=float(clip.get('engagement_score', clip.get('virality_score', 0))),
                            flow_score=float(clip.get('flow_score', clip.get('virality_score', 0))),
                            value_score=float(clip.get('value_score', clip.get('virality_score', 0))),
                            trend_score=float(clip.get('trend_score', 5.0)),
                            emotional_arc_score=float(clip.get('emotional_arc_score', clip.get('engagement_score', 5.0))),
                            mood=str(clip.get('mood', 'neutral')),
                            hook_type=str(clip.get('hook_type', 'unknown')),
                            ideal_platform=str(clip.get('ideal_platform', 'tiktok')),
                            edit_recipe=str(clip.get('edit_recipe', '')),
                        ))
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Clip inválido ignorado: {e}")
                        continue
                
                # Validate clips (talking-head quality gate)
                video_duration = transcription.get('duration', float('inf'))
                clips = self.validate_clips(clips, clip_duration_range[0], clip_duration_range[1], video_duration)
                
                # Ordenar por score descendente
                clips.sort(key=lambda x: x.virality_score, reverse=True)
                
                update_progress(1.0, f" {len(clips)} clips identificados!")
                logger.info(f"Análisis completado: {len(clips)} clips identificados")
                for clip in clips:
                    logger.info(f"  Clip: {clip.start:.1f}s-{clip.end:.1f}s | Score: {clip.virality_score:.1f}")
                
                return clips
                
            except Exception as e:
                last_error = e
                logger.warning(f"Error en intento {attempt + 1}: {e}")
                
                if attempt < retry_attempts - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff = 2 ** attempt
                    update_progress(0.1 * attempt, f" Error, reintentando en {backoff}s...")
                    logger.info(f"Esperando {backoff}s antes de reintentar...")
                    time.sleep(backoff)
                    continue
                else:
                    update_progress(0.0, f" Fallaron todos los intentos")
                    logger.error(f"Fallaron todos los intentos: {e}")
                    raise RuntimeError(f"No se pudo analizar el video tras {retry_attempts} intentos: {last_error}")
        
        return []  # Nunca debería llegar aquí
    
    def analyze_with_frames(
        self,
        transcription: Dict[str, Any],
        frames: List[Any],
        num_clips: int = 3,
        clip_duration_range: tuple = (15, 60),
        retry_attempts: int = 3,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        custom_prompt: str = "",
    ) -> List[ViralClip]:
        """
        Analiza transcripción + frames de video para identificar clips virales.
        
        Args:
            transcription: Dict con 'text', 'segments', 'duration'
            frames: Lista de PIL.Image objects (keyframes del video)
            num_clips: Cuántos clips identificar
            clip_duration_range: (min_seg, max_seg) duración de clips
            retry_attempts: Intentos en caso de error
            progress_callback: Función opcional(progress: float, message: str)
            custom_prompt: Instrucción adicional del usuario
            
        Returns:
            Lista de objetos ViralClip ordenados por score
        """
        if self.model is None:
            raise RuntimeError("Modelo no inicializado")
        
        def update_progress(p: float, msg: str):
            if progress_callback:
                progress_callback(p, msg)
        
        # Build enhanced prompt with visual context
        full_text = transcription.get('text', '')[:2000]  # Limit text
        min_dur, max_dur = clip_duration_range
        
        visual_context = f"""
Se han adjuntado {len(frames)} frames representativos del video, distribuidos uniformemente en el tiempo.
Analiza estos frames junto con la transcripción para identificar momentos visualmente impactantes.
""" if frames else ""

        prompt = f"""Eres un Senior Virality Engineer especializado en contenido de PERSONAS HABLANDO. Analiza los frames adjuntos + transcripción para identificar los {num_clips} MEJORES clips virales para TikTok/Reels/Shorts.

{visual_context}

### VIDEO - TRANSCRIPCIÓN
{full_text}

### ANÁLISIS VISUAL DE FRAMES - PRIORIDADES
Al revisar los frames, enfócate en:
1. **EXPRESIONES FACIALES** (más importante): ¿Sorpresa? ¿Shock? ¿Emoción genuina? ¿Sonrisa contagiosa? ¿Mirada intensa?
2. **GESTOS CORPORALES**: ¿Manos en la cara? ¿Salto de sorpresa? ¿Postura de derrota/victoria?
3. **PEAK MOMENTS**: ¿Frame con máxima emoción? ¿Momento de reveal? ¿Twist?
4. **HOOK VISUAL en segundo 1**: ¿El frame inicial DETIENE el scroll?

### HOOK TYPES (elige el más apropiado según la expresión facial/gesto visible):
- facial_reveal: expresión facial impactante (sorpresa, shock, emoción)
- story_hook: "Cuando yo tenía..." story opening con stakes
- reaction_setup: setup → pausa → reacción (comedy/tension)
- controversial_take: opinión fuerte, cara seria/intensa
- relatable_confession: "yo también", sonrisa/emoción compartida
- wait_for_it: delayed payoff, suspense visible en frames
- transformation_reveal: before/after visible en frames
- intimate_moment: vulnerabilidad genuina, emoción real
- bold_statement: afirmación polémica con gesto firme
- pattern_interrupt: gesto/expresión que rompe expectativas

### CRITERIOS DE SELECCIÓN - TALKING HEAD
1. **Hook visual en segundo 1-2**: Frame inicial DEBE mostrar expresión facial interesante
2. **Emotional Arc**: Estado inicial → tensión/emoción → payoff/resolución
3. **Retention**: ¿Qué en los frames hace que el viewer NO scrollee away?
4. **Autocontenido**: Arco completo en 15-60s, no necesita contexto externo
5. **Expresiones genuinas**: NO forzadas. Reacciones reales > actuadas.

### FORMATO JSON (solo JSON, sin texto adicional):
{{
  "clips": [
    {{
      "start": float,
      "end": float,
      "virality_score": float,
      "hook_score": float,
      "engagement_score": float,
      "emotional_arc_score": float,
      "pacing_score": float,
      "value_score": float,
      "mood": "funny|shocking|educational|motivational|dramatic|controversial|wholesome|inspirational",
      "hook_type": "facial_reveal|story_hook|reaction_setup|controversial_take|relatable_confession|wait_for_it|transformation_reveal|intimate_moment|bold_statement|pattern_interrupt",
      "ideal_platform": "tiktok|reels|shorts",
      "hook": "texto EXACTO del hook",
      "reason": "explicación detallada considerando elementos visuales y expresiones faciales",
      "emotional_start": "calm|excited|frustrated|suspenseful",
      "tension_point": "descripción del momento de máxima emoción visible",
      "payoff": "descripción del payoff/resolución",
      "edit_recipe": "instrucciones específicas basadas en la expresión facial/gesto visible"
    }}
  ]
}}

- virality_score = hook_score*0.30 + engagement_score*0.25 + emotional_arc_score*0.20 + pacing_score*0.15 + value_score*0.10
- Scores 1.0-10.0
- Clips NO solapados (min 3s entre clips)
- Hooks genéricos = descartar

Identifica los {num_clips} mejores clips:"""

        if custom_prompt.strip():
            prompt += f"\n\n### INSTRUCCIÓN ADICIONAL:\n{custom_prompt}"

        last_error = None
        for attempt in range(retry_attempts):
            try:
                update_progress(0.1 * attempt, f" Analizando con Gemini + frames (intento {attempt + 1}/{retry_attempts})...")
                logger.info(f"Enviando análisis multimodal a Gemini (intento {attempt + 1}/{retry_attempts})...")
                
                # Build content list with prompt + frames
                content = [prompt]
                if frames:
                    content.extend(frames[:8])  # Max 8 frames to avoid token limit
                
                response = self.model.generate_content(
                    content,
                    generation_config=GenerationConfig(
                        temperature=0.7,  # Higher creativity for visual analysis
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json"
                    )
                )
                
                response_text = response.text
                logger.debug(f"Respuesta recibida: {response_text[:200]}...")
                
                update_progress(0.8, " Procesando respuesta...")
                
                # Parse response
                data = self._parse_response(response_text)
                
                # Convert to ViralClip objects v3
                clips = []
                for clip in data['clips']:
                    try:
                        clips.append(ViralClip(
                            start=float(clip['start']),
                            end=float(clip['end']),
                            virality_score=float(clip.get('virality_score', 0)),
                            reason=str(clip.get('reason', '')),
                            hook=str(clip.get('hook', '')),
                            hook_score=float(clip.get('hook_score', clip.get('virality_score', 0))),
                            pacing_score=float(clip.get('pacing_score', clip.get('virality_score', 0))),
                            engagement_score=float(clip.get('engagement_score', clip.get('virality_score', 0))),
                            flow_score=float(clip.get('flow_score', clip.get('virality_score', 0))),
                            value_score=float(clip.get('value_score', clip.get('virality_score', 0))),
                            trend_score=float(clip.get('trend_score', 5.0)),
                            emotional_arc_score=float(clip.get('emotional_arc_score', clip.get('engagement_score', 5.0))),
                            mood=str(clip.get('mood', 'neutral')),
                            hook_type=str(clip.get('hook_type', 'unknown')),
                            ideal_platform=str(clip.get('ideal_platform', 'tiktok')),
                            edit_recipe=str(clip.get('edit_recipe', '')),
                        ))
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Clip inválido ignorado: {e}")
                        continue

                # Validate clips (talking-head quality gate)
                video_duration = transcription.get('duration', float('inf'))
                clips = self.validate_clips(clips, clip_duration_range[0], clip_duration_range[1], video_duration)

                # Sort by score
                clips.sort(key=lambda x: x.virality_score, reverse=True)

                update_progress(1.0, f" {len(clips)} clips identificados con análisis visual!")
                logger.info(f"Análisis multimodal completado: {len(clips)} clips")
                for clip in clips:
                    logger.info(f"  Clip: {clip.start:.1f}s-{clip.end:.1f}s | Score: {clip.virality_score:.1f}")
                
                return clips
                
            except Exception as e:
                last_error = e
                logger.warning(f"Error en intento {attempt + 1}: {e}")
                
                if attempt < retry_attempts - 1:
                    backoff = 2 ** attempt
                    update_progress(0.1 * attempt, f" Error, reintentando en {backoff}s...")
                    logger.info(f"Esperando {backoff}s antes de reintentar...")
                    time.sleep(backoff)
                    continue
                else:
                    update_progress(0.0, f" Fallaron todos los intentos")
                    logger.error(f"Fallaron todos los intentos: {e}")
                    raise RuntimeError(f"No se pudo analizar el video tras {retry_attempts} intentos: {last_error}")
        
        return []

    def _clips_overlap(self, c1_start: float, c1_end: float, c2_start: float, c2_end: float, min_gap: float = 3.0) -> bool:
        """Check if two clips overlap or are too close."""
        return not (c1_end + min_gap <= c2_start or c2_end + min_gap <= c1_start)

    def _build_edit_recipe(self, hook_type: str, mood: str) -> str:
        """Generate edit recipe based on hook type for talking-head content."""
        recipes = {
            'facial_reveal': 'Zoom punch-in a cara en frame de reacción + freeze 0.5s + captions grandes del hook',
            'story_hook': 'B-roll del lugar/persona mencionada durante narración + captions con la frase clave destacada',
            'reaction_setup': 'Split screen: setup arriba, reacción abajo + SFX en el payoff + captions de impacto',
            'controversial_take': 'Text overlay con opinión en ROJO + zoom lento mientras habla + "COMENTA" CTA',
            'relatable_confession': 'Emoji reactions flotando + caption "¿A quién más le pasa?" + soft zoom',
            'wait_for_it': 'Cuenta regresiva overlay 3-2-1 + suspense music cue + payoff con zoom rápido + freeze',
            'transformation_reveal': 'Before/after split + text overlay del cambio + zoom out dramático en reveal',
            'intimate_moment': 'Soft lighting filter + captions elegantes serif + fade in/out lento + close-up',
            'pattern_interrupt': 'Jump cut brusco + color invertido 0.3s + texto grande del hook inmediato',
            'bold_statement': 'Texto del statement en BLANCO sobre NEGRO 1s + zoom in + SFX impacto',
            'curiosity_gap': 'Texto "Esto es lo que nadie te cuenta..." + delay 2s + reveal con captions pop',
            'social_proof': 'Contador de views/likes flotante + "Mira lo que dicen" + testimonials rápidos',
            'value_first': 'Captions bullet list animada + número grande del beneficio + color brand',
            'shock_humor': 'Record scratch SFX + zoom a cara confundida + caption WTF + laugh track leve',
            'question': 'Texto de la pregunta en pantalla 2s + viewer poll overlay + reveal después',
        }
        base = recipes.get(hook_type, f'Captions bold estilo {mood} + zoom suave en momentos clave')
        if mood in ['shocking', 'controversial', 'dramatic']:
            base += ' | Color grading: alto contraste, saturation boost 15%'
        elif mood in ['wholesome', 'inspirational']:
            base += ' | Color grading: warm tones, soft vignette'
        elif mood == 'funny':
            base += ' | SFX comedy en cada punchline, emojis animados'
        return base

    def validate_clips(self, clips: List[ViralClip], min_duration: float, max_duration: float, video_duration: float) -> List[ViralClip]:
        """
        Validate clips: no overlap, valid duration, scores > 7, hooks not generic.
        Returns only valid clips, sorted by score.
        """
        if not clips:
            return []
        
        valid = []
        for clip in clips:
            # Duration check
            clip_duration = clip.end - clip.start
            if clip_duration < min_duration or clip_duration > max_duration:
                logger.warning(f"Clip descartado por duración: {clip.start:.1f}s-{clip.end:.1f}s ({clip_duration:.1f}s)")
                continue
            
            # Boundaries within video
            if clip.start < 0 or clip.end > video_duration + 5:  # 5s tolerance
                logger.warning(f"Clip fuera de rango: {clip.start:.1f}s-{clip.end:.1f}s (video={video_duration:.1f}s)")
                continue
            
            # Virality score check (must be > 6.5 for talking head)
            if clip.virality_score < 6.5:
                logger.warning(f"Clip score muy bajo ({clip.virality_score:.1f}): {clip.start:.1f}s")
                continue
            
            # Hook quality check
            hook = clip.hook.strip().lower()
            if not hook or len(hook) < 5 or hook in ['hook impactante', 'sin hook', 'ninguno', '']:
                logger.warning(f"Clip hook genérico: {clip.start:.1f}s")
                continue
            
            valid.append(clip)
        
        # Remove overlaps (keep higher score)
        valid.sort(key=lambda c: c.virality_score, reverse=True)
        final = []
        for clip in valid:
            if not any(self._clips_overlap(clip.start, clip.end, f.start, f.end) for f in final):
                final.append(clip)
        
        logger.info(f"Validación: {len(clips)} → {len(valid)} → {len(final)} clips finales")
        return final

    def get_clip_transcript(
        self,
        transcription: Dict[str, Any],
        clip: ViralClip
    ) -> str:
        """
        Extrae el texto de la transcripción para un clip específico.
        
        Args:
            transcription: Dict con 'segments'
            clip: Objeto ViralClip con start/end
            
        Returns:
            Texto concatenado del clip
        """
        segments = transcription.get('segments', [])
        
        clip_texts = []
        for seg in segments:
            if seg['start'] >= clip.start and seg['end'] <= clip.end:
                clip_texts.append(seg['text'].strip())
            elif seg['start'] < clip.end and seg['end'] > clip.start:
                # Segmento parcialmente dentro del clip
                clip_texts.append(seg['text'].strip())
        
        return ' '.join(clip_texts)


def analyze_video_transcription(
    transcription: Dict[str, Any],
    api_key: Optional[str] = None,
    num_clips: int = 3
) -> List[ViralClip]:
    """
    Función de conveniencia para analizar una transcripción.
    
    Args:
        transcription: Resultado de transcriber.transcribe()
        api_key: API key opcional (usa hardcodeada por defecto)
        num_clips: Número de clips a identificar
        
    Returns:
        Lista de ViralClip
    """
    analyzer = GeminiAnalyzer(api_key=api_key)
    return analyzer.analyze_transcription(transcription, num_clips=num_clips)


if __name__ == "__main__":
    # Test del módulo con datos de ejemplo
    test_transcription = {
        "text": "Hola amigos bienvenidos a mi canal hoy vamos a hacer algo increíble. Primero necesitamos preparar los ingredientes. ¡Oh no! Se me cayó el huevo. Bueno, continuemos con la receta de todos modos. El resultado final es espectacular.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Hola amigos bienvenidos a mi canal hoy vamos a hacer algo increíble."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "Primero necesitamos preparar los ingredientes."},
            {"id": 2, "start": 10.0, "end": 13.0, "text": "¡Oh no! Se me cayó el huevo."},
            {"id": 3, "start": 13.0, "end": 18.0, "text": "Bueno, continuemos con la receta de todos modos."},
            {"id": 4, "start": 18.0, "end": 22.0, "text": "El resultado final es espectacular."}
        ],
        "duration": 22.0,
        "language": "es"
    }
    
    print("🧠 Probando GeminiAnalyzer...")
    print(f"   Transcripción de prueba: {len(test_transcription['segments'])} segmentos")
    
    try:
        analyzer = GeminiAnalyzer()
        clips = analyzer.analyze_transcription(test_transcription, num_clips=2)
        
        print(f"\n✅ Análisis exitoso!")
        print(f"   Clips identificados: {len(clips)}")
        for i, clip in enumerate(clips, 1):
            print(f"\n   Clip {i}:")
            print(f"      Tiempo: {clip.start:.1f}s - {clip.end:.1f}s")
            print(f"      Score: {clip.virality_score:.1f}/10")
            print(f"      Hook: {clip.hook}")
            print(f"      Razón: {clip.reason}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
