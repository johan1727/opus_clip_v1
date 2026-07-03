# Reglas de Estilo de Código - OpusClip Clone

## 1. Tipado Estricto en Python
- **OBLIGATORIO**: Usar type hints en TODAS las funciones y variables
- Usar `typing` module: `List`, `Dict`, `Optional`, `Tuple`, `Callable`
- Ejemplo: `def transcribe(video_path: str, model_size: str = "base") -> Dict[str, Any]:`

## 2. Manejo de Errores Robusto
- Usar `try/except` con tipos de error específicos
- Nunca dejar `except:` vacío - siempre loggear o re-raise
- Crear excepciones custom para errores de dominio
- Verificar precondiciones (archivos existen, variables no son None)

## 3. Modularidad
- **UI**: Archivo `app.py` - Solo interfaz Gradio, lógica mínima
- **Transcripción**: Archivo `transcriber.py` - Todo lo relacionado con Whisper
- **LLM**: Archivo `llm_analyzer.py` - Análisis con Gemini/OpenAI
- **Edición Video**: Archivo `video_editor.py` - FFmpeg/moviepy operations
- **Utils**: Archivo `utils.py` - Funciones compartidas
- Cada módulo expone una interfaz clara, sin dependencias circulares

## 4. Entornos Virtuales
- Usar `python -m venv venv` para crear entorno
- Activar con `.\venv\Scripts\activate` en Windows
- Nunca instalar paquetes globalmente
- `requirements.txt` debe tener versiones fijas

## 5. Configuración
- Usar archivo `.env` para configuraciones (aunque Gemini key esté hardcodeada para uso personal)
- Cargar con `python-dotenv`
- Validar configuración al inicio de la app

## 6. Logging
- Usar `logging` module, no `print()` para debug
- Formato: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Niveles apropiados: INFO para flujo normal, DEBUG para detalles

## 7. Estructura de Funciones
- Funciones pequeñas (<50 líneas idealmente)
- Una responsabilidad por función
- Nombres descriptivos en inglés: `extract_viral_clips()`, `burn_subtitles()`
- Docstrings con descripción, parámetros y return

## 8. CUDA / GPU
- Detectar CUDA automáticamente: `torch.cuda.is_available()`
- Configurar `device="cuda"` cuando esté disponible
- Cargar modelos en GPU con batch size optimizado para GTX 1080 (8GB VRAM)

## 9. Formato de Salida
- Videos en 9:16 (vertical) 1080x1920
- Subtítulos quemados con estilo legible
- Output en carpeta `output/` con timestamp

## 10. Código Limpiado
- Sin código comentado muerto
- Sin imports no usados
- PEP8 compliant (líneas <100 chars)
- Comentarios solo cuando la lógica no es obvia
