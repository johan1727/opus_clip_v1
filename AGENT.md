# AGENT.md — Contexto persistente para agentes de IA (Claude Code)

Este archivo es la puerta de entrada para cualquier sesión de Claude Code (u otro agente) que
trabaje en este proyecto. No duplica información — apunta a los archivos que ya la tienen y
explica el loop de mejora continua que hay que seguir.

## Cómo trabajar en este proyecto (loop de mejora continua)

1. **Antes de empezar**: leer `memory.md` completo (arquitectura, stack, estado actual, y sobre
   todo la sección de auditoría con fecha más reciente al final) y `claude.md` (reglas de estilo
   de código obligatorias).
2. **Mientras trabajás**: si encontrás un bug, una decisión de producto, una duda del usuario, o
   algo que no se sabía antes, anotalo en `memory.md` con la fecha del día, en la sección de
   auditoría correspondiente (o creando una nueva si no encaja en ninguna existente).
3. **Antes de cerrar la sesión**: actualizar la tabla de "Actualizaciones" en `memory.md` y dejar
   claro qué quedó pendiente para la próxima vez (no asumas que la próxima sesión recuerda nada
   de esta conversación — todo lo que importa tiene que quedar escrito ahí).

No se crean archivos de contexto nuevos por cada hallazgo — todo entra en `memory.md`. Este
archivo (`AGENT.md`) es el único que casi no cambia; es el índice, no el contenido.

## Archivos de referencia

| Archivo | Para qué sirve |
|---|---|
| `claude.md` | Reglas de estilo de código obligatorias (tipado, manejo de errores, modularidad, logging, etc.) |
| `memory.md` | Arquitectura, stack tecnológico, estado del proyecto, y el historial completo de auditorías (seguridad, bugs, UX, contenido viral) con fecha — **la fuente de verdad principal** |
| `requirements.txt` | Dependencias con versión — mantenido en sync con lo que realmente corre en el venv (ver notas de "version drift" en `memory.md` sobre gradio/moviepy) |
| `.env` / `.env.example` | Secretos (nunca en git — ver `.gitignore`) |

## Resumen rápido del proyecto

Clon de Opus Clip en Python: sube un video → Whisper transcribe → Gemini identifica momentos
virales → se recortan en 9:16 con subtítulos quemados, face tracking opcional, y presets por
plataforma (TikTok/Reels/Shorts/LinkedIn/Twitter/Landscape). UI en Gradio. Uso 100% personal,
un solo usuario, corre local en Windows con GPU NVIDIA (NVENC).

Para el estado detallado (qué está hecho, qué bugs se encontraron y arreglaron, qué falta) —
**siempre ir a `memory.md`**, no asumir nada desde acá.

## Reglas de seguridad no negociables

- Nunca hardcodear API keys en el código — van en `.env` (ver `GEMINI_API_KEYS` en `llm_analyzer.py`).
- Nunca commitear `.env`, videos fuente, ni nada de `output/`/`outputs/`/`temp/` — ya está en
  `.gitignore`, no tocar esas reglas sin pensarlo dos veces.
- Antes de cualquier `git push`, verificar que no haya secretos en lo que se va a subir
  (`git diff --cached | grep AIzaSy` como mínimo).
