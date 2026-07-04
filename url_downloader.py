"""
Descarga de video por URL (YouTube, TikTok, Instagram, X/Twitter) vía yt-dlp.
Módulo aislado: no importa nada de app.py/video_editor.py, produce un
archivo local y devuelve su ruta — el resto del pipeline lo trata igual
que un archivo subido manualmente.
"""
import logging
import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import yt_dlp

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = (
    "youtube.com", "youtu.be",
    "tiktok.com",
    "instagram.com",
    "x.com", "twitter.com",
)


def is_supported_url(url: str) -> bool:
    """
    True si `url` es una URL http(s) válida de un dominio soportado.
    Chequeo defensivo antes de invocar yt-dlp — evita pasarle cualquier
    string arbitrario a un proceso de descarga externo.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.netloc or "").lower()
    host = re.sub(r"^www\.", "", host)
    return any(host == d or host.endswith(f".{d}") for d in SUPPORTED_DOMAINS)


def download_video(
    url: str,
    output_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_duration_s: int = 7200,
) -> str:
    """
    Descarga `url` a `output_dir` usando la API de Python de yt-dlp (no
    subprocess con la URL interpolada en un comando — evita cualquier
    vector de inyección de argumentos).

    Args:
        url: link del video (YouTube/TikTok/Instagram/X)
        output_dir: carpeta destino (se crea si no existe)
        progress_callback: (progress 0-1, mensaje) opcional
        max_duration_s: duración máxima descargable en segundos (default 2h)
            — evita que un stream/video excesivamente largo sature disco/tiempo

    Returns:
        Ruta absoluta del archivo descargado.

    Raises:
        ValueError: si la URL no es de un dominio soportado, o el video
            excede max_duration_s.
        RuntimeError: si yt-dlp falla (video privado, eliminado, geo-bloqueado, etc.)
    """
    if not is_supported_url(url):
        raise ValueError(
            f"URL no soportada: {url}\n"
            "Dominios soportados: YouTube, TikTok, Instagram, X/Twitter."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _progress_hook(d):
        if not progress_callback:
            return
        if d.get('status') == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                progress_callback(downloaded / total, f"⬇️ Descargando... {downloaded / total * 100:.0f}%")
        elif d.get('status') == 'finished':
            progress_callback(1.0, "✅ Descarga completa, procesando...")

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': str(output_path / '%(title).100s.%(ext)s'),
        'progress_hooks': [_progress_hook],
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise RuntimeError(
                    "yt-dlp no devolvió información del video "
                    "(respuesta vacía del extractor)."
                )
            duration = info.get('duration') or 0
            if duration > max_duration_s:
                raise ValueError(
                    f"El video dura {duration/60:.0f} min, excede el máximo "
                    f"permitido de {max_duration_s/60:.0f} min."
                )
            ydl.download([url])
            filename = ydl.prepare_filename(info)
            logger.info(f"Video descargado: {filename}")
            return filename
    except (ValueError, RuntimeError):
        # Nuestras propias excepciones de contrato (URL/duración/info vacía)
        # ya vienen limpias — no las re-envolvemos.
        raise
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"No se pudo descargar el video: {e}")
    except Exception as e:
        # Cualquier otro error de yt-dlp (red, error interno, etc.) que no
        # sea DownloadError — no debe escapar como traceback crudo.
        raise RuntimeError(f"Error inesperado al descargar el video: {e}")
