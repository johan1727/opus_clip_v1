"""
Módulo de análisis de audio para mejorar la detección de momentos virales.
Calcula: loudness RMS, velocidad de habla, picos de energía y zoom punch-in cues.
Usado antes de Gemini para pre-filtrar los mejores segmentos (F2.1).
Provee datos para zoom dinámico con FFmpeg zoompan (F2.3).
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# librosa es opcional — si no está instalado se usa fallback con ffmpeg
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.info("librosa no instalado — análisis de audio en modo básico (ffmpeg)")


class AudioAnalyzer:
    """
    Analiza el audio de un video para detectar momentos de alta energía,
    velocidad de habla y silencios. Alimenta al Virality Engine v2.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Core: extract audio array
    # ------------------------------------------------------------------

    def _extract_audio_array(self, video_path: str) -> Optional[Tuple[Any, int]]:
        """
        Returns (audio_array, sr) using librosa if available,
        otherwise falls back to None (ffmpeg-only mode).
        """
        if not LIBROSA_AVAILABLE:
            return None
        try:
            y, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
            return y, sr
        except Exception as e:
            logger.warning(f"librosa load failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Loudness analysis
    # ------------------------------------------------------------------

    def get_rms_energy(
        self,
        video_path: str,
        frame_duration: float = 1.0,
    ) -> List[Dict[str, float]]:
        """
        Returns per-second RMS energy values for the video.

        Returns:
            List of {'time': float, 'rms': float} dicts sorted by time.
        """
        result = self._extract_audio_array(video_path)
        if result is None:
            return self._rms_via_ffmpeg(video_path, frame_duration)

        y, sr = result
        hop = int(frame_duration * sr)
        energies = []
        for i in range(0, len(y) - hop, hop):
            chunk = y[i: i + hop]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            energies.append({'time': i / sr, 'rms': rms})
        return energies

    def _rms_via_ffmpeg(self, video_path: str, frame_duration: float = 1.0) -> List[Dict[str, float]]:
        """Fallback: parse FFmpeg astats filter for RMS per second."""
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-af", f"astats=metadata=1:reset={int(frame_duration)},"
                       "ametadata=print:key=lavfi.astats.Overall.RMS_level",
                "-f", "null", "-"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            lines = proc.stderr.splitlines()
            energies = []
            t = 0.0
            for line in lines:
                if "RMS_level" in line:
                    try:
                        val = float(line.split("=")[-1].strip())
                        # Convert dBFS to linear
                        linear = 10 ** (val / 20)
                        energies.append({'time': t, 'rms': linear})
                        t += frame_duration
                    except ValueError:
                        pass
            return energies
        except Exception as e:
            logger.warning(f"FFmpeg RMS fallback failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Speech rate analysis
    # ------------------------------------------------------------------

    def get_speech_rate(
        self,
        segments: List[Dict[str, Any]],
        window: float = 5.0,
    ) -> List[Dict[str, float]]:
        """
        Calculates words-per-second in a rolling window over transcript segments.

        Args:
            segments: Whisper segments with 'start', 'end', 'text'
            window: Rolling window in seconds

        Returns:
            List of {'time': float, 'words_per_sec': float}
        """
        if not segments:
            return []

        # Build word timeline
        word_times = []
        for seg in segments:
            words = seg.get('text', '').split()
            if not words:
                continue
            dur = seg['end'] - seg['start']
            wdur = dur / len(words) if len(words) > 0 else 0
            for i, _ in enumerate(words):
                word_times.append(seg['start'] + i * wdur)

        if not word_times:
            return []

        results = []
        max_t = max(word_times)
        t = 0.0
        while t <= max_t:
            count = sum(1 for wt in word_times if t <= wt < t + window)
            results.append({'time': t, 'words_per_sec': count / window})
            t += 1.0
        return results

    # ------------------------------------------------------------------
    # High-energy moment detection
    # ------------------------------------------------------------------

    def get_energy_peaks(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Combines RMS loudness + speech rate to score each second.
        Returns the top_n highest-energy moments with their timestamps.

        Useful for pre-filtering segments before sending to Gemini.

        Returns:
            List of {'time': float, 'score': float, 'rms': float, 'wps': float}
            sorted by score descending.
        """
        rms_data = self.get_rms_energy(video_path)
        wps_data = self.get_speech_rate(segments)

        rms_map = {round(d['time'], 1): d['rms'] for d in rms_data}
        wps_map = {round(d['time'], 1): d['words_per_sec'] for d in wps_data}

        # Normalize
        rms_vals = list(rms_map.values()) or [1]
        wps_vals = list(wps_map.values()) or [1]
        max_rms = max(rms_vals) or 1
        max_wps = max(wps_vals) or 1

        all_times = sorted(set(list(rms_map.keys()) + list(wps_map.keys())))
        scored = []
        for t in all_times:
            rms = rms_map.get(t, 0)
            wps = wps_map.get(t, 0)
            score = 0.6 * (rms / max_rms) + 0.4 * (wps / max_wps)
            scored.append({'time': t, 'score': score, 'rms': rms, 'wps': wps})

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Zoom punch-in cues (F2.3)
    # ------------------------------------------------------------------

    def get_zoom_cues(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        threshold_percentile: float = 75.0,
        max_cues: int = 8,
        min_gap_s: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """
        Returns timestamps where a zoom punch-in should be applied.
        Uses RMS energy peaks above threshold_percentile.

        Returns:
            List of {'time': float, 'intensity': float}
            intensity 0-1 (how strong the zoom should be)
        """
        rms_data = self.get_rms_energy(video_path, frame_duration=0.5)
        if not rms_data:
            return []

        rms_vals = [d['rms'] for d in rms_data]
        if not rms_vals:
            return []

        try:
            import numpy as np_local
            threshold = float(np_local.percentile(rms_vals, threshold_percentile))
            max_rms = max(rms_vals) or 1
        except ImportError:
            sorted_vals = sorted(rms_vals)
            idx = int(len(sorted_vals) * threshold_percentile / 100)
            threshold = sorted_vals[min(idx, len(sorted_vals) - 1)]
            max_rms = sorted_vals[-1] or 1

        cues = []
        last_cue_t = -min_gap_s
        for d in rms_data:
            if d['rms'] >= threshold and (d['time'] - last_cue_t) >= min_gap_s:
                cues.append({
                    'time': d['time'],
                    'intensity': min(1.0, d['rms'] / max_rms),
                })
                last_cue_t = d['time']
                if len(cues) >= max_cues:
                    break

        return cues

    # ------------------------------------------------------------------
    # Segment energy scoring (for LLM pre-filter)
    # ------------------------------------------------------------------

    def score_segments(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Scores each transcript segment by its average RMS energy + speech rate.
        Adds 'audio_energy_score' (0-10) to each segment dict.

        Returns enriched segments list (does NOT mutate originals).
        """
        rms_data = self.get_rms_energy(video_path, frame_duration=0.5)
        rms_series = {round(d['time'], 1): d['rms'] for d in rms_data}
        max_rms = max(rms_series.values()) if rms_series else 1

        scored = []
        for seg in segments:
            start, end = seg['start'], seg['end']
            # Average RMS for this segment window
            window_rms = [
                v for t, v in rms_series.items() if start <= t < end
            ]
            avg_rms = (sum(window_rms) / len(window_rms)) if window_rms else 0
            # Words per second
            words = len(seg.get('text', '').split())
            dur = max(end - start, 0.1)
            wps = words / dur
            # Combined score 0-10
            rms_norm = (avg_rms / max_rms) * 10
            wps_norm = min(wps / 5.0, 1.0) * 10  # 5 wps = max
            audio_score = round(0.6 * rms_norm + 0.4 * wps_norm, 2)
            new_seg = dict(seg)
            new_seg['audio_energy_score'] = audio_score
            scored.append(new_seg)

        return scored
