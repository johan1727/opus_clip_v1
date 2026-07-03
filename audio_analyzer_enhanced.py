"""
Analizador de audio avanzado para detección de engagement.
"""
import logging, subprocess, json, math
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    start: float; end: float; rms_db: float; peak_db: float
    is_silence: bool; silence_duration: float = 0.0

@dataclass
class EngagementData:
    timestamp: float
    audio_energy_score: float
    speech_density_score: float
    scene_change_score: float
    silence_proximity_score: float
    combined_score: float

class AudioAnalyzerEnhanced:
    def __init__(self, silence_threshold_db: float = -40.0):
        self.silence_threshold_db = silence_threshold_db

    def analyze_audio_energy(self, video_path: str, window_secs: float = 1.0) -> List[AudioSegment]:
        video_path = Path(video_path)
        if not video_path.exists():
            return []
        try:
            cmd_probe = ['ffprobe','-v','error','-show_entries','format=duration',
                         '-of','default=noprint_wrappers=1:nokey=1', str(video_path)]
            result = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip()) if result.returncode == 0 else 0
            if duration <= 0:
                return []
            # Silencedetect
            cmd = ['ffmpeg','-i',str(video_path),'-af',
                   f'silencedetect=noise={self.silence_threshold_db}dB:d=0.3',
                   '-f','null','-']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            silences = self._parse_silencedetect(result.stderr)
            # pydub fallback
            return self._fallback_energy_analysis(video_path, duration, window_secs, silences)
        except Exception as e:
            logger.error(f"Error analizando audio: {e}")
            return []

    def _parse_silencedetect(self, stderr: str) -> List[Tuple[float,float]]:
        silences = []; silence_start = None
        for line in stderr.split('\n'):
            if 'silence_start:' in line:
                try: silence_start = float(line.split('silence_start:')[1].strip().split()[0])
                except: continue
            elif 'silence_end:' in line and silence_start is not None:
                try: silences.append((silence_start, float(line.split('silence_end:')[1].strip().split()[0]))); silence_start = None
                except: continue
        return silences

    def _fallback_energy_analysis(self, video_path, duration, window_secs, silences):
        try:
            from pydub import AudioSegment as PydubAudio
            temp_wav = video_path.parent / f"temp_audio_{video_path.stem}.wav"
            cmd = ['ffmpeg','-y','-i',str(video_path),'-vn','-acodec','pcm_s16le','-ar','16000','-ac','1',str(temp_wav)]
            subprocess.run(cmd, capture_output=True, timeout=120)
            if not temp_wav.exists(): raise RuntimeError("No audio")
            audio = PydubAudio.from_wav(str(temp_wav))
            segments = []; window_ms = int(window_secs * 1000)
            for i in range(0, len(audio), window_ms):
                chunk = audio[i:i+window_ms]; start = i/1000.0; end = min((i+window_ms)/1000.0, duration)
                rms = chunk.rms; rms_db = 20*math.log10(rms/32768.0) if rms > 0 else -96.0
                peak = chunk.max; peak_db = 20*math.log10(peak/32768.0) if peak > 0 else -96.0
                is_silence = rms_db < self.silence_threshold_db; silence_dur = 0.0
                for ss, se in silences:
                    os = max(start,ss); oe = min(end,se)
                    if oe > os: silence_dur += oe - os
                segments.append(AudioSegment(start,end,rms_db,peak_db,is_silence,silence_dur))
            try: temp_wav.unlink()
            except: pass
            return segments
        except ImportError:
            return self._basic_energy_estimate(duration, window_secs, silences)

    def _basic_energy_estimate(self, duration, window_secs, silences):
        segments = []
        for i in range(int(duration / window_secs) + 1):
            start = i * window_secs; end = min((i+1)*window_secs, duration)
            silence_dur = 0.0
            for ss, se in silences:
                os = max(start,ss); oe = min(end,se)
                if oe > os: silence_dur += oe - os
            is_silence = silence_dur > (end-start)*0.3
            segments.append(AudioSegment(start,end,-50.0 if is_silence else -20.0,
                                          -40.0 if is_silence else -10.0,is_silence,silence_dur))
        return segments

    def calculate_engagement_scores(self, segments, scene_changes, transcription_segments):
        if not segments: return []
        rms_vals = [s.rms_db for s in segments if not s.is_silence]
        min_rms, max_rms = (min(rms_vals), max(rms_vals)) if rms_vals else (-60, -20)
        rms_range = max_rms - min_rms if max_rms > min_rms else 1.0
        # word density map
        word_density = {}
        for seg in transcription_segments:
            s, e = seg.get('start',0), seg.get('end',0)
            wps = len(seg.get('text','').split()) / (e-s) if e > s else 0
            for t in range(int(s), int(e)+1):
                word_density[t] = max(word_density.get(t,0), wps)
        max_wps = max(word_density.values()) if word_density else 1.0
        result = []
        for seg in segments:
            mid = (seg.start + seg.end) / 2
            audio_score = 1.0 if seg.is_silence else max(0,min(10,((seg.rms_db-min_rms)/rms_range)*10))
            wps = word_density.get(int(mid),0)
            speech_score = min(10,(wps/max_wps)*10) if max_wps > 0 else 5.0
            scene_score = max((10-abs(mid-sc)*5) for sc in scene_changes if abs(mid-sc)<2) if scene_changes else 0.0
            silence_score = 0.0  # calculated below
            for ss, se in [(s.rms_db,s.silence_duration) for s in segments if s.is_silence]:
                pass  # simplified
            combined = audio_score*0.35 + speech_score*0.30 + scene_score*0.25 + silence_score*0.10
            result.append(EngagementData(mid,audio_score,speech_score,scene_score,silence_score,min(10,combined)))
        return result

    def find_peak_engagement_windows(self, engagement_data, window_size=30.0, top_n=20):
        if not engagement_data: return []
        duration = max(e.timestamp for e in engagement_data)
        candidates = []
        step = 2.0
        t = 0.0
        while t + 15 <= duration:
            scores = [e.combined_score for e in engagement_data if t <= e.timestamp <= t+window_size]
            if scores:
                avg = sum(scores)/len(scores); peak = max(scores)
                # Weight peak score more
                score = avg*0.4 + peak*0.6
                candidates.append((t, t+window_size, score))
            t += step
        candidates.sort(key=lambda x: x[2], reverse=True)
        # Non-overlapping greedy selection
        selected = []
        for c in candidates:
            if len(selected) >= top_n: break
            if not any(abs(c[0]-s[0])<5 or abs(c[1]-s[1])<5 for s in selected):
                selected.append(c)
        return selected
