"""
Módulo de seguimiento facial para recorte dinámico 9:16.
Detecta rostros en el video y calcula offsets óptimos de crop.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import numpy as np
    import cv2
    # Verify cv2 actually works with current NumPy (NumPy 2.x incompatibility check)
    _test = cv2.CascadeClassifier()
    del _test
    CV2_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    CV2_AVAILABLE = False
    try:
        import numpy as np
    except Exception:
        import numpy as np  # fallback, will raise naturally if truly missing
    logging.warning("OpenCV no disponible o incompatible con NumPy actual. Intentando mediapipe...")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

FACE_TRACKING_AVAILABLE = CV2_AVAILABLE or MEDIAPIPE_AVAILABLE
if not FACE_TRACKING_AVAILABLE:
    logging.warning("Face tracking no disponible (ni OpenCV ni mediapipe instalados).")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FacePosition:
    """Posición de un rostro en un frame específico."""
    frame_idx: int
    timestamp: float
    x: int  # Centro del rostro
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class CropOffset:
    """Offset de crop calculado para un momento específico."""
    timestamp: float
    x_offset: int  # Offset horizontal desde el centro
    y_offset: int  # Offset vertical (normalmente 0)


class FaceTracker:
    """
    Rastreador de rostros para mantener hablantes centrados en videos 9:16.
    
    Usa mediapipe como primera opción, OpenCV Haar Cascade como fallback.
    """
    
    def __init__(self, sample_fps: float = 2.0, smoothing_window: float = 1.0):
        """
        Inicializa el tracker de rostros.
        
        Args:
            sample_fps: FPS para muestreo de frames (default 2 = analiza 2 frames/segundo)
            smoothing_window: Ventana de suavizado en segundos
        """
        self.sample_fps = sample_fps
        self.smoothing_window = smoothing_window
        self.face_detector = None
        self.mp_detector = None
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                logger.info("✅ mediapipe FaceDetection cargado")
                return
            except Exception as e:
                logger.warning(f"mediapipe init error: {e}")
        
        if not CV2_AVAILABLE:
            logger.error("❌ Ni OpenCV ni mediapipe disponibles para face tracking")
            return
        
        self._init_face_detector()
    
    def _init_face_detector(self) -> None:
        """Inicializa el detector de rostros DNN de OpenCV."""
        try:
            # Usar DNN face detector (más preciso que Haar Cascade)
            # Descargar modelos si no existen
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            # URLs de modelos OpenCV (descargar si no existen)
            model_urls = {
                model_file: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                config_file: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            }
            
            # Por ahora usar Haar Cascade como fallback (incluido en opencv-python)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector is None or self.face_detector.empty():
                logger.error("❌ No se pudo cargar Haar Cascade")
                self.face_detector = None
            else:
                logger.info("✅ Detector de rostros Haar Cascade cargado")
                
        except Exception as e:
            logger.error(f"Error inicializando detector: {e}")
            self.face_detector = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostros en un frame usando mediapipe o OpenCV.
        
        Args:
            frame: Imagen numpy array (BGR)
            
        Returns:
            Lista de (x, y, w, h) para cada rostro detectado
        """
        # Intentar mediapipe primero
        if self.mp_detector is not None:
            try:
                import cv2 as _cv2
                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                results = self.mp_detector.process(rgb)
                faces = []
                if results.detections:
                    h, w = frame.shape[:2]
                    for det in results.detections:
                        bb = det.location_data.relative_bounding_box
                        x = int(bb.xmin * w)
                        y = int(bb.ymin * h)
                        fw = int(bb.width * w)
                        fh = int(bb.height * h)
                        faces.append((x, y, fw, fh))
                return faces
            except Exception as e:
                logger.warning(f"mediapipe detect error: {e}")
                return []
        
        # Fallback: OpenCV Haar Cascade
        if self.face_detector is None or not CV2_AVAILABLE:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.warning(f"Error detectando rostros: {e}")
            return []
    
    def track_faces(self, video_path: str) -> List[FacePosition]:
        """
        Rastrea rostros a lo largo del video completo.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            Lista de FacePosition con timestamp y posición
        """
        if not CV2_AVAILABLE or self.face_detector is None:
            logger.warning("Face tracking no disponible")
            return []
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        logger.info(f"🔍 Analizando rostros en: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular cada cuántos frames samplear
        sample_interval = int(video_fps / self.sample_fps)
        if sample_interval < 1:
            sample_interval = 1
        
        face_positions = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Samplear solo cada N frames
                if frame_idx % sample_interval == 0:
                    timestamp = frame_idx / video_fps
                    faces = self.detect_faces(frame)
                    
                    for (x, y, w, h) in faces:
                        # Calcular centro del rostro
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        face_positions.append(FacePosition(
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            x=center_x,
                            y=center_y,
                            width=w,
                            height=h,
                            confidence=1.0  # Haar no da confidence
                        ))
                
                frame_idx += 1
                
                # Log progreso cada 10%
                if frame_idx % (total_frames // 10 + 1) == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.debug(f"Análisis facial: {progress:.0f}%")
        
        finally:
            cap.release()
        
        logger.info(f"✅ Detectados {len(face_positions)} rostros en {len(set(f.timestamp for f in face_positions))} frames")
        return face_positions
    
    def get_optimal_crop_offsets(
        self,
        face_positions: List[FacePosition],
        video_width: int,
        video_height: int,
        target_aspect: float = 9/16
    ) -> List[CropOffset]:
        """
        Calcula offsets óptimos de crop para mantener rostros centrados.
        
        Args:
            face_positions: Lista de posiciones de rostros
            video_width: Ancho original del video
            video_height: Alto original del video
            target_aspect: Ratio objetivo (default 9:16 = 9/16)
            
        Returns:
            Lista de CropOffset con timestamp y offsets
        """
        if not face_positions:
            logger.info("No se detectaron rostros - usando crop centrado")
            return []
        
        # Calcular dimensiones del crop
        if video_width / video_height > target_aspect:
            # Video muy ancho, crop horizontal
            crop_width = int(video_height * target_aspect)
            crop_height = video_height
        else:
            # Video muy alto, crop vertical
            crop_width = video_width
            crop_height = int(video_width / target_aspect)
        
        # Agrupar por timestamp (puede haber múltiples rostros por frame)
        faces_by_time: Dict[float, List[FacePosition]] = {}
        for face in face_positions:
            if face.timestamp not in faces_by_time:
                faces_by_time[face.timestamp] = []
            faces_by_time[face.timestamp].append(face)
        
        # Calcular posición promedio de rostros por timestamp
        timestamps = sorted(faces_by_time.keys())
        crop_offsets = []
        
        for t in timestamps:
            faces = faces_by_time[t]
            
            # Calcular centro promedio de todos los rostros
            avg_x = sum(f.x for f in faces) / len(faces)
            avg_y = sum(f.y for f in faces) / len(faces)
            
            # Calcular offset para mantener rostros centrados
            # El crop debe centrarse en avg_x
            center_x = video_width / 2
            x_offset = int(avg_x - center_x)
            
            # Limitar offset para no salir del video
            max_offset = (video_width - crop_width) // 2
            x_offset = max(-max_offset, min(max_offset, x_offset))
            
            y_offset = 0  # Mantener centro vertical
            
            crop_offsets.append(CropOffset(
                timestamp=t,
                x_offset=x_offset,
                y_offset=y_offset
            ))
        
        # Aplicar suavizado (moving average)
        crop_offsets = self._smooth_offsets(crop_offsets)
        
        logger.info(f"✅ Calculados {len(crop_offsets)} offsets de crop")
        return crop_offsets
    
    def _smooth_offsets(self, offsets: List[CropOffset]) -> List[CropOffset]:
        """
        Aplica suavizado a los offsets para transiciones suaves.
        
        Args:
            offsets: Lista de CropOffset
            
        Returns:
            Lista suavizada de CropOffset
        """
        if len(offsets) < 3:
            return offsets
        
        smoothed = []
        window_size = max(1, int(self.smoothing_window * len(offsets) / (offsets[-1].timestamp - offsets[0].timestamp + 0.001)))
        window_size = min(window_size, 5)  # Máximo 5 frames de ventana
        
        for i, offset in enumerate(offsets):
            # Tomar ventana alrededor del punto actual
            start = max(0, i - window_size // 2)
            end = min(len(offsets), i + window_size // 2 + 1)
            
            window = offsets[start:end]
            avg_x = sum(o.x_offset for o in window) / len(window)
            avg_y = sum(o.y_offset for o in window) / len(window)
            
            smoothed.append(CropOffset(
                timestamp=offset.timestamp,
                x_offset=int(avg_x),
                y_offset=int(avg_y)
            ))
        
        return smoothed
    
    def get_crop_at_timestamp(
        self,
        timestamp: float,
        crop_offsets: List[CropOffset],
        video_width: int,
        video_height: int,
        target_width: int = 1080,
        target_height: int = 1920
    ) -> Tuple[int, int, int, int]:
        """
        Obtiene parámetros de crop para un timestamp específico.
        
        Args:
            timestamp: Tiempo en segundos
            crop_offsets: Lista de offsets calculados
            video_width: Ancho original
            video_height: Alto original
            target_width: Ancho objetivo
            target_height: Alto objetivo
            
        Returns:
            (x, y, width, height) para el crop
        """
        # Calcular dimensiones base del crop
        if video_width / video_height > target_width / target_height:
            crop_width = int(video_height * target_width / target_height)
            crop_height = video_height
        else:
            crop_width = video_width
            crop_height = int(video_width * target_height / target_width)
        
        # Encontrar offset más cercano al timestamp
        if not crop_offsets:
            # Sin tracking - usar centro
            x = (video_width - crop_width) // 2
            y = (video_height - crop_height) // 2
            return (x, y, crop_width, crop_height)
        
        # Interpolar entre offsets
        closest_offset = min(crop_offsets, key=lambda o: abs(o.timestamp - timestamp))
        
        # Calcular posición con offset
        center_x = video_width // 2
        target_center_x = center_x + closest_offset.x_offset
        
        x = target_center_x - crop_width // 2
        y = (video_height - crop_height) // 2 + closest_offset.y_offset
        
        # Asegurar límites
        x = max(0, min(x, video_width - crop_width))
        y = max(0, min(y, video_height - crop_height))
        
        return (x, y, crop_width, crop_height)
