import subprocess
import cv2
import numpy as np
import time
from ultralytics import YOLO
import subprocess

# Mata el proceso PTPCamera si está en ejecución (macOS)
subprocess.run(['killall', 'PTPCamera'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 1) Carga del modelo entrenado
model = YOLO('runs/detect/train/weights/best.pt')  # asegúrate de que yolov8s.pt esté en el mismo directorio

# 2) Arranca gphoto2 en modo Live View (MJPEG)
proc = subprocess.Popen(
    ['gphoto2', '--capture-movie', '--stdout'],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)

data = b''  # buffer para juntar bytes MJPEG

try:
    while True:
        # 3) Lee un trozo del flujo
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        data += chunk

        # 4) Busca final de un JPEG (FFD9)
        if b'\xff\xd9' in data:
            frame_data, data = data.split(b'\xff\xd9', 1)
            frame_data += b'\xff\xd9'

            # 5) Decodifica el frame con OpenCV
            img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 6) Inferencia con YOLOv8
            results = model(img)[0]
            # Para detección: revisa las clases de los bounding boxes
            cls_ids = [int(c) for c in results.boxes.cls] if len(results.boxes) else []

            # 7) Si detecta la clase 1 ("uno_solo"), dispara
            if any(c == 1 for c in cls_ids):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                print(f"[!] Fotograma único detectado – disparando y guardando {filename}")
                subprocess.run([
                    'gphoto2',
                    '--capture-image-and-download',
                    '--set-config', 'liveviewsize=0',
                    '--filename', filename,
                    '--force-overwrite'
                ], stdout=subprocess.DEVNULL)
                # pequeña pausa para no disparar varias veces en el mismo fotograma
                time.sleep(2)

            # 8) Muestra el Live View con la detección
            cv2.imshow('LiveView', img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break

finally:
    # 9) Limpieza
    proc.terminate()
    cv2.destroyAllWindows()