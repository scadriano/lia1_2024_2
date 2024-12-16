import os
import time
import logging
from telegram_bot import (
    send_telegram, 
    send_telegram_photo, 
    send_alert, 
    get_latest_messages,  
    handle_gpt_conversation
)
from gpt4omini import get_gpt4o_mini_response
import cv2
from ultralytics import YOLO

# Variáveis de ambiente
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallDetector:
    def __init__(self):  # Corrigido de _init_ para __init__
        self.model = YOLO('models/yolo11n-pose.pt')
        self.fall_history = [False] * 5
        self.fall_alert_sent = False
        self.last_alert_time = 0
        self.last_chat_check = 0
        self.chat_check_interval = 5
        self.last_update_id = None
        self.skeleton = [
            [15,13], [13,11], [16,14], [14,12], [11,12],
            [5,11], [6,12], [5,6], [5,7], [6,8], [7,9],
            [8,10], [1,2], [0,1], [0,2], [1,3], [2,4],
            [3,5], [4,6]
        ]

    def draw_skeleton(self, frame, keypoints):
        for kp in keypoints:
            # Converter keypoints para formato correto
            kp = kp.reshape(-1, 3)
            
            # Desenhar pontos
            for x, y, conf in kp:
                if conf > 0.5:  # Filtrar por confiança
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            
            # Desenhar linhas do esqueleto
            for p1, p2 in self.skeleton:
                if kp[p1][2] > 0.5 and kp[p2][2] > 0.5:
                    pt1 = (int(kp[p1][0]), int(kp[p1][1]))
                    pt2 = (int(kp[p2][0]), int(kp[p2][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    def detect_fall(self, frame):
        results = self.model.predict(frame, conf=0.3)
        fall_detected = False

        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Desenhar esqueleto
                self.draw_skeleton(frame, result.keypoints.data)
                
                boxes = result.boxes
                if len(boxes) > 0:
                    # Obter keypoints da pessoa
                    kp = result.keypoints.data[0].reshape(-1, 3)
                    
                    # Cálculos para detecção de queda
                    # Distância vertical (quadril-ombro)
                    hip_shoulder_height = abs(kp[11][1] - kp[5][1])
                    # Distância horizontal (quadril-quadril)
                    hip_width = abs(kp[11][0] - kp[12][0])
                    
                    # Razão altura/largura (vertical = >1.5, horizontal = <1.0)
                    ratio = hip_shoulder_height / (hip_width + 1e-6)
                    
                    # Atualizar histórico
                    self.fall_history.pop(0)
                    self.fall_history.append(ratio < 1.0)
                    
                    # Detectar queda se pessoa estiver horizontal por N frames
                    if sum(self.fall_history) >= 3:  # 3 de 5 frames mostram posição horizontal
                        fall_detected = True
                        
                    # Debug info
                    cv2.putText(frame, f'Ratio: {ratio:.2f}', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Status na tela
                status_text = 'QUEDA DETECTADA!' if fall_detected else 'Monitorando...'
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 255) if fall_detected else (0, 255, 0), 2)

        return frame, fall_detected

    def check_chat(self):
        current_time = time.time()
        if current_time - self.last_chat_check >= self.chat_check_interval:
            try:
                messages = get_latest_messages(self.last_update_id)
                if messages:
                    for update_id, chat_id, text in messages:
                        self.last_update_id = update_id
                        if text:
                            response = get_gpt4o_mini_response(text)
                            if response:
                                send_telegram(response)
            except Exception as e:
                logger.error(f"Erro no chat: {e}")
            self.last_chat_check = current_time

def main():
    detector = FallDetector()
    cap = cv2.VideoCapture('video/vd01.mp4')

    if not cap.isOpened():
        logger.error("Erro ao abrir vídeo")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame, fall_detected = detector.detect_fall(frame)
            
            # Verificar chat periodicamente
            detector.check_chat()

            # Processar queda detectada
            if fall_detected and not detector.fall_alert_sent:
                try:
                    cv2.imwrite('img/queda_detectada.jpg', frame)
                    send_telegram_photo('img/queda_detectada.jpg')
                    send_alert()
                    detector.fall_alert_sent = True
                    detector.last_alert_time = time.time()
                except Exception as e:
                    logger.error(f"Erro ao processar queda: {e}")

            cv2.imshow('Detector de Quedas', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Erro na execução: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()