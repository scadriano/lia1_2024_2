import cv2
from ultralytics import YOLO

# Carregar modelo YOLO
model = YOLO('models/bestcinto10.pt')

# Capturar vídeo
video = cv2.VideoCapture('C:/Users/ADMIN/AppData/Local/Programs/Python/Python310/projetopedrobrom/envyolov11/videoprofessor.mp4')

# Configurações para salvar o vídeo (compatível com WhatsApp)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_resolution = (1280, 720)  # Resolução máxima para WhatsApp
fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
output_file = 'output_video.mp4'

# Redimensionar para resolução máxima, se necessário
if frame_width > max_resolution[0] or frame_height > max_resolution[1]:
    frame_width, frame_height = max_resolution

# Configuração do VideoWriter com codec H.264
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec H.264
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Fim do vídeo ou erro ao ler o quadro.")
        break

    # Redimensionar o quadro para a resolução máxima (opcional)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Realizar predições
    results = model.predict(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Coordenadas da bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])         # Classe predita
            conf = round(float(box.conf[0]), 2)  # Confiança
            class_name = model.names[cls] # Nome da classe

            # Adicionar texto e retângulo no quadro
            text = f'{class_name} - {conf}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # Escrever o frame processado no vídeo
    out.write(frame)

    # Mostrar o quadro
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:  # Tecla ESC para sair
        break

# Liberar recursos
video.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo salvo como: {output_file}")
