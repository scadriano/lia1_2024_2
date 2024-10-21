import cv2
from ultralytics import YOLO

video_source = "file"

# Carregar o modelo YOLO
model = YOLO('C:/Python/venvs/models/env_yolov11/Cards_best_V2_21.10.24.pt')

# Ler a webcam
if video_source == "webcam":
    video = cv2.VideoCapture(2)

# Ler o arquivo de vídeo 
if video_source == "file":
    video = cv2.VideoCapture(r'C:\Python\venvs\videos\env_yolov11\20241021_120126_1080p.mp4')

# Verifica se o vídeo foi aberto corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

if video_source == "file":
    # Obtém informações sobre o vídeo
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define o codec e cria o objeto VideoWriter para salvar o vídeo
    output_video = cv2.VideoWriter(r'C:\Python\venvs\videos\env_yolov11\cartas_1080p.avi',
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   fps, (frame_width, frame_height))

# Dicionário de nomes das classes
class_names = model.names

while True:
    check, img = video.read()
    if not check:  # Se não conseguir ler o frame, interrompe o loop
        print("Não foi possível ler o frame. Finalizando...")
        break
    
    # Realizar a predição na imagem
    results = model.predict(img, verbose=False)

    # Extrair detecções e desenhar na imagem
    for result in results:
        for box in result.boxes:
            # Obter coordenadas, confiança e classe
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertendo coordenadas para inteiros
            confidence = box.conf[0]
            cls = int(box.cls[0])
            
             # Desenhar a bounding box apenas se a confiança for maior que 0.5
            if confidence >= 0.5:
                # Obter o nome da classe
                class_name = class_names[cls] if cls in class_names else 'Desconhecido'
                
                # Desenhar o retângulo na imagem
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # O último valor é a espessura
                # Adicionar o nome da classe e a confiança
                cv2.putText(img, f'{class_name} ({confidence:.2f})', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # A cor é em BGR, não RGB.

    # Redimensionar a imagem para 1280x720
    img_resized = cv2.resize(img, (1280, 720))

    # Mostrar o vídeo com as detecções
    cv2.imshow('Detectando Cartas', img_resized)

    # Escreve o frame no vídeo de saída
    output_video.write(img)

    # Pressionar 'Esc' para sair
    if cv2.waitKey(1) == 27:
        break

# Liberar a captura e destruir todas as janelas
video.release()

# Se um arquivo de vídeo foi aberto, liberar o VideoWriter
if video_source == "file":
    output_video.release()

cv2.destroyAllWindows()
# PRECISA DESSE FINAL PARA LIBERAR MEMÓRIA