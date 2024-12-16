import cv2
from collections import deque
from ultralytics import YOLO

# Configurações iniciais
video_source = "webcam"             # "webcam" ou "file"
width, height = 1920, 1080          # Define a resolução da câmera usada

# Tamanho do Buffer, maior signific mais precisão, mas com mais delay
buffer_size = 12

# Localização do vídeo se video_source for "file"
video_input = r'C:\Python\venvs\videos\env_yolov11\20241021_120126_1080p.mp4'
video_output = r'C:\Python\venvs\videos\env_yolov11\cartas_1080p.avi'

# Carregar o modelo YOLO
model = YOLO('C:/Python/venvs/models/env_yolov11/Cards_best_V4_v11l_30-10-24.pt')

# Hierarquia de poder das cartas
power_groups = [
    ['4P'],                      # Maior poder
    ['7C'],                      
    ['AE'],                      
    ['7O'],                      
    ['3P', '3O', '3C', '3E'],    # Mesmo poder para todas as "3"
    ['2P', '2O', '2C', '2E'],    # Mesmo poder para todas as "2"
    ['AP', 'AO', 'AC'],          # Mesmo poder para todos os "A" (exceto "AE")
    ['KP', 'KO', 'KC', 'KE'],    # Rei
    ['QP', 'QO', 'QC', 'QE'],    # Dama
    ['JP', 'JO', 'JC', 'JE'],    # Valete
]

# Criar o dicionário de poderes
power_dict = {}
for idx, group in enumerate(power_groups):
    for card in group:
        power_dict[card] = idx

# Inicializar contadores de pontos
left_score = 0
right_score = 0

# Ler a webcam ou o arquivo de vídeo
if video_source == "webcam":
    video = cv2.VideoCapture(2)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
elif video_source == "file":
    video = cv2.VideoCapture(video_input)

if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

if video_source == "file":
    output_video = cv2.VideoWriter(video_output,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   fps, (frame_width, frame_height))

class_names = model.names
cv2.namedWindow('Detectando Cartas', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Detectando Cartas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Buffers para calcular média
left_buffer = deque(maxlen=buffer_size)
right_buffer = deque(maxlen=buffer_size)

while True:
    check, img = video.read()
    if not check:
        print("Não foi possível ler o frame. Finalizando...")
        break

    results = model.predict(img, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            if confidence >= 0.4:
                class_name = class_names[cls] if cls < len(class_names) else 'Desconhecido'
                detections.append((class_name, x1, y1, x2, y2))

    # Gerenciar duplicatas e hierarquia de poder
    unique_detections = {}
    for class_name, x1, y1, x2, y2 in detections:
        if class_name not in unique_detections:
            unique_detections[class_name] = [(x1, y1, x2, y2)]
        else:
            unique_detections[class_name].append((x1, y1, x2, y2))

    # Determinar a carta mais forte
    min_power = float('inf')  # Menor índice no dicionário é a carta mais forte
    strongest_cards = []
    left_has_strongest = False
    right_has_strongest = False

    for class_name in unique_detections:
        if class_name in power_dict:
            card_power = power_dict[class_name]
            if card_power < min_power:
                min_power = card_power
                strongest_cards = [class_name]
            elif card_power == min_power:
                strongest_cards.append(class_name)

    # Variáveis do buffer
    left_current = False
    right_current = False

    # Desenhar as bounding boxes
    for class_name, boxes in unique_detections.items():
        if class_name not in power_dict:
            color = (128, 128, 128)  # Cinza para cartas fora da hierarquia
        else:
            if class_name in strongest_cards:
                # Checar se há empate
                if len(strongest_cards) > 1:
                    color = (0, 255, 255)  # Amarelo para cartas empatadas
                else:
                    color = (0, 255, 0)  # Verde para a carta mais forte
            else:
                color = (0, 0, 192)  # Vermelho para cartas mais fracas

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Determinar se as cartas mais fortes estão à esquerda ou à direita
            center_x = (x1 + x2) // 2
            if class_name in strongest_cards:
                if center_x < frame_width // 2:
                    left_has_strongest = True
                    left_current = True
                else:
                    right_has_strongest = True
                    right_current = True

    # Verificar empate
    if len(strongest_cards) > 1 and left_has_strongest and right_has_strongest:
        left_has_strongest = False
        right_has_strongest = False

    left_buffer.append(left_current)
    right_buffer.append(right_current)

    left_strongest_avg = sum(left_buffer) / len(left_buffer)
    right_strongest_avg = sum(right_buffer) / len(right_buffer)

    if left_strongest_avg > 0.5 and right_strongest_avg > 0.5:
        line_color_left = line_color_right = (0, 255, 255)  # Empate
    elif left_strongest_avg > 0.5:
        line_color_left = (0, 255, 0)  # Verde
        line_color_right = (0, 0, 255)  # Vermelho
    elif right_strongest_avg > 0.5:
        line_color_left = (0, 0, 255)  # Vermelho
        line_color_right = (0, 255, 0)  # Verde
    else:
        line_color_left = line_color_right = (0, 0, 255)  # Nenhuma carta forte

    cv2.line(img, (0, 0), (0, frame_height), line_color_left, 5)
    cv2.line(img, (frame_width - 1, 0), (frame_width - 1, frame_height), line_color_right, 5)

    # Desenhar linha separadora e pontuações
    cv2.line(img, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 255, 255), 2)
    cv2.putText(img, f'Esquerda: {left_score}', (50, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Direita: {right_score}', (frame_width - 300, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Exibir vídeo
    cv2.imshow('Detectando Cartas', img)

    if video_source == "file":
        output_video.write(img)

    # Ler teclado
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32 or key == ord('1'):  # Espaço
        if left_has_strongest:
            left_score += 1
        elif right_has_strongest:
            right_score += 1
    elif key == ord('3'):
        if left_has_strongest:
            left_score += 3
        elif right_has_strongest:
            right_score += 3
    elif key == ord('6'):
        if left_has_strongest:
            left_score += 6
        elif right_has_strongest:
            right_score += 6
    elif key == ord('9'):
        if left_has_strongest:
            left_score += 9
        elif right_has_strongest:
            right_score += 9
    elif key == ord('a'):
        right_score += 1
    elif key == ord('e'):
        left_score += 1
    elif key == ord('q'):
        right_score -= 1
    elif key == ord('e'):
        right_score -= 1
    elif key == ord('r'):
        left_score = 0
        right_score = 0

video.release()
if video_source == "file":
    output_video.release()
cv2.destroyAllWindows()
