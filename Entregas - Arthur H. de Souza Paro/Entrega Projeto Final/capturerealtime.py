"""Código referente ao trabalho de Laboratório de Inovação e Automação.
O código é feito para abrir uma captura de tela (preguiça de baixar vídeos, mais prático), identificar se uma vaca está deitada, 
e a depender do tempo, tocar um alarme.
Alterações que podem ser feitas: Mudança no tempo do alarme, atualização do dataset para funcionalidade de densidade de população e isolamento
(deve-se usar imagens de drone). """


import cv2
import time
import numpy as np
from ultralytics import YOLO
import mss  # Para captura de tela
import math  # Para calcular a distância Euclidiana
from datetime import datetime  # Para registrar a hora de eventos

# Carregar o modelo YOLOv8 treinado
model = YOLO('C:\\Users\\arthu\\OneDrive\\Documentos\\Python_LIA\\models\\best (1).pt')

# Função para calcular IoU entre duas bounding boxes
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Coordenadas da interseção
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    # Área da interseção
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Áreas das bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Área da união
    union_area = box1_area + box2_area - inter_area

    # IoU
    return inter_area / union_area if union_area > 0 else 0

# Função para calcular a distância Euclidiana entre dois pontos
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Função principal
def process_screen():
    lying_down_timers = {}  # Dicionário para rastrear tempo de vacas deitadas
    previous_boxes = []  # Armazena bounding boxes do quadro anterior
    events = []  # Lista para armazenar eventos (hora de deitar, alarme)

    # Configuração de captura de tela
    sct = mss.mss()
    monitor = sct.monitors[1]  # Captura o monitor principal (substitua se necessário)

    while True:
        # Capturar a tela
        screen = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)  # Converter para formato BGR

        # Detecção com YOLO
        results = model(frame)
        detections = results[0].boxes

        current_boxes = []  # Armazena bounding boxes do quadro atual
        centers = []  # Armazena os centros das bounding boxes

        # Processar as detecções
        for box in detections:
            coords = box.xyxy[0].cpu().numpy()  # Coordenadas da bounding box
            class_id = int(box.cls.cpu().numpy()[0])  # Classe da detecção
            conf = float(box.conf.cpu().numpy()[0])  # Confiança da detecção

            if conf > 0.5:  # Filtrar por confiança
                x1, y1, x2, y2 = map(int, coords)
                current_boxes.append((x1, y1, x2, y2))

                if class_id == 0:  # Classe 'Lying-down'
                    matched = False

                    # Verificar se a bounding box corresponde a uma anterior
                    for prev_box in previous_boxes:
                        if calculate_iou((x1, y1, x2, y2), prev_box) > 0.5:  # Limite de IoU
                            if prev_box in lying_down_timers:
                                lying_down_timers[(x1, y1, x2, y2)] = lying_down_timers.pop(prev_box)
                            matched = True
                            break

                    # Se não houver correspondência, iniciar o timer
                    if not matched:
                        lying_down_timers[(x1, y1, x2, y2)] = time.time()

                    # Calcular o tempo deitado
                    elapsed = time.time() - lying_down_timers.get((x1, y1, x2, y2), time.time())
                    cv2.putText(frame, f"Deitada: {elapsed:.1f}s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Verificar limite de tempo
                    if elapsed > 16:
                        # Hora do alarme
                        alarm_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, "ALERTA: Muito tempo deitada!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"ALERTA: Vaca em ({x1}, {y1}, {x2}, {y2}) deitada por mais de 16 segundos!")

                        # Armazenar o evento (hora de deitar e hora do alarme)
                        lying_down_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        events.append({
                            "vaca": (x1, y1, x2, y2),
                            "hora_deitada": lying_down_time,
                            "hora_alarme": alarm_time,
                            "distancia_para_flock": "N/A"  # Adicionar se necessário
                        })

                # Desenhar a bounding box
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Verde para 'Lying-down'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Calcular o centro da bounding box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                centers.append(center)

        # Se houver mais de uma vaca detectada, calcular o centro do rebanho
        if len(centers) > 1:
            # Calcular o centro médio do rebanho
            avg_center = np.mean(centers, axis=0)

            # Identificar a vaca isolada (maior distância do centro do rebanho)
            max_distance = 0
            isolated_vaca = None
            for center in centers:
                distance = calculate_distance(center, avg_center)
                if distance > max_distance:
                    max_distance = distance
                    isolated_vaca = center

            if isolated_vaca:
                # Calcular a distância da vaca isolada para o centro do rebanho
                distance_to_flock = max_distance
                cv2.putText(frame, f"Distância da vaca isolada: {distance_to_flock:.1f} px", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Atualizar bounding boxes anteriores
        previous_boxes = current_boxes

        # Remover bounding boxes que não estão mais sendo detectadas
        lying_down_timers = {box: lying_down_timers[box] for box in current_boxes if box in lying_down_timers}

        # Exibir a imagem processada
        cv2.imshow('Detecção de Vacas (Gravação de Tela)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Gerar relatório ao final
    generate_report(events)

    cv2.destroyAllWindows()

# Função para gerar o relatório
def generate_report(events):
    if events:
        print("\nRelatório de Vacas Deitadas e Alarmes:")
        for event in events:
            print(f"\nVaca: {event['vaca']}")
            print(f"Hora em que o alarme foi acionado: {event['hora_alarme']}")
            print(f"Distância para o centro do rebanho: {event['distancia_para_flock']}")
    else:
        print("Nenhum evento registrado.")

# Executar captura de tela
process_screen()
