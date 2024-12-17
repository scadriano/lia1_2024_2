from ultralytics import YOLO
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Configuração inicial
door_line = 250  # Linha vertical representando a "porta"
tracked_ids = {}  # Armazena informações de cada ID (posição e status)
current_count = 0  # Contador de pessoas dentro da sala

# Carregar o modelo YOLOv8
model = YOLO("C:\\Users\\arthu\\OneDrive\\Documentos\\LIA_Last\\models\\bestColab.pt")

def detect_people(frame):
    """
    Detecta pessoas no frame usando YOLOv8.
    Retorna uma lista de detecções no formato [x1, y1, x2, y2].
    """
    results = model(frame)  # Realiza a inferência no frame
    detections = []  # Lista para armazenar as detecções de pessoas

    for result in results:  # Itera pelos resultados retornados pelo YOLO
        if result.boxes is not None:  # Verifica se existem bounding boxes
            for box in result.boxes:  # Para cada bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordenadas da bounding box
                conf = box.conf[0].item()  # Confiança da detecção
                cls = int(box.cls[0].item())  # Classe da detecção (0 para "person")

                if cls == 0 and conf > 0.5:  # Apenas considera pessoas com confiança > 50%
                    detections.append([x1, y1, x2, y2])  # Adiciona a bounding box à lista

    return detections  # Retorna as detecções de pessoas

def update_count(detections, door_line):
    """
    Atualiza a contagem de pessoas dentro da sala com base em suas detecções.
    """
    global tracked_ids, current_count  # Usa variáveis globais para rastrear IDs e contagem

    for detection in detections:  # Itera pelas detecções no frame
        x1, y1, x2, y2 = detection  # Coordenadas da bounding box
        center_x = (x1 + x2) // 2  # Calcula o centro horizontal da bounding box
        center_y = (y1 + y2) // 2  # Calcula o centro vertical da bounding box

        # Verifica se a pessoa já está sendo rastreada
        person_id = None  # Inicializa o ID como não encontrado
        for tracked_id, (prev_center, status) in tracked_ids.items():
            prev_x, prev_y = prev_center  # Posição anterior da pessoa
            if abs(center_x - prev_x) < 50 and abs(center_y - prev_y) < 50:
                # Se a pessoa está próxima da posição anterior, considera a mesma pessoa
                person_id = tracked_id
                break

        if person_id is None:  # Se a pessoa não foi encontrada, atribui um novo ID
            person_id = len(tracked_ids) + 1  # Novo ID é o próximo disponível
            tracked_ids[person_id] = [(center_x, center_y), None]  # Armazena a nova posição e sem status

        # Verifica o cruzamento da linha da porta
        prev_center, last_status = tracked_ids[person_id]  # Posição e status anteriores
        if last_status != "entry" and center_x > door_line > prev_center[0]:
            # Pessoa entrou (cruzou da esquerda para a direita)
            current_count += 1  # Incrementa o contador de pessoas na sala
            tracked_ids[person_id] = [(center_x, center_y), "entry"]  # Atualiza o status para "entry"
            print(f"Pessoa ID {person_id} entrou. Contagem atual: {current_count}")
        elif last_status != "exit" and center_x < door_line < prev_center[0]:
            # Pessoa saiu (cruzou da direita para a esquerda)
            current_count -= 1  # Decrementa o contador de pessoas na sala
            tracked_ids[person_id] = [(center_x, center_y), "exit"]  # Atualiza o status para "exit"
            print(f"Pessoa ID {person_id} saiu. Contagem atual: {current_count}")

        # Atualiza a posição atual da pessoa
        tracked_ids[person_id] = [(center_x, center_y), tracked_ids[person_id][1]]

    return current_count  # Retorna a contagem atualizada

# Processamento do vídeo
cap = cv2.VideoCapture("videoentradasaida5.mp4")  # Carrega o vídeo
if not cap.isOpened():  # Verifica se o vídeo foi aberto com sucesso
    print("Erro ao abrir o vídeo.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)  # Obtém os FPS do vídeo
logs = []  # Lista para armazenar os logs de contagem e timestamps

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))


def generate_detailed_report(logs, output_path="detailed_report.csv"):
    """
    Gera um relatório detalhado com timestamps e quantidade de pessoas na sala.
    Inclui o momento em que houve a maior quantidade de pessoas.
    """
    # Encontra o maior número de pessoas e o timestamp correspondente
    max_count = max(logs, key=lambda x: x[1])  # Log com a maior quantidade de pessoas
    max_timestamp, max_people = max_count  # Timestamp e quantidade máxima de pessoas

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp (s)", "Quantidade de Pessoas na Sala"])  # Cabeçalho do relatório
        writer.writerows(logs)  # Escreve os logs

        # Adiciona uma linha extra com o momento de maior ocupação
        writer.writerow([])
        writer.writerow(["Momento de Maior Ocupacao;"])
        writer.writerow(["Timestamp (s):"])
        writer.writerow([max_timestamp])
        writer.writerow(["Maior quantidade de pessoas registradas:"])
        writer.writerow([max_people])

    print(f"Relatório detalhado salvo em {output_path}")
    print(f"Maior quantidade de pessoas: {max_people} no timestamp: {max_timestamp} s")


def generate_plot(logs, output_path="occupancy_plot.png"):
    """
    Gera um gráfico mostrando a variação da quantidade de pessoas na sala.
    """
    timestamps = [log[0] for log in logs]  # Extrai os timestamps dos logs
    counts = [log[1] for log in logs]  # Extrai as contagens dos logs

    plt.figure(figsize=(10, 6))  # Configura o tamanho do gráfico
    plt.plot(timestamps, counts, marker="o", linestyle="-", color="b", label="Quantidade de Pessoas")
    plt.title("Quantidade de Pessoas na Sala ao Longo do Tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Quantidade de Pessoas")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)  # Salva o gráfico em um arquivo
    plt.close()
    print(f"Gráfico salvo em {output_path}")

while True:
    ret, frame = cap.read()  # Lê um frame do vídeo
    if not ret:  # Se não houver mais frames, encerra o loop
        break

    # Detecta pessoas no frame
    detections = detect_people(frame)

    # Atualiza a contagem de pessoas na sala
    current_count = update_count(detections, door_line)

    # Calcula o timestamp com base no frame atual
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Obtém o número do frame atual
    timestamp = frame_number / fps  # Converte o frame para tempo em segundos

    # Salva o log para o relatório detalhado
    logs.append((timestamp, current_count))

    # Exibição gráfica no vídeo
    frame = cv2.line(frame, (door_line, 0), (door_line, frame.shape[0]), (0, 255, 0), 2)  # Desenha a linha da porta
    for detection in detections:
        x1, y1, x2, y2 = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Desenha a bounding box

    cv2.putText(frame, f"Pessoas na Sala: {current_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Contagem de Pessoas", frame)  # Exibe o frame processado

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Encerra se a tecla 'q' for pressionada
        break

cap.release()  # Libera o vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas

# Gera relatório detalhado e gráfico
generate_detailed_report(logs)  # Cria o CSV com os dados
generate_plot(logs)  # Cria o gráfico com os dados
