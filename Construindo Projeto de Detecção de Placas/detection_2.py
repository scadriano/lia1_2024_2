import os
import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO(r'Construindo Projeto de Detecção de Placas/models/best.pt')

# Abre o vídeo para processamento
video = cv2.VideoCapture(r'Construindo Projeto de Detecção de Placas/video/cars.mp4')

# Verifica se o vídeo foi aberto corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Obtém informações sobre o vídeo
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Certifique-se de que o diretório de saída existe
output_dir = r'Construindo Projeto de Detecção de Placas/video'
os.makedirs(output_dir, exist_ok=True)

# Define o codec e cria o objeto VideoWriter para salvar o vídeo
output_path = os.path.join(output_dir, 'saved_predictions.avi')
output_video = cv2.VideoWriter(output_path,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               fps, (frame_width, frame_height))
# Verifica se o VideoWriter foi aberto corretamente
if not output_video.isOpened():
    print(f"Erro ao abrir o arquivo de saída de vídeo: {output_path}")
    exit()

while True:
    # Lê um frame do vídeo
    check, img = video.read()
    if not check:  # Se não conseguir ler o frame, interrompe o loop
        print("Não foi possível ler o frame. Finalizando...")
        break

    # Realiza a predição utilizando o modelo YOLO
    results = model.predict(img, verbose=False, save=False)

    # Itera sobre os resultados de detecção
    for obj in results:
        nomes = obj.names  # Obtém os nomes das classes

        # Para cada bounding box detectada
        for item in obj.boxes:
            # Converte as coordenadas para inteiros
            x1, y1, x2, y2 = item.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Identifica a classe do objeto
            cls = int(item.cls[0])
            nomeClasse = nomes[cls]

            # Desenha a bounding box e o nome da classe na imagem
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, nomeClasse, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Escreve o frame processado no vídeo de saída
    output_video.write(img)

    # Exibir o frame processado
    cv2.imshow('Detecção de Objetos', img)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video.release()
output_video.release()
cv2.destroyAllWindows()