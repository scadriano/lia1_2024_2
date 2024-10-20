import cv2
from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO('models/best_v2FI.pt')

# Abre o vídeo para processamento
video = cv2.VideoCapture('videos/cooking1.mp4')

# Verifica se o vídeo foi aberto corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Obtém informações sobre o vídeo
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define o codec e cria o objeto VideoWriter para salvar o vídeo
output_video = cv2.VideoWriter('videos/predictions/cooking1_predictions.avi',
                               cv2.VideoWriter_fourcc(*'XVID'),
                               fps, (frame_width, frame_height))

while True:
    # Lê um frame do vídeo
    check, img = video.read()
    if not check:  # Se não conseguir ler o frame, interrompe o loop
        print("Não foi possível ler o frame. Finalizando...")
        break

    # Realiza a predição utilizando o modelo YOLO
    results = model.predict(img, verbose=False, save=True)

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

            # Confiança da predição
            conf = round(float(item.conf[0]), 2)
            texto = f'{nomeClasse} - {conf}'

            # Adiciona o texto e bounding box na imagem para predições maiores que 0.6
            if conf >= 0.6:
                if nomeClasse == 'cabelo_solto':
                    cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if conf >= 0.86:
                    # Desenha um retângulo em torno do objeto com base na classe
                    if nomeClasse in ('com_avental', 'com_luva','cabelo_preso'):
                        cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    elif nomeClasse in ('sem_avental', 'sem_luva'):
                        cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)


    # Exibe o frame com as detecções
    cv2.imshow('IMG', img)

    # Escreve o frame no vídeo de saída
    output_video.write(img)

    # Sai do loop se a tecla 'ESC' for pressionada
    if cv2.waitKey(1) == 27:
        break

# Libera o vídeo, o vídeo de saída e fecha as janelas
video.release()
output_video.release()
cv2.destroyAllWindows()