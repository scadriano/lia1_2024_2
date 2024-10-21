import cv2
from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO('model/PRSbest_v1_2110.pt')

# Abre o vídeo para processamento
video = cv2.VideoCapture('video/PRSvideo - Trim.mp4')

# Verifica se o vídeo foi aberto corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Obtém informações sobre o vídeo
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define o codec e cria o objeto VideoWriter para salvar o vídeo
output_video = cv2.VideoWriter('video/saved_predictions.avi',
                               cv2.VideoWriter_fourcc(*'XVID'),
                               fps, (frame_width, frame_height))

# Reduz a resolução (opcional)
new_width = int(frame_width / 2)
new_height = int(frame_height / 2)

while True:
    # Lê um frame do vídeo
    check, img = video.read()
    if not check:  # Se não conseguir ler o frame, interrompe o loop
        print("Não foi possível ler o frame. Finalizando...")
        break

    # Redimensiona o frame
    img = cv2.resize(img, (new_width, new_height))

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

            # Define a cor do retângulo e do texto com base na classe
            if nomeClasse == 'Papel':
                cor = (0, 255, 0)  # Verde
            elif nomeClasse == 'Pedra':
                cor = (255, 0, 0)  # Azul
            elif nomeClasse == 'Tesoura':
                cor = (0, 0, 255)  # Vermelho
            else:
                cor = (255, 255, 255)  # Branco para classes desconhecidas

            # Adiciona o texto na imagem
            cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, cor, 2)

            # Desenha um retângulo em torno do objeto com a cor definida
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)

    # Exibe o frame com as detecções
    cv2.imshow('IMG', img)

    # Escreve o frame no vídeo de saída
    output_video.write(img)

    # Sai do loop se a tecla 'ESC' for pressionada
    if cv2.waitKey(1) == 27:  # Ajuste o valor se necessário
        break

# Libera o vídeo, o vídeo de saída e fecha as janelas
video.release()
output_video.release()
cv2.destroyAllWindows()
