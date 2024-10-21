import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('model/Libras_best.pt')


# Ler a webcam
video = cv2.VideoCapture(0)

# Dicionário de nomes das classes
class_names = model.names

while True:
    check, img = video.read()
    
    # Realizar a predição na imagem
    results = model.predict(img, verbose=False)

    # Extrair detecções e desenhar na imagem
    for result in results:
        for box in result.boxes:
            # Obter coordenadas, confiança e classe
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertendo coordenadas para inteiros
            confidence = box.conf[0]
            cls = int(box.cls[0])
            
            # Obter o nome da classe
            class_name = class_names[cls] if cls in class_names else 'Desconhecido'
            
            # Desenhar o retângulo na imagem(Bounding box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Adicionar o nome da classe e a confiança
            cv2.putText(img, f'{class_name} ({confidence:.2f})', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o vídeo com as detecções
    cv2.imshow('Detecção de Objetos', img)

    # Pressionar 'Esc' para sair
    if cv2.waitKey(1) == 27:
        break

# Liberar a captura e destruir todas as janelas SEMPRE TER ISSO NO CODIGO
video.release()
cv2.destroyAllWindows()