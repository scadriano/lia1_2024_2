import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('models/modeloBase.pt')

# Ler o vídeo de entrada
video = cv2.VideoCapture("videos/videoteste5.mp4")

# Dicionário de códigos de cores para as classes
custom_codes = {
    0: '#3D2C1D',  # Black
    1: '#7F4B28',  # Brown
    2: '#5C4033',  # Dark Brown
    3: '#F5D1C2',  # Fair
    4: '#F8E0D2',  # Light
    5: '#D6A692',  # Medium
    6: '#8E6E5C',  # Olive
    7: '#AE7C5A',  # Tan
    8: '#422C24',  # Very Dark Brown
    9: '#FDE9D9'   # Very Light
}

# Dicionário de nomes das classes do modelo
class_names = model.names

while True:
    check, img = video.read()
    
    cv2.namedWindow('Detecção de Objetos', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detecção de Objetos', img.shape[1], img.shape[0])

    # Realizar a predição na imagem
    results = model.predict(img, verbose=False)

    # Extrair detecções e desenhar na imagem
    for result in results:
        for box in result.boxes:
            # Obter coordenadas, confiança e classe
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertendo coordenadas para inteiros
            confidence = box.conf[0]
            cls = int(box.cls[0])
            
            # Obter o código de cor da classe
            class_code = custom_codes.get(cls, 'Desconhecido')
            
            # Desenhar o retângulo na imagem (Bounding box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Adicionar o código de cor e a confiança
            cv2.putText(img, f'{class_code} ({confidence:.2f})', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o vídeo com as detecções
    cv2.imshow('Detecção de Objetos', img)

    # Pressionar 'Esc' para sair
    if cv2.waitKey(1) == 27:
        break

# Liberar a captura e destruir todas as janelas SEMPRE TER ISSO NO CÓDIGO
video.release()
cv2.destroyAllWindows()
