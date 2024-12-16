import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('model/yolo11n-cls.pt')

# Ler a imagem de entrada
image = cv2.imread('Entregas - Diogo de Paula Quintão\Construindo Projeto de Classificação\image\img03.png')

# Realizar a predição na imagem
result = model.predict(image, verbose=False)

# Mostrar a imagem com as detecções
for obj in result:
    obj.show()