import cv2
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO('C:/Users/quint/Projetos/lia1_2024_2/Entregas - Diogo de Paula Quintão/Construindo Projeto de Detecção de Placas/models/best.pt')

# Ler a imagem de entrada para detecção
image = cv2.imread('C:/Users/quint/Projetos/lia1_2024_2/Entregas - Diogo de Paula Quintão/Construindo Projeto de Detecção de Placas/image/car3.jpeg')

# Realizar a predição na imagem 
result = model.predict(image, verbose=False)

# Mostrar a imagem com as detecções  
for obj in result:
    obj.show()