import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carregar o modelo salvo no formato Keras
classifier = load_model('modelo_classificador.keras')

# Carregar class_indices do arquivo JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Carregando a imagem de teste
test_image = image.load_img('C:/Users/vitor/Desktop/Nova pasta/test1/7.JPG', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Fazendo a previsão usando o modelo classifier
result = classifier.predict(test_image)

# Corrigir a comparação usando threshold 0.5
if result[0][0] > 0.5:
    prediction = 'Cachorro'
    accuracy = round(result[0][0] * 100, 2)
else:
    prediction = 'Gato'
    accuracy = round((1 - result[0][0]) * 100, 2)

# Exibindo a previsão e a acurácia
print("Previsão:", prediction)
print("Acurácia:", accuracy, "%")

# Exibindo a imagem
img = mpimg.imread('C:/Users/vitor/Desktop/Nova pasta/test1/7.JPG')
plt.imshow(img)
plt.title(f'Previsão: {prediction}, Acurácia: {accuracy}%')
plt.axis('off')  # Ocultar os eixos
plt.show()
