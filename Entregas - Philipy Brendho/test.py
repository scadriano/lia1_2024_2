import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from IPython.display import Image

# Carregar o modelo treinado
classifier = load_model('modelo_gatos_cachorros.h5')

# Carregar a imagem de teste
image_path = r'C:\\Python\\venvs\\test1\\test1\\312.jpg'
test_image = image.load_img(image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Fazer a previsão usando o modelo carregado
result = classifier.predict(test_image)

# Como seu conjunto de treino tinha duas classes, associamos o índice com as classes
if result[0][0] >= 0.5:
    prediction = 'Cachorro.'
    accuracy = round(result[0][0] * 100, 2)
else:
    prediction = 'Gato.'
    accuracy = round((1 - result[0][0]) * 100, 2)

# Exibir a previsão e a acurácia
print("Previsão:", prediction)
print("Acurácia:", accuracy, "%.")
