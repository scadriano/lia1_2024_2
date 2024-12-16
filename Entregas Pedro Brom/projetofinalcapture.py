# Capturar imagens de uma webcam e salvar na pasta: images
import cv2

video = cv2.VideoCapture(0)

name = input("Digite o nome do objeto: ")
id = 0
#print("Pressione 's' para salvar uma imagem ou 'q' para sair.")

while True:
    check,img = video.read()
    
    cv2.imshow('Capturing Images',img)
    
    if cv2.waitKey(1) & 0xff == ord('s'):
        cv2.imwrite(f'images/{name} {id}.jpg',img)
        id+=1
        print(id)

    if id == 40:
        print("Captura concluída!")
        break

#Fechar janelas e liberar a câmera
video.release()
cv2.destroyAllWindows()
cinto