import os
import cv2
import pytesseract
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np  # Importação de numpy, necessário para a conversão de imagem

# Configurações do pytesseract (ajuste o caminho, se necessário)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Caminhos
folder_path = r'C:\Users\Philpy\Desktop\Projects\Python_IA\venvs\cod'
excel_path = r'C:\Users\Philpy\Desktop\Projects\Python_IA\venvs\cod.xlsx'

# Configurações gerais
save_dir = 'save_videos'
os.makedirs(save_dir, exist_ok=True)
output_name = input("Digite o nome do arquivo de saída (sem extensão): ")
output_file = os.path.join(save_dir, f"{output_name}.mp4")

# Configuração do vídeo
video = cv2.VideoCapture(0)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Modelo YOLO
model_objetos = YOLO('models/melhor2.pt')

# Variáveis para contagem de itens
itens_enviados = 0
itens_coletados = 0

# Variáveis globais
classe_identificada = None
regiao_selecionada = None
class_name = None  # Definindo class_name para evitar erro

# Função para listar os arquivos da pasta
def list_files_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

# Função para o usuário escolher um arquivo
def select_file_from_folder(folder_path):
    files = list_files_in_folder(folder_path)
    if not files:
        print("Nenhum arquivo encontrado na pasta.")
        return None

    print("Arquivos disponíveis:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}: {file}")

    while True:
        try:
            choice = int(input("Digite o número do arquivo que deseja selecionar: "))
            if 1 <= choice <= len(files):
                return os.path.join(folder_path, files[choice - 1])
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número.")

# Função para processar a imagem e extrair texto
def extract_code_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
    return text

# Função para verificar o código na planilha
def check_code_in_excel(code, excel_path):
    try:
        # Carrega a planilha
        df = pd.read_excel(excel_path)

        # Verifica se o código está na coluna CODT
        if 'CODT' not in df.columns:
            print("A coluna 'CODT' não foi encontrada na planilha.")
            return False

        # Remove espaços em branco para comparação
        df['CODT'] = df['CODT'].astype(str).str.strip()
        return code in df['CODT'].values, df
    except Exception as e:
        print(f"Erro ao carregar a planilha: {e}")
        return False, None

# Função para adicionar o código na planilha
import openpyxl

# Adiciona código e classe identificada à planilha
def add_code_to_excel(extracted_text, excel_path):
    global classe_identificada  # Classe identificada anteriormente na etapa1

    # Carregar ou criar o arquivo Excel
    try:
        workbook = openpyxl.load_workbook(excel_path)
        sheet = workbook.active
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        # Cria os cabeçalhos caso o arquivo não exista
        sheet.append(["Código", "Classe Identificada"])
    
    # Adicionar o código extraído e a classe identificada
    sheet.append([extracted_text, classe_identificada])
    workbook.save(excel_path)
    print(f"Código '{extracted_text}' e classe '{classe_identificada}' adicionados à planilha!")



# Função para desenhar texto
def draw_text_with_pillow(img, text, position, font_size=32, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "arial.ttf"  # Certifique-se de que o arquivo está no caminho correto
    font = ImageFont.truetype(font_path, font_size)

    # Tamanho máximo permitido para o texto
    max_width = pil_img.width - position[0]

    # Quebra o texto em várias linhas, se necessário
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        line_width = draw.textbbox((0, 0), test_line, font=font)[2]

        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Desenha as linhas na imagem
    y_offset = position[1]
    for line in lines:
        draw.text((position[0], y_offset), line, font=font, fill=color)
        y_offset += font_size + 5  # Incrementa a posição vertical para a próxima linha

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Etapa 1: Identificação de objetos
def etapa1():
    global classe_identificada, class_name

    while True:
        check, img = video.read()
        if not check:
            print("Erro ao capturar vídeo")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        contador_texto = f"Enviados: {itens_enviados} | Coletados: {itens_coletados}"
        img = draw_text_with_pillow(img, contador_texto, (10, 10), font_size=24, color=(0, 255, 0))
        mensagem = "Identificando objeto..."
        results_objetos = model_objetos.predict(img, verbose=False)

        for result in results_objetos:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = model_objetos.names[cls]  # Atribuindo a class_name aqui
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                mensagem = f"Objeto detectado: {class_name}. Pressione ESPAÇO para confirmar."
                classe_identificada = class_name

        img = draw_text_with_pillow(img, mensagem, (10, frame_height // 2), font_size=32, color=(0, 0, 255))
        cv2.imshow('Sistema de Detecção', img)
        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            etapa2()
            break
        
# Etapa 2: Verificar código na imagem
def etapa2():
    global regiao_selecionada

    arquivo_selecionado = select_file_from_folder(folder_path)
    if arquivo_selecionado:
        print(f"Arquivo selecionado: {arquivo_selecionado}")
        extracted_text = extract_code_from_image(arquivo_selecionado)
        print(f"Texto extraído: {extracted_text}")

        is_code_found, _ = check_code_in_excel(extracted_text, excel_path)
        if is_code_found:
            mensagem = "Código encontrado na planilha! Enviar para a região de Envio."
            regiao_selecionada = "E"
        else:
            mensagem = "Código NÃO encontrado na planilha. Adicionando e enviando para a região de Coleta."
            add_code_to_excel(extracted_text, excel_path)
            regiao_selecionada = "C"

        print(mensagem)

        etapa3()

# Etapa 3: Verificar colocação
def etapa3():
    global class_name, regiao_selecionada, classe_identificada, itens_enviados, itens_coletados

    while True:
        check, img = video.read()
        if not check:
            print("Erro ao capturar vídeo")
            break

        mensagem = "Nenhum objeto detectado."  # Mensagem padrão
        results_objetos = model_objetos.predict(img, verbose=False)

        objeto_detectado = False  # Variável de controle

        # Desenhar as regiões de envio e coleta
        altura_regiao = int(frame_height * 0.1)
        coleta_cor = (255, 0, 0)  # Azul
        envio_cor = (0, 255, 0)  # Verde

        cv2.rectangle(img, (0, frame_height - altura_regiao), (frame_width, frame_height), coleta_cor, 2)
        cv2.putText(img, "Coleta", (10, frame_height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, coleta_cor, 2)
        
        cv2.rectangle(img, (0, 0), (frame_width, altura_regiao), envio_cor, 2)
        cv2.putText(img, "Envio", (10, altura_regiao - 10), cv2.FONT_HERSHEY_COMPLEX, 1, envio_cor, 2)
        """cv2.rectangle(img, (0, frame_height - int(frame_height * 0.1)), (frame_width, frame_height), (255, 0, 0), -1)
        cv2.putText(img, "COLETA", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.rectangle(img, (0, 0), (frame_width, int(frame_height * 0.1)), (0, 255, 0), -1)
        cv2.putText(img, "ENVIO", (10, int(frame_height * 0.1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)"""

        global Sair_Etapa3 
        Sair_Etapa3 = False # voltar para etapa1
        for result in results_objetos:
            for box in result.boxes:
                objeto_detectado = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                detected_class_name = model_objetos.names[cls]
                
                cv2.putText(img, detected_class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Verificação da classe identificada
                if detected_class_name != classe_identificada:
                    mensagem = "Erro! Objeto detectado não corresponde ao identificado."
                else:
                    if regiao_selecionada == "C":
                        if y2 > frame_height - altura_regiao:
                            mensagem = "Coletado com sucesso!"
                            itens_coletados += 1    
                            Sair_Etapa3 = True
                            break
                        elif y1 < altura_regiao:
                            mensagem = "Erro! Objeto na região de Envio, deveria estar na região de Coleta."
                        else:
                            mensagem = f"Coloque o objeto na região de {regiao_selecionada}."
                    elif regiao_selecionada == "E":
                        if y1 < altura_regiao:
                            mensagem = "Enviado com sucesso!"
                            itens_enviados += 1
                            Sair_Etapa3 = True
                            break
                        elif y2 > frame_height - altura_regiao:
                            mensagem = "Erro! Objeto na região de Coleta, deveria estar na região de Envio."
                        else:
                            mensagem = f"Coloque o objeto na região de {regiao_selecionada}."
            if Sair_Etapa3:
                break

        if not objeto_detectado:
            mensagem = "Nenhum objeto detectado."

        # Exibir imagem com a mensagem
        img = draw_text_with_pillow(img, mensagem, (10, frame_height // 2), font_size=32, color=(0, 0, 255))
        cv2.imshow("Sistema de Detecção", img)

        # Interromper o loop se a flag estiver definida
        if Sair_Etapa3:
            etapa1()
            break

        # Encerrar ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Iniciar processo
if __name__ == "__main__":
    try:
        print("Iniciando o sistema...")
        etapa1()
    except KeyboardInterrupt:
        print("Sistema interrompido pelo usuário.")
    finally:
        video.release()
        out.release()
        cv2.destroyAllWindows()
        print("Recursos liberados.")

print(f"Vídeo salvo como: {output_file}")