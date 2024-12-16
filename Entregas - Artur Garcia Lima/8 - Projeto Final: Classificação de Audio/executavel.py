import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
import tkinter as tk


class MessageDisplay:
    """
    Classe para gerenciar uma única janela de mensagens interativas usando tkinter.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Classificação de Áudio")
        self.root.geometry("800x400")
        
        # Label para exibir mensagens
        self.label = tk.Label(self.root, text="", font=("Arial", 20), wraplength=700, justify="center")
        self.label.pack(expand=True)
        
        # Campo de entrada para interações
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(self.root, textvariable=self.input_var, font=("Arial", 16))
        self.input_entry.pack()
        
        # Botão para confirmar a entrada
        self.button = tk.Button(self.root, text="Confirmar", font=("Arial", 16), command=self.confirm_input)
        self.button.pack()
        
        self.user_input = None

    def update_message(self, message):
        """
        Atualiza a mensagem exibida na janela.
        """
        self.label.config(text=message)
        self.input_var.set("")  # Limpa o campo de entrada
        self.root.update_idletasks()

    def confirm_input(self):
        """
        Captura a entrada do usuário e armazena na variável `user_input`.
        """
        self.user_input = self.input_var.get().strip().lower()
        self.root.quit()

    def get_user_input(self, message):
        """
        Exibe uma mensagem e aguarda a entrada do usuário.
        """
        self.update_message(message)
        self.root.mainloop()  # Espera o usuário interagir
        return self.user_input


def list_audio_devices():
    """
    Lista dispositivos de entrada e saída de áudio disponíveis no sistema.
    """
    print(sd.query_devices())


def record_audio_from_speakers(filename, duration=15, samplerate=44100, device=None, display=None):
    """
    Grava áudio diretamente dos alto-falantes (Stereo Mix ou Realtek) e salva como WAV.
    """
    display.update_message(f"Gravando áudio por {duration} segundos...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32', device=device)
    sd.wait()
    display.update_message("Gravação concluída.")
    write(filename, samplerate, (audio * 32767).astype(np.int16))


def preprocess_audio_for_model(filename, n_mfcc=40):
    """
    Carrega o áudio e extrai MFCCs para entrada no modelo.
    """
    audio, sample_rate = librosa.load(filename, sr=None)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1, -1)


def classify_audio_keras(model_path, processed_audio, label_encoder):
    """
    Classifica o áudio processado usando um modelo `.keras`.
    """
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(processed_audio)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = label_encoder.inverse_transform(top_indices)
    top_probabilities = predictions[top_indices]
    return top_classes, top_probabilities


def build_model(input_shape, num_classes):
    """
    Constrói a arquitetura do modelo para carregar pesos `.weights.h5`.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


def classify_audio_weights(model_path, processed_audio, label_encoder, input_shape, num_classes):
    """
    Classifica o áudio processado usando pesos `.weights.h5`.
    """
    model = build_model(input_shape, num_classes)
    model.load_weights(model_path)
    predictions = model.predict(processed_audio)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = label_encoder.inverse_transform(top_indices)
    top_probabilities = predictions[top_indices]
    return top_classes, top_probabilities


if __name__ == "__main__":
    list_audio_devices()

    # Configuração inicial
    DEVICE_ID = 22
    AUDIO_FILENAME = "recorded_audio.wav"
    DURATION = 15

    # Inicialização da janela de mensagens
    display = MessageDisplay()

    # Escolha do modelo
    model_type = display.get_user_input("Projeto final de LIA\n Professor: Adriano Cesar Santana\nAluno: Artur Garcia Lima\nPara começar, digite: ~keras~")
    if model_type == "keras":
        MODEL_PATH = "saved_models/audio_classification.keras"
        model_loader = classify_audio_keras
    elif model_type == "weights":
        MODEL_PATH = "saved_models/audio_classification.weights.h5"
        INPUT_SHAPE = (40,)
        NUM_CLASSES = 4  # Ajuste conforme o número de classes originais do treinamento
        model_loader = lambda model_path, processed_audio, label_encoder: classify_audio_weights(
            model_path, processed_audio, label_encoder, INPUT_SHAPE, NUM_CLASSES
        )
    else:
        display.update_message("Tipo de modelo inválido. Encerrando.")
        display.root.mainloop()
        exit()

    # Configuração das classes e LabelEncoder
    classes = ["Bateria", "Violino", "Piano", "Flauta"]
    LE = LabelEncoder()
    LE.fit(classes)

    while True:
        # Gravação do áudio
        record_audio_from_speakers(AUDIO_FILENAME, duration=DURATION, device=DEVICE_ID, display=display)

        # Pré-processamento do áudio
        processed_audio = preprocess_audio_for_model(AUDIO_FILENAME)

        # Classificação
        try:
            top_classes, top_probabilities = model_loader(MODEL_PATH, processed_audio, LE)
        except Exception as e:
            display.update_message(f"Erro ao classificar áudio: {e}")
            break

        # Exibição dos resultados
        message = (
            f"Classe prevista: {top_classes[0]} com precisão de {top_probabilities[0]:.2%}\n"
            f"2ª: {top_classes[1]} com precisão de {top_probabilities[1]:.2%}\n"
            f"3ª: {top_classes[2]} com precisão de {top_probabilities[2]:.2%}\n"
            "Deseja prever novamente? (s/n):"
        )
        user_input = display.get_user_input(message)

        # Verificar se o usuário deseja continuar
        if user_input == "n":
            display.update_message("Encerrando o programa. Até mais!")
            display.root.mainloop()
            break
