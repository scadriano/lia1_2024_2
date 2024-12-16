import os
import cv2
import time
import mediapipe as mp
import pygetwindow as gw  # Para gerenciar janelas do sistema
import pyautogui    # Para pressionamento de teclas multimídia

mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)


# Caminho para o aplicativo do YouTube Music
youtube_music_path = r"C:\Users\ryanr\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Aplicativos do Brave\Youtube Music.lnk"

# Função para abrir o YouTube Music
def open_youtube_music():
    if os.path.exists(youtube_music_path):
        print("Abrindo YouTube Music...")
        os.startfile(youtube_music_path)
    else:
        print("Caminho para o YouTube Music não encontrado!")

# Função para focar na página do Youtube Music
def focus_brave():
    windows = gw.getWindowsWithTitle('YouTube Music')  # Procura pela janela do YouTube Music
    if windows:
        if not windows[0].isActive:  # Verifica se a janela já está ativa
            windows[0].activate()    # Alterna o foco para a janela
        return True
    else:
        open_youtube_music()  # Abre o YouTube Music se não estiver aberto
        time.sleep(5)  # Aguarda o aplicativo abrir
        return False
    
# Função para retornar à janela anterior com tratamento de erro
def return_to_previous_window(previous_window_title):
    windows = gw.getWindowsWithTitle(previous_window_title)
    if windows:
        try:
            windows[0].activate()
        except Exception as e:
            print(f"Erro ao ativar a janela: {e}")
            # Alternativa: usar pyautogui para clicar na janela
            rect = windows[0].box
            pyautogui.click(rect.left + 10, rect.top + 10)
    else:
        print("Janela anterior não encontrada.")

# Função que verifica se a mão está fechada
def is_hand_fist(hand_landmarks):
    # Coordenadas dos pontos de referência dos dedos
    thumb_tip = hand_landmarks.landmark[4].y
    thumb_base = hand_landmarks.landmark[3].y
    index_tip = hand_landmarks.landmark[8].y
    index_base = hand_landmarks.landmark[7].y
    middle_tip = hand_landmarks.landmark[12].y
    middle_base = hand_landmarks.landmark[11].y
    ring_tip = hand_landmarks.landmark[16].y
    ring_base = hand_landmarks.landmark[15].y
    pinky_tip = hand_landmarks.landmark[20].y
    pinky_base = hand_landmarks.landmark[19].y

    # Verificar se todos os dedos, exceto o polegar, estão fechados
    all_fingers_closed = (
        index_tip > index_base and
        middle_tip > middle_base and
        ring_tip > ring_base and
        pinky_tip > pinky_base
    )

    # Verificar se o polegar está dobrado em relação à palma
    thumb_folded = thumb_tip > thumb_base

    # Retornar o valor lógico "and" dos dois valores
    return all_fingers_closed and thumb_folded

# Função para alternar o estado de comandos
def toggle_commands():
    global allow_commands, start_first_time, previous_window_title

    if allow_commands:  # Desabilitar comandos
        print("Comandos desabilitados.")
        if previous_window_title:  # Tentar voltar à janela anterior, se existir
            try:
                return_to_previous_window(previous_window_title)
            except Exception as e:
                print(f"Erro ao retornar à janela anterior: {e}")
        previous_window_title = None
    else:  # Habilitar comandos
        print("Comandos habilitados.")
        active_window = gw.getActiveWindow()
        previous_window_title = active_window.title if active_window else None
        if not focus_brave():  # Tenta focar ou abrir o YouTube Music
            print("Não foi possível focar ou abrir o YouTube Music.")

    allow_commands = not allow_commands  # Alterna o estado
    start_first_time = time.time()

# Função para identificar a mão dominante
def get_hand_type(hand_landmarks, handedness):
    for hand_index, hand_label in enumerate(handedness):
        if hand_label.classification[0].label == "Right":
            return "Right"
        elif hand_label.classification[0].label == "Left":
            return "Left"
    if not handedness:
        return None
    return None

# Detectar o gesto com base nas posições dos dedos
def detect_gesture(hand_landmarks):
    # Coordenadas y (altura) dos pontos de referência dos dedos
    thumb_tip = hand_landmarks.landmark[4].y
    thumb_base = hand_landmarks.landmark[3].y
    index_tip = hand_landmarks.landmark[8].y
    index_middle = hand_landmarks.landmark[6].y
    middle_tip = hand_landmarks.landmark[12].y
    middle_base = hand_landmarks.landmark[10].y
    ring_tip = hand_landmarks.landmark[16].y
    ring_base = hand_landmarks.landmark[14].y
    pinky_tip = hand_landmarks.landmark[20].y
    pinky_base = hand_landmarks.landmark[18].y

    # Condições para gestos específicos
    if index_tip < index_middle and middle_tip < middle_base:  # Indicador e médio levantados
        return "V"
    elif ring_tip < ring_base and pinky_tip < pinky_base:  # Anelar e mindinho levantados
        return "U"
    elif middle_tip < middle_base and ring_tip < ring_base: # Médio e anelar levantados
        return "M"
    elif index_tip < index_middle:  # Indicador levantado
        return "Point"
    elif middle_tip < middle_base:  # Dedo médio levantado
        return "Middle"
    elif ring_tip < ring_base:  # Anelar levantado
        return "Ring"
    return None

# Comandos para mão esquerda
def handle_left_hand(gesture):
    if gesture == "V":  # Indicador e médio levantados
        pyautogui.hotkey('tab')
        print("Tab")
        time.sleep(0.1)
    elif gesture == "U":  # Anelar e mindinho levantados
        pyautogui.hotkey('shift', 'tab')
        print("Shift+Tab")
        time.sleep(0.1)
    elif gesture == "Point":  # Indicador levantado
        pyautogui.press('enter')
        print("Enter")
    elif gesture == "Middle":  # Médio levantado  
        pyautogui.press('l')   # Avançar 15 segundos
        print("L")
    elif gesture == "Ring":  # Mindinho levantados
        pyautogui.press('h') # Retornar 15 segundos 
        print("H")

# Comandos para mão direita (já existentes)
def handle_right_hand(gesture):
    global is_paused # Adicionado para acessar a variável global

    if gesture == "Middle":
        pyautogui.press("prevtrack")
        print("Polegar levantado: shift+p.")
        time.sleep(0.3)  # Evitar ações repetidas muito rápidas

    elif gesture == "Ring":
        pyautogui.press("nexttrack")
        print("Dedo mindinho levantado: shift+n.")
        time.sleep(0.3)  # Evitar ações repetidas muito rápidas

    elif gesture == "Point":
        if is_paused:
            print("Despausando música...")
            pyautogui.press("playpause")
            is_paused = False
        else:
            print("Pausando música...")
            pyautogui.press("playpause")
            is_paused = True
        time.sleep(1)

    elif gesture == "V":
        print("Aumentando volume...")
        pyautogui.press('volumeup')
        time.sleep(0.01)  # Evitar ações repetidas muito rápidas

    elif gesture == "U":
        print("Diminuindo volume...")
        pyautogui.press("volumedown")
        time.sleep(0.01)  # Evitar ações repetidas muito rápidas

    elif gesture == "M":    
        print("Desativando o som...")
        pyautogui.press("volumemute")
        time.sleep(0.2)


# Variáveis de estado
allow_commands = False
start_first_time = None
is_paused = False
previous_window_title = None


while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)  # Inverte a imagem da webcam
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Exibe texto sobre o estado dos comandos
    status_text = "Comandos habilitados" if allow_commands else "Comandos desabilitados"
    status_color = (0, 255, 0) if allow_commands else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = get_hand_type(hand_landmarks, [handedness])
            
            # Se a mão estiver fechada
            if is_hand_fist(hand_landmarks):
                if start_first_time is None:
                    start_first_time = time.time()
                elif time.time() - start_first_time >= 3: # Por pelo menos 3 segundos
                    toggle_commands()                     # Mude o estado da variavel allow_commands
                    start_first_time = None
            else:
                # Reinicia o temporizador se a mão não estiver mais fechada
                start_first_time = None

            if allow_commands:
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    if hand_type == "Left":
                        handle_left_hand(gesture)
                    elif hand_type == "Right":
                        handle_right_hand(gesture)

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()