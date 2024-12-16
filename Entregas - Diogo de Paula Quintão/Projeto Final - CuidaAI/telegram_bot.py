from venv import logger
import requests
import os
import time
from gpt4omini import get_gpt4o_mini_response

# Carregar variáveis de ambiente
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

def send_telegram(message):
    """Envia uma mensagem de alerta pelo Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'   
    }
    response = requests.post(url, data=payload)
    return response.json()

def send_telegram_photo(photo_path):
    """Envia uma foto pelo Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {
            'chat_id': CHAT_ID
        }
        files = {
            'photo': photo
        }
        response = requests.post(url, data=payload, files=files)
    return response.json()

def get_latest_messages(last_update_id=None):
    """Obtém novas mensagens do usuário."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

    if last_update_id is not None:
        url += f"?offset={last_update_id + 1}"

    response = requests.get(url)
    updates = response.json()

    messages = []
    if updates.get("result"):
        for update in updates["result"]:
            message = update.get("message")
            if message:
                chat_id = message["chat"]["id"]
                text = message.get("text", "")
                update_id = update["update_id"]
                messages.append((update_id, chat_id, text))
    return messages

def clear_updates(last_update_id):
    """Limpa os updates antigos."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={last_update_id + 1}"
    response = requests.get(url)
    return response.json()

def handle_gpt_conversation(chat_id, text, update_id):
    """Gerencia conversas com GPT"""
    try:
        if chat_id and text and chat_id == int(CHAT_ID):
            response = get_gpt4o_mini_response(text)
            send_telegram(response)
            clear_updates(update_id)
            return True
        return False
    except Exception as e:
        logger.error(f"Erro na conversa com GPT: {str(e)}")
        send_telegram("Desculpe, tive um problema ao processar sua mensagem.")
        return False

def send_alert():
    """Envia alerta e gerencia respostas"""
    send_telegram("""
🚨 **Alerta de Queda Detectada** 🚨

Uma queda foi detectada. O cuidador precisa verificar a situação.

Por favor, responda:
- "Sim" se está indo ao local imediatamente.
- "Não" se não pode ir, para acionarmos uma ambulância.

**Tempo limite: 3 minutos** para chegar ao local.
""")

    timeout = 90  # 3 minutos
    start_time = time.time()
    last_update_id = None
    response_received = False

    while time.time() - start_time < timeout and not response_received:
        messages = get_latest_messages(last_update_id)

        if messages:
            for update in messages:
                update_id, chat_id, text = update
                last_update_id = update_id

                if chat_id and text and chat_id == int(CHAT_ID):
                    texto = text.lower().strip()
                    
                    # Verificar resposta sim/não
                    if texto in ["sim", "s", "si", "sí", "sI", "sim,", "sim.", "simon", "sim!", "siii", "S"]:
                        send_telegram("✅ Você está indo ao local. Agora você pode fazer perguntas sobre primeiros socorros.")
                        response_received = True
                        
                    elif texto in ["não", "nao", "nao!", "nao.", "não.", "nao,", "não!", "naõ", "n", "N"]:
                        send_telegram("🚑 Acionando ambulância. Agora você pode fazer perguntas sobre primeiros socorros.")
                        response_received = True
                        
                    else:
                        # Se não for sim/não, tratar como pergunta para GPT
                        handle_gpt_conversation(chat_id, text, update_id)
        
        time.sleep(0.5)

    if not response_received:
        send_telegram("⚠️ Tempo esgotado! Acionando ambulância por precaução.")
        return False

    return True