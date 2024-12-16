import openai
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def get_gpt4o_mini_response(prompt):
    """Obtém uma resposta do GPT-4o-mini para o prompt fornecido."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
           {"role": "system", "content": "Você é um assistente especializado em primeiros socorros. Sua função é fornecer instruções claras, rápidas e precisas relacionadas a emergências médicas, enquanto mantém uma linguagem simples e compreensível para qualquer pessoa. Você também pode responder a dúvidas gerais sobre como proceder em situações de queda ou outras emergências."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()