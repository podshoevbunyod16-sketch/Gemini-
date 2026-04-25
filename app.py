from flask import Flask, render_template, request, jsonify, send_file
from huggingface_hub import InferenceClient
import os
import base64
import uuid
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HF_TOKEN = os.environ.get('HF_TOKEN', None)
client = InferenceClient(token=HF_TOKEN)

# ... (остальной код такой же, но убрать строки с Image.open при ошибках)

@app.route('/api/generate/image', methods=['POST'])
def generate_image():
    """Генерация изображения через Stable Diffusion"""
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Пустой промпт'}), 400
    
    try:
        # Получаем изображение как байты напрямую
        image_bytes = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        # Если image_bytes уже в нужном формате
        if hasattr(image_bytes, 'read'):
            image_bytes = image_bytes.read()
        
        img_base64 = base64.b64encode(image_bytes).decode()
        return jsonify({'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Упрощённый `requirements.txt` без Pillow: