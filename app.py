from flask import Flask, render_template, request, jsonify, send_file
from huggingface_hub import InferenceClient
import os
import base64
from io import BytesIO
import requests
from PIL import Image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Hugging Face API Client
HF_TOKEN = os.environ.get('HF_TOKEN', None)  # Опционально: ваш токен с huggingface.co/settings/tokens
client = InferenceClient(token=HF_TOKEN)

# --- API Endpoints для генерации ---

@app.route('/api/generate/text', methods=['POST'])
def generate_text():
    """Генерация текста через LLM"""
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Пустой промпт'}), 400
    
    try:
        response = client.text_generation(
            prompt,
            model="google/gemma-2-2b-it",
            max_new_tokens=500,
            temperature=0.7
        )
        return jsonify({'text': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/image', methods=['POST'])
def generate_image():
    """Генерация изображения через Stable Diffusion"""
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Пустой промпт'}), 400
    
    try:
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        # Конвертируем в base64 для отправки на фронтенд
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/video', methods=['POST'])
def generate_video():
    """Генерация видео из изображения (Stable Video Diffusion)"""
    if 'image' not in request.files:
        return jsonify({'error': 'Изображение не загружено'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    try:
        # Сохраняем загруженное изображение
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.png")
        file.save(img_path)
        
        # Загружаем изображение
        image = Image.open(img_path)
        
        # Генерируем видео через SVD
        # Примечание: SVD требует локального запуска или отдельного эндпоинта
        # Этот эндпоинт демонстрирует концепцию, для продакшена используйте локальную модель
        
        # API Hugging Face для видео требует другой подход
        # Альтернатива: использовать diffusers библиотеку локально
        
        # Возвращаем заглушку с инструкцией
        os.remove(img_path)
        
        return jsonify({
            'video_url': None,
            'message': 'Видео генерация требует локального GPU. ' +
                       'Установите diffusers: pip install diffusers accelerate torch ' +
                       'и используйте модель "stabilityai/stable-video-diffusion-img2vid-xt"',
            'example_code': '''
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()
image = load_image("your_image.jpg")
frames = pipe(image, decode_chunk_size=8).frames[0]
export_to_video(frames, "output.mp4", fps=7)
            '''
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/audio', methods=['POST'])
def generate_audio():
    """Текст в речь (TTS)"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Пустой текст'}), 400
    
    try:
        # Используем TTS модель
        audio_bytes = client.text_to_speech(
            text,
            model="facebook/mms-tts-eng"
        )
        
        # Сохраняем аудио
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.wav")
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        return send_file(audio_path, mimetype='audio/wav', as_attachment=False)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Анализ изображения (распознавание объектов)"""
    if 'image' not in request.files:
        return jsonify({'error': 'Изображение не загружено'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    try:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.jpg")
        file.save(img_path)
        
        image = Image.open(img_path)
        
        # Используем модель для image-to-text
        response = client.image_to_text(
            image,
            model="google/vision-encoder-decoder-vit"
        )
        
        os.remove(img_path)
        
        return jsonify({'description': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Главная страница ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)