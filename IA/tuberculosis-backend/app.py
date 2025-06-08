from flask import Flask, request, jsonify
import joblib
import re
import logging
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
from pdfminer.high_level import extract_text
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from geopy.distance import geodesic
import json

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicialización de Flask
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app)

# Configuraciones
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Modelo CNN para imágenes
class TBImageModel(nn.Module):
    def __init__(self):
        super(TBImageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 256 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Preprocesador de texto mejorado
class TextPreprocessor:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or {
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se',
            'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es',
            'me', 'mi', 'tengo', 'con', 'que', 'y', 'ha', 'he', 'por', 'fiebre'
        }
        self.symptoms_keywords = {
            'tos', 'sangre', 'sudoración', 'sudoracion', 'nocturna', 'pérdida', 'perdida',
            'peso', 'fatiga', 'dolor', 'pecho', 'respirar', 'respiración', 'fiebre',
            'escalofríos', 'escalofrios', 'debilidad', 'flema', 'sudores', 'nocturnos'
        }
    
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\sáéíóúñ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_symptoms(self, text):
        words = set(self.clean_text(text).split())
        detected = []
        
        for word in words:
            if word in self.symptoms_keywords:
                detected.append(word)
            elif any(keyword.startswith(word[:4]) for keyword in self.symptoms_keywords):
                similar = [k for k in self.symptoms_keywords if k.startswith(word[:4])]
                detected.extend(similar)
        
        return list(set(detected))

# Cargar modelos
try:
    # Modelo de texto
    model_data = joblib.load("modelo_tb.pkl")
    text_model = model_data['pipeline']
    preprocessor = TextPreprocessor(model_data.get('stop_words'))
    
    # Modelo de imágenes
    image_model = TBImageModel()
    image_model.load_state_dict(torch.load('tb_radiography_model.pth', map_location=torch.device('cpu')))
    image_model.eval()
    
    logger.info("✅ Modelos cargados correctamente")
except Exception as e:
    logger.error(f"❌ Error cargando modelos: {e}")
    exit(1)

# Cargar datos de hospitales
with open('hospitals.json', 'r', encoding='utf-8') as f:
    hospitals = json.load(f)

# Transformaciones para imágenes
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_storage):
    try:
        # Crear directorio si no existe
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Guardar archivo temporal
        temp_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload.pdf')
        file_storage.save(temp_pdf_path)
        
        # Extraer texto
        text = extract_text_from_pdf(temp_pdf_path)
        
        # Eliminar temporal
        os.remove(temp_pdf_path)
        
        return text[:5000] if text else None
        
    except Exception as e:
        logger.error(f"Error procesando PDF: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF usando pdfminer"""
    output_string = StringIO()
    try:
        with open(pdf_path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                
        return output_string.getvalue()
    except Exception as e:
        logger.error(f"Error extrayendo texto de PDF: {e}")
        return None

def process_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = image_transform(img).unsqueeze(0)
        with torch.no_grad():
            prediction = image_model(img_tensor)
        return prediction.item()
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        return None

def find_nearest_hospital(lat, lon):
    try:
        user_location = (lat, lon)
        nearest = None
        min_distance = float('inf')
        
        for hospital in hospitals:
            hosp_location = (hospital['lat'], hospital['lon'])
            distance = geodesic(user_location, hosp_location).km
            if distance < min_distance:
                min_distance = distance
                nearest = hospital
        
        nearest['distance'] = round(min_distance, 2)
        return nearest
    except Exception as e:
        logger.error(f"Error buscando hospital: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Para texto directo
        if 'text' in request.form:
            texto = request.form['text'].strip()
            if len(texto) < 10:
                return jsonify({'error': 'El texto debe tener al menos 10 caracteres'}), 400
                
            texto_limpio = preprocessor.clean_text(texto)
            prob = float(text_model.predict_proba([texto_limpio])[0][1])
            symptoms = preprocessor.extract_symptoms(texto)
            
            return jsonify({
                'type': 'text',
                'probability': prob,
                'symptoms': symptoms,
                'processed_text': texto_limpio
            })
        
        # Para archivos (PDF o imagen)
        if 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({'error': 'Archivo no seleccionado'}), 400
                
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400
                
            ext = file.filename.rsplit('.', 1)[1].lower()
            
            if ext == 'pdf':
                text = process_pdf(file)
                if not text:
                    return jsonify({'error': 'Error al procesar PDF'}), 500
                    
                texto_limpio = preprocessor.clean_text(text)
                prob = float(text_model.predict_proba([texto_limpio])[0][1])
                symptoms = preprocessor.extract_symptoms(text)
                
                return jsonify({
                    'type': 'pdf',
                    'probability': prob,
                    'symptoms': symptoms,
                    'extracted_text': text[:500] + '...' if len(text) > 500 else text
                })
                
            elif ext in {'png', 'jpg', 'jpeg'}:
                img_bytes = file.read()
                prob = process_image(img_bytes)
                if prob is None:
                    return jsonify({'error': 'Error al procesar imagen'}), 500
                
                return jsonify({
                    'type': 'image',
                    'probability': prob,
                    'diagnosis': 'TB detected' if prob > 0.5 else 'Normal'
                })
        
        return jsonify({'error': 'Solicitud no válida'}), 400
    
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        return jsonify({'error': 'Error procesando la solicitud'}), 500

@app.route('/find-hospital', methods=['POST'])
def find_hospital():
    try:
        data = request.get_json()
        if not data or 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'Coordenadas requeridas (lat, lon)'}), 400
            
        lat = float(data['lat'])
        lon = float(data['lon'])
        
        hospital = find_nearest_hospital(lat, lon)
        if not hospital:
            return jsonify({'error': 'No se encontraron hospitales cercanos'}), 404
            
        return jsonify(hospital)
    
    except Exception as e:
        logger.error(f"Error en búsqueda de hospital: {str(e)}")
        return jsonify({'error': 'Error al buscar hospitales'}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)