from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- agrega esto
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # <--- habilita CORS

# Cargar modelo
model = load_model('modelo_covid19.keras')

# Procesar imagen
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_tensor = preprocess_image(img)
    prediction = model.predict(img_tensor)[0][0]

    result = 'COVID-19 POSITIVE' if prediction > 0.5 else 'NORMAL'
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

    return jsonify({'prediction': result, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
