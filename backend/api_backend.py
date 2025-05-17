from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import webbrowser
import threading
import os

# Load the model
MODEL_PATH = "models/plant_disease_model.keras"
model = keras.models.load_model(MODEL_PATH)

# Class names (update this list if needed)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
    'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

IMG_SIZE = (128, 128)

app = Flask(__name__)
CORS(app)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def format_class_name(raw_class):
    # Replace triple underscores and double underscores with comma and space, then single underscores with space
    name = raw_class.replace('___', ', ').replace('__', ', ').replace('_', ' ')
    return name

def parse_plant_and_status(class_label):
    # Attempt to split at triple underscore
    if '___' in class_label:
        plant, status = class_label.split('___', 1)
        # Clean up plant and status names
        plant = plant.replace('_', ' ').replace(',', ', ')
        status = status.replace('_', ' ').replace(',', ', ')
    else:
        plant = class_label.replace('_', ' ').replace(',', ', ')
        status = "Unknown"
    # Capitalize first letter of each word
    plant = plant.strip().title()
    status = status.strip().title()
    return plant, status

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)
        preds = model.predict(img_array)[0]
        pred_idx = int(np.argmax(preds))
        raw_label = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx]) * 100  # Convert to percent
        plant, status = parse_plant_and_status(raw_label)
        # Logging
        with open("prediction_log.txt", "a") as log_file:
            log_file.write(f"File: {file.filename}, Plant: {plant}, Status: {status}, Confidence: {confidence:.2f}%, Raw: {raw_label}\n")
        return jsonify({
            "plant": plant,
            "status": status,
            "confidence": f"{confidence:.2f}%",
            "raw_label": raw_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def root():
    # Serve the frontend HTML file
    return app.send_static_file("index.html")

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    return send_from_directory(assets_dir, filename)

@app.route("/about")
def about():
    return app.send_static_file("about.html")

# Serve static files from the correct frontend directory (sibling to backend)
app.static_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new(f"http://127.0.0.1:8000/")
    threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=8000, debug=True)
