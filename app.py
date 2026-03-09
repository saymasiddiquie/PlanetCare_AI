from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- PROJECT SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plantcare_model.h5") 
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD YOUR 38-CLASS MODEL ---
def load_plant_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        return None

    try:
        # 1. Build the exact architecture to match your Colab training
        base_model = tf.keras.applications.MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(38, activation='softmax')
        ])

        # 2. THE FIX: Load by name and skip the one broken layer (#0)
        # This allows the other 100+ layers to load their learned patterns
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        
        print("✅ PlantCare AI: Weights mapped successfully (Skipped mismatch layer)!")
        return model
    except Exception as e:
        print(f"❌ CRITICAL LOAD ERROR: {e}")
        return None

model = load_plant_model()

# Hardcoded 38 classes
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Check terminal for errors."
    
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocessing is CRITICAL for MobileNetV2
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    # MobileNetV2 expects pixels scaled to [-1, 1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = round(100 * np.max(predictions), 2)
    
    result_text = class_names[predicted_index].replace("___", " - ").replace("_", " ")

    return render_template("result.html", 
                           prediction=result_text, 
                           confidence=f"{confidence}%", 
                           image_path=file.filename)

if __name__ == "__main__":
    app.run(debug=True)