import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import io
from flask import Flask, request ,jsonify
# Dictionary to store models and their expected input shape
app = Flask(__name__)

MODEL_CONFIG = {
    'chest': {'file': 'models/Model-Chest2-151221.h5', 'type': 'binary', 'input_shape': (224, 224)},
    'skin': {'file': 'models/Model-skin-vgg-141221.h5', 'type': 'binary', 'input_shape': (224, 224)},
    'thyroid': {'file': 'models/Model-skin-vgg-141221.h5', 'type': 'binary', 'input_shape': (224, 224)},
    # Add other multi-class models here
    'breast': {'file': 'models/Model-Breast-Teach-161221.h5', 'type': 'multi-class', 'input_shape': (224, 224,3)},
    'Brain' : {'file': 'models/Model-BrainTumor-Teach-171221.h5', 'type': 'multi-class', 'input_shape' : (224, 224)},
    'Retino': {'file' : 'models/Model-RetinoD-Teach-121222.h5', 'type' : 'multi-class' , 'input_shape' : (224, 224)}
}

def load_and_preprocess_image(image_path, target_size):
    img_bytes = io.BytesIO(image_path.read())
    img = load_img(img_bytes, target_size=target_size[:2])
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        model_name = request.form['model']
        if model_name not in MODEL_CONFIG:
            return jsonify({"error": f"Model '{model_name}' not found"}), 400
    

    # Load model and config
        model_info = MODEL_CONFIG[model_name]
        model = load_model(model_info['file'], compile=False)
        target_size = model_info['input_shape']

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        image_file = request.files['image']

        img_array = load_and_preprocess_image(image_file, target_size)

        predictions = model.predict(img_array)
        # Load and preprocess image(s) based on model type
        if model_info['type'] == 'binary':
            confidence = predictions[0][0] * 100 if predictions[0][0] > 0.5 else (1 - predictions[0][0]) * 100
            label = 'Positive' if predictions[0][0] > 0.5 else 'Negative'
            return jsonify({"label": label, "confidence": f"{confidence:.2f}%"})

        elif model_info['type'] == 'multi-class':
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions[0]) * 100
            return jsonify({"class": int(predicted_class), "confidence": f"{confidence:.2f}%"})

        else:
            return jsonify({"error": "Unsupported model type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)

    
