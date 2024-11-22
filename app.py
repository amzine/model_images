from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io

app = Flask(__name__)

MODEL_CONFIG = {
    'chest': {'file': 'models/Model-Chest2-151221.h5', 'type': 'binary', 'input_shape': (224, 224), 'mean_pixel_value': 0.4799},
    'skin': {'file': 'models/Model-skin-vgg-141221.h5', 'type': 'binary', 'input_shape': (224, 224), 'mean_pixel_value': 0.62},
    'thyroid': {'file': 'models/Model-skin-vgg-141221.h5', 'type': 'binary', 'input_shape': (224, 224)},

    'breast': {'file': 'models/Model-Breast-Teach-161221.h5', 'type': 'multi-class', 'input_shape': (224, 224,3)},
    'Brain' : {'file': 'models/Model-BrainTumor-Teach-171221.h5', 'type': 'multi-class', 'input_shape' : (224, 224)},
    'Retino': {'file' : 'models/Model-RetinoD-Teach-121222.h5', 'type' : 'multi-class' , 'input_shape' : (224, 224)}
}
def load_and_preprocess_image(file_storage, target_size):
    # Convert FileStorage object to BytesIO
    img_bytes = io.BytesIO(file_storage.read())
    img = load_img(img_bytes, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img_array

def is_image_appropriate(img_array, mean_pixel_value, threshold=0.1):
    # Check if the mean pixel value of the image is within an acceptable range
    input_mean = np.mean(img_array)
    return abs(input_mean - mean_pixel_value) <= threshold

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form['model']
        if model_name not in MODEL_CONFIG:
            return jsonify({"error": f"Model '{model_name}' not found"}), 400

        model_info = MODEL_CONFIG[model_name]
        model = load_model(model_info['file'], compile=False)
        target_size = model_info['input_shape']
        mean_pixel_value = model_info['mean_pixel_value']

        # Get the uploaded file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        image_file = request.files['image']

        # Preprocess the image
        img_array_expanded, img_array = load_and_preprocess_image(image_file, target_size)

        # Validate the image against model-specific properties
        if not is_image_appropriate(img_array, mean_pixel_value):
            return jsonify({"error": "The provided image is not appropriate for this model"}), 400

        # Make a prediction
        predictions = model.predict(img_array_expanded)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
