from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from pyngrok import ngrok

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("wildlife_classifier.h5")

# Get class names from your dataset directory
# You'll need to adjust this path to match your actual dataset location
class_names = sorted(os.listdir("wildlife_dataset/train"))
print(f"Loaded classes: {class_names}")

def preprocess_image(image_bytes):
    """Process uploaded image bytes for model prediction"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # Resize to match model's expected input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read and preprocess the image
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        # Get top 3 predictions for display
        top_indices = predictions[0].argsort()[-3:][::-1]
        top_results = [
            {"class": class_names[i], "confidence": float(predictions[0][i])}
            for i in top_indices
        ]

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'top_results': top_results
        })

def run_with_ngrok():
    # Set up ngrok
    port = 5000
    public_url = ngrok.connect(port).public_url
    print(f" * Running on {public_url}")
    app.config["BASE_URL"] = public_url
    
    # Start Flask app
    app.run(debug=False, port=port)

if __name__ == '__main__':
    run_with_ngrok()