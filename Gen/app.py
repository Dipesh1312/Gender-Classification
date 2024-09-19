from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image 

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('gender_detection_3.h5')
 
# Define classes (if classification)
classes = ['man' , 'woman']  # Update with your class labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))  # Resize image if required by your model
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions[0])]

        return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
