
from flask import Flask,jsonify,request
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
model = tf.keras.models.load_model('ClassicCNN_v1_28_0.914.keras')
image_size = (32, 32)

app=Flask(__name__)
def preprocess_image_for_cnn(image_stream, image_size=(32, 32)):
    img = np.array(image_stream)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

    img = cv2.bitwise_not(img)

    img = img.astype('float32') / 255.0

    img = np.expand_dims(img, axis=-1)

    img = np.expand_dims(img, axis=0)

    return img
labels = {
    0: 'ain', 1: 'alef', 2: 'beh', 3: 'dad', 4: 'dal', 5: 'dhad', 6: 'feh', 7: 'ghain',
    8: 'hah', 9: 'heh', 10: 'jeem', 11: 'kaf', 12: 'khah', 13: 'lam', 14: 'meem', 15: 'noon',
    16: 'qaf', 17: 'reh', 18: 'sad', 19: 'seen', 20: 'sheen', 21: 'tah', 22: 'teh', 23: 'thal',
    24: 'theh', 25: 'waw', 26: 'yeh', 27: 'zain'
}
@app.route('/predictImage', methods=['POST'])
def predictImage():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        image = Image.open(file.stream)
        processed_image = preprocess_image_for_cnn(image)
        pred = model.predict(processed_image)
        y=labels[pred[0].argmax()]

        return jsonify({"prediction": y})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__=="__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
