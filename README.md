# Handwritten Arabic Letters Classification using Classic CNN 🧠🇸🇦

This project focuses on classifying **handwritten Arabic letters** using a custom-built **Convolutional Neural Network (CNN)** model implemented with **TensorFlow and Keras**.

## 🧩 Project Structure

- `ClassicCnnTraining.ipynb`: Jupyter notebook used to train the CNN model.
- `TestClassicCNN.ipynb`: Notebook for evaluating the model on test images.
- `main.py`: Flask API that accepts image files and returns the predicted Arabic character.
- `RunServer.py`: Alternative server file (used during testing); may include interface logic or experimental code.

## 🧠 Model Information

- **Model Name**: `ClassicCNN_v1_28_0.914.keras`
- **Accuracy Achieved**: **91.4%**
- **Input Size**: **32x32 grayscale images**
- **Output**: One of the 28 Arabic characters (e.g., **alef**, **ba**, **jeem**, ...)

## 🚀 How to Run

### 1. Install Requirements
Make sure to install the necessary Python packages:
```bash
pip install tensorflow flask pillow matplotlib opencv-python
python main.py
This will launch a local server at http://127.0.0.1:5001/
Send a POST request with an image:

curl -X POST -F image=@sample.jpg http://127.0.0.1:5001/predictImage
You will receive a JSON response: { "prediction": "jeem" }
## 🔤 Arabic Letters Classes
The model can classify the following 28 Arabic letters:
ain, alef, beh, dad, dal, dhad, feh, ghain, hah, heh,
jeem, kaf, khah, lam, meem, noon, qaf, reh, sad, seen,
sheen, tah, teh, thal, theh, waw, yeh, zain

# 📝 Notes
Images are preprocessed to grayscale, 32x32, inverted, and normalized.

The model was trained on a labeled dataset of handwritten Arabic characters.

# Handwritten Arabic Letters Classification using CNN

This project classifies handwritten Arabic letters using a CNN model. It is trained on 28 Arabic characters and achieves 91.4% accuracy. The project uses TensorFlow, Keras for model training and Flask to create an API for predicting letters from image input.

**Developed by Mohammed Hameed**
