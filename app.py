from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load Model CNN
interpreter = tf.lite.Interpreter(model_path="model_voice_cnn.tflite")
interpreter.allocate_tensors()

@app.route("/")
def home():
    return "Voice Recognition API Active!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # MFCC sudah diproses di ESP32
    input_data = np.array(data).reshape(1, 13, 44, 1)

    prediction = model.predict(input_data)[0]
    label = int(np.argmax(prediction))

    return jsonify({"label": label, "prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
