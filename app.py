from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import io
import soundfile as sf
import sys # PENTING: Untuk logging error

app = Flask(__name__)

# Load Model CNN
try:
    model = tf.keras.models.load_model("model_voice_cnn.h5")
    MODEL_TYPE = "Keras"
except:
    interpreter = tf.lite.Interpreter(model_path="model_voice_cnn.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    MODEL_TYPE = "TFLite"
    print("Menggunakan model TFLite Interpreter.")

# --- KONFIGURASI AUDIO SAMA DENGAN PELATIHAN MODEL ---
SAMPLE_RATE = 16000
MFCC_COUNT = 13
N_FFT = 2048
HOP_LENGTH = 512
TIME_STEPS = 44 

@app.route("/")
def home():
    return f"Voice Recognition API Active! Model: {MODEL_TYPE}"

def extract_mfcc(audio_data):
    # ... (fungsi ini tidak berubah)
    audio_data = audio_data.astype(np.float32)

    target_length = int(SAMPLE_RATE * 1.2)
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        padding = np.zeros(target_length - len(audio_data), dtype=np.float32)
        audio_data = np.concatenate([audio_data, padding])

    mfccs = librosa.feature.mfcc(
        y=audio_data,
        sr=SAMPLE_RATE,
        n_mfcc=MFCC_COUNT,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    if mfccs.shape[1] > TIME_STEPS:
        mfccs = mfccs[:, :TIME_STEPS]
    elif mfccs.shape[1] < TIME_STEPS:
        pad_width = TIME_STEPS - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

    mfccs = np.expand_dims(mfccs.T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    mfccs = mfccs.transpose(0, 2, 1, 3) 
    
    return mfccs

@app.route("/predict", methods=["POST"])
def predict():
    print("--- STARTING PREDICT REQUEST ---", file=sys.stderr) # Logging Awal
    try:
        # 1. Mengakses data mentah dari multipart/form-data
        if 'audio' not in request.files:
            print("ERROR: Missing audio file in request", file=sys.stderr)
            return jsonify({"error": "Missing audio file in request", "code": "E400-01"}), 400
        
        audio_file = request.files['audio']
        
        # Baca data mentah 32-bit (little-endian)
        raw_data = audio_file.read()
        print(f"DEBUG: Read {len(raw_data)} bytes of raw data.") # LOG 1: Cek ukuran data
        
        # Konversi buffer raw data (int32) ke numpy array (int32)
        # ESP32 mengirim 32-bit integer (i4), kita ubah ke numpy array
        audio_data_int32 = np.frombuffer(raw_data, dtype='<i4')
        print(f"DEBUG: Converted to {audio_data_int32.size} samples (expected ~19200).") # LOG 2: Cek jumlah sampel
        
        if audio_data_int32.size == 0:
             print("ERROR: Received empty audio data", file=sys.stderr)
             return jsonify({"error": "Received empty audio data", "code": "E400-02"}), 400

        # 2. Ekstraksi MFCC
        input_data = extract_mfcc(audio_data_int32)
        print("DEBUG: MFCC extracted successfully.") # LOG 3: Pra-pemrosesan sukses
        
        # 3. Prediksi
        if MODEL_TYPE == "Keras":
            prediction = model.predict(input_data)[0]
        else: # TFLite
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        label = int(np.argmax(prediction))
        
        # --- LOGIKA OTORISASI (Sesuaikan dengan label Anda) ---
        authorized_label = 0 
        access_granted = (label == authorized_label)

        print("--- REQUEST SUCCESSFUL, SENDING JSON ---", file=sys.stderr) # Logging Sukses

        return jsonify({
            "access": access_granted,
            "label_id": label,
            "prediction_scores": prediction.tolist(),
            "message": "Access Granted" if access_granted else "Access Denied"
        })

    except Exception as e:
        error_message = f"Critical Error during prediction: {e}"
        print(f"ERROR: {error_message}", file=sys.stderr) # LOG KRITIS: Cetak error ke log Railway
        return jsonify({"error": "Internal Server Error during processing", "details": str(e), "code": "E500"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
