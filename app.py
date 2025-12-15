from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import sys 
import io
import soundfile as sf

try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("Librosa imported successfully.", file=sys.stderr)
except Exception as e:
    LIBROSA_AVAILABLE = False
    print(f"FATAL ERROR: Failed to import librosa. Check requirements.txt. Details: {e}", file=sys.stderr)

app = Flask(__name__)

# Load Model CNN 
try:
    interpreter = tf.lite.Interpreter(model_path="model_voice_cnn.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_tensor_details()
    MODEL_TYPE = "TFLite"
    print("Menggunakan model TFLite Interpreter (model_voice_cnn.tflite).", file=sys.stderr)
except Exception as e:
    print(f"Error loading TFLite model: {e}", file=sys.stderr)

# --- KONFIGURASI AUDIO SAMA DENGAN PELATIHAN MODEL ---
SAMPLE_RATE = 16000
MFCC_COUNT = 40 
N_FFT = 2048
HOP_LENGTH = 512
TIME_STEPS = 50 
GAIN_FACTOR = 5.0
# Authorized Label (Sesuai hasil training: Kelas 1 = Authorized)
AUTHORIZED_LABEL = 1 

@app.route("/")
def home():
    if MODEL_TYPE == "TFLite":
        return f"Voice Recognition API Active! Model: {MODEL_TYPE}, Authorized Label: {AUTHORIZED_LABEL}"
    return "Voice Recognition API Failed to load model."

def extract_mfcc(audio_data):
    if not LIBROSA_AVAILABLE:
        raise ImportError("Librosa is not available on the server.")

    # 1. Konversi dari int32 (ESP32) ke float32 (Librosa)
    audio_data = audio_data.astype(np.float32)

    # 2. **KOREKSI KRITIS: HAPUS DC OFFSET (Mean Subtraction)**
    # Ini diperlukan karena mic ESP32 sering memiliki bias.
    audio_data = audio_data - np.mean(audio_data)

    # 3. **NORMALISASI (Penting untuk Librosa)**
    # Normalisasi amplitudo sinyal ke rentang [-1.0, 1.0]
    # librosa.util.normalize akan mencegah 'clipping' atau sinyal terlalu lemah
    audio_data = librosa.util.normalize(audio_data) 

    # 4. Padding/Cropping Sinyal Mentah ke panjang 1.2 detik (19200 samples)
    target_length = int(SAMPLE_RATE * 1.2)
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        padding = np.zeros(target_length - len(audio_data), dtype=np.float32)
        audio_data = np.concatenate([audio_data, padding])

    # 5. Ekstraksi MFCC
    mfccs = librosa.feature.mfcc(
        y=audio_data,
        sr=SAMPLE_RATE,
        n_mfcc=MFCC_COUNT,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # 6. Penyesuaian dimensi waktu (HARUS 50)
    if mfccs.shape[1] > TIME_STEPS: 
        mfccs = mfccs[:, :TIME_STEPS]
    elif mfccs.shape[1] < TIME_STEPS:
        pad_width = TIME_STEPS - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

    # 7. Reshape untuk input CNN: (1, 40, 50, 1)
    # Transpose (MFCC_COUNT, TIME_STEPS) -> (TIME_STEPS, MFCC_COUNT)
    mfccs = mfccs.T
    # Tambahkan dimensi batch dan channel: (1, TIME_STEPS, MFCC_COUNT, 1)
    mfccs = np.expand_dims(mfccs, axis=0) 
    mfccs = np.expand_dims(mfccs, axis=-1) 
    
    # Transpose final jika model mengharapkan MFCC di sumbu kedua: (1, 40, 50, 1)
    # Ini tergantung format pelatihan model Anda. Kita pertahankan format yang sudah ada 
    # yang berhasil lolos pengecekan shape terakhir: (1, 40, 50, 1)
    mfccs = mfccs.transpose(0, 2, 1, 3) 

    return mfccs

@app.route("/predict", methods=["POST"])
def predict():
    print("--- STARTING PREDICT REQUEST ---", file=sys.stderr) 
    
    if MODEL_TYPE != "TFLite" or not LIBROSA_AVAILABLE:
        error_msg = "Server not initialized correctly (Missing Model or Librosa)."
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return jsonify({"error": error_msg, "code": "E503"}), 503

    raw_data = None
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Missing 'audio' part in multipart request", "code": "E400-01"}), 400
        
        audio_file = request.files['audio']
        raw_data = audio_file.read()
        
        audio_data_int32 = np.frombuffer(raw_data, dtype='<i4')
        
        if audio_data_int32.size == 0:
             return jsonify({"error": "Received empty audio data", "code": "E400-02"}), 400

        # 2. Ekstraksi MFCC
        input_data = extract_mfcc(audio_data_int32)
        print(f"DEBUG: MFCC extracted successfully with final shape {input_data.shape}. Ready for prediction.") 
        
        # 3. Prediksi TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        label = int(np.argmax(prediction))
        
        # --- LOGIKA OTORISASI AKHIR ---
        access_granted = (label == AUTHORIZED_LABEL)

        print(f"DEBUG: Authorized Label is {AUTHORIZED_LABEL}. Model Predicted Label {label}.", file=sys.stderr)
        print(f"--- REQUEST SUCCESSFUL: Label {label} ({'ACCESS' if access_granted else 'DENIED'}) ---", file=sys.stderr)

        return jsonify({
            "access": access_granted,
            "label_id": label,
            "prediction_scores": prediction.tolist(),
            "message": "Access Granted" if access_granted else "Access Denied"
        })

    except Exception as e:
        error_message = f"Critical Error during prediction: {e}"
        print(f"ERROR: {error_message}", file=sys.stderr) 
        
        raw_data_size = len(raw_data) if raw_data is not None else 0
        print(f"DEBUG_INFO: Raw data size at crash: {raw_data_size} bytes.", file=sys.stderr)

        return jsonify({"error": "Internal Server Error during processing", "details": str(e), "code": "E500"}), 500

if __name__ == "__main__":
    AUTHORIZED_LABEL = 1 
    app.run(host="0.0.0.0", port=8000)
