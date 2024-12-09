from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
import tensorflow as tf

app = Flask(__name__)

# Muat model dan semua komponen yang disimpan
model = load_model('model/model_lstm.h5')

with open('model/words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('model/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

with open('model/le.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('model/tokenizers.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Muat data intents
with open('model/data.json', encoding='utf-8') as f:
    intents = json.load(f)

# Parameter untuk padding input
max_length = 14  # Sesuaikan dengan panjang input pada saat pelatihan
threshold = 0.6  # Threshold probabilitas minimal untuk jawaban yang meyakinkan

# Fungsi untuk prediksi kelas dan menghasilkan respons
def predict_class(sentence):
    # Ubah teks menjadi lowercase untuk konsistensi
    sentence = sentence.lower()
    
    # Ubah teks menjadi urutan numerik dengan tokenizer
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)
    
    # Prediksi dengan model LSTM
    prediction = model.predict(padded_sequences)
    class_index = np.argmax(prediction)  # Mendapatkan index kelas dengan probabilitas tertinggi
    confidence = np.max(prediction)  # Probabilitas tertinggi dari prediksi
    
    if confidence < threshold:
        # Jika confidence di bawah threshold, kembalikan 'no_match'
        return None, confidence
    else:
        intent_label = label_encoder.inverse_transform([class_index])[0]  # Mengubah index menjadi label kelas
        return intent_label, confidence

def chatbot_response(msg, time_of_day=None):
    intent, confidence = predict_class(msg)
    if intent is None:
        return ("Maaf, saya belum bisa memahami pertanyaan Anda. Anda bisa mencoba menyusun pertanyaan dengan cara lain, "
                "atau langsung menghubungi kami untuk bantuan lebih lanjut melalui:\n\n"
                "Email: admisi@tarumanagara.ac.id\nWhatsApp: +628117579727\nTelepon: 02156958723.")

    # Jika confidence lebih tinggi dari threshold dan intent adalah 'greeting', sesuaikan sapaan dengan waktu
    if intent == "greeting":
        return f"Selamat {time_of_day}, Tarumabot di sini, siap menjawab pertanyaan kamu seputar Universitas Tarumanagara :)"
    
    # Jika intent bukan greeting, cari respons sesuai di intents.json
    for i in intents['intents']:
        if i['tag'] == intent:
            return np.random.choice(i['responses'])
    
    return "Maaf, saya tidak mengerti. Bisakah Anda mencoba lagi?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chat')
def chat():
    return render_template('chatt.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    time_of_day = request.args.get('timeOfDay')  # Ambil waktu dari parameter
    return jsonify(chatbot_response(userText, time_of_day))

if __name__ == "__main__":
    app.run()
