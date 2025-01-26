import streamlit as st
import numpy as np
import pickle

# Load model dan scaler
with open('lstm_model.pkl', 'rb') as file:
    lstm_model = pickle.load(file)

with open('svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Judul aplikasi
st.title("Klasifikasi Status Gizi Balita")
st.markdown("Masukkan data balita untuk memprediksi status gizinya.")

# Input data dari pengguna
umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, step=1)
jenis_kelamin = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, step=0.1)

# Encode jenis kelamin (Laki-laki = 0, Perempuan = 1)
if jenis_kelamin == "Laki-laki":
    jenis_kelamin_encoded = 0
else:
    jenis_kelamin_encoded = 1

# Buat array input
input_data = np.array([[umur, jenis_kelamin_encoded, tinggi_badan]])

# Preprocessing input menggunakan scaler
scaled_data = scaler.transform(input_data)

# Ubah bentuk data untuk input ke model LSTM
lstm_input = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

# Prediksi menggunakan model LSTM
lstm_features = lstm_model.predict(lstm_input)

# Prediksi akhir menggunakan model SVM
prediction = svm_classifier.predict(lstm_features)

# Interpretasi hasil prediksi
if prediction == 0:
    status_gizi = "Gizi Baik"
elif prediction == 1:
    status_gizi = "Gizi Kurang"
else:
    status_gizi = "Gizi Buruk"

# Tampilkan hasil prediksi
st.subheader("Hasil Prediksi")
st.write(f"Status Gizi Balita: **{status_gizi}**")
