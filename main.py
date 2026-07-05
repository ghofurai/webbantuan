import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi BLT Desa Tegalkodo", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('model_blt_final.pkl')

try:
    model = load_model()
except:
    st.error("Model tidak ditemukan. Pastikan file 'model_blt_final.pkl' ada di folder yang sama.")
    st.stop()

# Header
st.title('📊 Sistem Klasifikasi Kelayakan Penerima BLT')
st.markdown('**Desa Tegalkodo** | Algoritma Gaussian Naïve Bayes')
st.markdown('---')

# Form Input
st.header('📝 Input Data Calon Penerima')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Data Demografi')
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki (1)', 'Perempuan (2)'])
    jk_val = 1 if 'Laki' in jenis_kelamin else 2
    
    pekerjaan = st.number_input('Pekerjaan (1-7)', min_value=1, max_value=7, value=1)
    jumlah_tanggungan = st.number_input('Jumlah Tanggungan', min_value=0, max_value=10, value=2)
    
    status_rumah = st.selectbox('Status Rumah', ['Milik Sendiri (1)', 'Kontrak/Sewa (2)', 'Menumpang (3)'])
    status_rumah_val = int(status_rumah.split('(')[1].replace(')', ''))
    
    jenis_lantai = st.number_input('Jenis Lantai (1-6)', min_value=1, max_value=6, value=2)

with col2:
    st.subheader('Kondisi Rumah')
    jenis_dinding = st.number_input('Jenis Dinding (1-4)', min_value=1, max_value=4, value=3)
    jenis_atap = st.number_input('Jenis Atap (1-5)', min_value=1, max_value=5, value=3)
    sumber_air = st.number_input('Sumber Air (1-4)', min_value=1, max_value=4, value=2)
    
    pkh = st.selectbox('Penerima PKH', ['Tidak (0)', 'Ya (1)'])
    pkh_val = 0 if 'Tidak' in pkh else 1
    
    desil = st.number_input('Desil Ekonomi (0-10)', min_value=0, max_value=10, value=2)

# Tombol Prediksi
st.markdown('---')
if st.button('🔍 Prediksi Kelayakan', type='primary', use_container_width=True):
    
    # Siapkan data input
    input_data = pd.DataFrame({
        'jenis kelamin': [jk_val],
        'pekerjaan': [pekerjaan],
        'jumlah tanggungan': [jumlah_tanggungan],
        'status rumah': [status_rumah_val],
        'jenis lantai': [jenis_lantai],
        'jenis dinding': [jenis_dinding],
        'jenis atap': [jenis_atap],
        'sumber air': [sumber_air],
        'pkh/tidak': [pkh_val],
        'desil': [desil]
    })
    
    # Prediksi
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    
    # Tampilkan Hasil
    st.subheader('Hasil Prediksi:')
    if prediction[0] == 1:
        st.success('✅ **LAYAK** menerima bantuan BLT')
        st.info(f'Probabilitas: Layak = {proba[0][0]:.2%}, Tidak Layak = {proba[0][1]:.2%}')
    else:
        st.error('❌ **TIDAK LAYAK** menerima bantuan BLT')
        st.info(f'Probabilitas: Layak = {proba[0][0]:.2%}, Tidak Layak = {proba[0][1]:.2%}')
