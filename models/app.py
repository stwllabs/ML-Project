import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="DSS Kesehatan Mental Mahasiswa", layout="wide")
st.title("Sistem Deteksi Stres & Intervensi Mahasiswa")
st.write("Aplikasi ini memprediksi tingkat stres berdasarkan kebiasaan Anda dan memberikan rekomendasi intervensi yang dipersonalisasi.")

# ==========================================
# 2. MEMUAT MODEL (CACHE AGAR CEPAT)
# ==========================================
@st.cache_resource
def load_models():
    # Load model klasifikasi (bisa diganti ke 'models/baseline_lr_model.pkl' jika ingin pakai Regresi Logistik)
    classifier = joblib.load('advanced_svm_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    
    # Load model clustering (dengan 3 fitur)
    kmeans = joblib.load('kmeans_model.pkl')
    scaler_cluster = joblib.load('scaler_cluster.pkl')
    
    return classifier, scaler, feature_cols, kmeans, scaler_cluster

classifier, scaler, feature_cols, kmeans, scaler_cluster = load_models()

# ==========================================
# 3. ANTARMUKA INPUT PENGGUNA (SIDEBAR)
# ==========================================
st.sidebar.header("Masukkan Data Perilaku Anda")

# Membagi input ke dalam beberapa kategori agar rapi
with st.sidebar.expander("Data Akademik", expanded=True):
    study_hours = st.number_input("Jam Belajar / Minggu", min_value=0, max_value=100, value=10)
    class_attendance = st.number_input("Tingkat Kehadiran Kelas (%)", min_value=0, max_value=100, value=80)
    assignment_load = st.slider("Beban Tugas (1-10)", min_value=1, max_value=10, value=5)
    exam_frequency = st.number_input("Frekuensi Ujian / Bulan", min_value=0, max_value=20, value=2)

with st.sidebar.expander("Gaya Hidup & Kesejahteraan", expanded=True):
    sleep_hours = st.number_input("Jam Tidur / Hari", min_value=0, max_value=24, value=7)
    screen_time = st.number_input("Waktu Layar (Jam/Hari)", min_value=0, max_value=24, value=6)
    social_media_use = st.number_input("Penggunaan Medsos (Jam/Hari)", min_value=0, max_value=24, value=3)
    physical_exercise = st.selectbox("Olahraga Fisik?", ["Yes", "No"])
    anxiety_level = st.slider("Tingkat Kecemasan (1-10)", min_value=1, max_value=10, value=3)
    peer_pressure = st.slider("Tekanan Teman Sebaya (1-10)", min_value=1, max_value=10, value=3)

with st.sidebar.expander("Profil Demografis", expanded=False):
    age = st.number_input("Usia", min_value=16, max_value=50, value=20)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    tuition = st.selectbox("Menerima Beasiswa / Bantuan SPP?", ["Yes", "No"])
    family_income = st.selectbox("Tingkat Pendapatan Keluarga", ["Low", "Medium", "High"])
    family_support = st.slider("Dukungan Keluarga (1-10)", min_value=1, max_value=10, value=7)
    university_type = st.selectbox("Jenis Universitas", ["National University", "Private University"])

# ==========================================
# 4. PROSES PREDIKSI & REKOMENDASI
# ==========================================
if st.sidebar.button("Analisis Stres Saya", type="primary"):
    
    # a. Mempersiapkan Data Input
    input_dict = {
        'Age': age, 'Gender': gender, 'Study_Hours': study_hours,
        'Class_Attendance': class_attendance, 'Tuition': tuition,
        'Exam_Frequency': exam_frequency, 'Assignment_Load': assignment_load,
        'Sleep_Hours': sleep_hours, 'Physical_Exercise': physical_exercise,
        'Social_Media_Use': social_media_use, 'Screen_Time': screen_time,
        'Family_Income_Level': family_income, 'Peer_Pressure': peer_pressure,
        'Family_Support': family_support, 'Anxiety_Level': anxiety_level,
        'University_Type': university_type
    }
    input_df = pd.DataFrame([input_dict])
    
    # Encoding input pengguna agar sesuai dengan model
    input_encoded = pd.get_dummies(input_df)
    
    # Menyesuaikan kolom dengan fitur saat training (penting untuk menghindari error)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]
    
    # Scaling data untuk klasifikasi
    input_scaled = scaler.transform(input_encoded)
    
    # b. Prediksi Tingkat Stres (Klasifikasi Fase 1/2)
    stress_pred = classifier.predict(input_scaled)[0]
    
    # c. Ekstraksi Fitur untuk Clustering (Fase 3) - HANYA 3 FITUR
    cluster_input = input_df[['Screen_Time', 'Assignment_Load', 'Sleep_Hours']]
    cluster_scaled = scaler_cluster.transform(cluster_input)
    cluster_id = kmeans.predict(cluster_scaled)[0]
    
    # Logika Rekomendasi (Sesuai Revisi Terakhir)
    def get_recommendation(cid):
        if cid == 0:
            return "Kategori: Kurang Tidur & Paparan Layar Tinggi\n\nIntervensi: Jam tidur Anda sangat kurang diiringi waktu layar yang tinggi. Saran: Kurangi menatap layar perangkat 1-2 jam sebelum tidur. Prioritaskan perbaikan 'Sleep Hygiene' agar mencapai minimal 6-7 jam tidur untuk menghindari kelelahan fisik (burnout)."
        elif cid == 1:
            return "Kategori: Gaya Hidup Akademik Seimbang\n\nIntervensi: Hebat! Anda memiliki manajemen waktu layar yang cukup baik dan jam tidur yang ideal. Saran: Pertahankan rutinitas sehat ini. Teruskan manajemen waktu yang baik antara istirahat dan beban tugas."
        elif cid == 2:
            return "Kategori: Jam Tidur Tinggi & Waktu Layar Berlebih\n\nIntervensi: Meskipun Anda memiliki jam tidur yang sangat cukup, waktu layar Anda sangat tinggi. Saran: Terapkan 'Digital Detox' secara berkala. Perbanyak aktivitas fisik atau kegiatan di luar ruangan untuk mencegah gaya hidup sedenter (kurang gerak) dan menjaga kesehatan mata."
        return "Hubungi layanan dukungan mahasiswa."
    
# ==========================================
# 5. MENAMPILKAN HASIL
# ==========================================
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Hasil Deteksi Stres")
        if stress_pred == "High":
            st.error(f"Tingkat Stres Terdeteksi: **{stress_pred}**")
        elif stress_pred == "Medium":
            st.warning(f"Tingkat Stres Terdeteksi: **{stress_pred}**")
        else:
            st.success(f"Tingkat Stres Terdeteksi: **{stress_pred}**")
            
    with col2:
        st.subheader("💡 Intervensi Personalisasi")
        st.info(get_recommendation(cluster_id))
        
    st.caption("Sistem ini dioptimalkan berdasarkan pengelompokan perilaku (Behavioral Profiling).")