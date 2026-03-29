import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('mindguard_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

model, scaler, le = load_models()

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="MindGuard AI", page_icon="🧠", layout="wide")

# --- 3. CUSTOM CSS (Clean & High Contrast) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    /* Font Global */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, label {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Memaksa Background Utama Terang agar Kontras */
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Sidebar Styling (Dark Theme for Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }

    /* Form Card (White Background) */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        padding: 40px !important;
        border-radius: 24px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Warna Teks di Dalam Form */
    div[data-testid="stForm"] label, div[data-testid="stForm"] p, div[data-testid="stForm"] h3 {
        color: #1E293B !important;
        font-weight: 600 !important;
    }

    /* Button Styling (Solid Blue) */
    div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        width: 100% !important;
        border-radius: 12px !important;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Result Box Custom Style */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #E2E8F0;
        background-color: #F1F5F9;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("<h1 style='text-align: center; color: #1E293B; margin-bottom: 0;'>🧠 MindGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.1rem;'>Student Well-being Predictive Analysis Dashboard</p>", unsafe_allow_html=True)
st.write("##")

# --- 5. MAIN FORM ---
if model is not None:
    with st.form("main_input"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### 🛋️ Lifestyle & Habits")
            sleep = st.select_slider("Jam Tidur Harian", options=list(range(0, 13)), value=7)
            steps = st.number_input("Estimasi Langkah Kaki", 0, 30000, 5000, 500)
            fruits = st.slider("Konsumsi Buah & Sayur (1-5)", 1, 5, 3)
            wlb = st.slider("Work-Life Balance Score (1-10)", 1, 10, 5)
            places = st.number_input("Tempat Dikunjungi (Mingguan)", 0, 20, 2)

        with col2:
            st.markdown("### 🤝 Social & Achievement")
            social = st.slider("Social Network Score (1-10)", 1, 10, 5)
            todo = st.slider("Efektivitas Tugas (1-10)", 1, 10, 6)
            shouting = st.select_slider("Tingkat Emosi/Marah", options=list(range(1, 11)), value=2)
            awards = st.number_input("Penghargaan Diraih", 0, 10, 0)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.form_submit_button("Analisis Kesehatan Mental")

    # --- 6. PREDICTION LOGIC ---
    if analyze_btn:
        # Menyesuaikan dengan 9 features yang kamu gunakan
        features = np.array([[sleep, steps, social, todo, shouting, fruits, wlb, places, awards]])
        
        # Scaling & Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        label = le.inverse_transform(prediction)[0]
        
        # Display Result
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #1E293B; margin-top: 0;'>Hasil Analisis: {label}</h2>", unsafe_allow_html=True)
        
        if label == 'Severe':
            st.error("**Peringatan:** Tingkat stres kamu sangat tinggi. Ambil waktu istirahat sejenak.")
        elif label == 'Moderate':
            st.warning("**Waspada:** Terdeteksi gejala kelelahan mental. Yuk, kurangi multitasking!")
        elif label == 'Normal':
            st.info("**Kondisi Baik:** Kamu berhasil menyeimbangkan kegiatan dengan kesehatan mental.")
        else: # Mild
            st.success("**Sangat Sehat:** Teruskan pola hidup positifmu ini!")
        st.markdown("</div>", unsafe_allow_html=True)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("## 📋 About")
    st.write("MindGuard AI menggunakan **Random Forest Classifier** untuk memprediksi tingkat kesejahteraan mahasiswa berdasarkan pola aktivitas harian.")
    st.divider()
    st.markdown(f"👤 **Developer:** Stella Budi S.")
    st.markdown(f"📊 **Accuracy:** ~53.4%")
    st.markdown(f"📅 **Semester:** 4 - Project ML")