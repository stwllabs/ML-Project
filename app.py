import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="MindGuard AI", page_icon="🧠", layout="wide")

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('mindguard_model.pkl')
        scaler_model = joblib.load('scaler_model.pkl')
        scaler_cluster = joblib.load('scaler_cluster.pkl')
        kmeans = joblib.load('kmeans_cluster.pkl')
        le = joblib.load('label_encoder.pkl')  # ✅ FIX 1: tambah load LabelEncoder
        return model, scaler_model, scaler_cluster, kmeans, le
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None, None, None

model, scaler_model, scaler_cluster, kmeans, le = load_models()

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

    html, body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F8FAFC !important; }

    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3,
    .stApp label, .stApp div, .stApp span {
        color: #1E293B;
    }

    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Form card */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        padding: 40px !important;
        border-radius: 24px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }

    /* Tombol */
    div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        width: 100% !important;
        padding: 12px !important;
    }

    /* Result box */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        margin-top: 20px;
        color: #1E293B;
    }

    /* Slider & selectbox label */
    .stSlider label, .stSelectbox label {
        color: #1E293B !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("<h1 style='text-align: center; color: #1E293B;'>🧠 MindGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Student Well-being Predictive Analysis Dashboard</p>", unsafe_allow_html=True)

# --- 5. MAIN FORM ---
if model is not None:
    with st.form("main_input"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### 📚 Academic & Lifestyle")
            study_h = st.slider("Study Hours (Daily)", 0, 12, 6)
            sleep_h = st.slider("Sleep Hours (Daily)", 0, 12, 7)
            screen_t = st.slider("Screen Time (Hours/Day)", 0, 16, 4)
            social_m = st.slider("Social Media Use (Hours/Day)", 0, 12, 2)
            exercise = st.selectbox("Physical Exercise", options=["Yes", "No"])

        with col2:
            st.markdown("### 🤝 Psychosocial Factors")
            anxiety = st.slider("Anxiety Level (1-10)", 1, 10, 4)
            peer_p = st.slider("Peer Pressure Level (1-10)", 1, 10, 3)
            tuition = st.selectbox("Extra Tuition?", options=["Yes", "No"])
            income = st.selectbox("Family Income Level", options=["Low", "Medium", "High"])
            uni_type = st.selectbox("University Type", options=["National University", "Private University"])

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.form_submit_button("Analyze Mental Well-being")

    # --- 6. PREDICTION LOGIC ---
    if analyze_btn:
        # Mapping input ke angka
        ex_val = 1 if exercise == "Yes" else 0
        tuit_val = 1 if tuition == "Yes" else 0
        inc_map = {"Low": 0, "Medium": 1, "High": 2}[income]
        uni_val = 0 if uni_type == "National University" else 1

        # ✅ FIX 2: Hitung interaction features (sama persis seperti di data_preprocessing.py)
        digital_engagement = screen_t + social_m
        sleep_quality_index = sleep_h + (ex_val * 2)
        academic_load_pressure = study_h * peer_p

        # ✅ FIX 3: Input lengkap 13 fitur sesuai urutan training
        inputs = np.array([[
            study_h, sleep_h, screen_t, social_m,
            anxiety, peer_p, tuit_val, ex_val, inc_map, uni_val,
            digital_engagement, sleep_quality_index, academic_load_pressure
        ]])

        # ✅ FIX 4: Gunakan scaler_model (bukan 'scaler' yang tidak terdefinisi)
        inputs_scaled = scaler_model.transform(inputs)

        # Predict Stress Level
        prediction = model.predict(inputs_scaled)
        label = le.inverse_transform(prediction)[0]

        # ✅ FIX 5: Cluster pakai 3 fitur behavioral + scaler_cluster yang benar
        cluster_input = np.array([[digital_engagement, sleep_quality_index, academic_load_pressure]])
        cluster_input_scaled = scaler_cluster.transform(cluster_input)
        cluster_id = kmeans.predict(cluster_input_scaled)[0]

        # --- DISPLAY RESULT ---
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Hasil Analisis: <span style='color:#2563EB'>{label} Stress</span></h2>", unsafe_allow_html=True)

        if label == 'High':
            st.error("**Rekomendasi Utama:** Tingkat stres kamu terdeteksi tinggi. Pertimbangkan untuk mengurangi beban SKS sementara atau berkonsultasi dengan dosen pembimbing/konselor.")
        elif label == 'Medium':
            st.warning("**Rekomendasi Utama:** Kamu berada di zona moderat. Cobalah untuk lebih rutin berolahraga dan atur jadwal tidur yang lebih konsisten.")
        else:
            st.success("**Rekomendasi Utama:** Kondisi mentalmu sangat stabil! Pertahankan keseimbangan antara akademik dan kehidupan sosialmu.")

        st.divider()
        st.markdown("#### 💡 Personalized Insights (Phase 3: Clustering)")
        if cluster_id == 0:
            st.info("🎯 **Profil Akademik Intensif:** Sistem mendeteksi kamu sangat fokus pada studi. Jangan lupa untuk mengambil jeda istirahat setiap 50 menit belajar.")
        elif cluster_id == 1:
            st.info("📱 **Profil Digital-Heavy:** Penggunaan layar kamu cukup dominan. Cobalah kurangi penggunaan gadget 1 jam sebelum tidur untuk kualitas istirahat lebih baik.")
        else:
            st.info("🏃 **Profil Gaya Hidup Seimbang:** Pola aktivitasmu cenderung stabil. Pertahankan interaksi sosial positif dengan teman sejawat.")

        st.markdown("</div>", unsafe_allow_html=True)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("## 📋 Project Info")
    st.write("MindGuard AI menggunakan **Integrated Framework** (Logistic Regression, SVM, Random Forest) untuk monitoring kesehatan mental mahasiswa.")
    st.divider()
    st.markdown("📊 **Model Accuracy:** 70.00%")
    st.markdown("📂 **Dataset Size:** 1997 samples")
    st.markdown("🎓 **Status:** Phase 3 Complete")