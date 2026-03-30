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
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        kmeans = joblib.load('kmeans_cluster.pkl')
        return model, scaler, le, kmeans
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None, None

model, scaler, le, kmeans = load_models()

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, label { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F8FAFC !important; }
    [data-testid="stSidebar"] { background-color: #1E293B !important; color: #FFFFFF !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    div[data-testid="stForm"] { background-color: #FFFFFF !important; padding: 40px !important; border-radius: 24px !important; border: 1px solid #E2E8F0 !important; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important; }
    div.stButton > button { background-color: #2563EB !important; color: #FFFFFF !important; border-radius: 12px !important; font-weight: 700 !important; width: 100% !important; padding: 12px !important; }
    .result-box { padding: 25px; border-radius: 15px; background-color: #F1F5F9; border: 1px solid #E2E8F0; margin-top: 20px; color: #1E293B; }
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
        # Mapping input ke angka sesuai data_preprocessing.py
        ex_val = 1 if exercise == "Yes" else 0
        tuit_val = 1 if tuition == "Yes" else 0
        inc_map = {"Low": 0, "Medium": 1, "High": 2}[income]
        uni_val = 0 if uni_type == "National University" else 1

        # Buat array input (10 fitur sesuai urutan training)
        inputs = np.array([[study_h, sleep_h, screen_t, social_m, anxiety, peer_p, tuit_val, ex_val, inc_map, uni_val]])
        
        # Scaling
        inputs_scaled = scaler.transform(inputs)
        
        # Predict Stress Level
        prediction = model.predict(inputs_scaled)
        label = le.inverse_transform(prediction)[0]
        
        # Predict Cluster (Recommender System Basis)
        cluster_id = kmeans.predict(inputs_scaled)[0]
        
        # --- DISPLAY RESULT ---
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Hasil Analisis: <span style='color:#2563EB'>{label} Stress</span></h2>", unsafe_allow_html=True)
        
        # Pesan berdasarkan Label
        if label == 'High':
            st.error("**Rekomendasi Utama:** Tingkat stres kamu terdeteksi tinggi. Pertimbangkan untuk mengurangi beban SKS sementara atau berkonsultasi dengan dosen pembimbing/konselor.")
        elif label == 'Medium':
            st.warning("**Rekomendasi Utama:** Kamu berada di zona moderat. Cobalah untuk lebih rutin berolahraga dan atur jadwal tidur yang lebih konsisten.")
        else: # Low
            st.success("**Rekomendasi Utama:** Kondisi mentalmu sangat stabil! Pertahankan keseimbangan antara akademik dan kehidupan sosialmu.")

        # Recommender System berdasarkan Cluster
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
    st.markdown("📊 **Model Accuracy:** 65.75%")
    st.markdown("📂 **Dataset Size:** 1997 samples")
    st.markdown("🎓 **Status:** Phase 3 Complete")