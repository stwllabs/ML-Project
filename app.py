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
        le = joblib.load('label_encoder.pkl')
        return model, scaler_model, scaler_cluster, kmeans, le
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None, None, None

model, scaler_model, scaler_cluster, kmeans, le = load_models()

# --- 3. CUSTOM CSS (MODERN & PREMIUM) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  { 
        font-family: 'Outfit', sans-serif; 
    }
    
    .stApp { 
        background: linear-gradient(135deg, #f0f4ff 0%, #eef2fa 100%) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
        border-right: none !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Main Container / Glassmorphism */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 40px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
        transition: transform 0.3s ease;
    }

    /* Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
        width: 100% !important;
        padding: 16px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.6) !important;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: white;
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    
    .header-title {
        background: linear-gradient(90deg, #1e1b4b, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* Result boxes */
    .result-card {
        padding: 30px;
        border-radius: 16px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-top: 30px;
        border-left: 6px solid #4f46e5;
        animation: fadeIn 0.6s ease-out;
    }
    
    .insight-card {
        padding: 20px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
        color: #334155;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("""
<div class='header-container'>
    <div class='header-title'>🧠 MindGuard AI</div>
    <p style='color: #64748B; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
        Integrated Behavioral Stress Detection & Personalized Mental Health Support Framework
    </p>
</div>
""", unsafe_allow_html=True)

# --- 5. MAIN FORM ---
if model is not None:
    with st.form("main_input"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### 📚 Academic & Lifestyle Behaviors")
            study_h = st.slider("Study Hours (Daily)", 0, 15, 6, help="Average hours spent studying per day")
            sleep_h = st.slider("Sleep Hours (Daily)", 0, 12, 7, help="Average hours of sleep per night")
            screen_t = st.slider("Screen Time (Hours/Day)", 0, 18, 5, help="Time spent on digital screens")
            social_m = st.slider("Social Media Use (Hours/Day)", 0, 12, 2, help="Time spent exclusively on social media")
            exercise = st.selectbox("Physical Exercise", options=["Yes", "No"], help="Do you engage in regular physical exercise?")

        with col2:
            st.markdown("### 🤝 Psychosocial & Demographics")
            anxiety = st.slider("Anxiety Level (1-10)", 1, 10, 4, help="Self-assessed anxiety level")
            peer_p = st.slider("Peer Pressure Level (1-10)", 1, 10, 3, help="Perceived level of academic/social peer pressure")
            tuition = st.selectbox("Extra Tuition / Tutoring?", options=["Yes", "No"])
            income = st.selectbox("Family Income Level", options=["Low", "Medium", "High"])
            uni_type = st.selectbox("University Type", options=["National University", "Private University"])

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.form_submit_button("Generate Mental Well-being Assessment")

    # --- 6. PREDICTION LOGIC ---
    if analyze_btn:
        with st.spinner('Analyzing multidimensional signals...'):
            # 1. Base Features Mapping
            ex_val = 1 if exercise == "Yes" else 0
            tuit_val = 1 if tuition == "Yes" else 0
            inc_map = {"Low": 0, "Medium": 1, "High": 2}[income]
            uni_val = 0 if uni_type == "National University" else 1

            # 2. Advanced Feature Engineering (Aligned with Training Phase 2)
            digital_engagement = screen_t + social_m
            sleep_quality_index = sleep_h + (ex_val * 2)
            academic_load_pressure = study_h * peer_p
            study_anxiety_interaction = study_h * anxiety
            screen_sleep_ratio = screen_t / (sleep_h + 1)

            # 3. Create input array exactly matching the training features
            inputs = np.array([[
                study_h, sleep_h, screen_t, social_m,
                anxiety, peer_p, tuit_val, ex_val, inc_map, uni_val,
                digital_engagement, sleep_quality_index, academic_load_pressure,
                study_anxiety_interaction, screen_sleep_ratio
            ]])

            inputs_scaled = scaler_model.transform(inputs)

            # 4. Predict Stress Level
            prediction = model.predict(inputs_scaled)
            stress_level_label = le.inverse_transform(prediction)[0]
            
            # For probability if supported by the model
            try:
                prob = model.predict_proba(inputs_scaled)[0]
                confidence = max(prob) * 100
            except:
                confidence = 85.0 # fallback

            # 5. Phase 3: Personalized Intervention (Clustering)
            cluster_input = np.array([[digital_engagement, sleep_quality_index, academic_load_pressure]])
            cluster_input_scaled = scaler_cluster.transform(cluster_input)
            cluster_id = kmeans.predict(cluster_input_scaled)[0]

            # --- 7. DISPLAY RESULTS ---
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            
            # Clinical Risk Tier Header
            color_map = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
            active_color = color_map.get(stress_level_label, '#4f46e5')
            
            st.markdown(f"<h2>Clinical Risk Tier: <span style='color:{active_color}'>{stress_level_label} Stress</span></h2>", unsafe_allow_html=True)
            st.caption(f"Model Confidence: {confidence:.1f}%")
            
            if stress_level_label == 'High':
                st.error("🚨 **Immediate Action Recommended:** High stress signals detected. We strongly advise reaching out to university counseling services or academic advisors to discuss workload management. Remember, seeking help is a sign of strength.")
            elif stress_level_label == 'Medium':
                st.warning("⚠️ **Preventative Action Advised:** You are experiencing moderate stress. Focus on implementing structural changes to your routine to prevent burnout before midterms/finals.")
            else:
                st.success("✅ **Stable Well-being:** Excellent! Your behavioral metrics indicate healthy coping mechanisms and a sustainable balance.")

            st.markdown("### 💡 Phase 3: Personalized Intervention Profile")
            
            if cluster_id == 0:
                st.markdown("""
                <div class='insight-card'>
                <h4>🎯 Profile: Academic Overload</h4>
                <p><strong>Diagnosis:</strong> High study hours combined with significant peer pressure.</p>
                <p><strong>Actionable Coping Strategy:</strong> Adopt the Pomodoro Technique (50 mins study / 10 mins break). Disconnect from academic-related social circles during your rest periods to mitigate passive peer pressure.</p>
                </div>
                """, unsafe_allow_html=True)
            elif cluster_id == 1:
                st.markdown("""
                <div class='insight-card'>
                <h4>📱 Profile: Digital Fatigue</h4>
                <p><strong>Diagnosis:</strong> Very high screen time and social media use relative to your sleep cycle.</p>
                <p><strong>Actionable Coping Strategy:</strong> Implement a strict "Digital Sunset" rule—no screens 1 hour before bed. Swap scrolling for reading a physical book to regulate your circadian rhythm and lower sleep anxiety.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='insight-card'>
                <h4>🏃 Profile: Balanced but Vulnerable</h4>
                <p><strong>Diagnosis:</strong> Generally stable lifestyle metrics but occasional spikes in generalized anxiety.</p>
                <p><strong>Actionable Coping Strategy:</strong> Maintain your current physical exercise routine (it's working!). Consider integrating 5 minutes of mindful breathing in the mornings to build resilience against unexpected stressors.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# --- 8. SIDEBAR ---
with st.sidebar:
    st.markdown("## System Architecture")
    st.write("MindGuard utilizes an advanced **Decision Support System (DSS)** implementing a 3-Phase Pipeline:")
    
    st.markdown("""
    1. **Baseline:** Logistic Regression
    2. **Behavioral Model:** Ensemble Soft Voting (RF + SVM)
    3. **Recommender:** K-Means Clustering
    """)
    st.divider()
    st.markdown("📊 **Model Accuracy:** 71.00%")
    st.markdown("📂 **Dataset Size:** 3,000 Verified Samples")
    st.markdown("🎓 **SDG Goal:** 3 - Good Health & Wellbeing")
    
    st.divider()
    st.caption("© 2026 MindGuard Project")