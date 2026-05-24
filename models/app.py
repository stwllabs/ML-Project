import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="MindSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── TOKENS ─────────────────────────────────────── */
:root {
  --bg:      #0f1117;
  --card:    #161b22;
  --sidebar: #0d1117;
  --primary: #7c3aed;
  --accent:  #8b5cf6;
  --text:    #f0f2f8;
  --muted:   #8b92a5;
  --border:  rgba(255,255,255,0.07);
  --danger:  #ef4444;
  --warning: #f59e0b;
  --success: #10b981;
  --r: 10px;
  --t: .15s ease;
}

/* ── BASE ────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at bottom right, #233842 0%, #1b163b 40%, #130b26 80%) !important;
    background-attachment: fixed !important;
}

.gradient-text {
  background: linear-gradient(90deg, #8b5cf6 0%, #c084fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  color: transparent;
  font-weight: 800;
}

.hero-badge {
  background: linear-gradient(135deg, rgba(124,58,237,.2), rgba(192,132,252,.2)) !important;
  border: 1px solid rgba(124,58,237,.4) !important;
}
            
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: absolute !important;
    right: 10px !important;
    top: 10px !important;
    z-index: 9999 !important;
}
            

/* ── SIDEBAR ─────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
  background: var(--sidebar)!important;
  border-right: 1px solid var(--border)!important; padding-top: 0px !important; margin-top: 0px !important;
}

/* ── SIDEBAR WIDGETS ─────────────────────────────── */
[data-testid="stSidebar"] label {
  font-size:.72rem!important; font-weight:600!important;
  text-transform:uppercase!important; letter-spacing:.05em!important;
  color:var(--muted)!important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] {
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background:#1c2130!important; border:1px solid var(--border)!important;
  border-radius:6px!important; color:var(--text)!important; cursor: pointer !important
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background:#1c2130!important; border:1px solid var(--border)!important;
  border-radius:var(--r)!important; margin-bottom:6px!important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-size:.84rem!important; font-weight:600!important; color:var(--text)!important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background:var(--primary)!important; border:none!important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [class*="StyledTrack"] div:first-child {
  background:#3a289c!important;
}
[data-testid="stSidebar"] hr { border-color:var(--border)!important; }
            

/* ── MODEL SELECTOR CARD ─────────────────────────── */
.model-selector {
  background: rgba(124,58,237,.08);
  border: 1px solid rgba(124,58,237,.2);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 14px;
}
.model-selector-title {
  font-size: .68rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: var(--accent);
  margin-bottom: 8px;
}
/* Radio buttons inside model selector */
.model-selector [data-testid="stRadio"] label {
  color: var(--text) !important;
  font-size: .80rem !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
.model-selector [data-testid="stRadio"] > div {
  gap: 4px !important;
}
            

/* ── BUTTON ──────────────────────────────────────── */
.stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button {
  width:100%!important; background:#3b2e85!important;
  color:#fff!important; border:none!important; border-radius:8px!important;
  padding:10px 18px!important; font-size:.88rem!important; font-weight:600!important;
  transition:opacity var(--t)!important; margin-top: -10px !important; margin-bottom: 10px !important;
}

.stDownloadButton > button {
  background:transparent!important; border:1px solid var(--border)!important;
  border-radius:6px!important; color:var(--accent)!important;
  font-size:.82rem!important; font-weight:500!important;
  transition:border-color var(--t)!important;
}
.stDownloadButton > button:hover { border-color:var(--accent)!important; }
            

/* ── METRIC ──────────────────────────────────────── */
[data-testid="stMetric"] {
  background:var(--card)!important; border:1px solid var(--border)!important;
  border-radius:var(--r)!important; padding:14px 18px!important;
}
[data-testid="stMetricLabel"]  { font-size:.68rem!important; font-weight:600!important; text-transform:uppercase!important; letter-spacing:.06em!important; color:var(--muted)!important; }
[data-testid="stMetricValue"]  { font-size:1.5rem!important; font-weight:700!important; color:var(--text)!important; }
            

/* ── EXPANDER (main) ─────────────────────────────── */
[data-testid="stExpander"] {
  background:var(--card)!important; border:1px solid var(--border)!important;
  border-radius:var(--r)!important;
}
[data-testid="stExpander"] summary { color:var(--text)!important; }
            

/* ── DIVIDER ─────────────────────────────────────── */
[data-testid="stDivider"] hr { background:var(--border)!important; border:none!important; height:1px!important; }
            

/* ── DATAFRAME ───────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius:var(--r)!important; border:1px solid var(--border)!important; overflow:hidden!important; }
            

/* ── SCROLLBAR ───────────────────────────────────── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:99px; }
            

/* ── CUSTOM COMPONENTS ───────────────────────────── */
/* Hero */
.hero { text-align:center; padding:3rem 1rem 2rem; }
.hero-badge {
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(124,58,237,.12); border:1px solid rgba(124,58,237,.25);
  border-radius:99px; padding:4px 14px;
  font-size:.70rem; font-weight:700; letter-spacing:.07em;
  color:white; text-transform:uppercase; margin-bottom:16px;
}
.hero-title { font-size:clamp(1.7rem,3.2vw,2.5rem); font-weight:800; letter-spacing:-.03em; color:var(--text); margin-bottom:10px; line-height:1.15; }
.hero-title .hl { color:var(--accent); }
.hero-sub   { font-size:.94rem; color:var(--muted); max-width:480px; margin:0 auto 1.6rem; line-height:1.65; }
            

/* ── Info cards row ── */
.info-cards { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:0; }
.info-card  {
  flex:1; min-width:170px; max-width:220px;
  border-radius:10px; padding:14px 16px; text-align:left;
  transition:transform .15s;
}
.info-card:hover { transform:translateY(-2px); }
.info-card-icon { font-size:1.3rem; margin-bottom:7px; display:block; }
.info-card-name { font-size:.80rem; font-weight:700; line-height:1.2; margin-bottom:2px; }
.info-card-type { font-size:.62rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; opacity:.7; margin-bottom:6px; }
.info-card-desc { font-size:.72rem; color:var(--muted); line-height:1.55; }
            

/* Section label */
.slabel {
  font-size:.7rem; font-weight:700; letter-spacing:.10em; text-transform:uppercase;
  color:var(--muted); margin-bottom:10px;
  display:flex; align-items:center; gap:8px;
}
.slabel::after { content:''; flex:1; height:1px; background:var(--border); }
            

/* Card */
.card {
  background:var(--card); border:1px solid var(--border);
  border-radius:var(--r); padding:20px 22px; margin-bottom:14px;
}
            

/* Result */
.res-row  { display:flex; align-items:center; gap:14px; margin-bottom:16px; }
.res-orb  { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex-shrink:0; }
.orb-high   { background:rgba(239,68,68,.15); }
.orb-medium { background:rgba(245,158,11,.15); }
.orb-low    { background:rgba(16,185,129,.15); }
.res-level  { font-size:1.4rem; font-weight:800; letter-spacing:-.02em; }
.res-sub    { font-size:.78rem; color:var(--muted); margin-top:2px; }
            

/* Badge */
.badge { display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:99px; font-size:.70rem; font-weight:700; letter-spacing:.04em; margin-top:6px; }
.b-high   { background:rgba(239,68,68,.12);  color:#f87171; border:1px solid rgba(239,68,68,.22); }
.b-medium { background:rgba(245,158,11,.12); color:#fbbf24; border:1px solid rgba(245,158,11,.22); }
.b-low    { background:rgba(16,185,129,.12); color:#34d399; border:1px solid rgba(16,185,129,.22); }

.model-pill {
  display:inline-flex; align-items:center; gap:5px;
  background:rgba(124,58,237,.15); border:1px solid rgba(124,58,237,.3);
  border-radius:99px; padding:2px 9px;
  font-size:.65rem; font-weight:700; letter-spacing:.05em;
  color:#a78bfa; margin-bottom:6px;
}

            
/* Prob bars */
.prob-row { display:flex; align-items:center; gap:10px; margin-bottom:7px; }
.prob-lbl { font-size:.72rem; font-weight:600; color:var(--muted); width:60px; flex-shrink:0; }
.prob-bg  { flex:1; height:5px; background:rgba(255,255,255,.06); border-radius:99px; overflow:hidden; }
.prob-fill { height:100%; border-radius:99px; }
.pf-high   { background:var(--danger); }
.pf-medium { background:var(--warning); }
.pf-low    { background:var(--success); }
.prob-val  { font-size:.68rem; color:var(--muted); width:30px; text-align:right; flex-shrink:0; }

            
/* Rec card */
.rec-card  { background:rgba(124,58,237,.08); border:1px solid rgba(124,58,237,.18); border-radius:var(--r); padding:18px 20px; margin-bottom:12px; }
.rec-cat   { font-size:.63rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:5px; }
.rec-title { font-size:.94rem; font-weight:700; color:var(--text); margin-bottom:7px; }
.rec-body  { font-size:.83rem; color:var(--muted); line-height:1.65; }

            
/* Alert */
.alert { border-radius:8px; padding:12px 16px; margin-bottom:12px; font-size:.82rem; line-height:1.6; }
.al-t  { font-weight:700; font-size:.72rem; margin-bottom:3px; }
.a-high   { background:rgba(239,68,68,.07);  border:1px solid rgba(239,68,68,.18); color:#fca5a5; }
.a-medium { background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.18); color:#fcd34d; }
.a-low    { background:rgba(16,185,129,.07); border:1px solid rgba(16,185,129,.18); color:#6ee7b7; }
.a-high   .al-t { color:#f87171; }
.a-medium .al-t { color:#fbbf24; }
.a-low    .al-t { color:#34d399; }

            
/* Chips */
.chip { display:inline-flex; background:rgba(255,255,255,.05); border:1px solid var(--border); border-radius:6px; padding:5px 11px; font-size:.77rem; color:var(--muted); font-weight:500; margin:3px 3px 3px 0; }

/* Cluster strip */
.cl-strip { display:flex; align-items:center; gap:12px; background:rgba(16,185,129,.06); border:1px solid rgba(16,185,129,.15); border-radius:8px; padding:10px 14px; margin-bottom:12px; }
.cl-icon  { width:28px; height:28px; border-radius:6px; background:rgba(139,92,246,.25); display:flex; align-items:center; justify-content:center; font-size:.85rem; flex-shrink:0; }
.cl-lbl   { font-size:.62rem; font-weight:700; color:#34d399; text-transform:uppercase; letter-spacing:.06em; }
.cl-val   { font-size:.84rem; font-weight:600; color:var(--text); }

            
/* ── STEPS ──────────────────────────────── */
.steps { display:flex; align-items:flex-start; margin-bottom:2rem; }
.step  { flex:1; text-align:center; position:relative; }
            
.step::after {
  content:''; position:absolute;
  top:18px; left:calc(50% + 20px);
  width:calc(100% - 40px); height:1px;
  background:var(--border);
}
.step:last-child::after { display:none; }
            

/* dot states */
.step-dot {
  width:36px; height:36px; border-radius:50%;
  background:var(--card); border:1px solid var(--border);
  display:flex; align-items:center; justify-content:center;
  font-size:.95rem; font-weight:700; color:var(--muted);
  margin:0 auto 6px; position:relative; z-index:1;
  transition: background .2s, border-color .2s;
}
.step-dot.done   { background:var(--success); border-color:transparent; color:#fff; font-size:.85rem; }
.step-dot.active { background:var(--primary); border-color:transparent; color:#fff; }
.step-lbl { font-size:.78rem; font-weight:600; color:var(--muted); line-height:1.3; }
.step-lbl.done-lbl   { color:var(--success); }
.step-lbl.active-lbl { color:var(--accent); }
            

/* Empty */
.empty { text-align:center; padding:4rem 1rem; }
.empty-ico { font-size:2.8rem; display:block; margin-bottom:14px; }
.empty-h { font-size:1.1rem; font-weight:700; color:var(--text); margin-bottom:8px; }
.empty-s { font-size:.86rem; color:var(--muted); max-width:340px; margin:0 auto; line-height:1.6; }
            

/* Brand (sidebar) */
.brand { 
  display: flex !important; 
  align-items: center !important;
  justify-content: flex-start !important; 
  gap: 16px !important;
  padding: 0px 10px 12px 10px !important;
  margin: -50px 0px 20px 0px !important; 
  border-bottom: 1px solid var(--border) !important;
}
.brand-ico { 
  width: 44px !important; height: 44px !important; border-radius: 10px !important; 
  background: var(--border) !important; display: flex !important; 
  align-items: center !important; justify-content: center !important; 
  font-size: 1.5rem !important; flex-shrink: 0 !important; 
}
.brand-name { font-size:1.2rem; font-weight:700; color:var(--text); }
.brand-sub  { font-size:.90rem; color:var(--muted); }
            

/* Note */
.note { margin-top:12px; padding:9px 12px; background:rgba(124,58,237,.08); border-radius:7px; border-left:2px solid #5240b8; font-size:.80rem; color:white; line-height:1.6; }

            
/* Footer */
.footer { font-size:.68rem; color:var(--muted); text-align:center; margin-top:2.5rem; padding-top:1.25rem; border-top:1px solid var(--border); line-height:1.8; opacity:.7; }
</style>""", unsafe_allow_html=True)


# ── MODEL LOADING ─────────────────────────────────────────────
MODEL_META = {
    "SVM": {
        "icon": "⚡",
        "name": "Support Vector Machine",
        "type": "Supervised · Klasifikasi",
        "desc": "Memisahkan kelas menggunakan hyperplane optimal. Tangguh terhadap data berdimensi tinggi.",
        "color": "#8b5cf6",
        "bg":    "rgba(139,92,246,.10)",
        "border":"rgba(139,92,246,.25)",
    },
    "Random Forest": {
        "icon": "🌲",
        "name": "Random Forest",
        "type": "Ensemble · Klasifikasi",
        "desc": "Menggabungkan banyak decision tree untuk prediksi yang stabil dan tahan overfitting.",
        "color": "#10b981",
        "bg":    "rgba(16,185,129,.10)",
        "border":"rgba(16,185,129,.25)",
    },
    "Logistic Regression": {
        "icon": "📈",
        "name": "Logistic Regression",
        "type": "Linear · Klasifikasi",
        "desc": "Model probabilistik linear yang cepat dan mudah diinterpretasi untuk prediksi biner/multi-kelas.",
        "color": "#f59e0b",
        "bg":    "rgba(245,158,11,.10)",
        "border":"rgba(245,158,11,.25)",
    },
}

@st.cache_resource
def load_models():
    base = Path(__file__).parent
    try:
        clf_lr  = joblib.load(base / 'baseline_lr_model.pkl')
        clf_svm = joblib.load(base / 'advanced_svm_model.pkl')
        clf_rf  = joblib.load(base / 'advanced_rf_model.pkl')

        scl   = joblib.load(base / 'scaler.pkl')
        fcols = joblib.load(base / 'feature_columns.pkl')
        km    = joblib.load(base / 'kmeans_model.pkl')
        sclk  = joblib.load(base / 'scaler_cluster.pkl')
    except FileNotFoundError as e:
        st.error(f"❌ Model tidak ditemukan: **{e.filename}**")
        st.stop()
    return clf_lr, clf_svm, clf_rf, scl, fcols, km, sclk

clf_lr, clf_svm, clf_rf, scaler, feature_cols, kmeans, scaler_cluster = load_models()

all_models = {
    "SVM":                 clf_svm,
    "Random Forest":       clf_rf,
    "Logistic Regression": clf_lr,
}

# ── HELPERS ───────────────────────────────────────────────────
def stress_meta(pred):
    return {
        "High":   ("🔥", "Tinggi", "#f87171", "b-high", "orb-high", "a-high", "🚨 Perhatian", "Stres terdeteksi <strong>tinggi</strong>."),
        "Medium": ("⚠️", "Sedang", "#fbbf24", "b-medium", "orb-medium", "a-medium", "⚡ Waspada", "Stres <strong>sedang</strong>."),
        "Low":    ("✅", "Rendah", "#34d399", "b-low", "orb-low", "a-low", "✨ Status Sehat", "Stres <strong>rendah</strong>.")
    }.get(pred, ("❓", pred, "#f0f2f8", "b-low", "orb-low", "a-low", "ℹ️ Info", ""))

RECS = {
    0: {"cat":"Pola Tidur & Layar","title":"Kurang Tidur & Layar Tinggi","icon":"😴",
        "body":"Jam tidur kurang + waktu layar tinggi adalah pemicu utama burnout. Kurangi layar 1–2 jam sebelum tidur dan buat jadwal tidur konsisten minimal 6–7 jam.",
        "tips":["🌙 Sleep hygiene rutin","📵 No-screen 1 jam sebelum tidur","⏰ Jadwal tidur konsisten"]},
    1: {"cat":"Keseimbangan Akademik","title":"Gaya Hidup Akademik Seimbang","icon":"⚖️",
        "body":"Pola manajemen waktu Anda sudah sehat. Pertahankan ritme ini dan tambahkan jeda aktif (Pomodoro) di setiap sesi belajar.",
        "tips":["⏱️ Teknik Pomodoro","🏃 Olahraga 3×/minggu","🧘 Mindfulness 10 mnt/hari"]},
    2: {"cat":"Gaya Hidup Digital","title":"Tidur Cukup, Layar Berlebih","icon":"📱",
        "body":"Tidur sudah ideal tapi waktu layar sangat tinggi. Terapkan Digital Detox terjadwal dan perbanyak aktivitas fisik.",
        "tips":["🌿 Digital Detox 1×/minggu","🚶 Jalan kaki 30 mnt/hari","👁️ Aturan 20-20-20 untuk mata"]},
}

def get_rec(cid):
    return RECS.get(cid, {"cat":"Umum","title":"Konsultasikan dengan Konselor","icon":"🩺","body":"Hubungi layanan dukungan mahasiswa.","tips":["🏥 Kunjungi klinik kampus"]})

def make_report(inp, pred, prob_map, cid, model_name):
    rec = get_rec(cid)
    L = ["="*50, f"  MindSense AI — Laporan Deteksi Stres", f"  Model: {model_name}", "="*50, f"\nTINGKAT STRES : {pred}"]
    if prob_map:
        L.append("\nPROBABILITAS:")
        for k, v in prob_map.items():
            L.append(f"  {k:<8} {'█'*int(v*20)}{'░'*(20-int(v*20))}  {v:.2%}")
    L += [f"\nKLUSTER : {rec['title']}", f"\nREKOMENDASI:\n{rec['body']}", "\nTIPS:", *[f"  • {t}" for t in rec["tips"]], "-"*50, "INPUT:"]
    for k, v in inp.items(): L.append(f"  {k:<26}: {v}")
    L += ["="*50, "  ⚠️  Bersifat indikatif, bukan pengganti profesional.", "="*50]
    return "\n".join(L)

def steps_html(state, model_label="Model"):
    """
    state: 'idle' | 'done'
    3 steps: Input → Prediksi (model name) → Rekomendasi
    """
    if state == "idle":
        s1 = ("done",   "done-lbl",   "✓")
        s2 = ("",       "",           "2")
        s3 = ("",       "",           "3")
    else:
        s1 = ("done",   "done-lbl",   "✓")
        s2 = ("done",   "done-lbl",   "✓")
        s3 = ("active", "active-lbl", "✓")

    return f"""<div class="steps">
  <div class="step">
    <div class="step-dot {s1[0]}">{s1[2]}</div>
    <div class="step-lbl {s1[1]}">Input</div>
  </div>
  <div class="step">
    <div class="step-dot {s2[0]}">{s2[2]}</div>
    <div class="step-lbl {s2[1]}">{model_label}</div>
  </div>
  <div class="step">
    <div class="step-dot {s3[0]}">{s3[2]}</div>
    <div class="step-lbl {s3[1]}">Rekomendasi</div>
  </div>
</div>"""

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="brand">
      <div class="brand-ico">🧠</div>
      <div><div class="brand-name">MindSense AI</div><div class="brand-sub">Stress Intelligence</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Model selector ──
    st.markdown('<div class="slabel">🔮 Pilih Model</div>', unsafe_allow_html=True)
    selected_model_key = st.radio(
        "Model Klasifikasi",
        ["SVM", "Random Forest", "Logistic Regression"],
        label_visibility="collapsed",
    )
    selected_model_short = str(selected_model_key)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="slabel">⚙️ Data Input</div>', unsafe_allow_html=True)

    with st.expander("📚 Akademik", expanded=True):
        study_hours      = st.number_input("Jam Belajar / Minggu",    0, 100, 10)
        class_attendance = st.number_input("Kehadiran Kelas (%)",      0, 100, 80)
        assignment_load  = st.slider("Beban Tugas (1–10)",             1, 10,  5)
        exam_frequency   = st.number_input("Frekuensi Ujian / Bulan", 0, 20,  2)

    with st.expander("🌿 Gaya Hidup", expanded=True):
        sleep_hours       = st.number_input("Jam Tidur / Hari",       0, 24, 7)
        screen_time       = st.number_input("Waktu Layar (Jam/Hari)", 0, 24, 6)
        social_media_use  = st.number_input("Media Sosial (Jam/Hari)",0, 24, 3)
        physical_exercise = st.selectbox("Rutin Olahraga?",           ["Yes","No"])
        anxiety_level     = st.slider("Kecemasan (1–10)",             1, 10, 3)
        peer_pressure     = st.slider("Tekanan Sebaya (1–10)",        1, 10, 3)

    with st.expander("👤 Profil", expanded=True):
        age             = st.number_input("Usia",                16, 50, 20)
        gender          = st.selectbox("Jenis Kelamin",          ["Male","Female"])
        tuition         = st.selectbox("Penerima Beasiswa?",     ["Yes","No"])
        family_income   = st.selectbox("Pendapatan Keluarga",    ["Low","Medium","High"])
        family_support  = st.slider("Dukungan Keluarga (1–10)",  1, 10, 7)
        university_type = st.selectbox("Jenis Universitas",      ["National University","Private University"])

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Analisis Stres Saya", type="primary", use_container_width=True)
    st.markdown('<div class="note">⚠️ Hasil bersifat indikatif.<br>Konsultasikan dengan profesional.</div>', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
mm = MODEL_META[selected_model_key]
st.markdown(f"""<div class="hero">
  <div class="hero-badge">🧠 AI-Powered Mental Health</div>
  <div class="hero-title">Deteksi Stres &amp; <span class="gradient-text">Intervensi Cerdas</span><br>untuk Mahasiswa</div>
  <div class="hero-sub">Platform ML untuk mendeteksi tingkat stres dan memberikan rekomendasi intervensi personal berdasarkan 16 fitur perilaku.</div>

  <div class="info-cards">
    <div class="info-card" style="background:{mm['bg']};border:1px solid {mm['border']};">
      <span class="info-card-icon">{mm['icon']}</span>
      <div class="info-card-name" style="color:{mm['color']}">{mm['name']}</div>
      <div class="info-card-type" style="color:{mm['color']}">Model Aktif · {mm['type']}</div>
      <div class="info-card-desc">{mm['desc']}</div>
    </div>
    <div class="info-card" style="background:rgba(99,102,241,.10);border:1px solid rgba(99,102,241,.25);">
      <span class="info-card-icon">🔵</span>
      <div class="info-card-name" style="color:#818cf8">K-Means Clustering</div>
      <div class="info-card-type" style="color:#818cf8">Unsupervised · 3 Kluster</div>
      <div class="info-card-desc">Mengelompokkan mahasiswa ke 3 profil perilaku: kurang tidur, seimbang, dan layar berlebih.</div>
    </div>
    <div class="info-card" style="background:rgba(255,255,255,.04);border:1px solid var(--border);">
      <span class="info-card-icon">📊</span>
      <div class="info-card-name" style="color:var(--text)">16 Fitur Input</div>
      <div class="info-card-type" style="color:var(--muted)">Akademik · Gaya Hidup · Profil</div>
      <div class="info-card-desc">Meliputi jam tidur, beban tugas, kecemasan, dukungan keluarga, dan kebiasaan digital.</div>
    </div>
    <div class="info-card" style="background:rgba(255,255,255,.04);border:1px solid var(--border);">
      <span class="info-card-icon">🎯</span>
      <div class="info-card-name" style="color:var(--text)">3 Label Output</div>
      <div class="info-card-type" style="color:var(--muted)">Low · Medium · High</div>
      <div class="info-card-desc">Tingkat stres diprediksi dalam tiga kategori disertai probabilitas dan rekomendasi personal.</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────
if not run:
    st.markdown(steps_html("idle", selected_model_short), unsafe_allow_html=True)
    st.markdown("""<div class="empty">
      <span class="empty-ico">🔬</span>
      <div class="empty-h">Siap untuk Menganalisis</div>
      <div class="empty-s">Lengkapi data di panel kiri, pilih model, lalu klik <strong>Analisis Stres Saya</strong>.</div>
    </div>""", unsafe_allow_html=True)

else:
    inp = {'Age':age,'Gender':gender,'Study_Hours':study_hours,'Class_Attendance':class_attendance,
           'Tuition':tuition,'Exam_Frequency':exam_frequency,'Assignment_Load':assignment_load,
           'Sleep_Hours':sleep_hours,'Physical_Exercise':physical_exercise,
           'Social_Media_Use':social_media_use,'Screen_Time':screen_time,
           'Family_Income_Level':family_income,'Peer_Pressure':peer_pressure,
           'Family_Support':family_support,'Anxiety_Level':anxiety_level,'University_Type':university_type}
    df  = pd.DataFrame([inp])
    enc = pd.get_dummies(df)
    for c in feature_cols:
        if c not in enc.columns: enc[c] = 0
    enc    = enc[feature_cols]
    scaled = scaler.transform(enc)

    classifier = all_models[selected_model_key]
    pred = classifier.predict(scaled)[0]
    try:
        prob_map = dict(zip(classifier.classes_, classifier.predict_proba(scaled)[0]))
    except Exception:
        prob_map = None

    cid = kmeans.predict(scaler_cluster.transform(df[['Screen_Time','Assignment_Load','Sleep_Hours']]))[0]
    rec = get_rec(cid)
    
    emoji, level, color, badge_cls, orb_cls, alert_cls, al_title, al_body = stress_meta(pred)

    st.markdown(steps_html("done", selected_model_short), unsafe_allow_html=True)
    st.markdown('<div class="slabel">📊 Hasil Analisis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.05, 0.95], gap="large")

    with col_l:
        with col_l:
          st.markdown(f"""
            <div class="card">
                <div class="res-row">
                    <div class="res-orb {orb_cls}" style="display:flex; align-items:center; justify-content:center; font-size: 1.5rem;">
                        {emoji}
                    </div>
                    <div>
                        <div style="font-size:.63rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;">Tingkat Stres Terdeteksi</div>
                        <div class="res-level" style="color:{color}">{level}</div>
                        <div class="res-sub">Analisis <strong>{selected_model_short}</strong> pada 16 fitur perilaku</div>
                        <span class="badge {badge_cls}">● {pred.upper()}</span>
                        <span class="model-pill">🔮 {selected_model_short}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if prob_map:
            pf = {"High":"pf-high","Medium":"pf-medium","Low":"pf-low"}
            html = '<div class="card"><div style="font-size:.63rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:11px;">Distribusi Probabilitas</div>'
            for k, v in prob_map.items():
                html += f'<div class="prob-row"><div class="prob-lbl">{k}</div><div class="prob-bg"><div class="prob-fill {pf.get(k,"")} " style="width:{v*100:.1f}%"></div></div><div class="prob-val">{v:.2f}</div></div>'
            st.markdown(html + "</div>", unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:4px;">📈 Metrik Kunci</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("😴 Tidur/Hari",     f"{sleep_hours}j",  delta="ideal" if sleep_hours>=7 else "kurang",   delta_color="normal" if sleep_hours>=7 else "inverse")
        m2.metric("📚 Belajar/Minggu", f"{study_hours}j")
        m3.metric("😰 Kecemasan",      f"{anxiety_level}/10", delta="tinggi" if anxiety_level>=7 else "aman", delta_color="inverse" if anxiety_level>=7 else "normal")

        st.markdown('<div class="slabel" style="margin-top:6px;">📉 Fitur Clustering</div>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Nilai":[float(screen_time),float(assignment_load),float(sleep_hours)]},
                                  index=["Screen Time","Beban Tugas","Jam Tidur"]),
                     color="#7c3aed", height=190)

        with st.expander("🔍 Pratinjau Data Input"):
            st.dataframe(pd.DataFrame.from_dict(inp, orient="index", columns=["Nilai"]), use_container_width=True)

    with col_r:
        st.markdown(f"""<div class="cl-strip">
          <div class="cl-icon">{rec["icon"]}</div>
          <div><div class="cl-lbl">Kluster Perilaku #{cid}</div><div class="cl-val">{rec["title"]}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="rec-card">
          <div class="rec-cat">💡 {rec["cat"]}</div>
          <div class="rec-title">{rec["title"]}</div>
          <div class="rec-body">{rec["body"]}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:.63rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">Langkah Aksi</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom:12px;">'+"".join(f'<span class="chip">{t}</span>' for t in rec["tips"])+"</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="alert {alert_cls}"><div class="al-t">{al_title}</div><div>{al_body}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="slabel">⬇️ Unduh Laporan</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button("📄 Laporan TXT", make_report(inp, pred, prob_map, cid, selected_model_short), "mindsense_laporan.txt", "text/plain", use_container_width=True)
        with dc2:
            out = df.copy(); out["Predicted_Stress"]=pred; out["Cluster_ID"]=int(cid); out["Cluster_Label"]=rec["title"]; out["Model_Used"]=selected_model_short
            st.download_button("📊 Data CSV", out.to_csv(index=False).encode(), "mindsense_hasil.csv","text/csv", use_container_width=True)

    st.markdown(f"""<div class="footer">
      🧠 <strong>MindSense AI</strong> · {selected_model_short} + K-Means Clustering<br>
      <span>Hasil bersifat indikatif dan tidak menggantikan konsultasi profesional.</span>
    </div>""", unsafe_allow_html=True)