import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
file_path = 'Wellbeing_and_lifestyle_data_Kaggle.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print(f"File '{file_path}' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit() # Berhenti jika file tidak ada

# --- DATA CLEANING ---
# Membersihkan spasi di nama kolom 
df.columns = df.columns.str.strip()

# 2. Konversi DAILY_STRESS ke numerik 
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce')

# Hapus baris yang DAILY_STRESS-nya kosong (NaN) setelah konversi
df = df.dropna(subset=['DAILY_STRESS'])

# 3. Cek Missing Values di seluruh kolom
print("\n--- Data Kosong per Kolom ---")
print(df.isnull().sum())

# --- FEATURE ENGINEERING ---

# 4. Fungsi mapping sesuai standar 4 Tier di proposal
def map_stress(val):
    if val <= 1: return 'Normal'
    elif val <= 3: return 'Mild'
    elif val <= 4: return 'Moderate'
    else: return 'Severe'

# Terapkan ke kolom baru
df['STRESS_LEVEL'] = df['DAILY_STRESS'].apply(map_stress)

# 5. Lihat distribusi target baru
print("\n--- Distribusi Level Stres (Target) ---")
print(df['STRESS_LEVEL'].value_counts())

# 6. Menyiapkan Fitur (X) sesuai Proposal
# Pastikan fitur-fitur ini juga dikonversi ke numerik
#features = ['SLEEP_HOURS', 'DAILY_STEPS', 'SOCIAL_NETWORK', 'TODO_COMPLETED', 'DAILY_SHOUTING']
features = [
    'SLEEP_HOURS', 'DAILY_STEPS', 'SOCIAL_NETWORK', 'TODO_COMPLETED', 
    'DAILY_SHOUTING', 'FRUITS_VEGGIES', 'WORK_LIFE_BALANCE_SCORE', 
    'PLACES_VISITED', 'PERSONAL_AWARDS'
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Bersihkan lagi jika ada fitur yang kosong setelah konversi numerik
df = df.dropna(subset=features)

# --- 7. LABEL ENCODING & SCALING ---

# Ubah Target Teks -> Angka (Normal=0, Mild=1, dst)
le = LabelEncoder()
y = le.fit_transform(df['STRESS_LEVEL'])

# Ambil Fitur (X)
#X = df[['SLEEP_HOURS', 'DAILY_STEPS', 'SOCIAL_NETWORK', 'TODO_COMPLETED', 'DAILY_SHOUTING']]
X = df[[
    'SLEEP_HOURS', 'DAILY_STEPS', 'SOCIAL_NETWORK', 'TODO_COMPLETED', 
    'DAILY_SHOUTING', 'FRUITS_VEGGIES', 'WORK_LIFE_BALANCE_SCORE', 
    'PLACES_VISITED', 'PERSONAL_AWARDS'
]]

# Split Data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (PENTING untuk Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 8. TRAINING PHASE 1: LOGISTIC REGRESSION (BASELINE) ---

print("\nTraining Baseline Model (Logistic Regression)...")
model_lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

# --- 9. EVALUASI ---

y_pred = model_lr.predict(X_test_scaled)

print("\n--- PREPROCESSING & PHASE 1 SELESAI ---")
print(f"Total data yang diolah: {len(df)} baris")
print(f"Akurasi Baseline: {accuracy_score(y_test, y_pred):.2%}")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Training Random Forest (Phase 2)
print("\nTraining Phase 2: Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train_scaled, y_train)

# 2. Evaluasi
y_pred_rf = model_rf.predict(X_test_scaled)
print(f"Akurasi Random Forest: {accuracy_score(y_test, y_pred_rf):.2%}")
print("\nLaporan Klasifikasi Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# 3. Simpan Hasil untuk Dashboard (Sangat Penting!)
# Kita simpan model yang paling akurat (biasanya Random Forest)
joblib.dump(model_rf, 'mindguard_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nPhase 2 Selesai. Model 'mindguard_model.pkl' telah siap!")


from sklearn.model_selection import GridSearchCV
print("\n--- Memulai Hyperparameter Tuning (Grid Search) ---")

# 1. Tentukan kombinasi parameter yang ingin dicoba
param_grid = {
    'n_estimators': [100, 200],         # Jumlah pohon
    'max_depth': [10, 20, None],        # Kedalaman pohon agar tidak overfitting
    'min_samples_split': [2, 5],        # Batas minimal data untuk membelah cabang
    'criterion': ['gini', 'entropy']    # Cara mengukur kualitas pembelahan
}

# 2. Inisialisasi GridSearchCV
# cv=5 artinya data dibagi 5 kali untuk validasi silang
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=3, 
    n_jobs=-1, # Pakai semua core prosesor laptopmu agar cepat
    verbose=1,
    scoring='f1_weighted' # Kita fokus ke F1-Score karena data tidak seimbang
)

# 3. Training ulang dengan Grid Search
grid_search.fit(X_train_scaled, y_train)

# 4. Ambil Model Terbaik
best_model = grid_search.best_params_
print(f"\nParameter Terbaik: {best_model}")

final_model = grid_search.best_estimator_

# 5. Evaluasi Ulang
y_pred_tuned = final_model.predict(X_test_scaled)
print(f"\nAkurasi Setelah Tuning: {accuracy_score(y_test, y_pred_tuned):.2%}")
print("\nLaporan Klasifikasi Akhir:")
print(classification_report(y_test, y_pred_tuned, target_names=le.classes_))

# 6. Simpan Model Terbaik yang SUDAH DI-TUNING
joblib.dump(final_model, 'mindguard_model.pkl')
print("\nModel terbaik hasil tuning telah disimpan!")