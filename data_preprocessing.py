import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, silhouette_score

# 1. LOAD DATA
df = pd.read_csv('university_student_stress_dataset.csv')
df.columns = df.columns.str.strip() 

# --- 2. MAPPING SESUAI ISI CSV ---
mapping_dict = {
    'Tuition': {'Yes': 1, 'No': 0},
    'Physical_Exercise': {'Yes': 1, 'No': 0},
    'Family_Income_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'University_Type': {'National University': 0, 'Private University': 1} # Disesuaikan dengan gambar
}

for col, mapping in mapping_dict.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# 3. SETUP FEATURES (Urutan sesuai gambar untuk konsistensi)
features = [
    'Study_Hours', 'Sleep_Hours', 'Screen_Time', 'Social_Media_Use', 
    'Anxiety_Level', 'Peer_Pressure', 'Tuition', 'Physical_Exercise', 
    'Family_Income_Level', 'University_Type'
]

# Bersihkan data (dropna)
df = df.dropna(subset=features + ['Stress_Level'])
print(f"Berhasil memproses {len(df)} baris data.")

X = df[features]
le = LabelEncoder()
y = le.fit_transform(df['Stress_Level'])

# 4. SPLIT & SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- TRAINING ---
print("\n--- Phase 1: Logistic Regression ---")
lr = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
print(f"Accuracy: {accuracy_score(y_test, lr.predict(X_test_scaled)):.2%}")

print("\n--- Phase 2: SVM & Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
print(f"RF Accuracy: {accuracy_score(y_test, rf.predict(X_test_scaled)):.2%}")

# Phase 3: Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X_full_scaled = scaler.transform(X)
df['Cluster'] = kmeans.fit_predict(X_full_scaled)
print(f"Silhouette Score: {silhouette_score(X_full_scaled, df['Cluster']):.3f}")

# 5. EXPORT
joblib.dump(rf, 'mindguard_model.pkl') 
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(kmeans, 'kmeans_cluster.pkl') 

print("\nSemua aset AI berhasil disimpan! 🚀")