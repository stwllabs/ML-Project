import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, silhouette_score

# Agar terminal Windows tidak error saat print karakter unik
sys.stdout.reconfigure(encoding='utf-8')

# 1. LOAD DATA
df = pd.read_csv('university_student_stress_dataset.csv')
df.columns = df.columns.str.strip()

# 2. MAPPING & FEATURE ENGINEERING
mapping_dict = {
    'Tuition': {'Yes': 1, 'No': 0},
    'Physical_Exercise': {'Yes': 1, 'No': 0},
    'Family_Income_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'University_Type': {'National University': 0, 'Private University': 1}
}
for col, mapping in mapping_dict.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Interaction Labels (Sesuai Chowdhury et al., 2025)
df['Digital_Engagement'] = df['Screen_Time'] + df['Social_Media_Use']
df['Sleep_Quality_Index'] = df['Sleep_Hours'] + (df['Physical_Exercise'] * 2)
df['Academic_Load_Pressure'] = df['Study_Hours'] * df['Peer_Pressure']

features = [
    'Study_Hours', 'Sleep_Hours', 'Screen_Time', 'Social_Media_Use', 
    'Anxiety_Level', 'Peer_Pressure', 'Tuition', 'Physical_Exercise', 
    'Family_Income_Level', 'University_Type', 
    'Digital_Engagement', 'Sleep_Quality_Index', 'Academic_Load_Pressure'
]

df = df.dropna(subset=features + ['Stress_Level'])
X = df[features]
le = LabelEncoder()
y = le.fit_transform(df['Stress_Level'])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PHASE 1: LOGISTIC REGRESSION ---
print("\n--- PHASE 1: LOGISTIC REGRESSION (Baseline) ---")
lr_baseline = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_baseline.fit(X_train_scaled, y_train)
print(f"Baseline Accuracy: {accuracy_score(y_test, lr_baseline.predict(X_test_scaled)):.2%}")

# --- PHASE 2: ADVANCED MODELING (Optimized for High Recall) ---
print("\n--- PHASE 2: RANDOM FOREST (Optimized for Recall) ---")

# Menambahkan class_weight='balanced_subsample' sangat penting untuk menaikkan Recall
rf_model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    class_weight='balanced_subsample', 
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
print(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")
print("\nClassification Report (Targeting High Recall):")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# --- PHASE 3: CLUSTERING ---
print("\n--- PHASE 3: K-MEANS CLUSTERING ---")
X_cluster_scaled = scaler.fit_transform(df[['Digital_Engagement', 'Sleep_Quality_Index', 'Academic_Load_Pressure']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Behavioral_Cluster'] = kmeans.fit_predict(X_cluster_scaled)

print(f"Silhouette Coefficient: {silhouette_score(X_cluster_scaled, df['Behavioral_Cluster']):.3f}")

# EXPORT
joblib.dump(rf_model, 'mindguard_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nModel Pipeline Integrated & Assets Saved Successfully!")