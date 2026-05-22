import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

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

# Interaction Features
df['Digital_Engagement'] = df['Screen_Time'] + df['Social_Media_Use']
df['Sleep_Quality_Index'] = df['Sleep_Hours'] + (df['Physical_Exercise'] * 2)
df['Academic_Load_Pressure'] = df['Study_Hours'] * df['Peer_Pressure']
df['Study_Anxiety_Interaction'] = df['Study_Hours'] * df['Anxiety_Level']
df['Screen_Sleep_Ratio'] = df['Screen_Time'] / (df['Sleep_Hours'] + 1)

features = [
    'Study_Hours', 'Sleep_Hours', 'Screen_Time', 'Social_Media_Use',
    'Anxiety_Level', 'Peer_Pressure', 'Tuition', 'Physical_Exercise',
    'Family_Income_Level', 'University_Type',
    'Digital_Engagement', 'Sleep_Quality_Index', 'Academic_Load_Pressure',
    'Study_Anxiety_Interaction', 'Screen_Sleep_Ratio'
]

processed_df = df.dropna(subset=features + ['Stress_Level'])
X = processed_df[features]

le = LabelEncoder()
y = le.fit_transform(processed_df['Stress_Level'])

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit Scaler
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

# --- PHASE 1: LOGISTIC REGRESSION ---
print('\n--- PHASE 1: LOGISTIC REGRESSION (Baseline) ---')
lr_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_baseline.fit(X_train_scaled, y_train)
y_pred_lr = lr_baseline.predict(X_test_scaled)
print(f'Baseline Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')

# --- PHASE 2: ADVANCED BEHAVIORAL MODELING (Ensemble: RF & SVM) ---
print('\n--- PHASE 2: ENSEMBLE LEARNING (RF + SVM) ---')

rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight='balanced_subsample')
svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42, class_weight='balanced')

# Soft Voting Classifier perfectly implements the proposal requirement
voting_clf = VotingClassifier(estimators=[('rf', rf), ('svm', svm)], voting='soft')
voting_clf.fit(X_train_scaled, y_train)

y_pred_final = voting_clf.predict(X_test_scaled)
acc_final = accuracy_score(y_test, y_pred_final)

print(f'\nFinal Test Accuracy (Voting Classifier): {acc_final:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

# Save the final model and scalers
joblib.dump(voting_clf, 'mindguard_model.pkl')
joblib.dump(scaler_model, 'scaler_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# --- PHASE 3: K-MEANS CLUSTERING ---
print('\n--- PHASE 3: K-MEANS CLUSTERING ---')
cluster_features = ['Digital_Engagement', 'Sleep_Quality_Index', 'Academic_Load_Pressure']
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(processed_df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
processed_df['Behavioral_Cluster'] = kmeans.fit_predict(X_cluster_scaled)

print(f'Silhouette Coefficient: {silhouette_score(X_cluster_scaled, processed_df["Behavioral_Cluster"]):.4f}')

joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
joblib.dump(kmeans, 'kmeans_cluster.pkl')

print('\nModel Pipeline Integrated & Assets Saved Successfully!')
