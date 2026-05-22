import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, silhouette_score

# optional imports (XGBoost + imbalanced-learn). We'll fall back if missing.
try:
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_XGB_IMB = True
except Exception:
    HAS_XGB_IMB = False

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

print('\nLabel distribution:')
print(processed_df['Stress_Level'].value_counts())

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

# --- PHASE 1: LOGISTIC REGRESSION ---
print('\n--- PHASE 1: LOGISTIC REGRESSION (Baseline) ---')
lr_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_baseline.fit(X_train_scaled, y_train)
print(f'Baseline Accuracy: {accuracy_score(y_test, lr_baseline.predict(X_test_scaled)):.2%}')
print(f'Baseline Balanced Accuracy: {balanced_accuracy_score(y_test, lr_baseline.predict(X_test_scaled)):.2%}')

# --- PHASE 2: BALANCING + XGBOOST (preferred) OR RF fallback ---
print('\n--- PHASE 2: RESAMPLING + CLASSIFIER TUNING ---')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if HAS_XGB_IMB:
    print('XGBoost and imbalanced-learn detected — running SMOTE + XGBoost GridSearch (scoring=f1_macro)')
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [4, 6],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    # Fit on training data (SMOTE will be applied inside CV properly)
    grid.fit(X_train, y_train)

    print('\nBest pipeline params:')
    print(grid.best_params_)
    print(f'Grid search best CV (f1_macro): {grid.best_score_:.2%}')

    # Extract best classifier and build final pipeline without SMOTE for inference
    best_clf = grid.best_estimator_.named_steps['clf']
    final_pipeline = ImbPipeline([('scaler', StandardScaler()), ('clf', best_clf)])
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    print(f'Final Test accuracy: {accuracy_score(y_test, y_pred):.2%}')
    print(f'Final Test balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.2%}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # export final pipeline (scaler + classifier)
    joblib.dump(final_pipeline, 'mindguard_model.pkl')
    joblib.dump(grid.best_estimator_.named_steps['scaler'], 'scaler_model.pkl')

else:
    print('XGBoost or imbalanced-learn not available — falling back to RandomForest with SMOTE via imbalanced-learn if present.')
    try:
        from imblearn.over_sampling import SMOTE as _SMOTE
        from imblearn.pipeline import Pipeline as _ImbPipeline
        sm = _SMOTE(random_state=42)
        rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
        pipeline = _ImbPipeline([('scaler', StandardScaler()), ('smote', sm), ('clf', rf)])
        param_grid = {'clf__n_estimators':[100,200], 'clf__max_depth':[10,15,None]}
        grid = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        print('\nBest params (RF fallback):', grid.best_params_)
        y_pred = best.predict(X_test)
        print(f'RF-fallback Test accuracy: {accuracy_score(y_test, y_pred):.2%}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        joblib.dump(best, 'mindguard_model.pkl')
    except Exception:
        print('imbalanced-learn missing — training RandomForest without resampling (class_weight balanced_subsample).')
        rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced_subsample', random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        print(f'RF Test accuracy: {accuracy_score(y_test, y_pred):.2%}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        joblib.dump(rf, 'mindguard_model.pkl')

# --- PHASE 3: K-MEANS CLUSTERING ---
print('\n--- PHASE 3: K-MEANS CLUSTERING ---')
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(
    processed_df[['Digital_Engagement', 'Sleep_Quality_Index', 'Academic_Load_Pressure']]
)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
processed_df.loc[:, 'Behavioral_Cluster'] = kmeans.fit_predict(X_cluster_scaled)
print(f'Silhouette Coefficient: {silhouette_score(X_cluster_scaled, processed_df["Behavioral_Cluster"]):.3f}')

# EXPORT MODEL & ARTIFACTS
# mindguard_model.pkl already written above depending on branch; ensure scaler_cluster/le/kmeans saved
joblib.dump(scaler_model, 'scaler_model.pkl')
joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(kmeans, 'kmeans_cluster.pkl')

print('\nModel Pipeline Integrated & Assets Saved Successfully!')
