import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('university_student_stress_dataset.csv')
df.columns = df.columns.str.strip()

mapping_dict = {
    'Tuition': {'Yes': 1, 'No': 0},
    'Physical_Exercise': {'Yes': 1, 'No': 0},
    'Family_Income_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'University_Type': {'National University': 0, 'Private University': 1}
}
for col, mapping in mapping_dict.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
print(f"LR Baseline: {accuracy_score(y_test, lr.predict(X_test_scaled)):.4f}")

# Experiment 1: RF without SMOTE, but with class_weight
rf1 = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
param1 = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
g1 = GridSearchCV(rf1, param1, cv=cv, scoring='accuracy', n_jobs=-1)
g1.fit(X_train_scaled, y_train)
print(f"RF (No SMOTE, balanced_subsample): {accuracy_score(y_test, g1.best_estimator_.predict(X_test_scaled)):.4f}")

# Experiment 2: RF with SMOTE
pipe2 = ImbPipeline([('smote', SMOTE(random_state=42)), ('rf', RandomForestClassifier(random_state=42))])
param2 = {
    'rf__n_estimators': [100, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
}
g2 = GridSearchCV(pipe2, param2, cv=cv, scoring='accuracy', n_jobs=-1)
g2.fit(X_train_scaled, y_train)
print(f"RF (With SMOTE): {accuracy_score(y_test, g2.best_estimator_.predict(X_test_scaled)):.4f}")

# Experiment 3: SVM without SMOTE, balanced
svm1 = SVC(class_weight='balanced', random_state=42)
param3 = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly']
}
g3 = GridSearchCV(svm1, param3, cv=cv, scoring='accuracy', n_jobs=-1)
g3.fit(X_train_scaled, y_train)
print(f"SVM (No SMOTE, balanced): {accuracy_score(y_test, g3.best_estimator_.predict(X_test_scaled)):.4f}")

# Try dropping interaction features to see if original features perform better
X_train_orig = X_train.iloc[:, :10]
X_test_orig = X_test.iloc[:, :10]
scaler2 = StandardScaler()
X_train_orig_scaled = scaler2.fit_transform(X_train_orig)
X_test_orig_scaled = scaler2.transform(X_test_orig)

g4 = GridSearchCV(rf1, param1, cv=cv, scoring='accuracy', n_jobs=-1)
g4.fit(X_train_orig_scaled, y_train)
print(f"RF (Original 10 features, No SMOTE): {accuracy_score(y_test, g4.best_estimator_.predict(X_test_orig_scaled)):.4f}")
