import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
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

df = df.dropna(subset=features + ['Stress_Level'])
X = df[features]
y = LabelEncoder().fit_transform(df['Stress_Level'])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- Testing configurations for 80% ---")

# 1. Try Polynomial Features
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('select', SelectKBest(f_classif, k=30)),
    ('rf', RandomForestClassifier(random_state=42))
])

param = {
    'select__k': [20, 30, 40, 'all'],
    'rf__n_estimators': [200, 500],
    'rf__max_depth': [10, 15, None],
    'rf__class_weight': ['balanced', 'balanced_subsample']
}

g = GridSearchCV(pipe, param, cv=5, scoring='accuracy', n_jobs=-1)
g.fit(X_tr, y_tr)
print("Poly + RF Best CV:", g.best_score_)
print("Poly + RF Test Acc:", accuracy_score(y_te, g.best_estimator_.predict(X_te)))

# 2. Try SVM with highly tuned RBF
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42, class_weight='balanced'))
])
param_svm = {
    'svm__C': [0.1, 1, 10, 50, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
}
g_svm = GridSearchCV(pipe_svm, param_svm, cv=5, scoring='accuracy', n_jobs=-1)
g_svm.fit(X_tr, y_tr)
print("SVM Tuned CV:", g_svm.best_score_)
print("SVM Tuned Test Acc:", accuracy_score(y_te, g_svm.best_estimator_.predict(X_te)))

# 3. Best Voting combinations
best_rf = g.best_estimator_.named_steps['rf']
best_poly = g.best_estimator_.named_steps['poly']
best_select = g.best_estimator_.named_steps['select']

# Actually applying voting might be tricky with poly, let's just do base features for Voting
voting = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight='balanced_subsample')),
    ('svm', g_svm.best_estimator_.named_steps['svm'])
], voting='soft', weights=[2, 1])

scaler = StandardScaler()
voting.fit(scaler.fit_transform(X_tr), y_tr)
print("Voting (Weights 2:1) Test Acc:", accuracy_score(y_te, voting.predict(scaler.transform(X_te))))
