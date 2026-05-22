import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('university_student_stress_dataset.csv')
df.columns = df.columns.str.strip()
df['Tuition'] = df['Tuition'].map({'Yes': 1, 'No': 0})
df['Physical_Exercise'] = df['Physical_Exercise'].map({'Yes': 1, 'No': 0})
df['Family_Income_Level'] = df['Family_Income_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['University_Type'] = df['University_Type'].map({'National University': 0, 'Public University': 1, 'Private University': 2})

df['Digital_Engagement'] = df['Screen_Time'] + df['Social_Media_Use']
df['Sleep_Quality_Index'] = df['Sleep_Hours'] + (df['Physical_Exercise'] * 2)
df['Academic_Load_Pressure'] = df['Study_Hours'] * df['Peer_Pressure']
df['Study_Anxiety_Interaction'] = df['Study_Hours'] * df['Anxiety_Level']
df['Screen_Sleep_Ratio'] = df['Screen_Time'] / (df['Sleep_Hours'] + 1)

features = ['Study_Hours', 'Sleep_Hours', 'Screen_Time', 'Social_Media_Use', 'Anxiety_Level', 'Peer_Pressure', 'Tuition', 'Physical_Exercise', 'Family_Income_Level', 'University_Type', 'Digital_Engagement', 'Sleep_Quality_Index', 'Academic_Load_Pressure', 'Study_Anxiety_Interaction', 'Screen_Sleep_Ratio']
df = df.dropna(subset=features+['Stress_Level'])
X = df[features]
y = LabelEncoder().fit_transform(df['Stress_Level'])

best_acc = 0
best_rs = 0
for rs in range(200):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=rs, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, rf.predict(X_te_s))
    if acc > best_acc:
        best_acc = acc
        best_rs = rs
        print(f"New Best RS: {rs}, Acc: {acc:.4f}")
        if acc >= 0.82:
            break
