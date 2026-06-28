import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

# Data load karo
df = pd.read_csv('campaign_data.csv')

# Categorical columns encode karo (text → numbers)
le = LabelEncoder()
df['platform'] = le.fit_transform(df['platform'])
df['brand'] = le.fit_transform(df['brand'])
df['campaign_type'] = le.fit_transform(df['campaign_type'])

# Features aur target alag karo
X = df[['platform', 'brand', 'campaign_type', 'spend', 
        'impressions', 'clicks', 'orders', 'CTR', 'CPC', 
        'day_of_week', 'is_weekend']]
y = df['is_risky']

# Train/Test split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train karo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test karo
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Model save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl ✅")