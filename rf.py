import sklearn
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import glob
import re
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics         import accuracy_score, confusion_matrix, f1_score, classification_report


df = pd.read_csv('all_subjects_segmented.csv')
df = df.drop(columns=["subject", "window_start"])
df = df[df['class'] != -1]

imputer = SimpleImputer(strategy='mean')
df_filled = imputer.fit_transform(df)

df_final = pd.DataFrame(df_filled)

df_final = df_final.fillna(0)
df_final.columns = df.columns

#df_final.iloc[0]

x = df_final.drop(columns=['class'])
y = df_final['class']

iso = IsolationForest(contamination=0.01, random_state=42)
mask = iso.fit_predict(x) == 1  # 1 = inlier
x = x[mask]
y = y[mask]

#nomalization 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

param_dist = {
    'n_estimators':      [100, 200],
    'max_depth':         [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
    'max_features':      ['sqrt']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=4,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)


search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("▶ Best hyper-parameters:", search.best_params_)

y_pred = best_rf.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nWeighted F1-score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

