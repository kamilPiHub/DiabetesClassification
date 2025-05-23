import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Wczytanie danych
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Przedstawienie zmiennych
print("Zmienna docelowa:", "Diabetes_012")
print("Zmiennie objaśniające:", list(df.columns.difference(['Diabetes_012'])))

# Statystyki opisowe
desc = df.describe().T
desc["skewness"] = df.skew()
print(desc[["mean", "50%", "min", "max", "std", "skewness"]])

# Podstawowa wizualizacja
plt.figure(figsize=(10, 5))
sns.countplot(x="Diabetes_012", data=df)
plt.title("Rozkład klas")
plt.show()

df.drop('Diabetes_012', axis=1).hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Obsługa braków danych
print("Braki danych:\n", df.isnull().sum().sum())  # Brak braków

# Skośność i transformacja
skewed_cols = df.drop("Diabetes_012", axis=1).apply(lambda x: skew(x)).sort_values(ascending=False)
print("Skośne kolumny (>1):\n", skewed_cols[skewed_cols > 1])

# Log transformacja silnie skośnych cech
df_log = df.copy()
log_cols = skewed_cols[skewed_cols > 1].index
df_log[log_cols] = df_log[log_cols].apply(lambda x: np.log1p(x))

# Skalowanie danych
X = df_log.drop(columns=["Diabetes_012"])
y = df_log["Diabetes_012"].apply(lambda x: 0 if x == 0 else 1)  # Binarna klasyfikacja

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Obserwacje odstające (na podstawie IQR)
Q1 = X_scaled.quantile(0.25)
Q3 = X_scaled.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).any(axis=1)
X_scaled = X_scaled[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

# SMOTE - oversampling
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Modele
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_res, y_train_res)

model_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
model_tree.fit(X_train_res, y_train_res)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_res, y_train_res)

# Predykcje
probs_rf = model_rf.predict_proba(X_test)[:, 1]
probs_tree = model_tree.predict_proba(X_test)[:, 1]
probs_knn = model_knn.predict_proba(X_test)[:, 1]

    # Dynamiczne ważenie modeli zgodnie z ich skutecznością
    # Ocena dokładności każdego modelu
acc_rf = accuracy_score(y_test, model_rf.predict(X_test))
acc_tree = accuracy_score(y_test, model_tree.predict(X_test))
acc_knn = accuracy_score(y_test, model_knn.predict(X_test))

    # Obliczanie wag proporcjonalnie do dokładności
total_acc = acc_rf + acc_tree + acc_knn
w_rf = acc_rf / total_acc
w_tree = acc_tree / total_acc
w_knn = acc_knn / total_acc

print(f"Wagi modeli (RF: {w_rf:.2f}, Tree: {w_tree:.2f}, kNN: {w_knn:.2f})")

    # Hybrydowe prawdopodobieństwo (ważone dokładnością)
ensemble_probs = w_rf * probs_rf + w_tree * probs_tree + w_knn * probs_knn
ensemble_pred = (ensemble_probs > 0.5).astype(int)

# Ocena modeli
def evaluate_model(y_true, y_pred, name):
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print()

evaluate_model(y_test, model_knn.predict(X_test), "kNN")
evaluate_model(y_test, model_tree.predict(X_test), "Decision Tree")
evaluate_model(y_test, model_rf.predict(X_test), "Random Forest")
evaluate_model(y_test, ensemble_pred, "Model Hybrydowy")

# Walidacja krzyżowa
scores = cross_val_score(model_rf, X_scaled, y, cv=5, scoring='accuracy')
print("Random Forest - Accuracy (CV=5):", scores.mean(), "\n")

# Przykładowa predykcja - osoba zdrowa
print("=== Dane wejściowe dla osoby zdrowej ===")
sample = pd.DataFrame([{
    'HighBP': 0, 'HighChol': 0, 'CholCheck': 0, 'BMI': 20, 'Smoker': 1, 'Stroke': 0,
    'HeartDiseaseorAttack': 0, 'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 1, 'GenHlth': 1,
    'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 0, 'Age': 22, 'Education': 3, 'Income': 3
}])
for col, val in sample.iloc[0].items():
    print(f"{col}: {np.round(val, 3)}")

sample[log_cols] = np.log1p(sample[log_cols])
sample_scaled = scaler.transform(sample)
print("Model hybrydowy przewiduje:", "Cukrzyca\n" if model_rf.predict(sample_scaled) == 1 else "brak cukrzycy\n")

# Przykładowa predykcja - osoba z ryzykiem
print("=== Dane wejściowe dla osoby z ryzykiem cukrzycy ===")
sample_risk = pd.DataFrame([{
    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 31, 'Smoker': 1, 'Stroke': 0,
    'HeartDiseaseorAttack': 0, 'PhysActivity': 0, 'Fruits': 0, 'Veggies': 0,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 3,
    'MentHlth': 5, 'PhysHlth': 5, 'DiffWalk': 0, 'Sex': 0, 'Age': 8, 'Education': 3, 'Income': 3
}])
for col, val in sample_risk.iloc[0].items():
    print(f"{col}: {np.round(val, 3)}")

sample_risk[log_cols] = np.log1p(sample_risk[log_cols])
sample_risk_scaled = scaler.transform(sample_risk)
print("Model hybrydowy przewiduje:", "Cukrzyca" if model_rf.predict(sample_risk_scaled) == 1 else "brak cukrzycy")
