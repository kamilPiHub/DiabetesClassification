# Ścieżka robocza i wczytanie danych
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv("C:\\Users\\kvmil\\Desktop\\projekt2mad\\dane.txt", sep=",")

# Podstawowe informacje
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())

# Wykresy
sns.scatterplot(data=df, x="totChol", y="sysBP", hue=df["TenYearCHD"].astype(str))
plt.show()

sns.scatterplot(data=df, x="BMI", y="heartRate", hue=df["TenYearCHD"].astype(str))
plt.show()

print(df["male"].value_counts())

# Podział danych
train, test = train_test_split(df, test_size=0.3, random_state=23)

# Model 1
X1 = train[["totChol", "sysBP"]]
X1 = sm.add_constant(X1)
y1 = train["TenYearCHD"]
model1 = sm.Logit(y1, X1).fit()
print(model1.summary())

print(np.exp(0.021877))  # OR dla sysBP
print(np.exp(0.002002))  # OR dla totChol

# Model 2
X2 = train[["BMI", "heartRate"]]
X2 = sm.add_constant(X2)
y2 = train["TenYearCHD"]
model2 = sm.Logit(y2, X2).fit()
print(model2.summary())

# Model 3
X3 = train[["BMI", "sysBP"]]
X3 = sm.add_constant(X3)
y3 = train["TenYearCHD"]
model3 = sm.Logit(y3, X3).fit()
print(model3.summary())

# Wykres kolejny
sns.scatterplot(data=df, x="BMI", y="sysBP", hue=df["TenYearCHD"].astype(str))
plt.show()

# Model pełny
X_full = train.drop(columns="TenYearCHD").copy()
X_full = sm.add_constant(X_full)
y_full = train["TenYearCHD"]
full_model = sm.Logit(y_full, X_full, missing='drop').fit()
print(full_model.summary())

print(np.exp(0.439804))  # OR dla prevalentHyp
print(df["prevalentHyp"].value_counts())
print(df["prevalentHyp"].value_counts(normalize=True))

# Macierz konfuzji dla zbioru treningowego
train_preds = full_model.predict(X_full) > 0.5
conf_matrix_train = pd.crosstab(y_full, train_preds)
print(conf_matrix_train)

accuracy_train = (2165 + 31) / (2165 + 31 + 370 + 8)
print(accuracy_train)

# Predykcja na zbiorze testowym
X_test = test.drop(columns="TenYearCHD").copy()
X_test = sm.add_constant(X_test)
y_test = test["TenYearCHD"]
predict_test = full_model.predict(X_test)
conf_matrix_test = pd.crosstab(y_test, predict_test > 0.5)
print(conf_matrix_test)

accuracy_test = (923 + 12) / (144 + 5 + 923 + 12)
print(accuracy_test)

# Czułość i specyficzność
sensitivity = 12 / (12 + 144)
specificity = 923 / (923 + 5)
print("Czułość:", sensitivity)
print("Specyficzność:", specificity)

# Usunięcie braków danych
train1 = train.dropna()
X_full1 = sm.add_constant(train1.drop(columns="TenYearCHD"))
y_full1 = train1["TenYearCHD"]
full_model1 = sm.Logit(y_full1, X_full1).fit()
print(full_model1.summary())

# Wybór najlepszego modelu - analogia do step() (manualnie)
X_best = train1[["male", "age", "cigsPerDay", "prevalentHyp", "totChol", "sysBP", "diaBP", "glucose"]]
X_best = sm.add_constant(X_best)
y_best = train1["TenYearCHD"]
best_model = sm.Logit(y_best, X_best).fit()
print(best_model.summary())

# Macierz konfuzji dla najlepszego modelu
preds_best_train = best_model.predict(X_best) > 0.5
conf_matrix_best_train = pd.crosstab(y_best, preds_best_train)
print(conf_matrix_best_train)

# Accuracy (przykłady obliczeń)
print((2153 + 26) / (2153 + 9 + 359 + 26))
print((2168 + 29) / (2168 + 29 + 372 + 5))

# Predykcja na zbiorze testowym
X_test_best = test[["male", "age", "cigsPerDay", "prevalentHyp", "totChol", "sysBP", "diaBP", "glucose"]]
X_test_best = sm.add_constant(X_test_best)
predict_test_best = best_model.predict(X_test_best)
conf_matrix_best_test = pd.crosstab(y_test, predict_test_best > 0.5)
print(conf_matrix_best_test)

print((967 + 13) / (967 + 13 + 169 + 5))
print((975 + 10) / (975 + 10 + 154 + 4))

# Czułość i specyficzność
sensitivity_best = 10 / (10 + 154)
specificity_best = 975 / (975 + 4)
print("Czułość:", sensitivity_best)
print("Specyficzność:", specificity_best)

# ORy
print(np.exp(0.463056))  # OR dla male
print(np.exp(0.453646))  # OR dla prevalentHyp
