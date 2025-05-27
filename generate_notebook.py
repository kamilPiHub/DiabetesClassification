import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, classification_report, RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Utworzenie treści Jupyter Notebook w formie listy stringów
notebook_content = []

notebook_content.append("""
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Klasyfikacja Ryzyka Cukrzycy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Autorzy/Autor",
    "IMIĘ NAZWISKO (numer indeksu)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Streszczenie",
    "Niniejszy projekt ma na celu zbudowanie i porównanie modeli klasyfikacyjnych do przewidywania ryzyka cukrzycy na podstawie danych zdrowotnych i demograficznych. Wykorzystane zostaną dane z badania BRFSS2015. W ramach projektu przeprowadzono wstępną analizę danych, transformacje, identyfikację i obsługę braków danych oraz obserwacji odstających. Zastosowano trzy algorytmy klasyfikacyjne: K-Nearest Neighbors (KNN), Drzewo Decyzyjne oraz Las Losowy (Random Forest), a także model hybrydowy. Modele zostaną ocenione przy użyciu metryk takich jak Accuracy, F1-Score, AUC oraz macierz konfuzji. Przedstawione zostaną również przykłady użycia wytrenowanych modeli na nowych, nieznanych obserwacjach."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Słowa kluczowe",
    "Cukrzyca, Klasyfikacja, Machine Learning, KNN, Drzewo Decyzyjne, Random Forest, Analiza danych, Walidacja krzyżowa, SMOTE"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Wprowadzenie",
    "Cukrzyca jest poważnym problemem zdrowotnym na całym świecie, prowadzącym do wielu powikłań, takich jak choroby serca, udar, niewydolność nerek czy utrata wzroku. Wczesne wykrycie i interwencja są kluczowe dla zarządzania chorobą i poprawy jakości życia pacjentów. Rozwój modeli predykcyjnych opartych na danych medycznych i behawioralnych może znacząco przyczynić się do identyfikacji osób zagrożonych, umożliwiając wdrożenie działań zapobiegawczych. W tym projekcie skupiamy się na zastosowaniu algorytmów uczenia maszynowego do klasyfikacji ryzyka cukrzycy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Przedmiot badania i Cel",
    "**Przedmiot badania:** Zbiór danych 'diabetes_012_health_indicators_BRFSS2015.csv', zawierający dane zdrowotne i demograficzne ankietowanych osób z badania BRFSS (Behavioral Risk Factor Surveillance System) z 2015 roku, z informacją o statusie cukrzycowym (brak cukrzycy, pre-cukrzyca, cukrzyca).\n",
    "**Cel:** Głównym celem projektu jest opracowanie i ocena modeli klasyfikacyjnych, które będą w stanie skutecznie przewidywać ryzyko wystąpienia cukrzycy (lub jej braku) na podstawie dostępnych zmiennych. Dodatkowo, projekt ma na celu porównanie wydajności różnych algorytmów uczenia maszynowego oraz zrozumienie, które zmienne mają największy wpływ na predykcję ryzyka cukrzycy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Wstępna analiza danych"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Opis danych i zmiennych",
    "Wykorzystany zbiór danych `diabetes_012_health_indicators_BRFSS2015.csv` pochodzi z badania BRFSS 2015, zbierającego dane dotyczące zdrowia dorosłych Amerykanów. Zawiera on 22 zmienne, w tym zmienną docelową `Diabetes_012`, która klasyfikuje status cukrzycowy:\n",
    "- 0: brak cukrzycy\n",
    "- 1: pre-cukrzyca\n",
    "- 2: cukrzyca\n",
    "Dla uproszczenia problemu klasyfikacji, zmienna docelowa zostanie przekształcona na binarną, gdzie 0 oznacza brak cukrzycy, a 1 oznacza pre-cukrzycę lub cukrzycę. Pozostałe zmienne to wskaźniki zdrowotne i demograficzne. Poniżej przedstawiono listę zmiennych objaśniających:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Wczytanie danych\n",
    "df = pd.read_csv(\"diabetes_012_health_indicators_BRFSS2015.csv\")\n",
    "\n",
    "# Przekształcenie zmiennej docelowej na binarną\n",
    "df['Diabetes_012'] = df['Diabetes_012'].replace({1: 1, 2: 1})\n",
    "\n",
    "print(\"Zmienna docelowa:\", \"Diabetes_012\")\n",
    "print(\"Zmienne objaśniające:\", list(df.columns.difference(['Diabetes_012'])))\n",
    "print(\"\\nKształt zbioru danych:\", df.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Statystyki opisowe",
    "Poniżej przedstawiono statystyki opisowe dla każdej zmiennej, w tym średnią, medianę (50% kwantyl), minimum, maksimum, odchylenie standardowe oraz skośność. Skośność wskazuje na asymetrię rozkładu danych; wartości bliskie 0 oznaczają rozkład symetryczny, wartości dodatnie skośność prawostronną, a ujemne lewostronną."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "desc = df.describe().T\n",
    "desc[\"skewness\"] = df.skew()\n",
    "print(desc[[\"mean\", \"50%\", \"min\", \"max\", \"std\", \"skewness\"]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Podstawowa wizualizacja danych",
    "Wizualizacje pomagają zrozumieć rozkład danych i relacje między zmiennymi. Poniżej przedstawiono rozkład zmiennej docelowej oraz histogramy dla wybranych zmiennych numerycznych, aby zilustrować ich rozkłady."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Rozkład klas zmiennej docelowej\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.countplot(x=\"Diabetes_012\", data=df)\n",
    "plt.title(\"Rozkład klas zmiennej docelowej (0: Brak cukrzycy, 1: Cukrzyca/Pre-cukrzyca)\")\n",
    "plt.xlabel(\"Status Cukrzycowy\")\n",
    "plt.ylabel(\"Liczba obserwacji\")\n",
    "plt.xticks([0, 1], ['Brak cukrzycy', 'Cukrzyca/Pre-cukrzyca'])\n",
    "plt.show()\n",
    "\n",
    "# Histogramy dla wybranych zmiennych numerycznych\n",
    "num_cols = ['BMI', 'Age', 'MentHlth', 'PhysHlth', 'Education', 'Income']\n",
    "plt.figure(figsize=(15, 10))\n",
    "for i, col in enumerate(num_cols):\n",
    "    plt.subplot(2, 3, i + 1)\n",
    "    sns.histplot(df[col], kde=True)\n",
    "    plt.title(f'Rozkład zmiennej: {col}')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Boxploty dla wybranych zmiennych numerycznych w zależności od statusu cukrzycowego\n",
    "plt.figure(figsize=(15, 10))\n",
    "for i, col in enumerate(num_cols):\n",
    "    plt.subplot(2, 3, i + 1)\n",
    "    sns.boxplot(x='Diabetes_012', y=col, data=df)\n",
    "    plt.title(f'Boxplot {col} wg statusu cukrzycy')\n",
    "    plt.xticks([0, 1], ['Brak cukrzycy', 'Cukrzyca/Pre-cukrzyca'])\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Braki danych i obserwacje odstające",
    "Przed przystąpieniem do modelowania kluczowe jest sprawdzenie braków danych i obsługi obserwacji odstających. Brakujące wartości mogą prowadzić do błędnych wyników, a obserwacje odstające mogą zniekształcać modele. \n",
    "\n",
    "**Braki danych:** Sprawdzamy, czy w zbiorze danych występują brakujące wartości. Jeśli tak, w zależności od ich liczby i charakteru, można je uzupełnić (np. średnią, medianą, modą) lub usunąć wiersze/kolumny zawierające braki. W tym zbiorze danych nie ma brakujących wartości, co upraszcza preprocessing.\n",
    "\n",
    "**Obserwacje odstające:** Obserwacje odstające to punkty danych, które znacznie odbiegają od większości danych. Można je identyfikować za pomocą metod statystycznych (np. IQR) lub wizualizacji (np. boxploty). W przypadku wielu zmiennych z potencjalnymi wartościami odstającymi, takimi jak BMI, MentHlth czy PhysHlth, skalowanie danych (np. MinMaxScaler) oraz ewentualne transformacje (np. logarytmowanie) pomagają zmniejszyć ich wpływ na modele, zwłaszcza te wrażliwe na skalę zmiennych (np. KNN)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Liczba brakujących wartości w każdej kolumnie:\")\n",
    "print(df.isnull().sum())\n",
    "print(\"\\nBrak brakujących wartości w zbiorze danych.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Transformacje danych (Skalowanie i Logarytmowanie)",
    "**Logarytmowanie:** Zmienne o skośnym rozkładzie (np. `BMI`, `MentHlth`, `PhysHlth`, `Age`, `Education`, `Income`) mogą skorzystać z transformacji logarytmicznej, która pomaga zmniejszyć skośność i sprowadzić rozkład do bardziej zbliżonego do normalnego. Użyto `np.log1p`, która oblicza `log(1+x)`, co jest przydatne, gdy dane zawierają zera.\n",
    "\n",
    "**Skalowanie:** Algorytmy uczenia maszynowego oparte na odległości (np. KNN) są wrażliwe na skalę zmiennych. `MinMaxScaler` skaluje każdą zmienną do zakresu [0, 1], co zapewnia, że żadna zmienna nie dominuje nad innymi ze względu na jej większy zakres wartości."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Zmienne do logarytmowania (wybrane na podstawie analizy skośności)\n",
    "log_cols = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']\n",
    "for col in log_cols:\n",
    "    if col in df.columns:\n",
    "        df[col] = np.log1p(df[col])\n",
    "        print(f\"Zmienna {col} została zlogarytmowana.\")\n",
    "\n",
    "# Skalowanie danych\n",
    "scaler = MinMaxScaler()\n",
    "X = df.drop('Diabetes_012', axis=1)\n",
    "y = df['Diabetes_012']\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "X_scaled = pd.DataFrame(X_scaled, columns=X.columns)\n",
    "\n",
    "print(\"Dane zostały przeskalowane przy użyciu MinMaxScaler.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Macierz korelacji zmiennych",
    "Macierz korelacji pokazuje siłę i kierunek liniowych zależności między parami zmiennych. Wysoka korelacja między zmiennymi objaśniającymi (multicollinearity) może wpływać na stabilność i interpretowalność niektórych modeli. Dodatkowo, korelacja zmiennych objaśniających ze zmienną docelową wskazuje na ich potencjalną ważność w przewidywaniu."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(15, 12))\n",
    "correlation_matrix = df.corr()\n",
    "sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=\".2f\")\n",
    "plt.title(\"Macierz korelacji zmiennych\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Opis metod"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Podział danych (Treningowe/Testowe)",
    "Zbiór danych został podzielony na podzbiór treningowy (80%) i testowy (20%). Podzbiór treningowy jest używany do uczenia modeli, natomiast podzbiór testowy służy do oceny wydajności wytrenowanych modeli na niewidzianych danych, co pozwala oszacować ich zdolność do generalizacji. Zastosowano `stratify=y` aby zachować proporcje klas zmiennej docelowej w obu podzbiorach, co jest kluczowe w przypadku niezbalansowanych klas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(\"Kształt zbioru treningowego X:\", X_train.shape)\n",
    "print(\"Kształt zbioru testowego X:\", X_test.shape)\n",
    "print(\"Kształt zbioru treningowego y:\", y_train.shape)\n",
    "print(\"Kształt zbioru testowego y:\", y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Obsługa niezbalansowanych danych - SMOTE",
    "Zauważono, że klasy w zmiennej docelowej są niezbalansowane (znacznie więcej obserwacji bez cukrzycy niż z cukrzycą). Niezbalansowane klasy mogą prowadzić do modeli, które są biasowe w stronę klasy większościowej i słabo radzą sobie z przewidywaniem klasy mniejszościowej. Aby temu zaradzić, zastosowano metodę **SMOTE (Synthetic Minority Over-sampling Technique)** na zbiorze treningowym. SMOTE tworzy syntetyczne próbki klasy mniejszościowej na podstawie istniejących próbek, co pomaga wyrównać rozkład klas i poprawić wydajność modelu dla klasy mniejszościowej."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Rozkład klas przed SMOTE:\")\n",
    "print(y_train.value_counts())\n",
    "\n",
    "sm = SMOTE(random_state=42)\n",
    "X_train_res, y_train_res = sm.fit_resample(X_train, y_train)\n",
    "\n",
    "print(\"\\nRozkład klas po SMOTE:\")\n",
    "print(y_train_res.value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Modele Klasyfikacyjne"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### K-Nearest Neighbors (KNN)",
    "**Opis:** KNN to algorytm klasyfikacyjny oparty na odległości. Klasyfikuje nową obserwację na podstawie większości głosów jej 'k' najbliższych sąsiadów w przestrzeni cech. Działa na zasadzie 'leniwego uczenia' (lazy learning), co oznacza, że nie buduje modelu podczas fazy treningowej, a jedynie przechowuje dane treningowe i wykonuje obliczenia dopiero w momencie predykcji.  Wartość `k` (liczba sąsiadów) jest kluczowym hiperparametrem. \n",
    "**Referencje:** Fix, E., & Hodges, J. L. (1951). *Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties.* USAF School of Aviation Medicine, Randolph Field, Texas.  (Choć formalnie praca jest z 1951, koncepcja KNN była rozwijana w latach 60. i 70. przez m.in. T. Covera i P. Hart'a).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_knn = KNeighborsClassifier(n_neighbors=5)\n",
    "model_knn.fit(X_train_res, y_train_res)\n",
    "y_pred_knn = model_knn.predict(X_test)\n",
    "y_proba_knn = model_knn.predict_proba(X_test)[:, 1]\n",
    "\n",
    "print(\"K-Nearest Neighbors (KNN) - Macierz Konfuzji:\")\n",
    "print(confusion_matrix(y_test, y_pred_knn))\n",
    "print(\"\\nRaport klasyfikacyjny KNN:\")\n",
    "print(classification_report(y_test, y_pred_knn))\n",
    "print(\"AUC KNN:\", roc_auc_score(y_test, y_proba_knn))\n",
    "print(\"Accuracy KNN:\", accuracy_score(y_test, y_pred_knn))\n",
    "print(\"F1-Score KNN:\", f1_score(y_test, y_pred_knn))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Drzewo Decyzyjne (Decision Tree)",
    "**Opis:** Drzewo decyzyjne to model klasyfikacyjny, który konstruuje drzewo, gdzie każdy węzeł wewnętrzny reprezentuje test na atrybucie, każda gałąź reprezentuje wynik testu, a każdy liść (węzeł końcowy) reprezentuje etykietę klasy. Proces budowy drzewa polega na rekurencyjnym dzieleniu danych na podgrupy na podstawie cech, które najlepiej rozdzielają klasy. \n",
    "**Referencje:** Quinlan, J. R. (1986). *Induction of decision trees.* Machine learning, 1(1), 81-106.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_dt = DecisionTreeClassifier(random_state=42)\n",
    "model_dt.fit(X_train_res, y_train_res)\n",
    "y_pred_dt = model_dt.predict(X_test)\n",
    "y_proba_dt = model_dt.predict_proba(X_test)[:, 1]\n",
    "\n",
    "print(\"Drzewo Decyzyjne - Macierz Konfuzji:\")\n",
    "print(confusion_matrix(y_test, y_pred_dt))\n",
    "print(\"\\nRaport klasyfikacyjny Drzewa Decyzyjnego:\")\n",
    "print(classification_report(y_test, y_pred_dt))\n",
    "print(\"AUC Drzewa Decyzyjnego:\", roc_auc_score(y_test, y_proba_dt))\n",
    "print(\"Accuracy Drzewa Decyzyjnego:\", accuracy_score(y_test, y_pred_dt))\n",
    "print(\"F1-Score Drzewa Decyzyjnego:\", f1_score(y_test, y_pred_dt))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Wizualizacja Drzewa Klasyfikacji (Przykładowe małe drzewo)",
    "Dla lepszego zrozumienia działania drzewa decyzyjnego, poniżej przedstawiono wizualizację małego drzewa zbudowanego na podzbiorze danych. Pozwala to na podgląd reguł decyzyjnych."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(20, 10))\n",
    "plot_tree(model_dt, feature_names=X.columns, class_names=['Brak cukrzycy', 'Cukrzyca'], filled=True, rounded=True, fontsize=8, max_depth=3)\n",
    "plt.title(\"Przykładowe Drzewo Decyzyjne (max_depth=3)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Las Losowy (Random Forest)",
    "**Opis:** Random Forest to algorytm zespołowy (ensemble learning), który buduje wiele drzew decyzyjnych podczas treningu i wyprowadza klasę, która jest modą klas (klasyfikacja) lub średnią predykcji (regresja) poszczególnych drzew. Kluczową ideą jest losowe wybieranie podzbiorów cech i podpróbek danych treningowych dla każdego drzewa, co zwiększa różnorodność i redukuje wariancję, prowadząc do lepszej generalizacji. \n",
    "**Referencje:** Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_rf = RandomForestClassifier(random_state=42)\n",
    "model_rf.fit(X_train_res, y_train_res)\n",
    "y_pred_rf = model_rf.predict(X_test)\n",
    "y_proba_rf = model_rf.predict_proba(X_test)[:, 1]\n",
    "\n",
    "print(\"Random Forest - Macierz Konfuzji:\")\n",
    "print(confusion_matrix(y_test, y_pred_rf))\n",
    "print(\"\\nRaport klasyfikacyjny Random Forest:\")\n",
    "print(classification_report(y_test, y_pred_rf))\n",
    "print(\"AUC Random Forest:\", roc_auc_score(y_test, y_proba_rf))\n",
    "print(\"Accuracy Random Forest:\", accuracy_score(y_test, y_pred_rf))\n",
    "print(\"F1-Score Random Forest:\", f1_score(y_test, y_pred_rf))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model Hybrydowy (Ensemble - Średnia Ważona Prawdopodobieństw)",
    "**Opis:** Model hybrydowy łączy predykcje z wielu modeli bazowych w celu uzyskania lepszej ogólnej wydajności niż pojedynczy model. W tym przypadku zastosowano proste uśrednianie prawdopodobieństw przewidywanych przez każdy z trzech modeli (KNN, Drzewo Decyzyjne, Random Forest). Można również zastosować średnią ważoną, jeśli z eksperymentów wynika, że niektóre modele są bardziej wiarygodne. Klasyfikacja odbywa się poprzez zaokrąglenie średniego prawdopodobieństwa do najbliższej liczby całkowitej (0 lub 1). \n",
    "**Referencje:** Rokach, L. (2010). *Ensemble-based classifiers*. The Data Mining and Knowledge Discovery Handbook, 193-219. (Ogólna koncepcja modeli zespołowych).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Przewidywanie prawdopodobieństw dla wszystkich modeli\n",
    "y_proba_knn = model_knn.predict_proba(X_test)[:, 1]\n",
    "y_proba_dt = model_dt.predict_proba(X_test)[:, 1]\n",
    "y_proba_rf = model_rf.predict_proba(X_test)[:, 1]\n",
    "\n",
    "# Stworzenie modelu hybrydowego (uśrednienie prawdopodobieństw)\n",
    "y_proba_hybrid = (y_proba_knn + y_proba_dt + y_proba_rf) / 3\n",
    "y_pred_hybrid = (y_proba_hybrid > 0.5).astype(int)\n",
    "\n",
    "print(\"Model Hybrydowy - Macierz Konfuzji:\")\n",
    "print(confusion_matrix(y_test, y_pred_hybrid))\n",
    "print(\"\\nRaport klasyfikacyjny Modelu Hybrydowego:\")\n",
    "print(classification_report(y_test, y_pred_hybrid))\n",
    "print(\"AUC Modelu Hybrydowego:\", roc_auc_score(y_test, y_proba_hybrid))\n",
    "print(\"Accuracy Modelu Hybrydowego:\", accuracy_score(y_test, y_pred_hybrid))\n",
    "print(\"F1-Score Modelu Hybrydowego:\", f1_score(y_test, y_pred_hybrid))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Ważność zmiennych (Feature Importance)",
    "Analiza ważności zmiennych pozwala zidentyfikować, które cechy mają największy wpływ na predykcję modelu. Jest to szczególnie przydatne w przypadku modeli opartych na drzewach, takich jak Random Forest, które naturalnie dostarczają informacji o ważności cech. Poniżej przedstawiono tabelę z ważnością zmiennych dla modelu Random Forest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model_rf.feature_importances_})\n",
    "feature_importances = feature_importances.sort_values(by='Importance', ascending=False)\n",
    "print(\"Tabela z ważnością zmiennych (Random Forest):\")\n",
    "print(feature_importances)\n",
    "\n",
    "plt.figure(figsize=(12, 7))\n",
    "sns.barplot(x='Importance', y='Feature', data=feature_importances)\n",
    "plt.title('Ważność Zmiennych (Random Forest)')\n",
    "plt.xlabel('Ważność')\n",
    "plt.ylabel('Zmienna')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Rezultaty"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Porównanie metryk modeli",
    "Poniżej przedstawiono podsumowanie kluczowych metryk (Accuracy, F1-Score, AUC) dla wszystkich wytrenowanych modeli. Pozwala to na bezpośrednie porównanie ich wydajności."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = pd.DataFrame({\n",
    "    'Model': ['KNN', 'Decision Tree', 'Random Forest', 'Hybrid Model'],\n",
    "    'Accuracy': [\n",
    "        accuracy_score(y_test, y_pred_knn),\n",
    "        accuracy_score(y_test, y_pred_dt),\n",
    "        accuracy_score(y_test, y_pred_rf),\n",
    "        accuracy_score(y_test, y_pred_hybrid)\n",
    "    ],\n",
    "    'F1-Score': [\n",
    "        f1_score(y_test, y_pred_knn),\n",
    "        f1_score(y_test, y_pred_dt),\n",
    "        f1_score(y_test, y_pred_rf),\n",
    "        f1_score(y_test, y_pred_hybrid)\n",
    "    ],\n",
    "    'AUC': [\n",
    "        roc_auc_score(y_test, y_proba_knn),\n",
    "        roc_auc_score(y_test, y_proba_dt),\n",
    "        roc_auc_score(y_test, y_proba_rf),\n",
    "        roc_auc_score(y_test, y_proba_hybrid)\n",
    "    ]\n",
    "})\n",
    "print(\"Podsumowanie wydajności modeli:\")\n",
    "print(results.set_index('Model'))\n",
    "\n",
    "# Wizualizacja ROC Curve dla wszystkich modeli\n",
    "plt.figure(figsize=(10, 8))\n",
    "ax = plt.gca()\n",
    "RocCurveDisplay.from_estimator(model_knn, X_test, y_test, ax=ax, name='KNN')\n",
    "RocCurveDisplay.from_estimator(model_dt, X_test, y_test, ax=ax, name='Decision Tree')\n",
    "RocCurveDisplay.from_estimator(model_rf, X_test, y_test, y_proba=y_proba_rf, ax=ax, name='Random Forest') # Prawdopodobieństwa RF są już dostępne\n",
    "plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess')\n",
    "plt.title('Krzywe ROC dla Modeli Klasyfikacyjnych')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Sposób walidacji (Walidacja Krzyżowa - Cross-Validation)",
    "Walidacja krzyżowa, w szczególności k-krotna walidacja krzyżowa (k-fold cross-validation), jest robustną metodą oceny wydajności modelu i pomaga uniknąć przeuczenia (overfittingu). Zbiór danych jest dzielony na 'k' podzbiorów. Model jest trenowany 'k' razy, za każdym razem używając innego podzbioru jako zbioru walidacyjnego, a pozostałe 'k-1' podzbiorów jako zbioru treningowego. Wyniki są następnie uśredniane. W ten sposób ocena modelu jest mniej zależna od konkretnego podziału danych na zbiór treningowy i testowy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Wyniki walidacji krzyżowej (5-krotna) - Accuracy:\")\n",
    "cv_scores_knn = cross_val_score(model_knn, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)\n",
    "print(f\"KNN: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std():.4f})\")\n",
    "\n",
    "cv_scores_dt = cross_val_score(model_dt, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)\n",
    "print(f\"Decision Tree: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})\")\n",
    "\n",
    "cv_scores_rf = cross_val_score(model_rf, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)\n",
    "print(f\"Random Forest: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})\")\n",
    "\n",
    "# Dla modelu hybrydowego walidacja krzyżowa jest bardziej złożona, wymaga niestandardowej funkcji lub iteracji\n",
    "# Poniżej przykład prostej walidacji krzyżowej dla hybrydy, jednak dla pełnej oceny zaleca się bardziej zaawansowane podejście ensemble learning cross-validation.\n",
    "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
    "hybrid_cv_scores = []\n",
    "for train_index, val_index in kf.split(X_scaled):\n",
    "    X_train_fold, X_val_fold = X_scaled.iloc[train_index], X_scaled.iloc[val_index]\n",
    "    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]\n",
    "\n",
    "    # Zastosowanie SMOTE na danych treningowych w każdej fałdzie\n",
    "    X_train_res_fold, y_train_res_fold = sm.fit_resample(X_train_fold, y_train_fold)\n",
    "\n",
    "    model_knn.fit(X_train_res_fold, y_train_res_fold)\n",
    "    model_dt.fit(X_train_res_fold, y_train_res_fold)\n",
    "    model_rf.fit(X_train_res_fold, y_train_res_fold)\n",
    "\n",
    "    y_proba_knn_fold = model_knn.predict_proba(X_val_fold)[:, 1]\n",
    "    y_proba_dt_fold = model_dt.predict_proba(X_val_fold)[:, 1]\n",
    "    y_proba_rf_fold = model_rf.predict_proba(X_val_fold)[:, 1]\n",
    "\n",
    "    y_proba_hybrid_fold = (y_proba_knn_fold + y_proba_dt_fold + y_proba_rf_fold) / 3\n",
    "    y_pred_hybrid_fold = (y_proba_hybrid_fold > 0.5).astype(int)\n",
    "    hybrid_cv_scores.append(accuracy_score(y_val_fold, y_pred_hybrid_fold))\n",
    "\n",
    "hybrid_cv_scores = np.array(hybrid_cv_scores)\n",
    "print(f\"Hybrid Model: {hybrid_cv_scores.mean():.4f} (+/- {hybrid_cv_scores.std():.4f})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Przykłady użycia modeli na nieznanych obserwacjach",
    "Poniżej przedstawiono przykłady predykcji dla dwóch sztucznie stworzonych obserwacji: jednej reprezentującej osobę zdrową o niskim ryzyku cukrzycy, a drugiej osobę z czynnikami ryzyka. Dane wejściowe są najpierw logarytmowane (dla wybranych zmiennych), a następnie skalowane przy użyciu tego samego `MinMaxScaler`, który został dopasowany do danych treningowych. Dzięki temu, modele mogą dokonywać predykcji na danych o podobnym zakresie i rozkładzie."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=== Dane wejściowe dla osoby zdrowej ===\")\n",
    "sample_healthy = pd.DataFrame([{\n",
    "    'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 22, 'Smoker': 0, 'Stroke': 0,\n",
    "    'HeartDiseaseorAttack': 0, 'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,\n",
    "    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 2,\n",
    "    'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 0, 'Age': 6, 'Education': 6, 'Income': 8\n",
    "}], columns=X.columns) # Ważne aby kolumny zgadzały się z X.columns\n",
    "\n",
    "print(\"Nieprzetworzone dane:\")\n",
    "for col, val in sample_healthy.iloc[0].items():\n",
    "    print(f\"{col}: {np.round(val, 3)}\")\n",
    "\n",
    "sample_healthy_processed = sample_healthy.copy()\n",
    "for col in log_cols:\n",
    "    if col in sample_healthy_processed.columns:\n",
    "        sample_healthy_processed[col] = np.log1p(sample_healthy_processed[col])\n",
    "\n",
    "sample_healthy_scaled = scaler.transform(sample_healthy_processed)\n",
    "\n",
    "print(\"\\nPredykcje dla osoby zdrowej:\")\n",
    "print(f\"KNN przewiduje: {'Cukrzyca' if model_knn.predict(sample_healthy_scaled)[0] == 1 else 'Brak cukrzycy'}\")\n",
    "print(f\"Drzewo Decyzyjne przewiduje: {'Cukrzyca' if model_dt.predict(sample_healthy_scaled)[0] == 1 else 'Brak cukrzycy'}\")\n",
    "print(f\"Random Forest przewiduje: {'Cukrzyca' if model_rf.predict(sample_healthy_scaled)[0] == 1 else 'Brak cukrzycy'}\")\n",
    "\n",
    "proba_hybrid_healthy = (model_knn.predict_proba(sample_healthy_scaled)[:, 1] + \\\n",
    "                      model_dt.predict_proba(sample_healthy_scaled)[:, 1] + \\\n",
    "                      model_rf.predict_proba(sample_healthy_scaled)[:, 1]) / 3\n",
    "print(f\"Model Hybrydowy przewiduje: {'Cukrzyca' if (proba_hybrid_healthy > 0.5)[0] else 'Brak cukrzycy'}\\n\")\n",
    "\n",
    "\n",
    "print(\"=== Dane wejściowe dla osoby z ryzykiem cukrzycy ===\")\n",
    "sample_risk = pd.DataFrame([{\n",
    "    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 35, 'Smoker': 1, 'Stroke': 0,\n",
    "    'HeartDiseaseorAttack': 1, 'PhysActivity': 0, 'Fruits': 0, 'Veggies': 0,\n",
    "    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 4,\n",
    "    'MentHlth': 20, 'PhysHlth': 30, 'DiffWalk': 1, 'Sex': 1, 'Age': 9, 'Education': 4, 'Income': 3\n",
    "}], columns=X.columns)\n",
    "\n",
    "print(\"Nieprzetworzone dane:\")\n",
    "for col, val in sample_risk.iloc[0].items():\n",
    "    print(f\"{col}: {np.round(val, 3)}\")\n",
    "\n",
    "sample_risk_processed = sample_risk.copy()\n",
    "for col in log_cols:\n",
    "    if col in sample_risk_processed.columns:\n",
    "        sample_risk_processed[col] = np.log1p(sample_risk_processed[col])\n",
    "\n",
    "sample_risk_scaled = scaler.transform(sample_risk_processed)\n",
    "\n",
    "print(\"\\nPredykcje dla osoby z ryzykiem:\")\n",
    "print(f\"KNN przewiduje: {'Cukrzyca' if model_knn.predict(sample_risk_scaled)[0] == 1 else 'Brak cukrzycy'}\")\n",
    "print(f\"Drzewo Decyzyjne przewiduje: {'Cukrzyca' if model_dt.predict(sample_risk_scaled)[0] == 1 else 'Brak cukrzycy'}\")\n",
    "print(f\"Random Forest przewiduje: {'Cukrzyca' if model_rf.predict(sample_risk_scaled)[0] == 1 else 'Brak cukrzycy'}\")\n",
    "\n",
    "proba_hybrid_risk = (model_knn.predict_proba(sample_risk_scaled)[:, 1] + \\\n",
    "                   model_dt.predict_proba(sample_risk_scaled)[:, 1] + \\\n",
    "                   model_rf.predict_proba(sample_risk_scaled)[:, 1]) / 3\n",
    "print(f\"Model Hybrydowy przewiduje: {'Cukrzyca' if (proba_hybrid_risk > 0.5)[0] else 'Brak cukrzycy'}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bibliografia",
    "\\* Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.\n",
    "\\* Fix, E., & Hodges, J. L. (1951). *Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties.* USAF School of Aviation Medicine, Randolph Field, Texas.\n",
    "\\* Quinlan, J. R. (1986). *Induction of decision trees.* Machine learning, 1(1), 81-106.\n",
    "\\* Rokach, L. (2010). *Ensemble-based classifiers*. The Data Mining and Knowledge Discovery Handbook, 193-219.\n",
    "\\* Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: synthetic minority over-sampling technique*. Journal of artificial intelligence research, 16, 321-357.\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
""")

# Zapisz zawartość do pliku .ipynb
with open("Diabetes_Risk_Classification_Project.ipynb", "w", encoding="utf-8") as f:
    f.write("".join(notebook_content[0]))

print("Plik 'Diabetes_Risk_Classification_Project.ipynb' został pomyślnie wygenerowany.")
print("Możesz go otworzyć w środowisku Jupyter.")