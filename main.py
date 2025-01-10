import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 1. Wczytanie danych
data = pd.read_csv("data_new.csv")

# Atrybut decyzyjny i opisowe
y = data.iloc[:, 0]  # Decyzyjny
X = data.iloc[:, 1:]  # Opisowe

# 2. Kodowanie atrybutów tekstowych
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X = X.apply(lambda col: encoder.fit_transform(col) if col.dtypes == 'object' else col)

# 3. Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Generowanie danych syntetycznych (zwiększenie liczby próbek)
X_synthetic, y_synthetic = make_classification(
    n_samples=500,  # Ilość próbek
    n_features=X_scaled.shape[1],  # Tyle samo cech, co w oryginalnych danych
    n_informative=5,  # Liczba cech istotnych
    n_classes=len(np.unique(y)),  # Liczba klas
    random_state=42
)

# 5. Podział danych (z wykorzystaniem syntetycznych danych) - 80% danych to dane treningowe, 20% to testowe
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# 6. Definicja modeli
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# 7. Trenowanie i ocena
results = []

for name, model in models.items():
    print(f"Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Miary jakości
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    # ROC-AUC 
    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except ValueError:
            roc_auc = "N/A"
    
    # Zapisanie wyników
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": report["macro avg"]["precision"],
        "Recall": report["macro avg"]["recall"],
        "F1-score": report["macro avg"]["f1-score"],
        "ROC-AUC": roc_auc
    })
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {report['macro avg']['precision']:.4f}")
    print(f"Recall: {report['macro avg']['recall']:.4f}")
    print(f"F1-score : {report['macro avg']['f1-score']:.4f}")
    if roc_auc != "N/A":
        print(f"ROC-AUC: {roc_auc:.4f}")
    else:
        print("ROC-AUC: brak")
    print("-" * 40)

# 8. Podsumowanie
results_df = pd.DataFrame(results)
print("\n=== Podsumowanie wyników ===")
print(results_df)
