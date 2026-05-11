import mlflow
import mlflow.sklearn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
import dagshub

# --- KONFIGURASI DAGSHUB (ADVANCE) ---
# Ganti dengan username dan repo anda
DAGSHUB_USER = "luhung004"
DAGSHUB_REPO = "SMSML-Luhung-Pandyaska-Suyi"

# Inisialisasi Dagshub untuk MLflow tracking
dagshub.init(repo_owner='luhung004', repo_name='Eksperimen_SML_Luhung-Pandyaska-Suyi', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/luhung004/Eksperimen_SML_Luhung-Pandyaska-Suyi.mlflow")

# Untuk lokal/latihan, gunakan tracking uri lokal jika DagsHub belum siap
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart Disease Tuning")

# Load data
data_path = os.path.join(os.path.dirname(__file__), 'dataset_processed.joblib')
X_train, X_test, y_train, y_test = joblib.load(data_path)

# Pastikan target biner (0 = Sehat, 1 = Sakit)
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)

# Definisikan Hyperparameter Space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'random_state': [42]
}

# --- MANUAL LOGGING (SKILLED/ADVANCE) ---
with mlflow.start_run(run_name="RF_Tuning_GridSearch"):

    # Grid Search
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log Best Params Manual
    for param_name, param_value in best_params.items():
        mlflow.log_param(f"best_{param_name}", param_value)

    # Evaluasi
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log Metrics Manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # --- ARTEFAK TAMBAHAN (ADVANCE) ---

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - Best Model')
    plt.savefig('tuning_cm.png')
    mlflow.log_artifact('tuning_cm.png')

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('tuning_roc.png')
    mlflow.log_artifact('tuning_roc.png')

    # 3. Model Save
    mlflow.sklearn.log_model(best_model, "best_rf_model")

    # Cleanup local files
    os.remove('tuning_cm.png')
    os.remove('tuning_roc.png')

    print("Training Selesai. Best F1:", f1)
