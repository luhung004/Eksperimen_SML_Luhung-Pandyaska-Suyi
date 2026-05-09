import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set tracking URI ke lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Latihan Model heart_disease_uci")

# Load data hasil preprocessing dari kriteria 1
X_train, X_test, y_train, y_test = joblib.load('')

# Jika y_train berisi nilai numerik (math score), ubah jadi binary
if len(set(y_train)) > 2:
    y_train = (y_train >= 60).astype(int)
    y_test = (y_test >= 60).astype(int)
    print("Target diubah menjadi binary (lulus >=60)")

# Aktifkan autolog MLflow (akan mencatat parameter default, metrik, dan model)
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik tambahan secara manual
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrik manual
    mlflow.log_metric("manual_accuracy", acc)
    mlflow.log_metric("manual_precision", prec)
    mlflow.log_metric("manual_recall", rec)
    mlflow.log_metric("manual_f1", f1)
    
    # Log confusion matrix sebagai artefak
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    os.remove('confusion_matrix.png')  # hapus file lokal setelah di-log
    
    print(f"Model selesai dilatih. Accuracy: {acc:.4f}, F1: {f1:.4f}")

print("Cek MLflow UI di http://127.0.0.1:5000")