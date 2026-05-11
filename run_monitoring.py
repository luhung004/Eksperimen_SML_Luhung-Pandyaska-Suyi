import subprocess
import time
import os
import requests

# Konfigurasi
PROMETHEUS_PATH = "C:\\path\\to\\prometheus.exe" # GANTI KE PATH PROMETHEUS KAMU
MLFLOW_MODEL_PATH = "runs:/<YOUR_RUN_ID>/model"   # GANTI KE RUN_ID MODEL KAMU

def start_processes():
    print("1. Menjalankan Prometheus Exporter...")
    exporter = subprocess.Popen(["python", "Monitoring dan Logging/3.prometheus_exporter.py"])

    print("2. Menjalankan MLflow Model Serving (Port 5001)...")
    # mlflow_serve = subprocess.Popen(["mlflow", "models", "serve", "-m", MLFLOW_MODEL_PATH, "-p", "5001", "--no-conda"])

    print("3. Menjalankan Prometheus Server...")
    # prom = subprocess.Popen([PROMETHEUS_PATH, "--config.file=Monitoring dan Logging/2.prometheus.yml"])

    print("\nSemua sistem jalan. Tunggu 10 detik untuk traffic...")
    time.sleep(10)

    print("4. Mengirim traffic simulasi (7.inference.py)...")
    for i in range(10):
        try:
            subprocess.run(["python", "Monitoring dan Logging/7.inference.py"])
            print(f"Request ke-{i+1} terkirim.")
            time.sleep(2)
        except:
            print("Gagal kirim request. Pastikan model serve sudah jalan.")

    print("\n--- SELESAI ---")
    print("Sekarang buka Grafana (localhost:3000) dan ambil screenshot!")
    print("Tekan Ctrl+C untuk stop exporter.")

    try:
        exporter.wait()
    except KeyboardInterrupt:
        exporter.terminate()

if __name__ == "__main__":
    start_processes()
