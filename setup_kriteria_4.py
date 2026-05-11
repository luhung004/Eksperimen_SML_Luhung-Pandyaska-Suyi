import subprocess
import time
import os
import sys

# === KONFIGURASI ===
# 1. Buka MLflow UI, copy Run ID model RandomForest kamu, paste di sini:
RUN_ID = "PASTE_RUN_ID_DISINI"

def run_kriteria_4():
    if RUN_ID == "PASTE_RUN_ID_DISINI":
        print("ERROR: Kamu harus isi RUN_ID dulu di dalam script ini!")
        print("Buka MLflow UI (mlflow ui), copy Run ID-nya.")
        return

    print("--- MEMULAI MONITORING & SERVING ---")

    # 1. Jalankan Exporter (Port 8000)
    print("\n[1/3] Menjalankan Prometheus Exporter...")
    exporter = subprocess.Popen([sys.executable, "Monitoring dan Logging/3.prometheus_exporter.py"])
    time.sleep(2)

    # 2. Jalankan Model Serving (Port 5001)
    print(f"\n[2/3] Menjalankan Model Serving untuk Run ID: {RUN_ID}...")
    # Pakai --no-conda agar lebih cepat jika library sudah terinstall di env sekarang
    cmd_serve = f'mlflow models serve -m "runs:/{RUN_ID}/model" -p 5001 --no-conda'
    serving = subprocess.Popen(cmd_serve, shell=True)

    print("Menunggu Model Serving siap (15 detik)...")
    time.sleep(15)

    # 3. Kirim Traffic (Inference) secara berulang
    print("\n[3/3] Mengirim Traffic ke Model (Simulasi User)...")
    print("Tekan Ctrl+C untuk berhenti.")

    try:
        count = 1
        while True:
            # Jalankan script inference.py
            subprocess.run([sys.executable, "Monitoring dan Logging/7.inference.py"])
            print(f"Request ke-{count} sukses terkirim.")
            count += 1
            time.sleep(3) # Kirim tiap 3 detik supaya data di Prometheus/Grafana update terus
    except KeyboardInterrupt:
        print("\nBerhenti mengirim traffic.")
    finally:
        print("Mematikan proses...")
        exporter.terminate()
        serving.terminate()

if __name__ == "__main__":
    run_kriteria_4()
