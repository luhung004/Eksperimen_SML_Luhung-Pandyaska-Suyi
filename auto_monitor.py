import subprocess
import time
import os
import sys
import glob

def get_latest_run_id():
    # Cari folder mlruns lokal
    mlruns_dir = "mlruns/0" # Default experiment ID 0
    if not os.path.exists(mlruns_dir):
        print("ERROR: Folder 'mlruns' tidak ditemukan. Jalankan 'modelling.py' dulu!")
        return None

    # Cari folder run terbaru berdasarkan waktu modifikasi
    runs = [d for d in glob.glob(os.path.join(mlruns_dir, "*/")) if os.path.isdir(d)]
    if not runs:
        print("ERROR: Belum ada hasil training di MLflow.")
        return None

    latest_run = max(runs, key=os.path.getmtime)
    return os.path.basename(os.path.normpath(latest_run))

def run_all():
    run_id = get_latest_run_id()
    if not run_id: return

    print(f"--- MEMULAI MONITORING (Run ID: {run_id}) ---")

    # 1. Jalankan Exporter
    print("\n[1/3] Menjalankan Prometheus Exporter (Port 8000)...")
    exporter = subprocess.Popen([sys.executable, "Monitoring dan Logging/3.prometheus_exporter.py"])

    # 2. Jalankan Serving
    print(f"\n[2/3] Menjalankan Model Serving (Port 5001)...")
    cmd_serve = f'mlflow models serve -m "runs:/{run_id}/model" -p 5001 --no-conda'
    serving = subprocess.Popen(cmd_serve, shell=True)

    print("Menunggu sistem siap (10 detik)...")
    time.sleep(10)

    # 3. Kirim Traffic & Minta Screenshot
    print("\n[3/3] MENGIRIM TRAFFIC...")
    print(">>> SEKARANG BUKA GRAFANA (localhost:3000) <<<")
    print(">>> SCREENSHOT DASHBOARD DENGAN NAMA KAMU <<<")

    try:
        while True:
            subprocess.run([sys.executable, "Monitoring dan Logging/7.inference.py"])
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nBerhenti.")
    finally:
        exporter.terminate()
        serving.terminate()

if __name__ == "__main__":
    run_all()
