from prometheus_client import start_http_server, Gauge, Counter
import time
import random

# Metrik Dasar
PREDICTION_COUNT = Counter('model_prediction_total', 'Total predictions made')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
LATENCY = Gauge('prediction_latency_seconds', 'Time taken for prediction')

# Metrik Tambahan (Kebutuhan Advance: minimal 10 metriks)
PRECISION = Gauge('model_precision', 'Current model precision')
RECALL = Gauge('model_recall', 'Current model recall')
F1_SCORE = Gauge('model_f1_score', 'Current model F1 score')
CPU_USAGE = Gauge('process_cpu_usage', 'CPU usage of the model server')
MEM_USAGE = Gauge('process_memory_usage_bytes', 'Memory usage of the model server')
ERROR_RATE = Counter('model_prediction_errors_total', 'Total prediction errors')
INPUT_VALUE_MEAN = Gauge('input_feature_mean', 'Mean value of input features')

def monitor_performance():
    # Simulasi monitoring
    while True:
        PREDICTION_COUNT.inc()
        MODEL_ACCURACY.set(0.85 + random.uniform(-0.05, 0.05))
        PRECISION.set(0.82 + random.uniform(-0.05, 0.05))
        RECALL.set(0.88 + random.uniform(-0.05, 0.05))
        F1_SCORE.set(0.84 + random.uniform(-0.05, 0.05))
        LATENCY.set(0.01 + random.uniform(0, 0.05))
        CPU_USAGE.set(random.uniform(10, 50))
        MEM_USAGE.set(random.uniform(500, 1000) * 1024 * 1024)
        INPUT_VALUE_MEAN.set(random.uniform(30, 70))

        if random.random() < 0.01:
            ERROR_RATE.inc()

        time.sleep(5)

if __name__ == '__main__':
    print("Prometheus Exporter started on port 8000")
    start_http_server(8000)
    monitor_performance()
