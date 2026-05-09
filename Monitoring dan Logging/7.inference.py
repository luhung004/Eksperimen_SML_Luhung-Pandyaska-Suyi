import requests
import json
import pandas as pd
import joblib
import os

# Sample data
data = {
    "dataframe_split": {
        "columns": ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
        "data": [[50, 140, 230, 150, 1.0, 1, 2, 0, 1, 0, 1, 0, 2]]
    }
}

def test_inference(url="http://127.0.0.1:5001/invocations"):
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Prediction:", response.json())
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Failed to connect to model server:", e)

if __name__ == "__main__":
    test_inference()
