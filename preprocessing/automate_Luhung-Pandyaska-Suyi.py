import pandas as pd
import os 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

def run_preprocessing(file_path):
    print("Membaca data dari:", file_path)
    df = pd.read_csv(file_path)
    
    # Drop kolom id dan dataset (id adalah identifier, dataset adalah lokasi data)
    df = df.drop(columns=['id', 'dataset'])
    
    # Target variable adalah 'num' (penyakit jantung)
    X = df.drop(columns=['num'])
    y = df['num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fitur numerik dari dataset heart disease
    numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    # Fitur kategorikal dari dataset heart disease
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # LabelEncoder untuk setiap kolom kategorikal
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        # Fit pada data train, transform pada data train dan test
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le  # simpan encoder untuk kemungkinan penggunaan di masa depan
    
    # Kolom numerik discaling, kolom kategorikal (sudah di-label-encode) dibiarkan lewat (passthrough)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    print("Membersihkan dan mengubah data dengan Label Encoding...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Simpan data train/test split sebagai joblib
    output_path = os.path.join(os.path.dirname(__file__), 'dataset_processed.joblib')
    joblib.dump((X_train_processed, X_test_processed, y_train, y_test), output_path)
    print("Berhasil! Data bersih disimpan sebagai 'dataset_processed.joblib'")

if __name__ == "__main__":
    # Tentukan path yang benar tergantung di mana script dijalankan
    paths_to_try = [
        'heart_disease_uci.csv',  # Jika dijalankan dari root
        'heart_disease_uci.csv'  # Jika dijalankan dari preprocessing/
    ]
    
    file_path = None
    for path in paths_to_try:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError("heart_disease.csv tidak ditemukan di mana pun!")
    
    run_preprocessing(file_path)
