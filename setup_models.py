import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Configuration
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_dummy_models():
    print("🚀 Starting model generation...")
    
    # 1. Determine Feature Count based on your main.py documentation
    # MFCC(40) + Spec(6) + ZCR(2) + Chroma(24) + Contrast(14) + Mel(256) = 342 features
    n_features = 342
    n_samples = 50  # Dummy data samples

    print(f"   - Generating dummy data ({n_samples} samples, {n_features} features)...")
    # Generate random training data
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary target (0 or 1)

    # 2. Create and Train Scaler (CRITICAL for main.py)
    print("   - Training Scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler (Note: main.py expects 'scaler.pkl', not 'scaler_model.pkl')
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 3. Create and Train Random Forest
    print("   - Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_scaled, y)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # 4. Create and Train SVM
    print("   - Training SVM...")
    # probability=True is REQUIRED because main.py uses .predict_proba()
    svm_model = SVC(probability=True, kernel='rbf', random_state=42)
    svm_model.fit(X_scaled, y)
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.pkl"))

    print("\n✅ SUCCESS! The following files were created in 'models/':")
    print(f"   1. {os.path.join(MODEL_DIR, 'rf_model.pkl')}")
    print(f"   2. {os.path.join(MODEL_DIR, 'svm_model.pkl')}")
    print(f"   3. {os.path.join(MODEL_DIR, 'scaler.pkl')}")
    print("\nYou can now run your FastAPI app!")

if __name__ == "__main__":
    generate_dummy_models()