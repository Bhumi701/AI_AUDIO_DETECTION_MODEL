#MODEL USES RANDOM FOREST, SVM, ENEMBLE === BUT GUESS WHAT RANDOM FOREST PRETTY WELL RATHER THAN REST.
    
    
import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED FEATURE EXTRACTION ---
def extract_advanced_features(file_path, augment=False):
    """
    Extract comprehensive audio features including:
    - MFCCs (Mel-frequency cepstral coefficients)
    - Spectral features (centroid, bandwidth, rolloff)
    - Zero crossing rate
    - Chroma features
    - Spectral contrast
    """
    try:
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=2.0)
        
        # Data Augmentation (optional)
        if augment:
            # Add slight noise
            noise = np.random.randn(len(audio)) * 0.003
            audio = audio + noise
            
            # Pitch shifting (small variations)
            if np.random.random() > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=np.random.randint(-2, 3))
        
        # 1. MFCCs (20 coefficients for better resolution)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 2. Spectral Centroid (indicates where the "center of mass" of the spectrum is)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)
        
        # 3. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 6. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # 7. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        
        # 8. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_spec_mean = np.mean(mel_spec.T, axis=0)
        mel_spec_std = np.std(mel_spec.T, axis=0)
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            [spectral_centroid_mean, spectral_centroid_std],
            [spectral_bandwidth_mean, spectral_bandwidth_std],
            [spectral_rolloff_mean, spectral_rolloff_std],
            [zcr_mean, zcr_std],
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            mel_spec_mean, mel_spec_std
        ])
        
        return features
        
    except Exception as e:
        print(f"[Error] Could not process file: {os.path.basename(file_path)} - {e}")
        return None


# --- SETUP PATHS ---
base_path = r'C:\Users\Bhumi Bhardwaj\Downloads\for-2sec\for-2seconds'

paths = {
    "train_real": os.path.join(base_path, 'training/real'),
    "train_fake": os.path.join(base_path, 'training/fake'),
    "test_real":  os.path.join(base_path, 'testing/real'),
    "test_fake":  os.path.join(base_path, 'testing/fake')
}

# --- LOAD & PROCESS DATA ---
X_train, y_train = [], []
X_test, y_test = [], []

print("Loading and extracting advanced features... (This may take longer)")

def process_folder(folder_path, label, is_training=False, augment=False):
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for idx, file in enumerate(files):
        full_path = os.path.join(folder_path, file)
        
        # Extract features
        features = extract_advanced_features(full_path, augment=augment)
        
        if features is not None:
            if is_training:
                X_train.append(features)
                y_train.append(label)
                
                # Data augmentation: create augmented version
                if augment:
                    aug_features = extract_advanced_features(full_path, augment=True)
                    if aug_features is not None:
                        X_train.append(aug_features)
                        y_train.append(label)
            else:
                X_test.append(features)
                y_test.append(label)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(files)} files from {os.path.basename(folder_path)}")

# Load Training Data with augmentation
print("\n--- Processing Training Data ---")
process_folder(paths["train_real"], 0, is_training=True, augment=True)
process_folder(paths["train_fake"], 1, is_training=True, augment=True)

# Load Testing Data (no augmentation)
print("\n--- Processing Testing Data ---")
process_folder(paths["test_real"], 0, is_training=False, augment=False)
process_folder(paths["test_fake"], 1, is_training=False, augment=False)

print(f"\nData Loaded Successfully!")
print(f"Training samples: {len(X_train)} (with augmentation)")
print(f"Testing samples: {len(X_test)}")

# --- FEATURE SCALING ---
if len(X_train) == 0:
    print("Error: No training data found! Check your paths.")
else:
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nFeature dimension: {X_train.shape[1]}")
    
    # Standardize features (important for SVM and neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- TRAIN MULTIPLE MODELS ---
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Model 1: Enhanced Random Forest
    print("\n1. Training Enhanced Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"   Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    
    # Model 2: SVM (often very effective for audio classification)
    print("\n2. Training Support Vector Machine (SVM)...")
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"   SVM Accuracy: {svm_accuracy * 100:.2f}%")
    
    # --- ENSEMBLE PREDICTION (Voting) ---
    print("\n3. Creating Ensemble Model (RF + SVM)...")
    rf_proba = rf_model.predict_proba(X_test_scaled)
    svm_proba = svm_model.predict_proba(X_test_scaled)
    
    # Average probabilities
    ensemble_proba = (rf_proba + svm_proba) / 2
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"   Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")
    
    # --- SELECT BEST MODEL ---
    models = {
        'Random Forest': (rf_model, rf_accuracy, rf_pred),
        'SVM': (svm_model, svm_accuracy, svm_pred),
        'Ensemble': (None, ensemble_accuracy, ensemble_pred)
    }
    
    best_model_name = max(models, key=lambda x: models[x][1])
    best_model, best_accuracy, best_pred = models[best_model_name]
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"ACCURACY: {best_accuracy * 100:.2f}%")
    print("="*60)
    
    # --- DETAILED EVALUATION ---
    print("\n--- Classification Report ---")
    print(classification_report(y_test, best_pred, target_names=["Real", "Fake"]))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, best_pred)
    print(f"True Negatives (Real as Real): {cm[0][0]}")
    print(f"False Positives (Real as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake as Real): {cm[1][0]}")
    print(f"True Positives (Fake as Fake): {cm[1][1]}")
    
    # --- PREDICTION FUNCTION ---
    def predict_new_audio(file_path, use_ensemble=True):
        """Predict if an audio file is real or fake"""
        features = extract_advanced_features(file_path)
        
        if features is None:
            print("Could not process file for prediction.")
            return
        
        features_scaled = scaler.transform([features])
        
        if use_ensemble and best_model_name == 'Ensemble':
            # Use ensemble prediction
            rf_prob = rf_model.predict_proba(features_scaled)[0]
            svm_prob = svm_model.predict_proba(features_scaled)[0]
            avg_prob = (rf_prob + svm_prob) / 2
            prediction = np.argmax(avg_prob)
            confidence = np.max(avg_prob)
        else:
            # Use best single model
            model_to_use = rf_model if best_model_name == 'Random Forest' else svm_model
            prediction = model_to_use.predict(features_scaled)[0]
            confidence = np.max(model_to_use.predict_proba(features_scaled))
        
        result = "🚨 Fake Audio (AI-Generated)" if prediction == 1 else "✓ Real Audio (Human)"
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
        return prediction, confidence
    
    # --- TEST ON SAMPLE FILE ---
    print("\n" + "="*60)
    print("TESTING ON SAMPLE FILE")
    print("="*60)
    
    test_file = r'C:\Users\Bhumi Bhardwaj\Downloads\for-2sec\for-2seconds\testing\fake\file19.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav'
    if os.path.exists(test_file):
        predict_new_audio(test_file)
    else:
        print("Sample test file not found.")
    
    # --- FEATURE IMPORTANCE (for Random Forest) ---
    print("\n--- Top 10 Most Important Features ---")
    feature_importance = rf_model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
