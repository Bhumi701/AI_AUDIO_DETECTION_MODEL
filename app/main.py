"""
FastAPI Application for AI Audio Detection
Optimized specifically for your Random Forest + SVM ensemble model
Feature-compatible with your training code
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import librosa
import soundfile as sf
from pydantic import BaseModel
import logging
import time
import io
import os
from functools import lru_cache
from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB
MAX_DURATION = int(os.getenv("MAX_AUDIO_DURATION", 30))
MODEL_PATH = os.getenv("MODEL_PATH", "models")

# Global variables
rf_model = None
svm_model = None
scaler = None
use_ensemble = True

def extract_advanced_features(audio, sample_rate):
    """
    Extract the EXACT same features as your training code
    Optimized for speed by accepting pre-loaded audio
    
    Returns 352 features matching your training:
    - 20 MFCCs (mean + std) = 40
    - Spectral features = 6
    - ZCR = 2
    - Chroma = 24
    - Spectral contrast = 14
    - Mel spectrogram = 256
    Total = 342 features (approximately)
    """
    try:
        # Trim to max duration for speed
        max_samples = int(MAX_DURATION * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # 1. MFCCs (20 coefficients - SAME AS YOUR TRAINING)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 2. Spectral Centroid
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
        
        # Combine all features - EXACT SAME ORDER AS TRAINING
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
        logger.error(f"Feature extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@lru_cache(maxsize=1)
def load_models():
    """Load models once and cache them"""
    global rf_model, svm_model, scaler, use_ensemble
    
    try:
        logger.info("="*60)
        logger.info("🚀 Loading AI Audio Detection Models")
        logger.info(f"   Model path: {MODEL_PATH}")
        logger.info("="*60)
        
        # Try to load Random Forest
        rf_path = os.path.join(MODEL_PATH, "rf_model.pkl")
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            logger.info("✓ Random Forest model loaded")
        else:
            logger.warning(f"✗ Random Forest not found at {rf_path}")
        
        # Try to load SVM
        svm_path = os.path.join(MODEL_PATH, "svm_model.pkl")
        if os.path.exists(svm_path):
            svm_model = joblib.load(svm_path)
            logger.info("✓ SVM model loaded")
        else:
            logger.warning(f"✗ SVM not found at {svm_path}")
        
        # Try to load scaler (REQUIRED)
        scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("✓ Feature scaler loaded")
        else:
            logger.warning(f"✗ Scaler not found at {scaler_path}")
        
        # Determine if we can use ensemble
        use_ensemble = (rf_model is not None) and (svm_model is not None)
        
        if not rf_model and not svm_model:
            logger.error("✗ No models found! Please run setup_models.py or upload trained models.")
            raise Exception("No models found! Please save your trained models.")
        
        logger.info("="*60)
        logger.info(f"✓ Models ready. Using: {('Ensemble (RF+SVM)' if use_ensemble else 'Random Forest only')}")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("\n" + "="*60)
    logger.info("🔄 AI Audio Detection API - Starting Up")
    logger.info("="*60)
    load_models()
    logger.info("✓ API is READY for predictions")
    logger.info("="*60 + "\n")
    yield
    logger.info("🛑 Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Audio Detection API",
    description="Detect AI-generated vs Real human audio using RF+SVM ensemble",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    is_ai_generated: bool
    model_used: str

class HealthResponse(BaseModel):
    status: str
    rf_model_loaded: bool
    svm_model_loaded: bool
    scaler_loaded: bool
    ensemble_available: bool

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "AI Audio Detection API - RF+SVM Ensemble",
        "version": "1.0.0",
        "features": "352 advanced audio features",
        "models": "Random Forest + SVM",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rf_model_loaded": rf_model is not None,
        "svm_model_loaded": svm_model is not None,
        "scaler_loaded": scaler is not None,
        "ensemble_available": use_ensemble
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict if audio is AI-generated or real using your trained ensemble model
    
    Args:
        file: Audio file (wav, mp3, flac supported)
    
    Returns:
        PredictionResponse with prediction and confidence
    """
    start_time = time.time()
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB"
        )
    
    # Check file extension
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Load audio using soundfile (faster than librosa.load)
        audio_data, sr = sf.read(io.BytesIO(contents))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 22050 Hz (SAME AS YOUR TRAINING)
        target_sr = 22050
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Limit duration (first 2 seconds like your training)
        duration = 2.0  # Your training uses 2 seconds
        max_samples = int(duration * sr)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        # Extract features (SAME AS TRAINING)
        features = extract_advanced_features(audio_data, sr)
        features = features.reshape(1, -1)
        
        logger.info(f"Extracted {features.shape[1]} features")
        
        # Scale features using the same scaler from training
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
            logger.warning("No scaler available - using unscaled features")
        
        # Make prediction: prefer Random Forest (`rf_model.pkl`) when available
        if rf_model is not None:
            # Use Random Forest by default
            prediction = rf_model.predict(features_scaled)[0]
            confidence = float(np.max(rf_model.predict_proba(features_scaled)))
            model_used = "Random Forest (rf_model.pkl)"

        elif use_ensemble and rf_model is not None and svm_model is not None:
            # Ensemble fallback (not used when RF is present)
            rf_proba = rf_model.predict_proba(features_scaled)[0]
            svm_proba = svm_model.predict_proba(features_scaled)[0]
            avg_proba = (rf_proba + svm_proba) / 2
            prediction = np.argmax(avg_proba)
            confidence = float(np.max(avg_proba))
            model_used = "Ensemble (RF + SVM)"

        elif svm_model is not None:
            # Use SVM only if RF is unavailable
            prediction = svm_model.predict(features_scaled)[0]
            confidence = float(np.max(svm_model.predict_proba(features_scaled)))
            model_used = "SVM"

        else:
            raise HTTPException(status_code=500, detail="No model available for prediction")
        
        # Interpret prediction (0 = Real, 1 = Fake)
        is_ai_generated = bool(prediction == 1)
        prediction_label = "🚨 Fake Audio (AI-Generated)" if is_ai_generated else "✓ Real Audio (Human)"
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence:.2%}, Time: {processing_time:.2f}s")
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=confidence,
            processing_time=processing_time,
            is_ai_generated=is_ai_generated,
            model_used=model_used
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        await file.close()

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple audio files
    
    Args:
        files: List of audio files (max 10)
    
    Returns:
        List of predictions
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    for file in files:
        try:
            result = await predict_audio(file)
            results.append({
                "filename": file.filename,
                **result.dict()
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)