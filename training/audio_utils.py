import librosa
import numpy as np
import soundfile as sf
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_audio(file_source, target_sr=22050, duration=2.0):
    """
    Standardized audio loading and preprocessing.
    Ensures Training and API use the exact same input format.
    
    Args:
        file_source: File path (str) or file-like object (BytesIO)
        target_sr: Sampling rate (default 22050 Hz)
        duration: Duration to keep in seconds (default 2.0s)
        
    Returns:
        audio: Preprocessed audio numpy array
        sr: Sampling rate
    """
    try:
        # 1. Load audio (supports paths and file-like objects)
        # Using soundfile is faster than librosa.load
        audio, sr = sf.read(file_source)
        
        # 2. Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # 3. Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        # 4. Trim/Pad to fixed duration
        # This is CRITICAL for maintaining consistent feature shapes
        max_samples = int(duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            # Optional: Pad with zeros if too short (prevents crashes on short clips)
            padding = max_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            
        return audio, sr
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise

def extract_advanced_features(audio, sample_rate):
    """
    Extracts the exact 342 features used by the RF+SVM model.
    
    Features:
    - MFCC (40)
    - Spectral Centroid (2)
    - Spectral Bandwidth (2)
    - Spectral Rolloff (2)
    - Zero Crossing Rate (2)
    - Chroma (24)
    - Spectral Contrast (14)
    - Mel Spectrogram (256)
    """
    try:
        # 1. MFCCs (20 coefficients -> 40 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)
        
        # 3. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        bandwidth_mean = np.mean(spectral_bandwidth)
        bandwidth_std = np.std(spectral_bandwidth)
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_std = np.std(spectral_rolloff)
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 6. Chroma Features (12 bins -> 24 features)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # 7. Spectral Contrast (7 bands -> 14 features)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        
        # 8. Mel Spectrogram (128 bands -> 256 features)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_mean = np.mean(mel_spec.T, axis=0)
        mel_std = np.std(mel_spec.T, axis=0)
        
        # Combine all features into a single 1D array
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            [centroid_mean, centroid_std],
            [bandwidth_mean, bandwidth_std],
            [rolloff_mean, rolloff_std],
            [zcr_mean, zcr_std],
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            mel_mean, mel_std
        ])
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None