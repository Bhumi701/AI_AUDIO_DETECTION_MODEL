from audio_utils import preprocess_audio, extract_advanced_features

# Inside your data loading loop:
audio, sr = preprocess_audio(file_path, duration=2.0)
features = extract_advanced_features(audio, sr)