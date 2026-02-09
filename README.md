
# 🎙️ AI Audio Detection API

A high-performance **FastAPI** application designed to detect AI-generated (spoof) vs. Real (human) audio. This system utilizes a machine learning ensemble approach (Random Forest + SVM) trained on **352 advanced spectral and temporal audio features**.

Built for the **Buildathon 2026**, this API handles Base64 audio input, performs real-time feature extraction, and provides classification with confidence scores and explainability.

---

## 🚀 Key Features

* **Ensemble Learning:** capable of using Random Forest and SVM models in tandem for higher accuracy.
* **Advanced Feature Extraction:** Extracts **352 features** using `librosa`, including:
* MFCCs (Mean & Std)
* Mel Spectrograms
* Spectral Contrast, Centroid, Bandwidth, & Rolloff
* Chroma Features & Zero Crossing Rate


* **Production Ready:** Includes robust error handling, logging, and input validation.
* **Secure:** Header-based API Key authentication.
* **Scalable:** Async request handling with optimized LRU caching for model loading.

---

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python 3.9+)
* **Server:** Uvicorn
* **Audio Processing:** Librosa, SoundFile, NumPy
* **ML Backend:** Scikit-learn (Joblib)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <https://github.com/Bhumi701/AI_AUDIO_DETECTION_MODEL>
cd ai-audio-detection-api

```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

Create a `requirements.txt` with the following content and install it:

```text
fastapi
uvicorn
numpy
joblib
librosa
soundfile
pydantic
python-multipart
scikit-learn

```

```bash
pip install -r requirements.txt

```

### 4. Model Setup (CRITICAL)

The API expects a `models/` directory containing your trained `.pkl` files. You must place your trained models here before running the app.

Structure:

```text
/
├── main.py
├── models/
│   ├── rf_model.pkl       # Random Forest Model
│   ├── svm_model.pkl      # SVM Model
│   └── scaler.pkl         # StandardScaler (Required)

```

---

## 🏃‍♂️ Running the Application

You can start the server using Uvicorn:

```bash
# Development mode with hot reload
uvicorn main:app --reload

# Production mode
python main.py

```

The API will start at `http://0.0.0.0:8000`.

---

## 🔌 API Documentation

### Authentication

All requests to `/detect` require the `buildathon2026` header.

* **Default Key:** `buildathon2026`
* *(Configurable via .env)*

### 1. Detect Audio

**Endpoint:** `POST /detect`

Classifies audio as `AI_GENERATED` or `HUMAN`.

**Headers:**

```http
Content-Type: application/json
x-api-key: buildathon2026

```

**Request Body:**

```json
{
  "language": "en",
  "audioFormat": "wav",
  "audioBase64": "UklGRi..." // Base64 encoded audio string
}

```

**Response:**

```json
{
  "status": "success",
  "language": "en",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.9852,
  "explanation": "Strong synthetic speech patterns and consistent spectral characteristics detected"
}

```

### 2. Health Check

**Endpoint:** `GET /health`

Checks if the API is running and which models are successfully loaded.

**Response:**

```json
{
  "status": "healthy",
  "rf_model_loaded": true,
  "svm_model_loaded": true,
  "scaler_loaded": true,
  "ensemble_available": true
}

```

---

## 🔧 Configuration (Environment Variables)

You can configure the application using environment variables (or a `.env` file):

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | `buildathon2026` | The secret key required in headers. |
| `MODEL_PATH` | `models` | Directory containing `.pkl` files. |
| `MAX_FILE_SIZE` | `10485760` | Max upload size in bytes (10MB). |
| `MAX_AUDIO_DURATION` | `30` | Max audio duration in seconds to process. |

---

## 🧠 Feature Extraction Details

To ensure compatibility with the training phase, the API extracts the exact same feature vector (shape: `1x352`) as the training script:

1. **MFCC (40):** 20 coefficients (Mean + Std).
2. **Spectral Centroid (2):** Mean + Std.
3. **Spectral Bandwidth (2):** Mean + Std.
4. **Spectral Rolloff (2):** Mean + Std.
5. **Zero Crossing Rate (2):** Mean + Std.
6. **Chroma (24):** 12 bins (Mean + Std).
7. **Spectral Contrast (14):** 7 bands (Mean + Std).
8. **Mel Spectrogram (256):** 128 bands (Mean + Std).

**Total Features:** 352

---

## ⚠️ Troubleshooting

* **Error: "No models found!"**
* Ensure you have created a `models` folder.
* Ensure `rf_model.pkl`, `svm_model.pkl`, and `scaler.pkl` are inside that folder.


* **Error: "Feature extraction failed"**
* Ensure the uploaded audio is a valid format (WAV/MP3) encoded in Base64.
* Install `ffmpeg` on your system if `librosa` fails to load MP3 files (`sudo apt install ffmpeg`).



---

## 📄 License

This project is licensed for the **Buildathon 2026** competition.
