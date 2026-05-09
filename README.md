# Speech Emotion Recognition – Streamlit App

## Setup

```bash
pip install -r requirements.txt
# also install ffmpeg (needed by pydub for format conversion):
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

## Model files required (same directory as emotion_app.py)

| File | Source |
|------|--------|
| `ser_mlp_model.pkl` | Notebook 1 output |
| `ser_scaler.pkl` | Notebook 1 output |
| `ser_label_encoder.pkl` | Notebook 1 output |
| `best_wav2vec2_ser.pt` | Notebook 2 output |
| `wav2vec2_config.json` | Notebook 2 output |
| `wav2vec2_label_encoder.pkl` | Notebook 2 output |

## Run

```bash
streamlit run emotion_app.py
```

## Features

| Tab | Description |
|-----|-------------|
| 🎤 Hold-to-Record | Browser mic button (like WhatsApp). Record then hit Analyse. |
| ⚡ Real-time Stream | Continuous analysis in rolling windows (sounddevice). |
| 📂 Upload Audio | Any format (mp3/ogg/flac/m4a/wav…) → auto-converted to WAV. |

Both models run in parallel and results are shown side-by-side with
colour-coded emotion cards and probability bars.
