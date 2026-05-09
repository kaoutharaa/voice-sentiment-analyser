"""
Speech Emotion Recognition - Streamlit App
Supports: real-time streaming | hold-to-record | file upload
Models: Classical MLP + fine-tuned wav2vec 2.0 (from notebook_3_inference_comparison)
"""

import os, io, json, warnings, tempfile, time, threading, queue
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import soundfile as sf
import sounddevice as sd
import librosa
import noisereduce as nr
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pydub import AudioSegment

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
)

# ──────────────────────────────────────────────
# CSS – WhatsApp/Messenger-style mic button
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* global */
body { font-family: 'Segoe UI', sans-serif; }

/* mic button */
.mic-btn {
    width: 80px; height: 80px; border-radius: 50%;
    background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 36px;
    box-shadow: 0 4px 15px rgba(37,211,102,0.4);
    transition: all 0.2s ease;
    margin: 0 auto;
}
.mic-btn:hover { transform: scale(1.08); box-shadow: 0 6px 20px rgba(37,211,102,0.6); }
.mic-btn.recording {
    background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
    box-shadow: 0 4px 15px rgba(255,68,68,0.5);
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); }
    50%      { transform: scale(1.12); }
}

/* emotion cards */
.emotion-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 20px;
    color: white; text-align: center; margin: 10px 0;
}
.emotion-label { font-size: 2rem; font-weight: 700; margin: 0; }
.emotion-conf  { font-size: 1rem; opacity: 0.85; }

/* progress bars for probabilities */
.prob-bar-container { margin: 4px 0; }
.prob-bar-label { display: flex; justify-content: space-between; font-size: 0.85rem; }
.prob-bar-bg    { background: #e0e0e0; border-radius: 6px; height: 12px; }
.prob-bar-fill  { height: 12px; border-radius: 6px;
                  background: linear-gradient(90deg,#667eea,#764ba2); }

/* tabs */
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants (must match training notebooks)
# ──────────────────────────────────────────────
SR_CLASSICAL = 22050
SR_WAV2VEC   = 16000
MAX_DURATION = 4.0
PRE_EMPHASIS = 0.97
TRIM_TOP_DB  = 30

EMOTION_EMOJI = {
    "angry":   "😡", "disgust": "🤢", "fear":    "😨",
    "happy":   "😄", "neutral": "😐", "ps":       "😲",
    "sad":     "😢", "surprise":"😲",
}

# ──────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────
def to_wav_bytes(uploaded_file) -> bytes:
    """Convert any audio format to WAV bytes using pydub."""
    ext = os.path.splitext(uploaded_file.name)[-1].lower().lstrip(".")
    audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format=ext or "wav")
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def load_and_clean(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    buf = io.BytesIO(audio_bytes)
    y, sr = librosa.load(buf, sr=target_sr, mono=True)
    
    # FIX: Do not use the first 0.5s as noise profile, because in live 
    # recordings the user is often already speaking. Instead, use 
    # stationary noise reduction over the whole clip.
    if len(y) > sr * 0.1: # safety check for very short clips
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.5, stationary=True)

    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y) == 0:
        buf.seek(0)
        y, sr = librosa.load(buf, sr=target_sr, mono=True)
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y * (0.1 / rms)
    return y.astype(np.float32)


def extract_features_classical(y: np.ndarray, sr: int) -> np.ndarray:
    feats = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats.append(np.mean(mfccs, axis=1))
    feats.append(np.mean(librosa.feature.delta(mfccs), axis=1))
    stft_mag = np.abs(librosa.stft(y))
    feats.append(np.mean(librosa.feature.chroma_stft(S=stft_mag, sr=sr), axis=1))
    feats.append(np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1))
    feats.append([np.mean(librosa.feature.zero_crossing_rate(y))])
    feats.append([np.mean(librosa.feature.rms(y=y))])
    return np.hstack(feats).reshape(1, -1)


# ──────────────────────────────────────────────
# wav2vec model definition (must match Notebook 2)
# ──────────────────────────────────────────────
class Wav2Vec2ForEmotionClassification(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec2.feature_extractor._freeze_parameters()
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, input_values, attention_mask=None):
        out    = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


# ──────────────────────────────────────────────
# Model loader (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models(mlp_path, scaler_path, le_mlp_path,
                w2v_weights_path, w2v_config_path, le_w2v_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp    = joblib.load(mlp_path)
    scaler = joblib.load(scaler_path)
    le_mlp = joblib.load(le_mlp_path)

    with open(w2v_config_path) as f:
        w2v_cfg = json.load(f)

    le_w2v    = joblib.load(le_w2v_path)
    processor = Wav2Vec2Processor.from_pretrained(w2v_cfg["model_name"])
    w2v_model = Wav2Vec2ForEmotionClassification(
        w2v_cfg["model_name"], w2v_cfg["num_classes"]
    ).to(device)
    w2v_model.load_state_dict(torch.load(w2v_weights_path, map_location=device))
    w2v_model.eval()

    return mlp, scaler, le_mlp, w2v_model, processor, le_w2v, w2v_cfg, device


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict_classical(audio_bytes, mlp, scaler, le_mlp):
    y = load_and_clean(audio_bytes, SR_CLASSICAL)
    feats = extract_features_classical(y, SR_CLASSICAL)
    feats_scaled = scaler.transform(feats)
    proba = mlp.predict_proba(feats_scaled)[0]
    pred_label = le_mlp.classes_[np.argmax(proba)]
    probs = {cls: float(p) for cls, p in zip(le_mlp.classes_, proba)}
    return pred_label, probs


def predict_wav2vec(audio_bytes, w2v_model, processor, le_w2v, w2v_cfg, device):
    max_len   = w2v_cfg["max_length"]
    target_sr = w2v_cfg["target_sr"]
    y = load_and_clean(audio_bytes, target_sr)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    else:
        y = y[:max_len]
    inputs = processor(y, sampling_rate=target_sr, return_tensors="pt", padding=False)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        proba = F.softmax(w2v_model(input_values), dim=-1).cpu().numpy()[0]
    pred_label = le_w2v.classes_[np.argmax(proba)]
    probs = {cls: float(p) for cls, p in zip(le_w2v.classes_, proba)}
    return pred_label, probs


# ──────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────
EMOTION_COLORS = {
    "angry":"#FF4444","disgust":"#8B4513","fear":"#800080",
    "happy":"#FFD700","neutral":"#607D8B","ps":"#FF6B35",
    "sad":"#5B86E5","surprise":"#FF9800",
}

def render_emotion_card(label: str, probs: dict, model_name: str):
    emoji = EMOTION_EMOJI.get(label, "🎭")
    conf  = probs.get(label, 0)
    color = EMOTION_COLORS.get(label, "#667eea")
    st.markdown(f"""
    <div class="emotion-card" style="background: linear-gradient(135deg, {color}aa 0%, {color} 100%)">
        <div style="font-size:0.9rem;opacity:0.9;margin-bottom:4px">{model_name}</div>
        <div class="emotion-label">{emoji} {label.upper()}</div>
        <div class="emotion-conf">Confidence: {conf:.1%}</div>
    </div>""", unsafe_allow_html=True)

    # sorted probability bars
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    for emotion, p in sorted_probs:
        bar_w = int(p * 100)
        em    = EMOTION_EMOJI.get(emotion, "")
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">
                <span>{em} {emotion}</span><span>{p:.1%}</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{bar_w}%;background:{EMOTION_COLORS.get(emotion,'#667eea')}"></div>
            </div>
        </div>""", unsafe_allow_html=True)


def run_predictions(audio_bytes, models_loaded, models):
    mlp, scaler, le_mlp, w2v_model, processor, le_w2v, w2v_cfg, device = models
    results = {}
    if "MLP" in models_loaded:
        with st.spinner("Running Classical MLP…"):
            results["MLP"] = predict_classical(audio_bytes, mlp, scaler, le_mlp)
    if "wav2vec" in models_loaded:
        with st.spinner("Running wav2vec 2.0…"):
            results["wav2vec 2.0"] = predict_wav2vec(audio_bytes, w2v_model, processor, le_w2v, w2v_cfg, device)
    return results


# ──────────────────────────────────────────────
# Real-time recorder helper (using sounddevice)
# ──────────────────────────────────────────────
def record_audio(duration: float, sr: int = SR_CLASSICAL) -> bytes:
    frames = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, frames, sr, format="WAV")
    return buf.getvalue()


# ──────────────────────────────────────────────
# Sidebar – model paths & selection
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/voice-recognition-scan.png", width=72)
    st.title("🎙️ SER App")
    st.markdown("**Speech Emotion Recognition**")
    st.divider()

    st.subheader("📁 Model Paths")
    mlp_path      = st.text_input("MLP model (.pkl)",          "MLP_pt/ser_mlp_model.pkl")
    scaler_path   = st.text_input("Scaler (.pkl)",             "MLP_pt/ser_scaler.pkl")
    le_mlp_path   = st.text_input("MLP label encoder (.pkl)",  "MLP_pt/ser_label_encoder.pkl")
    w2v_weights   = st.text_input("wav2vec weights (.pt)",     "FT_pt/best_wav2vec2_ser.pt")
    w2v_config    = st.text_input("wav2vec config (.json)",    "FT_pt/wav2vec2_config.json")
    le_w2v_path   = st.text_input("wav2vec label encoder",     "FT_pt/wav2vec2_label_encoder.pkl")

    st.divider()
    st.subheader("🤖 Active Models")
    use_mlp = st.checkbox("Classical MLP", value=True)
    use_w2v = st.checkbox("wav2vec 2.0",   value=True)
    models_loaded = (["MLP"] if use_mlp else []) + (["wav2vec"] if use_w2v else [])

    load_btn = st.button("⚡ Load / Reload Models", use_container_width=True)

# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
models = None
if load_btn or "models_loaded" not in st.session_state:
    all_files = [mlp_path, scaler_path, le_mlp_path, w2v_weights, w2v_config, le_w2v_path]
    missing   = [f for f in all_files if not os.path.exists(f)]
    if missing:
        st.sidebar.error(f"Missing files:\n" + "\n".join(f"• {f}" for f in missing))
        st.sidebar.info("Place your model files in the same directory as this script, "
                        "or update the paths above and click **Load / Reload Models**.")
        st.stop()
    try:
        models = load_models(mlp_path, scaler_path, le_mlp_path,
                             w2v_weights, w2v_config, le_w2v_path)
        st.sidebar.success("✅ Models loaded!")
        st.session_state["models"] = models
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")
        st.stop()
else:
    models = st.session_state.get("models")
    if models is None:
        st.info("👈 Click **Load / Reload Models** in the sidebar to get started.")
        st.stop()

# ──────────────────────────────────────────────
# Main UI – three tabs
# ──────────────────────────────────────────────
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Analyse the emotion in your voice using two models: a **Classical MLP** and a **fine-tuned wav2vec 2.0**.")

tab_record, tab_realtime, tab_upload = st.tabs([
    "🎤 Hold-to-Record", "⚡ Real-time Stream", "📂 Upload Audio"
])

# ═══════════════════════════════════════════════
# TAB 1 – Hold-to-Record  (WAV via st.audio_input)
# ═══════════════════════════════════════════════
with tab_record:
    st.markdown("### 🎤 Record Your Voice")
    st.markdown("Press the microphone button below, speak, then press **Analyse**.")

    # st.audio_input is available in Streamlit ≥ 1.31
    audio_value = st.audio_input("Tap to record (hold and release)", key="recorder")

    if audio_value is not None:
        st.audio(audio_value, format="audio/wav")
        if st.button("🔍 Analyse Recording", use_container_width=True, key="analyse_rec"):
            audio_bytes = audio_value.read() if hasattr(audio_value, "read") else bytes(audio_value)
            results = run_predictions(audio_bytes, models_loaded, models)
            if results:
                cols = st.columns(len(results))
                for col, (name, (label, probs)) in zip(cols, results.items()):
                    with col:
                        render_emotion_card(label, probs, name)
    else:
        st.info("👆 Click the mic icon above to start recording.")

# ═══════════════════════════════════════════════
# TAB 2 – Real-time streaming (sounddevice chunks)
# ═══════════════════════════════════════════════
with tab_realtime:
    st.markdown("### ⚡ Real-time Emotion Detection")
    st.markdown("Audio is captured in rolling windows and analysed continuously.")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        chunk_sec = st.slider("Window (seconds)", 1.0, 5.0, 2.0, 0.5)
        overlap   = st.slider("Overlap (%)",       0,   80,  50,   5)

    rt_placeholder = st.empty()
    start_rt = col_l.button("▶ Start Real-time", use_container_width=True, key="rt_start")
    stop_rt  = col_l.button("⏹ Stop",           use_container_width=True, key="rt_stop")

    if "rt_running" not in st.session_state:
        st.session_state["rt_running"] = False

    if start_rt:
        st.session_state["rt_running"] = True
    if stop_rt:
        st.session_state["rt_running"] = False

    if st.session_state["rt_running"]:
        step = chunk_sec * (1 - overlap / 100)
        st.info(f"🔴 Recording… window={chunk_sec}s, step={step:.1f}s. Press **Stop** to end.")
        iter_count = 0
        while st.session_state["rt_running"] and iter_count < 60:   # safety cap
            try:
                audio_bytes = record_audio(chunk_sec, sr=SR_CLASSICAL)
            except Exception as e:
                rt_placeholder.error(f"Microphone error: {e}")
                break

            results = run_predictions(audio_bytes, models_loaded, models)
            with rt_placeholder.container():
                cols = st.columns(len(results))
                for col, (name, (label, probs)) in zip(cols, results.items()):
                    with col:
                        render_emotion_card(label, probs, name)

            time.sleep(max(0, step - chunk_sec))   # crude pacing
            iter_count += 1
    else:
        if not start_rt:
            st.info("Press **▶ Start Real-time** to begin continuous detection.")

# ═══════════════════════════════════════════════
# TAB 3 – Upload Audio (any format → WAV)
# ═══════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📂 Upload an Audio File")
    st.markdown("Any format is accepted (mp3, ogg, flac, m4a, wav, …) – converted to WAV automatically.")

    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav","mp3","ogg","flac","m4a","aac","wma","aiff","aif"],
        key="uploader"
    )

    if uploaded:
        with st.spinner("Converting to WAV…"):
            try:
                wav_bytes = to_wav_bytes(uploaded)
                st.success(f"✅ Converted **{uploaded.name}** → WAV ({len(wav_bytes)//1024} KB)")
                st.audio(wav_bytes, format="audio/wav")
            except Exception as e:
                st.error(f"Conversion failed: {e}")
                st.stop()

        if st.button("🔍 Analyse Audio", use_container_width=True, key="analyse_up"):
            results = run_predictions(wav_bytes, models_loaded, models)
            if results:
                cols = st.columns(len(results))
                for col, (name, (label, probs)) in zip(cols, results.items()):
                    with col:
                        render_emotion_card(label, probs, name)
    else:
        st.info("👆 Upload a file above to analyse its emotional content.")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;opacity:0.5;font-size:0.8rem'>"
    "Speech Emotion Recognition · Classical MLP + wav2vec 2.0 · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)