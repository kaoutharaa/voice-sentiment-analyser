"""
Speech Emotion Recognition - Streamlit App
Supports: file upload
Models: Classical MLP + fine-tuned wav2vec 2.0
Model weights loaded from HuggingFace Hub: Kaouthara/voice-emotion-detector
"""

import os, io, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import librosa
import noisereduce as nr
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pydub import AudioSegment
from huggingface_hub import hf_hub_download

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
body { font-family: 'Segoe UI', sans-serif; }


.emotion-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 20px;
    color: white; text-align: center; margin: 10px 0;
}
.emotion-label { font-size: 2rem; font-weight: 700; margin: 0; }
.emotion-conf  { font-size: 1rem; opacity: 0.85; }

.prob-bar-container { margin: 4px 0; }
.prob-bar-label { display: flex; justify-content: space-between; font-size: 0.85rem; }
.prob-bar-bg    { background: #e0e0e0; border-radius: 6px; height: 12px; }
.prob-bar-fill  { height: 12px; border-radius: 6px;
                  background: linear-gradient(90deg,#667eea,#764ba2); }

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

# Filenames as they exist in the HuggingFace repo
HF_FILES = {
    "mlp":        "ser_mlp_model.pkl",
    "scaler":     "ser_scaler.pkl",
    "le_mlp":     "ser_label_encoder.pkl",
    "w2v_weights":"best_wav2vec2_ser.pt",
    "w2v_config": "wav2vec2_config.json",
    "le_w2v":     "wav2vec2_label_encoder.pkl",
}

EMOTION_EMOJI = {
    "angry":   "😡", "disgust": "🤢", "fear":    "😨",
    "happy":   "😄", "neutral": "😐", "ps":       "😲",
    "sad":     "😢", "surprise":"😲",
}

EMOTION_COLORS = {
    "angry":"#FF4444","disgust":"#8B4513","fear":"#800080",
    "happy":"#FFD700","neutral":"#607D8B","ps":"#FF6B35",
    "sad":"#5B86E5","surprise":"#FF9800",
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
    if len(y) > sr * 0.1:
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
# HuggingFace download helper
# ──────────────────────────────────────────────
def download_from_hub(repo_id: str, filename: str, cache_dir: str | None) -> str:
    """Download a single file from a HF repo and return its local path."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir or None,
    )


# ──────────────────────────────────────────────
# Model loader (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading & loading models from HuggingFace…")
def load_models(repo_id: str, cache_dir: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Download all artefacts ──────────────────
    paths = {key: download_from_hub(repo_id, fname, cache_dir)
             for key, fname in HF_FILES.items()}

    # ── Classical MLP stack ─────────────────────
    mlp    = joblib.load(paths["mlp"])
    scaler = joblib.load(paths["scaler"])
    le_mlp = joblib.load(paths["le_mlp"])

    # ── wav2vec 2.0 stack ───────────────────────
    with open(paths["w2v_config"]) as f:
        w2v_cfg = json.load(f)

    le_w2v    = joblib.load(paths["le_w2v"])
    processor = Wav2Vec2Processor.from_pretrained(w2v_cfg["model_name"])
    w2v_model = Wav2Vec2ForEmotionClassification(
        w2v_cfg["model_name"], w2v_cfg["num_classes"]
    ).to(device)
    w2v_model.load_state_dict(torch.load(paths["w2v_weights"], map_location=device))
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
            results["wav2vec 2.0"] = predict_wav2vec(
                audio_bytes, w2v_model, processor, le_w2v, w2v_cfg, device
            )
    return results



# ──────────────────────────────────────────────
# Sidebar – HuggingFace repo config & model selection
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/voice-recognition-scan.png", width=72)
    st.title("🎙️ SER App")
    st.markdown("**Speech Emotion Recognition**")
    st.divider()

    st.subheader("🤗 HuggingFace Repository")
    repo_id = st.text_input(
        "Repo ID",
        value="Kaouthara/voice-emotion-detector",
        help="HuggingFace repo that contains your model artefacts.",
    )
    cache_dir = st.text_input(
        "Local cache directory (optional)",
        value="",
        help="Leave blank to use the default HF cache (~/.cache/huggingface).",
    )

    with st.expander("📄 Expected repo file structure", expanded=False):
        st.code("\n".join(HF_FILES.values()), language="text")

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
if load_btn or "models" not in st.session_state:
    if not repo_id.strip():
        st.sidebar.error("Please enter a HuggingFace repo ID.")
        st.stop()
    try:
        models = load_models(repo_id.strip(), cache_dir.strip() or None)
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
st.markdown(
    "Analyse the emotion in your voice using two models: "
    "a **Classical MLP** and a **fine-tuned wav2vec 2.0**."
)

# ═══════════════════════════════════════════════
# Upload Audio
# ═══════════════════════════════════════════════
st.markdown("### 📂 Upload an Audio File")
st.markdown("only .wav format is accepted (server does not contain ffmpeg 😢")

uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav"],
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
    unsafe_allow_html=True,
)
