import streamlit as st
from PIL import Image
import ollama
import re
import tempfile
import os

# YOLO imports
from ultralytics import YOLO

st.set_page_config(page_title="🧠 Offline Multi-Modal Assistant", layout="centered")

# ---------------- STYLES (unchanged) ----------------
st.markdown("""
    <style>
        .reportview-container {
            background: #1e1e2f;
            color: white;
        }
        h1, h2, h3 {
            color: #00FFAA;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .chat-box {
            background-color: #2b2b3d;
            color: white;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            line-height: 1.6;
            border-left: 4px solid #00FFAA;
            margin-top: 1rem;
        }
        .refresh-btn {
            display: flex;
            justify-content: center;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER (unchanged) ----------------
st.title("🧠 Offline Multi-Modal Assistant")
st.subheader("💡 Ask questions about uploaded images using local object detection + llama3")

# ---------------- YOLO LOADING ----------------
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """
    Loads YOLO model. If local weights exist at ./models/yolov8n.pt, uses them.
    Otherwise, Ultralytics will download yolov8n weights on first run.
    """
    local_weights = os.path.join("models", "yolov8n.pt")
    if os.path.exists(local_weights):
        return YOLO(local_weights)
    # Small & fast model
    return YOLO("yolov8n.pt")

def detect_objects_yolo(pil_image, conf=0.35, max_labels=20):
    """
    Runs YOLO on a PIL image and returns a deduplicated list of labels sorted by frequency.
    """
    model = load_yolo_model()
    results = model.predict(pil_image, conf=conf, verbose=False)
    if not results or len(results) == 0:
        return []

    res = results[0]
    names = model.names  # class index -> label name
    if res.boxes is None or len(res.boxes) == 0:
        return []

    cls_indices = res.boxes.cls.tolist()
    # Count frequency
    from collections import Counter
    labels = [names[int(i)] for i in cls_indices]
    freq = Counter(labels)
    # Sort by frequency desc, keep unique labels
    sorted_labels = [lab for lab, _ in freq.most_common(max_labels)]
    return sorted_labels

# ---------------- STREAMING (fixed to avoid metadata tail) ----------------
def safe_stream_llama3(prompt: str):
    """
    Stream llama3 output using the Ollama Python client.
    Only display clean text (no chunk metadata).
    Falls back to non-stream if streaming not supported.
    """
    try:
        stream_iter = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        collected = ""
        for chunk in stream_iter:
            piece = ""

            # Handle object response (newer clients)
            if hasattr(chunk, "message") and getattr(chunk.message, "content", None):
                piece = chunk.message.content
            # Handle dict response (some client versions)
            elif isinstance(chunk, dict):
                msg = chunk.get("message", {})
                if isinstance(msg, dict):
                    piece = msg.get("content", "") or ""
                else:
                    piece = chunk.get("content", "") or ""

            # ✅ Skip empty strings / metadata-only chunks
            if not piece or not piece.strip():
                continue

            collected += piece
            yield collected

    except TypeError:
        # stream argument not supported → synchronous fallback
        resp = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        content = ""
        if isinstance(resp, dict):
            content = resp.get("message", {}).get("content", "")
        elif hasattr(resp, "message"):
            content = getattr(resp.message, "content", "")
        if not content:
            content = str(resp)
        yield content

# ---------------- FORM (unchanged) ----------------
with st.form("image_form"):
    uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
    question = st.text_input("💬 What would you like to ask about the image?")
    submitted = st.form_submit_button("🧠 Get Answer")

# ---------------- PROCESSING (UI unchanged; logic improved) ----------------
if submitted and uploaded_image and question.strip():
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # 1) Detect objects with YOLO (offline after first weights download)
    with st.spinner("🔍 Detecting objects in the image..."):
        detected_objects = detect_objects_yolo(image, conf=0.35)
    detected_text = ", ".join(detected_objects) if detected_objects else "No recognizable objects detected."

    # 2) Ask llama3 with those detections (keeps answers professional & concise)
    prompt = (
        "You are a concise and professional assistant. "
        "Answer strictly based on the detected objects and the user's question. "
        "Avoid informal descriptors; use plain object names. "
        "If uncertain, state that briefly.\n\n"
        f"Detected objects in the image: {detected_text}\n"
        f"User question: {question}\n\n"
        "Provide a short, factual, unambiguous answer."
    )

    answer_placeholder = st.empty()
    with st.spinner("🤖 Thinking..."):
        collected_text = ""
        for partial in safe_stream_llama3(prompt):
            clean = re.sub(r'\s+', ' ', partial).strip()
            collected_text = clean
            answer_placeholder.markdown(
                f"<div class='chat-box'>{clean}</div>",
                unsafe_allow_html=True
            )

    # ✅ Refresh button (unchanged)
    st.markdown("""
        <div class='refresh-btn'>
            <form action="" method="get">
                <button type="submit" class="stButton">🔄 Upload another file</button>
            </form>
        </div>
    """, unsafe_allow_html=True)
