import streamlit as st
from PIL import Image
import ollama
import re
import tempfile
import os
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🧠 Offline Multi-Modal Assistant", layout="centered")

# ---------------- STYLES (enhanced but same dark theme) ----------------
st.markdown("""
    <style>
        .reportview-container {
            background: linear-gradient(135deg, #1e1e2f, #2a2a40);
            color: white;
        }
        h1, h2, h3 {
            color: #00FFAA;
            text-align: center;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 5px #00FFAA; }
            to { text-shadow: 0 0 15px #00FFAA, 0 0 25px #00FFAA; }
        }
        .stButton > button {
            background: linear-gradient(90deg, #00FFAA, #00DD88);
            color: black;
            font-weight: bold;
            border-radius: 8px;
            margin-top: 1rem;
            transition: all 0.3s ease;
            border: none;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #00DD88, #00FFAA);
        }
        .chat-box {
            background-color: #2b2b3d;
            color: white;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1.1rem;
            line-height: 1.6;
            border-left: 5px solid #00FFAA;
            margin-top: 1rem;
            box-shadow: 0 0 10px #00FFAA33;
            white-space: pre-wrap;
        }
        .refresh-btn {
            display: flex;
            justify-content: center;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("Multi-Modal Assistant")
st.subheader("💡 Ask away!!!")

# ---------------- YOLO LOADING ----------------
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Load YOLO model, prefer local weights."""
    local_weights = os.path.join("models", "yolov8n.pt")
    if os.path.exists(local_weights):
        return YOLO(local_weights)
    return YOLO("yolov8n.pt")

def detect_objects_yolo(pil_image, conf=0.35, max_labels=20):
    """Run YOLO and return list of detected labels."""
    model = load_yolo_model()
    results = model.predict(pil_image, conf=conf, verbose=False)
    if not results or len(results) == 0:
        return []
    res = results[0]
    names = model.names
    if res.boxes is None or len(res.boxes) == 0:
        return []
    from collections import Counter
    cls_indices = res.boxes.cls.tolist()
    labels = [names[int(i)] for i in cls_indices]
    freq = Counter(labels)
    sorted_labels = [lab for lab, _ in freq.most_common(max_labels)]
    return sorted_labels

# ---------------- STREAMING CLEAN OUTPUT ----------------
def safe_stream_llama3(prompt: str):
    """Stream clean text output from LLaMA 3."""
    try:
        stream_iter = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        collected = ""
        for chunk in stream_iter:
            piece = ""
            if hasattr(chunk, "message") and getattr(chunk.message, "content", None):
                piece = chunk.message.content
            elif isinstance(chunk, dict):
                msg = chunk.get("message", {})
                if isinstance(msg, dict):
                    piece = msg.get("content", "") or ""
                else:
                    piece = chunk.get("content", "") or ""
            if not piece or not piece.strip():
                continue
            collected += piece
            yield collected
    except TypeError:
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

# ---------------- FORM ----------------
with st.form("image_form"):
    uploaded_image = st.file_uploader("📷 Upload an image (optional)", type=["jpg", "jpeg", "png"])
    question = st.text_input("💬 Ask a question (about the image or anything):")
    submitted = st.form_submit_button("🧠 Get Answer")

# ---------------- PROCESSING ----------------
if submitted and question.strip():
    detected_text = ""

    # If an image is uploaded, detect objects first
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        with st.spinner("🔍 Detecting objects in the image..."):
            detected_objects = detect_objects_yolo(image, conf=0.35)
        detected_text = ", ".join(detected_objects) if detected_objects else "No recognizable objects detected."

    # Create prompt depending on whether an image is present
    if uploaded_image:
        prompt = (
            "You are a concise and professional assistant. "
            "Answer strictly based on the detected objects and user's question. "
            "Avoid informal tone; use precise language.\n\n"
            f"Detected objects in the image: {detected_text}\n"
            f"User question: {question}\n\n"
            "Provide a short, factual, and professional answer."
        )
    else:
        prompt = (
            "You are a concise and professional AI assistant. "
            "Answer the user's question clearly and factually, keeping tone formal and direct.\n\n"
            f"User question: {question}\n\n"
            "Provide a brief, accurate response."
        )

    # Stream response
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

    # Refresh button
    st.markdown("""
        <div class='refresh-btn'>
            <form action="" method="get">
                <button type="submit" class="stButton">🔄 Upload another file</button>
            </form>
        </div>
    """, unsafe_allow_html=True)
