# LLM-Multi-Modal-Assistant

# Learning Log: Multi-Modal Assistant Project

## Overview
This log documents the learning journey of building an **Offline Multi-Modal Assistant** using **Streamlit, Ollama, and YOLO**.  
The assistant is designed to process **images**, detect objects, and provide contextual answers using **local LLaMA models**.

---

## Topics Learned

### 1. **Streamlit UI Development**
- Built a **dark-themed** UI with chat-bubble style responses.
- Implemented **dynamic image upload** and auto-scrolling response display.
- Added refresh functionality for re-uploading files.

### 2. **OCR with Tesseract**
- Integrated **pytesseract** for text extraction from images.
- Encountered issues with Tesseract installation and Windows path configuration.
- Eventually moved away from OCR in favor of **object detection (YOLO)** for higher accuracy.

### 3. **Ollama + LLaMA3 Integration**
- Connected to **Ollama** to run local **LLaMA3** models.
- Encountered streaming response issues where partial responses like `"model='llama3'..."` were displayed.
- Fixed by **extracting only the assistant’s message content** and discarding debug/metadata.

### 4. **Vision Models (LLaVA / BakLLaVA)**
- Tried integrating **LLaVA** and **BakLLaVA** for direct vision-language understanding.
- Hit compatibility issues: Ollama v0.11.3 did not support the `--image` flag.
- Resolved by **switching to YOLO for object detection** instead of relying on unsupported multimodal features.

### 5. **YOLO Object Detection**
- Integrated **Ultralytics YOLOv8** for local object detection.
- Detected objects (e.g., *elephant*) and passed structured results to LLaMA3 for professional summaries.
- Solved the issue where LLaMA wrongly identified objects (e.g., calling an elephant a "cat").

### 6. **Error Handling & Dependencies**
- Fixed `ModuleNotFoundError: pytesseract` by installing correct packages.
- Installed `ultralytics` for YOLO: `pip install ultralytics`.
- Debugged Ollama installation conflicts (`ollama.exe` path, version mismatches).

---

## Issues Faced and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Look-behind regex error** | Python `re` requires fixed-width lookbehind | Simplified regex to format assistant answers |
| **pytesseract not found** | Missing module | Installed `pytesseract` and set Tesseract path |
| **LLaVA `--image` flag error** | Ollama v0.11.3 CLI doesn’t support `--image` | Replaced with YOLO detection instead |
| **Streaming debug logs in response** | Full response object printed | Extracted only `message.content` from LLaMA output |
| **Elephant misclassified as cat** | LLaMA guessing from poor OCR data | Integrated YOLOv8 for accurate object detection |
| **YOLO import error** | Missing dependency | Installed `ultralytics` via pip |

---

## Key Learnings
1. Streamlit customization for **chat-like UI**.
2. Importance of **structured detection data** before passing to LLMs.
3. **YOLOv8 is a reliable workaround** when multimodal models (like LLaVA) fail in Ollama.
4. Debugging Ollama requires checking **installation paths and versions** carefully.
5. Clean handling of LLaMA responses improves **professional accuracy**.

---

## Current State
- Fully working **offline assistant**.  
- Supports **image upload**, **object detection via YOLO**, and **contextual answering via LLaMA3**.  
- UI is **dark-themed, professional, and dynamic**.

---


