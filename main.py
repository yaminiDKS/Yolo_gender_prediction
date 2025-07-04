import streamlit as st
from ultralytics import YOLO
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os

# ------------------------------
# Load YOLOv8 face model
model_path = "yolov11n-face.pt"
if not os.path.exists(model_path):
    st.error("❌ Model file 'yolov8n-face.pt' not found. Please download it and place it next to this script.")
    st.stop()

model = YOLO(model_path)

# ------------------------------
# Streamlit UI
st.set_page_config(page_title="Smart Face Analyzer", layout="centered")
st.title("🧠 Smart Face Detection with Age, Gender & Emotion")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)

    # Detect faces using YOLOv8
    results = model(img_np)
    faces = results[0].boxes.xyxy.cpu().numpy()

    st.subheader(f"📸 Detected {len(faces)} face(s)")

    for i, (x1, y1, x2, y2) in enumerate(faces.astype(int)):
        face_crop = img_np[y1:y2, x1:x2]

        try:
            analysis = DeepFace.analyze(face_crop, actions=['gender', 'age', 'emotion'], enforce_detection=False)
            gender_scores = analysis[0]['gender']
            gender = max(gender_scores, key=gender_scores.get)
            gender_conf = gender_scores[gender] * 100

            age = analysis[0]['age']
            emotion_scores = analysis[0]['emotion']
            emotion = max(emotion_scores, key=emotion_scores.get)
            emotion_conf = emotion_scores[emotion] * 100

            label = f"{gender} ({gender_conf:.1f}%) | Age: {int(age)} | {emotion} ({emotion_conf:.1f}%)"
        except Exception as e:
            label = "Unknown"
            st.warning(f"⚠️ Face {i+1} analysis failed: {e}")

        # Draw box and label (smaller font to avoid overlap)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    st.image(img_np, caption="🔍 Analysis Result", channels="RGB", use_column_width=True)
