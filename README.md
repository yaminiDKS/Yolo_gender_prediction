# 🧠 Gender Prediction with YOLOv11 Face Detection

An intelligent computer vision pipeline that detects human faces using **YOLOv11** and classifies gender (Male/Female) with a pre-trained deep learning model. Built for real-time inference on images and video streams.

---

## 🚀 Project Overview

This project integrates **YOLOv11** for accurate face detection and a **CNN-based gender classifier** to predict gender from detected facial regions.

### 🔍 Key Features
- ⚡ High-speed, high-accuracy face detection via YOLOv11
- 👩‍🦰 Gender classification using pre-trained deep neural networks
- 📸 Supports image, webcam, and video stream input
- 📦 Easy to integrate into larger surveillance, marketing, or analytics pipelines

---
gender-prediction-yolov11/
│
├── weights/
│   ├── yolov11-face.pt            # Pretrained YOLOv11 face detection model
│   └── gender_model.pth           # Pretrained gender classifier
│
├── src/
│   ├── detector.py                # YOLOv11 face detection
│   ├── gender_classifier.py       # Gender prediction logic
│   └── main.py                    # Entry point for image/video inference
│
├── utils/
│   └── preprocessing.py           # Cropping, resizing, normalization
│
├── requirements.txt
├── README.md
└── app.py (optional Streamlit interface)
---