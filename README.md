🚗 AI-Based Driver Alertness Monitoring System

A real-time computer vision and machine learning system that detects driver fatigue and drowsiness using eye behavior analysis, blink patterns, and a hybrid ML + geometry-based approach.
The system provides live alerts, session analytics, and a web-based monitoring dashboard deployed on the cloud.

🔗 Live Demo:
https://driver-alertness-monitoring-system-2fmxzazppyoz7c9geytl6x.streamlit.app

📌 Problem Statement

Driver fatigue is one of the leading causes of road accidents.
Traditional systems rely on vehicle behavior, but this project focuses directly on the driver’s eyes and blinking patterns to detect loss of alertness.

This system continuously monitors:

Eye openness

Blink rate

Blink duration

Micro-sleep events

and triggers alerts when fatigue is detected.



🧠 System Overview

The project is built as a Hybrid AI System using three layers of intelligence:

Layer	Description

Computer Vision	Face and eye detection using MediaPipe
Geometry	Eye Aspect Ratio (EAR) to measure eye openness
Machine Learning	SVM model classifies eye as Open or Closed
Behavioral Analysis	Blink rate and eye-closure duration
Decision Engine	Hybrid logic to detect fatigue


⚙️ How the System Works

Webcam captures the driver’s face

MediaPipe detects facial landmarks

Eye regions are extracted

EAR (Eye Aspect Ratio) measures eye openness

ML model (SVM) predicts open/closed eye state

Blink rate & closure duration are calculated

A hybrid fatigue score is computed

If risk is high → alert is triggered

All data is logged and visualized on the dashboard


🖥️ Features

🔴 Real-time fatigue detection

👁️ Eye Aspect Ratio (EAR) based geometry

🤖 ML-based eye state classifier (SVM)

👀 Blink rate & duration analysis

🔊 Audio alert system

📊 Live dashboard with:

Total blinks

Average EAR

Fatigue events

Session duration

☁️ Cloud-deployed Streamlit interface



📊 Dashboard

The Streamlit dashboard displays:

Real-time session analytics

EAR and blink graphs

Fatigue timeline

System status

This allows supervisors or users to visually monitor driver alertness.



🧪 Dataset

A custom dataset of eye images was created:

Open eyes

Closed eyes

Images were captured using webcam and preprocessed for training the SVM classifier.

🧠 Machine Learning Model

Algorithm: Support Vector Machine (SVM)

Input: Grayscale 24×24 eye images

Output: Open (1) / Closed (0)

Combined with EAR for high accuracy



🛠️ Tech Stack

Python

OpenCV

MediaPipe

NumPy / Pandas

Scikit-Learn

Streamlit

GitHub + Streamlit Cloud




🚀 Deployment

The system is deployed on Streamlit Community Cloud and is accessible via browser.

Due to cloud limitations, live webcam access runs locally, while the deployed version shows analytics and system behavior using logs and demo video.



📂 Project Structure
eyeFatigue/
│
├── app.py                 # Streamlit dashboard
├── dataset/               # Open/closed eye images
├── models/                # Trained SVM and data
├── scripts/               # Real-time detection + training scripts
├── logs/                  # Session logs
└── README.md



📌 Future Improvements

Head pose detection

Yawn detection

CNN-based eye classifier

Mobile deployment

Cloud video input



👨‍💻 Author

Faiz Ahmad
Final Year B.Tech (CSE)
This project was built as a placement-level AI system with real-time deployment.