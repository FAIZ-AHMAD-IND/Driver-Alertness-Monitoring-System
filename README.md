# 🚗 AI-Based Driver Alertness Monitoring System

Real-time driver fatigue & drowsiness detection using **Computer Vision + Machine Learning**

---

## 📌 Project Overview

This system monitors a driver’s eye behavior in real-time and detects fatigue using a **hybrid AI model**:

- Eye Aspect Ratio (EAR)
- ML-based eye state classifier
- Blink rate & closure duration
- Hybrid fatigue decision logic

The goal is to prevent accidents caused by **drowsy driving**.

---

## 🧠 Technologies Used

- Python  
- OpenCV  
- MediaPipe  
- Scikit-Learn  
- Streamlit  
- NumPy, Pandas  

---

## ⚙️ How It Works

1. Camera captures face  
2. MediaPipe detects eye landmarks  
3. EAR measures eye openness  
4. ML model classifies eye as open/closed  
5. Blink behavior is analyzed  
6. Fatigue is detected & alarm is triggered  

---

## 🌐 Deployment

The system is deployed on **Streamlit Community Cloud**.

Due to cloud security limitations, **live webcam runs locally**, while the **deployed version shows**:

- Real session analytics  
- Blink & EAR graphs  
- Fatigue event logs  
- Demo video  

🔗 **Live App:**  
*(Paste your Streamlit URL here)*

---

## 📁 Project Structure

```text
eyeFatigue/
│
├── app.py              # Streamlit dashboard
├── dataset/            # Eye images (open / closed)
├── models/             # Trained SVM + NumPy data
├── scripts/            # Real-time detection & training
├── logs/               # Session CSV logs
└── README.md
```

---

## 📊 Dashboard Features

- Total blinks  
- Average EAR  
- Fatigue events  
- Session time  
- Live charts of EAR & blink rate  

---

## 🚀 Future Improvements

- Head pose detection  
- Yawn detection  
- CNN-based eye classifier  
- Mobile deployment  
- Cloud video input  

---

## 👨‍💻 Author

**Faiz Ahmad**  
Final Year B.Tech (CSE)  

This project was built as a **placement-level AI system** with real-time deployment.
