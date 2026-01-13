import streamlit as st
import pandas as pd
import time
import subprocess
import os

st.set_page_config(
    page_title="Driver Alertness Monitoring System",
    layout="wide",
    page_icon="🚗"
)

#Header
st.markdown("""
<h1 style='text-align:center;'>🚗 AI-Based Driver Alertness Monitoring System</h1>
<p style='text-align:center; color:gray;'>
Real-time drowsiness detection using Computer Vision & Machine Learning
</p>
<p style='text-align:center; color:gray;'>
Build for road safety, driver alertness and accident prevention
</p>
""", unsafe_allow_html=True)


st.markdown("---")

#---------------------------------SIDEBAR---------------------------------


st.sidebar.header("System Controls")
start=st.sidebar.button("▶ Start Detection")
stop=st.sidebar.button("⛔ Stop Detection")

st.sidebar.markdown("---")

st.sidebar.markdown("### Detection Pipeline")
st.sidebar.markdown("""
1. Camera captures the face
2. Facial landmarks detect Eye in the face
3. EAR measures eye openness
4. ML model confirms open/closed state of eye
5. Blink rate + closure duration detect fatigue
6. System Trigger Alert when the fatigue is detected

""")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🚘 Use Case")
st.sidebar.markdown("""
Designed for:
- Driver drowsiness detection  
- Vehicle safety systems  
- Fleet monitoring  
- Industrial operator safety  
""")


#______________________________________________________________________________________________________________________________
#writing logic for start and stop buton as streamlit rerun script on every button click(every time user interact with ui)
#this block initialize a persistent variable in streamlit
#st.session_state allows us to store valuses permanently across these reruns
#______________________________________________________________________________________________________________________________
if "process" not in st.session_state:  #process is just a key name
    st.session_state.process=None # no background app is running
# if 'process' DOES NOT EXIST in st.session_state it mean that fatigue detection script has never been started

#________________________________________________________________________________________________________________________________________________________________
#START BUTTON LOGIC
#This condition ensures: when startbutton is clicked , no  fatigue detection process is already running, this prevent multiple instances of the camera script:
#________________________________________________________________________________________________________________________________________________________________
if start and st.session_state.process is None:
    st.session_state.process=subprocess.Popen(
        ["Python","scripts/real_time_fatigue_blink.py"]
    )
    #Display success message in Streamlit UI
    st.success("Detection started")
#_______________________________________________________________________________________________________________________________________________________________
#Stop Button Logic
#_______________________________________________________________________________________________________________________________________________________________
if stop and st.session_state.process is not None:
    st.session_state.process.terminate() #terminate the running fatigue detction process , this safely stops camera access, Ml inference, and any active alerts
    st.session_state.process=None #Reset the store process reference, this allow to cleanly restart the process again
    #Displaying warning message
    st.warning("Detection stopped")

#Dashborads from logs
@st.cache_data(ttl=3)
def load_logs():
    try:
        return pd.read_csv("logs/session_log.csv")
    except PermissionError:
        return None
    except FileNotFoundError:
        return None


data=load_logs()

#----------------------------------MAIN DASHBOARD--------------------------------
st.subheader("📊 Live Driver State Analytics")
col1, col2, col3, col4 = st.columns(4)

if data is not None and not data.empty: # we wrote this line because load_logs can return None

    col1,col2,col3,col4=st.columns(4)
    col1.metric("Total Blinks",int(data["blnk_count"].max()))
    col2.metric("Avg EAR",round(data["avgEAR"].mean(),2))
    col3.metric("Fatigue Events",int(data["fatigue_flag"].sum()))
    col4.metric("Session Time (sec)", int(len(data)))

    
else:
    col1.metric("Total Blinks", 0)
    col2.metric("Avg EAR", 0)
    col3.metric("Fatigue Events", 0)
    col4.metric("Session Time", 0)

st.markdown("---")
# ---------------- CHARTS ----------------
if data is not None and not data.empty:

    st.subheader("📈 Eye & Fatigue Signals")

    st.line_chart(
        data[["avgEAR", "blink_rate_per_min"]]
    )

    st.markdown("### 🚨 Fatigue Timeline")
    st.bar_chart(data["fatigue_flag"])

# ---------------- SYSTEM EXPLANATION ----------------
st.markdown("---")
st.subheader("🛡️ Safety Logic")

st.markdown("""
The system continuously evaluates driver alertness using:

• **Eye Aspect Ratio (EAR)**  
Detects eye openness and micro-sleep events  

• **ML Eye Classifier**  
Distinguishes open vs closed eyes  

• **Blink Rate & Duration**  
Abnormal blinking = fatigue  

• **Hybrid Risk Model**  
Combines all signals to detect drowsiness  

When risk exceeds safe limits, the driver is alerted immediately.
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>AI Safety System | Built using Python, OpenCV, MediaPipe & Scikit-Learn</center>",
    unsafe_allow_html=True
)
