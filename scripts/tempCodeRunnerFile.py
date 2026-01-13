import cv2
import mediapipe as mp
import os
import numpy as np
import joblib
from scipy.spatial import distance as dist
import time
import csv
import threading
import platform
#______________________________________________________________________________________________________
#OPTIONAL ALARM SETUP( cross-platform)
#_______________________________________________________________________________________________________
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE=True
except:
    PLAYSOUND_AVAILABLE=False
if platform.system()=='Windows':
    try:
        import winsound
        WINSOUND_AVAILABLE=True
    except:
        WINSOUND_AVAILABLE=False
else:
    WINSOUND_AVAILABLE=False

#______________________________________________________________________________________________________________
#configuration/constants/path setup
#______________________________________________________________________________________________________________
BASE_DIR=os.path.dirname(os.path.dirname(__file__))
MODELS_DIR=os.path.join(BASE_DIR,"models")
MODEL_PATH=os.path.join(MODELS_DIR,"eye_state_svm.pkl")

IMG_SIZE=24        #Must match training 
EAR_THRESHOLD=0.22    #eye aspect ratio threshold
LOG_INTERVAL_SECOND=1.0  #logging frequency
LOG_INTERVAL_SECONDS = LOG_INTERVAL_SECOND
ALARM_AUDIO=os.path.join(MODELS_DIR,"alarm.wav")
FATIGUE_FRAME_THRESHOLD = 20

#_______________________________________________________________________________________________________________
#Load Training Model
#________________________________________________________________________________________________________________
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("SVM model not found")
model=joblib.load(MODEL_PATH)

#______________________________________________________________________________________________________________
#Mediapipe Inintialization
#_____________________________________________________________________________________________________________
mp_face_mesh=mp.solutions.face_mesh
#6 landmark indices for EAR calculation
L_EYE=[33,160,158,133,153,144]
R_EYE=[362,385,387,263,373,380]

#_______________________________________________________________________________________________________________
#Utility Functions
#_______________________________________________________________________________________________________________

def eye_aspect_ratio(eye_points):
    """
    EAR=(|P2-P6)|+|P3-P5|)/(2*|P1-P4|)
    Measure eye openness geometrically
    """
    A=dist.euclidean(eye_points[1],eye_points[5])
    B=dist.euclidean(eye_points[2],eye_points[4])
    C=dist.euclidean(eye_points[0],eye_points[3])
    if C==0:
        return 0.0
    return (A+B)/(2.0*C)

def preprocess_eye(eye_img_gray):
    """
    Convert ROI -> SVM input
    """
    try:
        eye_resized=cv2.resize(eye_img_gray,(IMG_SIZE,IMG_SIZE))
    except:
        return None
    eye_norm=eye_resized/255.0
    return eye_norm.flatten().reshape(1,-1)

def play_alarm_sound():
    """
     Non-blocking alarm(threaded)
    """
    try:
        if PLAYSOUND_AVAILABLE and os.path.exists(ALARM_AUDIO):
            threading.Thread(
                target=playsound,
                args=(ALARM_AUDIO,),
                daemon=True
            ).start()
        elif WINSOUND_AVAILABLE:
            threading.Thread(
                target=winsound.PlaySound,
                args=("SystemHand",winsound.SND_ALIAS),
                daemon=True
            ).start()
    except Exception as e:
        print("Alarm error:",e)

#________________________________________________________________________________________________________________
#Logging Setup
#_________________________________________________________________________________________________________________

LOG_PATH=os.path.join(MODELS_DIR,"session_log.csv")

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow([
            "timestamp","avgEAR","is_closed",
            "closed_frames","blnk_count",
            "blink_rate_per_min","average_blink_duration",
            "fatigue_flag"
        ])

#_________________________________________________________________________________________________________________________________________
#STATE VARIABLES
#_________________________________________________________________________________________________________________________________________
closed_frames=0
prev_eye_state=1       #1=open , 0=closed
closed_start=None
closed_durations=[]
blink_count=0
start_time=time.time()
last_log_time=time.time()

#__________________________________________________________________________________________________________________________________________
#Start camera and mediapipe
#__________________________________________________________________________________________________________________________________________
cap=cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        img_h,img_w=frame.shape[:2]
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(rgb_frame)

        status="Unknown"
        fatigue_text=""
        is_closed=False
        avgEAR=0.0

        #___________________________________________________________________________________________________________________________________________
        # FACE AND EYE PROCESSING
        #____________________________________________________________________________________________________________________________________________
        if results.multi_face_landmarks:
            face_landmarks=results.multi_face_landmarks[0]

            # convert landmarks to pixels coordinates
            lm=[]
            for pt in face_landmarks.landmark:
                lm.append((int(pt.x*img_w),int(pt.y*img_h)))

            left_eye_pts=[lm[i] for i in L_EYE]   
            right_eye_pts=[lm[i] for i in R_EYE]

            leftEAR=eye_aspect_ratio(left_eye_pts)
            rightEAR=eye_aspect_ratio(right_eye_pts)
            avgEAR=(leftEAR+rightEAR)/2.0

            #Eye bounding box
            lx,rx=min(p[0] for p in left_eye_pts),max(p[0] for p in left_eye_pts)
            ly,ry=min(p[1] for p in left_eye_pts),max(p[1] for p in left_eye_pts)

            pad=5
            x1,x2=max(0,lx-pad),min(img_w,rx+pad)
            y1,y2=max(0,ly-pad),min(img_h,ry+pad)

            eye_roi_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[y1:y2,x1:x2]

            #ML Prediction
            ml_pred=1
            if eye_roi_gray.size != 0:
                features = preprocess_eye(eye_roi_gray)
                if features is not None:
                    ml_pred = model.predict(features)[0]
        
            #HYBRID DECISION  
            if ml_pred==0 and avgEAR<EAR_THRESHOLD:
                is_closed=True
                status_text="Eyes Closed"
            else:
                status_text="Eyes Open"

            cv2.rectangle(
                frame,(x1,y1),(x2,y2),
                (0,0,255) if is_closed else (0,255,0),2
            )

        #____________________________________________________________________________________________________________________________________________
        #______________________Blink Detection_______________________________________________________________________________________________________
        #____________________________________________________________________________________________________________________________________________
        if is_closed:    # if eye is currently closed
            if prev_eye_state==1:  #if in previous frame eye was open this mean a new eye closure has just started
                closed_start=time.time()    #record closure start time
            closed_frames+=1 # count for how many consecutive frames eye is closed
        else:     # if eye is currently open
            if prev_eye_state==0 and closed_start is not None:  # if in the previous frame eye was closedthis mean blink has just completed
                blink_count+=1 #increment blink counter
            
                #calculate how long the eye was closed
                blink_duration=time.time()-closed_start
                closed_durations.append(blink_duration)

                closed_start=None #reset for nrxt blink
            closed_frames=max(0,closed_frames-2) #reset closed frame counter

        #update previous eye state for next frame
        #0=closed , 1=open
        prev_eye_state=0 if is_closed else 1

        #_____________________________________________________________________________________________________________________________________________________________________________________
        # Blink metric calculation
        #______________________________________________________________________________________________________________________________________________________________________________________
        elapsed=time.time()-start_time #total time since program started
        #blink rate calculation(blink per minute)
        #only calculate after 5 second to avoid unstable values
        blink_rates=(blink_count*60/elapsed) if elapsed>5 else 0
        # to calculate averaget blink duration use mean if atleast one blin occured
        avg_blink_duration=np.mean(closed_durations) if closed_durations else 0

        #_______________________Fatigue Decision Logic____________________________
        #Blink based fatigue condition
        #1) Average blink duration is too high
        #2) blink rate is abnormally low or abnormally high
        blink_fatigue=(
            avg_blink_duration>0.7     #eye closed too long ->drowsiness
            or blink_rates<5          # too few blink -> eye strain /fatifue
            or blink_rates>40         # too many blinks ->irritation /fatigue
        )

        #Final Fatigue Logic
        #Fatigue is detected if :
        #1) Eye remain closed for many consecutive frames(micro sleeps)
        #2) Blink behavior shows fatigue
        primary_fatigue = closed_frames >= FATIGUE_FRAME_THRESHOLD
        secondary_fatigue = blink_fatigue and closed_frames >= FATIGUE_FRAME_THRESHOLD//2

        fatigue_flag = primary_fatigue or secondary_fatigue

        # ---------------- DEBUG REASON (ADD HERE) ----------------
        debug_reason = ""

        if primary_fatigue:
            debug_reason = "Primary: Eye Closure"
        elif secondary_fatigue:
            debug_reason = "Secondary: Blink + Partial Closure"

        # ---------------- ALERT HANDLING ----------------
        if elapsed < 50: #ignore blink logic for first 50 sec , this will stabalize the system
           blink_fatigue = False
        # If fatigue is detected, show warning and play alarm
        if fatigue_flag:
            fatigue_text = "FATIGUE DETECTED!"
            play_alarm_sound()
            ALARM_COOLDOWN = 10  #Only allow alarm if:fatigue detected,AND last alarm was more than 10s ago

            



        # ---------------- DISPLAY ----------------

        cv2.putText(frame, f"Status: {status_text}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0) if not is_closed else (0,0,255), 2)

        cv2.putText(frame, f"Blinks: {blink_count}  Rate/min: {blink_rates:.1f}",
                        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"Avg Blink Dur: {avg_blink_duration:.2f}s",
                        (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        cv2.putText(frame, debug_reason, (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if fatigue_text:
            cv2.putText(frame, fatigue_text, (20,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        cv2.imshow("Hybrid Eye Fatigue Detector (Blink + EAR)", frame)

        # ---------------- LOGGING ----------------
        if time.time() - last_log_time >= LOG_INTERVAL_SECONDS:
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    round(time.time(),2), round(avgEAR,3), int(is_closed),
                    closed_frames, blink_count,
                    round(blink_rates,2), round(avg_blink_duration,3),
                    int(fatigue_flag)
                ])
            last_log_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
