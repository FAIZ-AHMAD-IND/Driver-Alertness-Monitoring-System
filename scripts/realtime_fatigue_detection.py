import cv2
import os
import numpy as np
import joblib

# Optional: sound alarm (works on Windows)
try:
    import winsound
    HAVE_WINSOUND = True
except ImportError:
    HAVE_WINSOUND = False

#_____________________________________________________________________________
# STEP 1: PATH SETUP AND LOADING THE MODELS
#_____________________________________________________________________________
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Paths to Haar Cascade files (for face and eye detection)
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_eye.xml")
EYEGLASS_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_eye_tree_eyeglasses.xml")

# Load the Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
eyeglass_cascade = cv2.CascadeClassifier(EYEGLASS_CASCADE_PATH)

# Load the trained SVM eye-state classification model
model = joblib.load(os.path.join(MODELS_DIR, "eye_state_svm.pkl"))

IMG_SIZE = 24  # must match training

closed_frames = 0  # counter for how many consecutive frames eyes are closed
FATIGUE_THRESHOLD = 20  # tune based on FPS (e.g., ~1 second if ~20 FPS)

# Start the webcam (0 = default camera)
cap = cv2.VideoCapture(0)


#_____________________________________________________________________________
# STEP 2: PREPROCESSING FUNCTION FOR EYE IMAGES
#_____________________________________________________________________________
def preprocess_eye(eye_img_gray):
    """
    Takes a grayscale eye image (ROI),
    resizes to 24x24,
    normalizes pixels to [0, 1],
    flattens to 1D,
    reshapes to (1, -1) for model.predict().
    This must match preprocessing used during training.
    """
    eye_resized = cv2.resize(eye_img_gray, (IMG_SIZE, IMG_SIZE))
    eye_norm = eye_resized / 255.0
    return eye_norm.flatten().reshape(1, -1)


#_____________________________________________________________________________
# STEP 3: READ FRAME BY FRAME FROM THE WEBCAM
#_____________________________________________________________________________
while True:
    ret, frame = cap.read()   # <-- use cap.read(), not cv2.read()
    if not ret:
        break  # if frame not captured properly, exit loop

    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Default text if nothing detected
    status_text = "Unknown"

    # We will only consider the first detected face for simplicity
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract ROI for face (both gray and color)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes inside the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Optional: if no eyes detected, try eyeglass cascade
        if len(eyes) == 0:
            eyes = eyeglass_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Use only the first detected eye (to avoid multiple predictions per frame)
        for (ex, ey, ew, eh) in eyes:
            # Extract eye region correctly: [rows, cols] = [y:y+eh, x:x+ew]
            eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]

            # Draw rectangle around the eye for better visualization
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            #______________________________________________
            # Preprocess Eye ROI and Predict the state
            #______________________________________________
            features = preprocess_eye(eye_roi_gray)  # shape: (1, 576)

            # Predict using trained SVM model (0 = closed, 1 = open)
            pred = model.predict(features)[0]

            if pred == 1:  # open
                status_text = "Eyes Open"
                closed_frames = 0  # reset counter when eye is open
            else:           # closed
                status_text = "Eyes Closed"
                closed_frames += 1  # increment counter when eye is closed

            # Use only the first eye
            break

        # Use only the first face
        break

    #_________________________________________________________________________
    # STEP 4: FATIGUE LOGIC BASED ON CONSECUTIVE CLOSED FRAMES
    #_________________________________________________________________________
    fatigue_text = ""
    if closed_frames >= FATIGUE_THRESHOLD:
        fatigue_text = "FATIGUE DETECTED!"

        # Play sound alarm (simple beep) if winsound is available (Windows only)
        if HAVE_WINSOUND:
            winsound.Beep(1000, 500)  # frequency=1000 Hz, duration=500 ms

    #_________________________________________________________________________
    # STEP 5: DISPLAY TEXT ON THE FRAME
    #_________________________________________________________________________
    # Status text: "Eyes Open" / "Eyes Closed" / "Unknown"
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if status_text == "Eyes Open" else (0, 0, 255),
        2
    )

    # If fatigue detected, show alert text
    if fatigue_text:
        cv2.putText(
            frame,
            fatigue_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # Show the frame in a window
    cv2.imshow("Eye Fatigue Detector", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#_________________________________________________________________________
# STEP 6: RELEASE RESOURCES
#_________________________________________________________________________
cap.release()
cv2.destroyAllWindows()
