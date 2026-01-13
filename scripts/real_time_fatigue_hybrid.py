
import os
import cv2
import numpy as np
import joblib
from scipy.spatial import distance as dist
import mediapipe as mp

# ---------------------------
# PATHS & MODEL LOADING
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "eye_state_svm.pkl")

# Load trained SVM model (fall back with clear error if missing)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ---------------------------
# CONSTANTS / HYPERPARAMS
# ---------------------------
IMG_SIZE = 24             # must match training preprocessing
FATIGUE_THRESHOLD = 20    # consecutive closed frames -> fatigue (tune by FPS)
EAR_THRESHOLD = 0.21      # eye aspect ratio threshold (tune per person)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Landmark indices (6 points used for EAR)
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]

# ---------------------------
# UTILITIES
# ---------------------------
def eye_aspect_ratio(eye_points):
    """
    eye_points: list of 6 (x,y) tuples in pixel coords
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Return 0.0 if denominator is zero (safe guard).
    """
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def preprocess_eye(eye_img_gray):
    """
    Convert a grayscale eye ROI to the vector expected by the SVM:
      - resize to IMG_SIZE x IMG_SIZE
      - normalize to [0,1]
      - flatten and reshape to (1, -1)
    Returns None on failure (too small ROI or resize error).
    """
    try:
        eye_resized = cv2.resize(eye_img_gray, (IMG_SIZE, IMG_SIZE))
    except Exception:
        return None
    eye_norm = eye_resized.astype(np.float32) / 255.0
    return eye_norm.flatten().reshape(1, -1)


# ---------------------------
# MAIN LOOP
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera index / permissions.")

closed_frames = 0  # consecutive closed-frame counter

# Use FaceMesh with refine_landmarks=True to get accurate eye landmarks
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "Unknown"
        fatigue_text = ""

        if results.multi_face_landmarks:
            # Take the first detected face
            face_landmarks = results.multi_face_landmarks[0]

            # Convert normalized landmarks to pixel (x, y) tuples
            lm = []
            for lm_i in face_landmarks.landmark:
                x_px = int(lm_i.x * img_w)
                y_px = int(lm_i.y * img_h)
                lm.append((x_px, y_px))

            # Build left & right eye point lists
            left_eye_pts = [lm[idx] for idx in L_EYE]
            right_eye_pts = [lm[idx] for idx in R_EYE]

            # Compute EAR per eye and average
            leftEAR = eye_aspect_ratio(left_eye_pts)
            rightEAR = eye_aspect_ratio(right_eye_pts)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # Compute bounding box around left eye (you can combine both eyes if desired)
            lx = min(p[0] for p in left_eye_pts)
            rx = max(p[0] for p in left_eye_pts)
            ly = min(p[1] for p in left_eye_pts)
            ry = max(p[1] for p in left_eye_pts)  # <-- must be max

            # Expand bbox slightly and clamp to image bounds
            pad = 6
            x1 = max(0, lx - pad)
            x2 = min(img_w, rx + pad)
            y1 = max(0, ly - pad)
            y2 = min(img_h, ry + pad)

            # Extract grayscale eye ROI safely
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eye_roi_gray = gray_frame[y1:y2, x1:x2]

            # Default ML prediction = open (1) in case of failure
            ml_pred = 1
            if eye_roi_gray.size != 0:
                features = preprocess_eye(eye_roi_gray)
                if features is not None:
                    try:
                        ml_pred = int(model.predict(features)[0])
                    except Exception:
                        # If model.predict fails, keep ml_pred = 1 (open)
                        ml_pred = 1

            # HYBRID DECISION:
            # Mark closed only when BOTH ML says closed (ml_pred==0) AND avgEAR < EAR_THRESHOLD
            if (ml_pred == 0) and (avgEAR < EAR_THRESHOLD):
                status_text = "Eyes Closed"
                closed_frames += 1
            else:
                status_text = "Eyes Open"
                closed_frames = 0

            # Draw rectangle around the eye ROI and show EAR for debugging
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"EAR:{avgEAR:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Fatigue detection based on consecutive closed frames
        if closed_frames >= FATIGUE_THRESHOLD:
            fatigue_text = "FATIGUE DETECTED!"

        # Overlay status and fatigue text on the frame
        cv2.putText(frame, f"Status: {status_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if "Open" in status_text else (0, 0, 255), 2)

        if fatigue_text:
            cv2.putText(frame, fatigue_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Hybrid Eye Fatigue Detector (MediaPipe)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
