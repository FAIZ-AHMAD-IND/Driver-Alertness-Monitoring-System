import cv2
import os

# step 1. path setup
# __file__ = path of this .py file
# os.path.dirname(__file__) = folder of this .py file
# os.path.dirname(os.path.dirname(__file__)) = parent of that folder (project root)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up from scripts/

# dataset/ folder inside project root
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# two folders inside dataset: one for open eyes and one for closed eyes
OPEN_DIR = os.path.join(DATASET_DIR, "open_eyes")
CLOSED_DIR = os.path.join(DATASET_DIR, "closed_eyes")  # <- use closed_eyes for consistency

# create the folders if they don't exist
os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

# step 2: load the Haar cascade models
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_eye.xml")
EYE_GLASS_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_eye_tree_eyeglasses.xml")

# Load the pretrained Haar Cascade classifiers from XML
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
eye_glass_cascade = cv2.CascadeClassifier(EYE_GLASS_CASCADE_PATH)

# step 3: open the camera
cap = cv2.VideoCapture(0)

# Count how many images already exist, so we continue filenames from there and don't overwrite
open_count = len(os.listdir(OPEN_DIR))
closed_count = len(os.listdir(CLOSED_DIR))

print("Press 'o' to save Open Eye image")
print("Press 'c' to capture Closed Eye image")
print("Press 'q' to quit")

# step 4: read frames and save images
while True:
    # frame will store the image, ret will store True or False
    ret, frame = cap.read()   # <- FIXED: removed (0)
    if not ret:
        break  # if no frame was captured, break the loop

    # Convert the captured image to grayscale because Haar cascades work on gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # STEP 5: Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest (ROI) for the face in gray and color images
        roi_gray = gray[y:y + h, x:x + w]       # gray face portion
        roi_color = frame[y:y + h, x:x + w]     # color face portion

        # STEP 6: Detect eyes inside the face region
        # We run eye detection only inside the face ROI to reduce noise and unnecessary scanning
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Optional: if no eyes detected, try eyeglass cascade
        if len(eyes) == 0:
            eyes = eye_glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            # Eye region in gray (for saving) and color (for drawing rectangle)
            eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_roi_color = roi_color[ey:ey + eh, ex:ex + ew]

            # Draw rectangle around the eyes on the face region (for visualization)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Show the eye ROI in a separate window
            cv2.imshow("EYE ROI", eye_roi_gray)

            # KEY HANDLING (inside eye loop)
            key = cv2.waitKey(1) & 0xFF

            # if 'o' is pressed, save this eye ROI as an open-eye image
            if key == ord('o'):  # ord gives ASCII code of the character
                img_path = os.path.join(OPEN_DIR, f"open_{open_count}.jpg")
                cv2.imwrite(img_path, eye_roi_gray)
                open_count += 1
                print(f"Saved Open eye: {img_path}")

            # if 'c' is pressed, save this eye ROI as a closed-eye image
            elif key == ord('c'):
                img_path = os.path.join(CLOSED_DIR, f"closed_{closed_count}.jpg")
                cv2.imwrite(img_path, eye_roi_gray)
                closed_count += 1
                print(f"Saved Closed eye: {img_path}")

            # if 'q' is pressed, exit immediately
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # show the main webcam frame with rectangles on the face/eyes
    cv2.imshow("Frame", frame)

    # allow quitting even if no face/eye is detected
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
