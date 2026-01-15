import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
import csv
import json
from datetime import datetime

# -----------------------------
# Load pre-trained model
# -----------------------------
emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# Folder setup to save faces
# -----------------------------
save_folder = "Detected_Faces"
os.makedirs(save_folder, exist_ok=True)

# CSV log file
log_file = "emotion_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Emotion", "Confidence", "Image_File"])

# JSON log file
json_log_file = "emotion_log.json"
log_data = []

# Emotion summary dictionary
emotion_count = {label: 0 for label in emotion_labels}

# -----------------------------
# Camera function
# -----------------------------
def open_camera():
    for index in range(0, 3):  # Try camera indexes 0,1,2
        cap = cv2.VideoCapture(index)
        time.sleep(2)
        if cap.isOpened():
            print(f"Camera opened at index {index}")
            return cap
        else:
            cap.release()
    return None

cap = open_camera()
if cap is None:
    print("Error: No camera detected. Please check camera or drivers.")
    exit()

print("Camera started. Press 'q' to exit, 's' to save a face manually, 'p' to pause, 'r' to resume.")

paused = False  # Flag for pause/resume

# -----------------------------
# Main loop
# -----------------------------
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            face_normalized = face_resized / 255.0
            face_input = np.reshape(face_normalized, (1, 64, 64, 1))

            prediction = emotion_model.predict(face_input, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            confidence = float(prediction[0][emotion_index])

            # Update session summary
            emotion_count[emotion] += 1

            # Draw rectangle and label with confidence
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Real Emotion Detection", frame)

    # -----------------------------
    # Key controls
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s') and not paused:  # Manual save current detected faces
        for i, (x, y, w, h) in enumerate(faces):
            face = gray[y:y+h, x:x+w]

            # Optional: Preview face before saving
            cv2.imshow("Save Preview", face)
            cv2.waitKey(500)
            cv2.destroyWindow("Save Preview")

            # Folder structure by date + emotion
            date_folder = datetime.now().strftime("%Y-%m-%d")
            emotion_folder = os.path.join(save_folder, date_folder, emotion)
            os.makedirs(emotion_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{emotion_folder}/{emotion}_{timestamp}_{i}.jpg"
            cv2.imwrite(image_filename, face)

            # CSV log
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, emotion, f"{confidence:.2f}", image_filename])

            # JSON log
            log_data.append({"timestamp": timestamp, "emotion": emotion,
                             "confidence": confidence, "file": image_filename})

            print(f"Manually saved: {image_filename}")
    elif key == ord('p'):
        paused = True
        print("Detection paused. Press 'r' to resume.")
    elif key == ord('r'):
        paused = False
        print("Detection resumed.")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# Save JSON log
with open(json_log_file, "w") as json_file:
    json.dump(log_data, json_file, indent=4)

# Print session summary
print("\n--- Emotion Summary This Session ---")
for em, count in emotion_count.items():
    print(f"{em}: {count}")

print("\nCamera stopped. Faces saved in 'Detected_Faces/' and logs updated in CSV & JSON.")
