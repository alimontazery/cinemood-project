
import cv2
import numpy as np
import keras
from collections import deque

# 1. Emotion classes — same order as training
CLASSES = ["angry", "happy", "neutral", "sad", "surprise"]

# 2. Load trained model
model = keras.models.load_model("fer2_model_72_percent.keras")

# 3. Face detector
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if cascade.empty():
    raise RuntimeError("Could not load the face detector!")

# Separate history for each tracked face
histories = {}
next_id = 0

# Webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )

try:
    while True:
        ok, frame = cam.read()

        if not ok:
            print("Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        # Match detected faces to previous face positions
        current_faces = []
        used_ids = set()

        for (x, y, w, h) in faces:
            cx = x + w / 2
            cy = y + h / 2

            best_id = None
            best_distance = float("inf")

            for face_id, data in histories.items():
                if face_id in used_ids:
                    continue

                old_cx, old_cy = data["center"]

                distance = np.sqrt(
                    (cx - old_cx) ** 2
                    + (cy - old_cy) ** 2
                )

                threshold = max(w, h, data["size"]) * 1.2

                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_id = face_id

            if best_id is None:
                best_id = next_id
                next_id += 1

                histories[best_id] = {
                    "center": (cx, cy),
                    "size": max(w, h),
                    "probs": deque(maxlen=8)
                }

            used_ids.add(best_id)

            data = histories[best_id]
            data["center"] = (cx, cy)
            data["size"] = max(w, h)

            # Crop and preprocess face
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = face.reshape(1, 48, 48, 1)

            # Predict emotion
            probs = model.predict(face, verbose=0)[0]

            # Keep independent prediction history
            data["probs"].append(probs)
            avg_probs = np.mean(data["probs"], axis=0)

            idx = int(np.argmax(avg_probs))
            confidence = float(avg_probs[idx])

            label = (
                f"Face {best_id + 1}: "
                f"{CLASSES[idx]} {confidence:.0%}"
            )

            # Draw face rectangle and prediction
            color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            text_y = max(y - 10, 25)

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )

        # Display number of detected faces
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("CineMood AI - Multi Face", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
