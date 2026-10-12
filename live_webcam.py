import cv2
import numpy as np
import keras
from collections import deque

CLASSES = ["angry", "happy", "neutral", "sad", "surprise"]  # same order as training
model = keras.models.load_model("model_v2_69.keras")
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

history = deque(maxlen=8)  # last 8 predictions, for smoothing

cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)) / 255.0
        probs = model.predict(face.reshape(1, 48, 48, 1), verbose=0)[0]

        history.append(probs)
        avg = np.mean(history, axis=0)       # average of last frames
        idx = int(np.argmax(avg))
        label = f"{CLASSES[idx]} {avg[idx]:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("CineMood live (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()