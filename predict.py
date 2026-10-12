import json
import cv2
import numpy as np
import keras

CLASSES = ["angry", "happy", "neutral", "sad", "surprise"]  # same order as training
model = keras.models.load_model("model2_v2_69.keras")
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_mood(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # biggest face
        gray = gray[y:y + h, x:x + w]

    gray = cv2.resize(gray, (48, 48)) / 255.0
    arr = gray.reshape(1, 48, 48, 1)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {"mood": CLASSES[idx], "confidence": round(float(probs[idx]), 2)}

for name in ["happy", "sad", "neutral"]:
    result = predict_mood(f"my_photos/{name}.jpg")
    print(name, "->", json.dumps(result))
