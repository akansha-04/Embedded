import streamlit as st 
import cv2
from PIL import Image
import numpy as np 
from model_01.model import FacialExpressionModel

st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = FacialExpressionModel("model_01/model.json", "model_01/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    predictions = []
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        predictions.append(pred)
        cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, len(faces), predictions

def main():
    st.title("Welcome to Emotion Detection!")

    # Capture video from webcam
    frame = st.camera_input("Take a picture")
    
    result_img, num_faces, predictions = detect_faces(frame)
    if result_img is not None:
        st.image(result_img, channels="BGR")
        st.success(f"Found {num_faces} face(s)")
        for prediction in predictions:
            if prediction in ['Happy', 'Neutral', 'Surprise']:
                st.subheader("Feeling relaxed and happy?")
            elif prediction in ['Angry', 'Sad', 'Disgust', 'Fear']:
                st.subheader("Feeling a bit stressed? Don't worry!")
            else:
                st.error("Uh Oh! We weren't able to detect an emotion properly. Please try again.")
main()
