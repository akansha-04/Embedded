import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from model_01.model import FacialExpressionModel
from pyserial import Serial  # Import the serial library

# Initialize the serial connection to the Arduino
ser = serial.Serial('COM5', 9600) 

st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier("frecog/haarcascade_frontalface_default.xml")
model = FacialExpressionModel("model_01/model.json", "model_01/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img, len(faces), pred

def main():
    st.title("Welcome to Emotion Detection!")
    image_file = st.file_uploader("Enable your video to capture the image:")

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
    else:
        st.error("No image uploaded yet")

    if st.button("Process"):
        result_img, num_faces, prediction = detect_faces(our_image)
        if result_img is not None:
            st.image(result_img)
            st.success(f"Found {num_faces} face(s)")
            if prediction in ['Happy', 'Neutral', 'Surprise']:
                st.subheader("Feeling relaxed and happy?")
                # Send a signal to the Arduino (e.g., turn on an LED)
                ser.write(b'H')  # Send 'H' to Arduino
            elif prediction in ['Angry', 'Sad', 'Disgust', 'Fear']:
                st.subheader("Feeling a bit stressed? Don't worry!")
                # Send a different signal to the Arduino (e.g., turn off the LED)
                ser.write(b'L')  # Send 'L' to Arduino
            else:
                st.error("Uh Oh! We weren't able to detect an emotion properly. Please try again.")

main()

