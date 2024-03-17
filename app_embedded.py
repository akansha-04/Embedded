import streamlit as st 
import serial
import path
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
import sys
dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)
from my_model.model import FacialExpressionModel
import time
from bokeh.models.widgets import Div

st.set_option('deprecation.showfileUploaderEncoding', False)
try:
    face_cascade = cv2.CascadeClassifier("./frecog/haarcascade_frontalface_default.xml")
    model = FacialExpressionModel("./my_model/model.json", "./my_model/model_weights.h5")
    font = cv2.FONT_HERSHEY_SIMPLEX
except FileNotFoundError:
    st.error("Error: Unable to find one or more required files.")
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
    st.title("Welcome to Bouquet!")
    image_file = st.camera_input("Enable your video to capture the image:")
    
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
                st.subheader("Feeling relaxed and happy? Let our scent diffuser set the mood for you! Sit back and enjoy the fragrance!")
            elif prediction in ['Angry', 'Sad', 'Disgust', 'Fear']:
                st.subheader("Feeling a bit stressed? Don't worry! Sit back, relax and let our diffuser take care of you!")
            else:
                st.error("Uh Oh! We weren't able to detect an emotion properly. Please try again.")
main()
