import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np

from my_model.model import FacialExpressionModel

# Set up the necessary objects and parameters
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier("C:/Users/Akansha/OneDrive/Desktop/frecog/haarcascade_frontalface_default.xml")
model = FacialExpressionModel("C:/Users/Akansha/OneDrive/Desktop/my_model/model.json", "C:/Users/Akansha/OneDrive/Desktop/my_model/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the function to detect faces
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        fc = gray[y:y + h, x:x + w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img, faces, pred

# Define the main function
def main():
    """Face Expression Detection App"""
    # Setting the app title & sidebar
    activities = ["Home", "Model Performance"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Home':
        st.title("Face Expression Detection App")
        st.caption("Upload an image or enable your camera to detect facial expressions.")

        # File uploader for uploading images
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

        # Camera input for capturing images
        picture = st.camera_input("Take a picture")

        # Handle image upload
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)
        elif picture:
            st.image(picture)
            our_image = Image.fromarray(picture)
        else:
            st.warning("Please upload an image or take a picture.")

        # Process the uploaded or captured image
        if st.button("Process"):
            result_img, result_faces, prediction = detect_faces(our_image)
            if result_img is not None:
                st.success("Found {} faces".format(len(result_faces)))
                st.image(result_img)
                if prediction in ['Happy', 'Neutral', 'Surprise']:
                    st.subheader("Feeling relaxed and happy?")
                elif prediction in ['Angry', 'Sad', 'Disgust', 'Fear']:
                    st.subheader("Feeling a bit stressed?")
                else:
                    st.error("Unable to detect emotion properly.")

    elif choice == 'Model Performance':
        # Model performance section, kept as it is
        st.title("Model Performance")
        st.text("We have used a convolutional neural network:")
        st.image('/content/drive/MyDrive/embedded/images/model.png', width=700)
        st.subheader("To train the model we used the FER2013 dataset")
        st.text("https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        st.image('/content/drive/MyDrive/embedded/images/dataframe.PNG', width=700)
        st.subheader("Lets look at the performance!")
        st.markdown("Accuracy :chart_with_upwards_trend: :")
        st.image("/content/drive/MyDrive/embedded/images/accuracy.PNG")
        st.markdown("Loss :chart_with_downwards_trend: : ")
        st.image("/content/drive/MyDrive/embedded/images/Loss.PNG")

main()
