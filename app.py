import streamlit as st 
import serial
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
from my_model.model import FacialExpressionModel
import time
from bokeh.models.widgets import Div



#importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("my_model/model.json", "my_model/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 

#face exp detecting function
def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:

			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))
			pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
			cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			return img,faces,pred 
#the main function	

def main():
	
	"""Face Expression Detection App"""
	#setting the app title & sidebar

	activities = ["Home" ,"Model Performance","Manual"]
	choice = st.sidebar.selectbox("Select Activity",activities)
	#if choosing to consult the cnn model performance
	if choice == 'Home':
		col1, col2 = st.columns([5,1],gap="medium")
		with col1:
			st.title("Welcome to Bouquet!")
			st.caption("Our Smart Scent Diffuser takes your emotions into consideration. Relax your mind with a unique blend of aromatherapy and artificial intelligence!")
		with col2:
			st.image("logo.png", width=160, use_column_width=None)

		st.subheader("Let's dive into the depths of your emotions with our cutting-edge technology!")
		st.caption("To ensure the most accurate detection, please capture the image in a well-lit environment.")
		image_file = st.camera_input("Enable your video to capture the image:")
    
	#if image if uploaded,display the progress bar +the image
		if image_file is not None:
				our_image = Image.open(image_file)
				st.text("Original Image")
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.01)
					progress.progress(i+1)
				st.image(our_image)
		if image_file is None:
			st.error("No image uploaded yet")

		# Face Detection
		task = ["Faces"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):
			if feature_choice == 'Faces':

				#process bar
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.05)
					progress.progress(i+1)
				#end of process bar
				ser = serial.Serial('COM5', 9600)
				time.sleep(2)
				result_img,result_faces,prediction = detect_faces(our_image)
				if st.image(result_img) :
					st.success("Found {} faces".format(len(result_faces)))	

					if prediction == 'Happy' or prediction == 'Neutral' or prediction == 'Surprise' :
						ser.write(b'0\n')
						st.subheader("Feeling relaxed and happy? Let our scent diffuser set the mood for you! Sit back and enjoy the fragrance!")


					elif prediction == 'Angry' or prediction == 'Sad' or prediction == 'Disgust' or prediction == 'Fear':
						ser.write(b'1\n')
						st.subheader("Feeling a bit stressed? Don't worry! Sit back, relax and let our diffuser take care of you!")
						
						

					else :
						st.error("Uh Oh! We weren't able to detect an emotion properly. Please try again.")
					
					ser.close()

				
	
	if choice == 'Manual':
				col1, col2 = st.columns([5,1],gap="medium")
				with col1:
					st.title("Welcome to Bouquet!")
					st.caption("Our Smart Scent Diffuser takes your emotions into consideration. Relax your mind with a unique blend of aromatherapy and artificial intelligence!")
				with col2:
					st.image("logo.png", width=160, use_column_width=None)
				
				ser = serial.Serial('COM5', 9600)
				time.sleep(2)
				button = st.button("Are you feeling Happy today?")
				if button:
							st.subheader("Feeling relaxed and happy? Let our scent diffuser set the mood for you! Sit back and enjoy the fragrance!")
							ser.write(b'0\n')
				button1 = st.button("Something got you Stressed?")
				if button1:
							st.subheader("Feeling a bit stressed? Don't worry! Sit back, relax and let our diffuser take care of you!")
							ser.write(b'1\n')
				ser.close()

	if choice == 'Model Performance':
		st.header("Learn more about the advanced model that powers our Smart Scent Diffuser!")
		st.text("We have used a convulutional neural network:")
		st.image('images/model.png', width=700)
		st.subheader("To train the model we used the FER2013 dataset")
		st.text(" https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
		st.image('images/dataframe.png', width=700)
		st.subheader("Lets look at the performance!")
		st.markdown("Accuracy :chart_with_upwards_trend: :")
		st.image("images/accuracy.png")
		st.markdown("Loss :chart_with_downwards_trend: : ")
		st.image("images/loss.png")
	#if choosing to detect your face exp , give access to upload the image

main()
#reset_output_buffer()