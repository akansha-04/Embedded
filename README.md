
# Emotion Detection and Connection to Arduino Uno
The objective of this project is to create a CNN-based emotion detection model that will be connected to a web server built using the Streamlit library. The model's output will be utilized to operate an Arduino board, thereby creating an integrated system.

To structure the project, it can be divided into three phases: creating the web application, configuring the hardware, and executing the integration between the web application and the hardware.

## Dataset Used
The [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset was used. The data consists of 48x48 pixel grayscale images of faces. The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

Each face is categorised into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

## Web Application Interface

- The web application aims to detect the user's emotion from camera input and use it to activate a smart scent diffuser.
- The system utilizes Haar cascades to detect faces in the camera input and face alignment techniques to center and align them.
- Emotion detection is done using a CNN that has been trained on a dataset of labelled facial expressions.
- The CNN uses convolutional layers to learn image features and fully connected layers to classify emotions.
- The CNN's output is a probability distribution over emotion labels, and the highest probability label is selected as the predicted emotion.
- Based on the predicted emotion, the system sends signals to the diffuser to release a fragrance tailored to the user's emotional state.

## Hardware
- For the smart scent diffuser, we used 2 servo motors, an Arduino Uno board, and a USB serial cable.
- Jumper wires were used to connect the servo motors to the Arduino board.
- The Arduino board was connected to a laptop running the web application using a USB serial cable.

## Integration 
- To connect the Arduino board and web application using PySerial, we first initialized a Serial object on the board with a specified baud rate using the Serial.begin() function.
- On the web application side, we used PySerial to establish a connection to the board by providing the name of the serial port and baud rate as arguments to the serial.Serial() constructor.
- With the serial connection established, we were able to send commands to the board using the Serial object and receive data back from the board using the Serial.println() function.
