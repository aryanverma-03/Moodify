import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import sys

# Configure default encoding for Windows environments
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = tf.keras.models.load_model('mood_train_3.h5')

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the list of emotion labels
emotion_list = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  # Resize to match model's input size
        face = face.astype('float32') / 255  # Normalize pixel values
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Add batch dimension (1, 48, 48, 1)

        # Predict the mood
        prediction = model.predict(face)
        mood_index = np.argmax(prediction)  # Get index of the highest confidence value
        mood_label = emotion_list[mood_index]

        # Display the mood label on the video
        cv2.putText(frame, mood_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detected faces and mood labels
    cv2.imshow('Mood Detection', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, quit the program
    if key == ord('q'):  # 'q' key
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# from flask import Flask, request, jsonify
# from pyngrok import ngrok
# import threading

# # Initialize the Flask app
# app = Flask(_name_)

# # Define an API route for predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     prediction = model.predict([data['features']])  # Modify according to your model’s input requirements
#     return jsonify({'prediction': prediction.tolist()})

# # Function to run Flask in a thread
# def run_flask():
#     app.run(port=5001)

# # Start Flask in a background thread
# thread = threading.Thread(target=run_flask)
# thread.start()

# # Start ngrok tunnel
# public_url = ngrok.connect(5001)
# print("Public URL:", public_url)