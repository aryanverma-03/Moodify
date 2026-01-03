import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the default camera (0 means the built-in webcam)
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

    # Convert the frame to grayscale (required for the face detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If the spacebar is pressed, capture the image
    if key == ord(' '):  # Spacebar key
        cv2.imwrite('captured_image.jpg', frame)
        print("Image is captured.")

    # If 'q' is pressed, quit the program
    elif key == ord('q'):  # 'q' key
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
