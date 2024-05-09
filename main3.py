import cv2

# Paths to the pre-trained Haar Cascade classifiers for face and smile detection
face_cascade_path = "model/haarcascade_frontalface_default.xml"
smile_cascade_path = "model/haarcascade_smile.xml"

# Load the Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Set the resolution of the video capture
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Check if the frame was successfully read
    if not success:
        print("Failed to read frame")
        break

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Get the region of interest (ROI) corresponding to the face
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect smiles in the ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)

        # Draw rectangles around the detected smiles
        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the image with detected smiles
    cv2.imshow("Smile Detection", img)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
