import cv2

# Paths to the pre-trained Haar Cascade classifiers for full body detection
fullbody_cascade_path = "model/haarcascade_fullbody.xml"

# Load the Haar Cascade classifier for full body detection
fullbody_cascade = cv2.CascadeClassifier(fullbody_cascade_path)

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Set the resolution of the video capture
cap.set(100, 640)  # Width
cap.set(100, 480)  # Height

while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Check if the frame was successfully read
    if not success:
        print("Failed to read frame")
        break

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect full bodies in the grayscale image
    fullbodies = fullbody_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over each detected full body
    for (x, y, w, h) in fullbodies:
        scale_factor = 0.9  # Example scale factor
        x_adjusted = int(x + (1 - scale_factor) * w / 2)
        y_adjusted = int(y + (1 - scale_factor) * h / 2)
        w_adjusted = int(w * scale_factor)
        h_adjusted = int(h * scale_factor)

        # Drawing the adjusted bounding box
        cv2.rectangle(img, (x_adjusted, y_adjusted), (x_adjusted + w_adjusted, y_adjusted + h_adjusted), (0, 255, 0), 2)
        
    # Display the image with detected full bodies
    cv2.imshow("Full Body Detection", img)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
