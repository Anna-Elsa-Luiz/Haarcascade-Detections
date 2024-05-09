import cv2

face_classifier_path = "model/haarcascade_frontalface_default.xml"
smile_classifier_path = "model/haarcascade_smile.xml"

face_classifier = cv2.CascadeClassifier(face_classifier_path)
smile_classifier = cv2.CascadeClassifier(smile_classifier_path)


def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return img

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract region of interest (ROI) for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        smile = smile_classifier.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in smile:
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Call the face_detector function
    output_frame = face_detector(frame)
    
    cv2.imshow('Face and smile Detection', output_frame)
    
    # Check for Enter key press to exit
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
