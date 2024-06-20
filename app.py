from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
import pickle
from pygame import mixer

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Initialize Pygame mixer for sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Global variables
score = 0
closed_eye_frames = 0
MAX_CLOSED_EYE_FRAMES = 10

def detect_drowsiness(frame):
    global score, closed_eye_frames

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y + h, x:x + w]
        
        # Detect eyes within the face ROI
        left_eye = leye_cascade.detectMultiScale(roi_gray)
        right_eye = reye_cascade.detectMultiScale(roi_gray)

        if len(left_eye) == 0 and len(right_eye) == 0:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        # If eyes are continuously closed for a certain duration, trigger alarm
        if closed_eye_frames >= MAX_CLOSED_EYE_FRAMES:
            score = 10  # Set score to 10 to trigger alarm
            cv2.putText(frame, 'Sleepy!', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            sound.play()
            return frame

        for (ex, ey, ew, eh) in left_eye:
            # Extract the region of interest (ROI) containing the left eye
            left_eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            left_eye_roi = cv2.resize(left_eye_roi, (24, 24))
            left_eye_roi = np.expand_dims(left_eye_roi, axis=2)
            left_eye_roi = np.expand_dims(left_eye_roi, axis=0)
            left_eye_roi = left_eye_roi / 255.0
            
            # Predict drowsiness using the trained model
            prediction = model.predict(left_eye_roi)
            
            # If prediction indicates drowsiness, increment score
            if prediction[0][0] > 0.5:
                score += 1
            else:
                score = max(0, score - 1)

        for (ex, ey, ew, eh) in right_eye:
            # Extract the region of interest (ROI) containing the right eye
            right_eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            right_eye_roi = cv2.resize(right_eye_roi, (24, 24))
            right_eye_roi = np.expand_dims(right_eye_roi, axis=2)
            right_eye_roi = np.expand_dims(right_eye_roi, axis=0)
            right_eye_roi = right_eye_roi / 255.0
            
            # Predict drowsiness using the trained model
            prediction = model.predict(right_eye_roi)
            
            # If prediction indicates drowsiness, increment score
            if prediction[0][0] > 0.5:
                score += 1
            else:
                score = max(0, score - 1)

        # Display captions based on the score
        if score > 10:
            cv2.putText(frame, 'Sleepy!', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            # Trigger alarm
            sound.play()
        else:
            cv2.putText(frame, 'Alert', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        # Display score
        cv2.putText(frame, 'Score: ' + str(score), (100, frame.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to capture frame from the camera.")
            break

        # Detect drowsiness and update frame
        frame = detect_drowsiness(frame)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
