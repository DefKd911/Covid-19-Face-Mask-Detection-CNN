

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\HP\OneDrive\Desktop\CV\Face Mask Detection\models\model_final_cnn (1).h5')
# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize variables for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_input(x):
    x = x.astype('float32')
    x /= 127.5
    x -= 1.
    return x

def detect_mask(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face for the model
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = preprocess_input(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Make prediction
        prediction = model.predict(face_roi)
        
        # Draw rectangle and text
        color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        label = "Mask" if prediction > 0.5 else "No Mask"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

frame_count = 0
def generate_frames():
    global frame_count
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            if frame_count % 3 == 0:  # Process every 3rd frame
                frame = detect_mask(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
