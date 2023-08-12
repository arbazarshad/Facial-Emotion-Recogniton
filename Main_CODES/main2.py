import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('model_mobilenet.h5')

# Define the emotion labels
emotion_labels = ['Neutral', 'Fear', 'Happy', 'Angry']

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the desired size of (48, 48)
    resized_frame = cv2.resize(frame, (48, 48))

    # Convert the image to the desired color space (e.g. RGB or BGR)
    color_space_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Add a new dimension to the resized frame to represent the batch size
    batched_frame = np.expand_dims(color_space_frame, axis=0)

    # Preprocess the image by scaling its pixel values to the range [0, 1]
    preprocessed_frame = batched_frame / 255.0

    # Use the model to make a prediction on the preprocessed image
    prediction = model.predict(preprocessed_frame)[0]
    
    print(prediction)

    # Get the emotion label with the highest predicted probability
    label = emotion_labels[np.argmax(prediction)]

    # Print the predicted emotion label on the screen
    cv2.putText(frame, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the resulting frame
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
