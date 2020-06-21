import os
import cv2
import numpy as np
import keras
import tensorflow as tf
from keras.models import model_from_json
from keras_preprocessing import image


model = model_from_json(
    open("saved\saved_model.json", "r").read())
model.load_weights("saved\saved_weights.h5")

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    if not ret:
        continue

    # converting coloured captured image to graycale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det_face = haar_cascade.detectMultiScale(grayscale_img, 1.32, 5)

    for (x, y, width, height) in det_face:
        cv2.rectangle(img, (x, y), (x + width, y + height),
                      (255, 0, 0), thickness=7)
        # Selecting the Region of Interest
        roi_grey = grayscale_img[y:y + width, x:x + height]
        # Matching our required size!
        roi_grey = cv2.resize(roi_grey, (48, 48))
        pixels = image.img_to_array(roi_grey)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = pixels / 255

        preds = model.predict(pixels)

        # Preds will be in the form of probabilties.
        # We need to convert the following data as per requirements

        max_index = np.argmax(preds[0])

        emotion = ('angry', 'disgust', 'fear', 'happy',
                   'sad', 'surprise', 'neutral')
        predicted_emotion = emotion[max_index]

        cv2.putText(img, predicted_emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    normal_image = cv2.resize(img, (1000, 700))
    cv2.imshow('Emotion Recognition System', normal_image)

    # Wait until "esc" key is pressed.
    if cv2.waitKey(33) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows
