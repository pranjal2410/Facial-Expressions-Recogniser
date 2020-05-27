from cv2 import cv2
from keras.models import load_model
import numpy as np
import os

model = load_model('models/modelv2.h5')

model.load_weights('models/modelv2.weights')

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = video.read()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(gray, (48, 48))
        image = np.array(image, dtype='float32')/255.0
        image = image.reshape(48, 48, 1)
        image = np.concatenate((image, np.zeros(shape=(48, 48, 2))), axis=2)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        index = np.argmax(pred, axis=1)
        print(pred, index)
        if index == 0:
            text = 'Angry'
            color = (0, 0, 255)
        elif index == 1:
            text = 'Happy'
            color = (0, 255, 0)
        elif index == 2:
            text = 'Sad'
            color = (211, 211, 211)
        elif index == 3:
            text = 'Neutral'
            color = (255, 255, 0)
        else:
            text = 'Surprise'
            color = (128, 0, 128)
        
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, h, w) in faces:
            cv2.rectangle(gray, (x, y), (x+w, y+h), color, 2)
            
        cv2.putText(gray, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.imshow('Video', gray)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()