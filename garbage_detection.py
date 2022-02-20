import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import requests


font = cv2.FONT_HERSHEY_SIMPLEX

model = tf.keras.models.load_model('./Saved Model and Cascading File/garbage_detection.h5')
garbage_haar_cascade = cv2.CascadeClassifier('./Saved Model and Cascading File/garbage_cascade.xml')
label_dict = {0 : 'Garbage', 1: 'Glass', 2: 'Mask', 3: 'Metal', 4: 'Non Garbage', 5: 'Organic', 6: 'Plastic'}

camera = cv2.VideoCapture(0)
while True:
    success, frame = camera.read()
    if success:
        garbage = garbage_haar_cascade.detectMultiScale(frame,
                                                scaleFactor= 1.2,
                                                minNeighbors=100,
                                                minSize=(24, 24),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in garbage:
            img = frame[y:y+h, x:x+w]
            resize_image = cv2.resize(img, (150,150))
            normalize_img = resize_image/225
            img_pixels = normalize_img.reshape(150,150,3)
            img_pixels = np.expand_dims(img_pixels, axis=0)
  
            predictions = model.predict(img_pixels)
            garbage_label = np.argmax(predictions)

            if garbage_label != 4:
                cv2.rectangle(frame, (x,y), (x+w,y+h),(255,255,255),2)
                cv2.putText(frame,'{}'.format(label_dict[garbage_label]), (x, y), font, 0.5,(0,0,225), 1)
        
            # Display the resulting frame
            cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
