import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('.\pneumonia_detection_model.h5')
img_path = './test/p1.jpeg' 
img = cv2.imread(img_path)
img = cv2.resize(img, (150, 150))  
img = np.expand_dims(img, axis=0)  


prediction = model.predict(img)
if prediction[0] > 0.5:
    print("Prediction: Pneumonia")
else:
    print("Prediction: Normal")