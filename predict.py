from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("tumor_classifier.h5")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Use dataset image
img = cv2.imread("dataset/Testing/glioma/Te-gl_10.jpg")

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img, (224,224)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

result = classes[np.argmax(pred)]
confidence = np.max(pred) * 100

print("Prediction:", result)
print("Confidence:", confidence)