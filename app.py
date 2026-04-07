import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model("tumor_classifier_v2.keras", compile=False)

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor Detection System")

file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (224,224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict
    pred = model.predict(img_resized)
    class_index = np.argmax(pred)
    result = classes[class_index]
    confidence = np.max(pred) * 100

    # Output
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
