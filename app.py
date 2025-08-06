import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("cat_dog_model.h5")
class_names = ['cat', 'dog']

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions)

    st.markdown(f"### Prediction: `{predicted_class}`")
    st.markdown(f"### Confidence: `{confidence * 100:.2f}%`")
