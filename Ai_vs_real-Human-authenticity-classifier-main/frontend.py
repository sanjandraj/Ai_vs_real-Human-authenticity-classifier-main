import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Load the trained model
try:
    model = tf.keras.models.load_model("final_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Streamlit UI
st.title("Human vs AI Face Classifier")
st.write("Upload an image to classify whether it's Real or AI-Generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        result = "Real" if prediction[0][0] > 0.5 else "AI Generated"
        st.write(f"Prediction: **{result}**")