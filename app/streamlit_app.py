import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Very Fresh', 'Fresh', 'Slightly Aged', 'Stale', 'Spoiled']

# Load model once
@st.cache_resource
def load_alexnet_model():
    model_path = "AlexNet_best_model.keras"
    return load_model(model_path)

model = load_alexnet_model()

# Preprocessing function (from first script)
def preprocess_image(image):
    """
    Takes a PIL image, resizes and preprocesses it for model input.
    """
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit App UI
st.title("🍎 Food Freshness Detection")
st.write("Upload an image to predict its freshness category.")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and show image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        # Output results
        st.success(f"🧠 Predicted Class: **{predicted_class}**")
        st.subheader("📊 Confidence Scores:")
        for cls, prob in zip(CLASS_NAMES, prediction[0]):
            st.write(f"{cls}: {prob:.4f}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")