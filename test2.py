import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io
import time
from streamlit_lottie import st_lottie
import json
import os

# Load models
DR_model = load_model(r"C:\Users\uday\OneDrive\Desktop\Btech\Major_proj\UI\Models\efficientnet_finetuned.h5")  # Diabetic Retinopathy model
Glaucoma_model = load_model(r"C:\Users\uday\OneDrive\Desktop\Btech\Major_proj\UI\Models\g_efficientnet_model.h5")  # Glaucoma model

# Constants
IMG_SIZE = 224
CLASS_NAMES_DR = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_NAMES_GLAUCOMA = ["Glaucomatous", "Healthy"]

remedies = {
    0: "✅ No signs of diabetic retinopathy detected. Maintain good blood sugar, blood pressure, and cholesterol levels. Get an eye checkup annually.",
    1: "⚠️ Early-stage DR detected. Strict blood sugar control, regular monitoring every 6-12 months, and a healthy lifestyle are recommended.",
    2: "⚠️ Moderate DR detected. Closer monitoring every 3-6 months, possible early intervention with laser therapy, and strict diabetes management are advised.",
    3: "🚨 Severe DR detected. High risk of vision loss—frequent eye exams, laser treatment, and anti-VEGF injections may be needed.",
    4: "🚨 Critical Condition: Advanced DR stage. Urgent treatment required, including laser therapy, vitrectomy surgery, and strict diabetes control."
}

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_dr(img_array):
    prediction = DR_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return predicted_class

def predict_glaucoma(img_array):
    prediction = Glaucoma_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return predicted_class

# Load Lottie animation from a local file
def load_lottiefile(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animations
animation_main = load_lottiefile(r"C:\Users\uday\OneDrive\Desktop\Btech\Major_proj\UI\Animations\Animation - 1741093748053.json")
animation_buffering = load_lottiefile(r"C:\Users\uday\OneDrive\Desktop\Btech\Major_proj\UI\Animations\Animation - 1741094290527.json")

# Streamlit UI
st.set_page_config(page_title="Retinal Blindness Detection", layout="centered")
st.title("🔍 Retinal Blindness Detection")
st_lottie(animation_main, height=200)
st.write("Upload a retinal fundus image to classify Diabetic Retinopathy and Glaucoma.")

uploaded_file = st.file_uploader("📤 Upload Fundus Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True, width=100)
    
    # Display buffering animation before prediction
    show_animation = True
    if show_animation:
        st_lottie(animation_buffering, height=90, key="buffering")
        time.sleep(2)
        show_animation = False  # Hide animation
    
    # Preprocess image
    img_array = preprocess_image(img)
    
    # DR Prediction
    dr_class = predict_dr(img_array)
    st.subheader(f"🩺 Diabetic Retinopathy Prediction: {CLASS_NAMES_DR[dr_class]}")
    st.write(remedies[dr_class])
    
    # Glaucoma Prediction
    glaucoma_class = predict_glaucoma(img_array)
    st.subheader(f"👁️ Glaucoma Prediction: {CLASS_NAMES_GLAUCOMA[glaucoma_class]}")
    
    if glaucoma_class == 0:
        st.warning("🚨 Signs of Glaucoma detected! Consult an ophthalmologist immediately for further examination and treatment.")
