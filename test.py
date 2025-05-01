import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from streamlit_lottie import st_lottie
import json

# Load models
dr_model = tf.keras.models.load_model("efficientnet_finetuned.h5")
glaucoma_model = tf.keras.models.load_model("g_efficientnet_model.h5")

# Define class labels
dr_classes = ["No DR (Class 0)", "Mild DR (Class 1)", "Moderate DR (Class 2)", "Severe DR (Class 3)", "Proliferative DR (Class 4)"]
glaucoma_classes = ["Healthy Eye", "Glaucomatous Eye"]

# Remedies and Suggestions based on classification
remedies = {
    0: "✅ No signs of diabetic retinopathy detected. Maintain good blood sugar, blood pressure, and cholesterol levels. Get an eye checkup annually.",
    1: "⚠️ Early-stage DR detected. Strict blood sugar control, regular monitoring every 6-12 months, and a healthy lifestyle are recommended.",
    2: "⚠️ Moderate DR detected. Closer monitoring every 3-6 months, possible early intervention with laser therapy, and strict diabetes management are advised.",
    3: "🚨 Severe DR detected. High risk of vision loss—frequent eye exams, laser treatment, and anti-VEGF injections may be needed.",
    4: "🚨 Critical Condition: Advanced DR stage. Urgent treatment required, including laser therapy, vitrectomy surgery, and strict diabetes control."
}

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)  # Resize
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to preprocess image for Glaucoma model (same as DR model preprocessing)
def preprocess_glaucoma_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)  # Resize
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load Lottie animation from a local file
def load_lottiefile(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_eye = load_lottiefile("C:/Users/uday/OneDrive/Desktop/Btech/Major_proj/UI/Animation - 1741093748053.json")

# Streamlit UI styles
st.set_page_config(page_title="Retinal Fundus Classification", page_icon="👁️", layout="wide")

# Title and Animation
st.markdown("<h1 style='text-align: center; font-size: 40px; color: black;'>Retinal Fundus Image Classification</h1>", unsafe_allow_html=True)
st_lottie(lottie_eye, height=200)

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], help="Upload a clear retinal fundus image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Display Uploaded Image directly (small, clear, professional size)
    st.image(image, caption="Uploaded Image", use_container_width=False, width=500)
    
    # Process and Classify
    with st.spinner("Processing image... Please wait."):
        time.sleep(2)  # Simulating processing delay
        
        # Preprocess and classify using DR model
        processed_image = preprocess_image(image)
        dr_prediction = dr_model.predict(processed_image)
        dr_class = np.argmax(dr_prediction, axis=1)[0]
        confidence_score = np.max(dr_prediction) * 100
        
        st.markdown(f"<div style='background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; color: black;'>🩺 <b>Diabetic Retinopathy Classification:</b> {dr_classes[dr_class]}<br><b>Confidence:</b> {confidence_score:.2f}%</div>", unsafe_allow_html=True)
        
        # Display suggestions based on classification
        st.markdown(f"<div style='background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px; text-align: left; font-size: 16px; color: black; margin-top: 10px;'><b>📌 Suggestions:</b> {remedies[dr_class]}</div>", unsafe_allow_html=True)
        
        # Preprocess and classify using Glaucoma model
        processed_glaucoma_image = preprocess_glaucoma_image(image)
        glaucoma_prediction = glaucoma_model.predict(processed_glaucoma_image)
        glaucoma_class = np.argmax(glaucoma_prediction, axis=1)[0]
        glaucoma_confidence = np.max(glaucoma_prediction) * 100
            
        st.markdown(f"<div style='background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; color: black;'>🔬 <b>Glaucoma Classification:</b> {glaucoma_classes[glaucoma_class]}<br><b>Confidence:</b> {glaucoma_confidence:.2f}%</div>", unsafe_allow_html=True)
    
    st.success("✅ Classification complete.")
