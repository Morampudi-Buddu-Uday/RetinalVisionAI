import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Libraries Importing
import os
from PIL import Image

# 2. Model Loading
model = load_model("efficientnet_finetuned.h5")  # Path to your fine-tuned model

# 3. Data Preprocessing (based on your preprocessing)
IMG_SIZE = 224  # This should match the image size used during training (check the `IMG_SIZE` in your training code)
BATCH_SIZE = 32  # Batch size used during training

# Define the preprocessing pipeline using ImageDataGenerator (matching your training preprocessing)
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Function to preprocess the image as required
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale as done during training
    return img_array

# 4. Passing the test image into the model
test_image_path = "DR_images/image3.png"  # Path to your test image

# Preprocess the image
processed_image = preprocess_image(test_image_path)

# 5. Prediction of the disease
prediction = model.predict(processed_image)

# Since the model uses softmax, we will use the class with the highest probability
predicted_class = np.argmax(prediction, axis=-1)

# Ensure predicted_class is an integer
predicted_class = int(predicted_class)

# 6. Output the prediction
if predicted_class == 0:
    print("The test image is not suffering from Diabetic Retinopathy.")
else:
    class_names = ["Mild", "Moderate", "Severe", "Proliferative DR"]
    print("The test image is suffering from Diabetic Retinopathy.")
    print(f"Predicted Class: {class_names[predicted_class - 1]}")


