import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# 1. Libraries Importing
from PIL import Image

# 2. Model Loading
model = load_model("g_efficientnet_model.h5")  # Path to your fine-tuned Glaucoma model

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
test_image_path = "C://Users//uday//OneDrive//Desktop//Btech//Major_proj//Glaucoma//test//Healthy//normal_1158.jpg"  # Path to your test image

# Preprocess the image
processed_image = preprocess_image(test_image_path)

# 5. Prediction of the disease (Glaucoma)
prediction = model.predict(processed_image)

# Since it's a binary classification (Glaucoma: Yes or No), we'll use softmax output
# Typically, the output will be a probability distribution for both classes
# Get the class index with the highest probability
predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability (0 for Healthy, 1 for Glaucoma)

# 6. Output the prediction
if predicted_class == 1:
    print("The test image is NOT suffering from Glaucoma.")
else:
    print("The test image IS suffering from Glaucoma.")
