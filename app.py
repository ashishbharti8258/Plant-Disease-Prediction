import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import cv2
model=load_model(r"Notebook\trained_model(plt_disease_prediction).keras")
def load_and_preprocess_image(img_path):
    img_input=tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    input_arr= tf.keras.preprocessing.image.img_to_array(img_input)
    input_arr=input_arr.reshape(1, 256, 256, 3)
    prediction=model.predict(input_arr)
    result=np.argmax(prediction, axis=1)
    return result[0]

class_names=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# model_porediction=class_names[load_and_preprocess_image(uploaded_file)]
# st.write("Model Prediction: ",model_porediction)


st.header("🌿🌿Plant Disease Prediction🌿🌿")
st.subheader("Upload an image of a plant leaf to check for diseases.")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if(st.button("Show Image")):
    if uploaded_file is not None:
        image= Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True, channels="RGB")
    else:
        st.write("Please upload an image.")

if(st.button("Predict")):
    if uploaded_file is not None:
        image= Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True, channels="RGB")
        model_prediction=class_names[load_and_preprocess_image(uploaded_file)]
        st.write("Model Prediction: ",model_prediction)
    else:
        st.write("Please upload an image.")

    