import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.models import load_model

# Define the home function
def home():
    st.write("## Introduction")
    imageha = mpimg.imread('image.jpg')     
    st.image(imageha)
    st.write("This app uses  convolutional neural network  to classify Natural Scene image into six different class category")
   
    
    st.write("This Data contains around 25k images of size 150x150 distributed under 6 categories.")
    st.write("'buildings' -> 0")
    st.write("'forest' -> 1")
    st.write("'glacier' -> 2")
    st.write("'mountain' -> 3")
    st.write("'sea' -> 4")
    st.write("'street' -> 5")
    
   
    st.info("Please select a tab on the left to get started.")

# Define the prediction function
def prediction():
    
    st.write("Predict the Nature Scene that is being represented in the image")
    
    # Define the input fields
    model = load_model("bestmodel.h5")


    
    uploaded_file = st.file_uploader(
        "Upload an image of a Nature Scene:", type="jpg"
    )
    predictions=-1
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        image1=image.smart_resize(image1,(150,150))
        img_array = image.img_to_array(image1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array/255.0
        predictions = model.predict(img_array)
    st.write("### Prediction Result")
    if st.button("Predict"): 
        labels = {0:"buildings",1:"forest",2:"glacier",3:"mountain",4:"sea",5:"street"}
        if prediction!=-1:
            if uploaded_file is not None:
                image1 = Image.open(uploaded_file)
                st.image(image1, caption="Uploaded Image", use_column_width=True)
                st.markdown(
                    f"<h2 style='text-align: center;'>Image of {labels[np.argmax(predictions)]}</h2>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("Please upload file or choose sample image.")

def main():
    st.set_page_config(page_title="Nature Scene Classification", page_icon=":heart:")
    st.markdown("<h1 style='text-align: center; color: white;'>Nature Scene Classification</h1>", unsafe_allow_html=True)
# Create the tab layout
    tabs = ["Home", "Classification"]
    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    if page == "Home":
        home()
    elif page == "Classification":
        prediction()
   
   
main()

