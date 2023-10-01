import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
from keras.models import load_model
import sys
from streamlit_option_menu import option_menu


# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload a video, and this app will generate a caption for it using a trained model."
)


# Load the pre-trained model
model = MobileNetV2(weights='imagenet', include_top=True)

# Function to extract features from the image
def extract_features(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features

# Function to perform image captioning
def image_captioning(image):
     # Generate caption using the model
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption

        # Generate caption
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Remove startseq and endseq
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")
    pass

# Streamlit app
st.title('Image Captioning on Video Clips')
uploaded_file = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov', 'mkv'], max_upload_size=2*1024*1024)

if uploaded_file is not None:
    video = cv2.VideoCapture(uploaded_file)
    success, image = video.read()
    while success:
        features = extract_features(image)
        caption = image_captioning(features)
        st.write(f'Caption: {caption}')
        success, image = video.read()
    video.release()
