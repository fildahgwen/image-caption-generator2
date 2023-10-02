import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
from keras.models import load_model
import sys
from streamlit_option_menu import option_menu
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image




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
    "Upload an image, and this app will generate a caption for it "
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features using VGG16
        image_features = mobilenet_model.predict(image, verbose=0)

        # Max caption length
        max_caption_length = 34
        
        # Define function to get word from index
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

                 # Function to process video frames
def process_video_frames(video_frames):
    captions = []
    for frame in video_frames:
        caption = generate_caption(frame)
        captions.append(caption)
    return captions

# Function to display video with captions
def display_video_with_captions(video_path, captions):
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        caption = captions.pop(0)
        cv2.putText(frame, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Video with Captions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
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

    # Display the generated caption with custom styling
st.markdown(
    f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
    f'<p style="font-style: italic;">“{generated_caption}”</p>'
    f'</div>',
    unsafe_allow_html=True
)
