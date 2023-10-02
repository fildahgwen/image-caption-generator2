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
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tokenizer import Tokenizer
python
from PIL import Image



# Load the model and tokenizer
#model = load_model('model.h5')
#tokenizer = Tokenizer.from_json('tokenizer.json')

# Load your trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the MobileNet model for feature extraction
mobile_net_model = load_model('mobilenet_model.h5')



# Load the MobileNet model for feature extraction
#mobile_net_model = load_model('mobilenet_model.h5')
# Load MobileNetV2 model
mobilenet_net_model = MobileNetV2(weights="imagenet")
mobilenet_net_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Function to generate captions from image
def generate_caption(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Extract features from the image using MobileNet
    features = mobile_net_model.predict(image)

    # Generate a caption using the model and tokenizer
    caption = ''
    for i in range(max_caption_length):
        sequence = tokenizer.sequences_to_texts([caption_sequence])[0]
        word_index = tokenizer.word_index[sequence.split()[-1]] + 1
        next_word_index = np.argmax(word_probabilities[i][word_index])
        next_word = tokenizer.index_word[next_word_index]
        if next_word == '<end>':
            break
        else:
            caption += ' ' + next_word
    return caption

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

# Streamlit app
st.title('Video Captioning App')
uploaded_file = st.file_uploader('Choose a video file', type=['mp4', 'avi'])
if uploaded_file is not None:
    video_path = 'temp.mp4'
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        video_frames.append(frame)
    video.release()
    captions = process_video_frames(video_frames)
    display_video_with_captions(video_path, captions)
