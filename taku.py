import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tempfile
import pickle
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
import sys
from streamlit_option_menu import option_menu

# Load the pre-trained ResNet-50 model
#resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Define the prediction function
def predict_caption(photo):
    in_text = "startseq"
    max_caption_length = 34
    max_len=34

    # Load word_to_idx dictionary from file
    #with open("saved_ixtoword.pkl", "rb") as f:
       # word_to_idx = pickle.load(f)
      
    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # Load idx_to_word dictionary from file
    #with open("saved_ixtoword.pkl", "rb") as f:
        #idx_to_word = pickle.load(f)

    # Placeholder for the image captioning model
    #model = load_model("final_mod.h5")

    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], max_len, padding='post')

        ypred = model.predict([photo, sequence])
        yhat = model.predict([image_features, sequence], verbose=0)
        ypred = model.predict([image_features, sequence], verbose=0)
        ypred = ypred.argmax()
        if ypred not in idx_to_word:
            break
        word = idx_to_word[ypred]
        in_text += ' ' + word

        if word == 'endseq':
            break

    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption

# Streamlit app
def main():
    st.title("Video Description Generator")

 

   
        

    # Upload video file
    video_file = st.file_uploader("Upload Video ", type=["mp4"])
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        # Convert video frames to images


      
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        frames.append(image)
        success, image = vidcap.read()

        # Process each frame and predict captions
     
    for i, frame in enumerate(frames):
            # Resize the frame to the input size of the ResNet-50 model
        frame = cv2.resize(frame, (224, 224))

            # Preprocess the image
        img = preprocess_input(frame)

            # Pass the image through the ResNet-50 model
        img_features = mobilenet_model.predict(np.expand_dims(img, axis=0))

            # Get the predicted caption
        caption = predict_caption(img_features)

            # Display the frame and the predicted caption
    st.image(frame, use_column_width=True)
    st.write(f" Frame {i+1}: {caption}")

# Run the Streamlit app
if __name__ == "__main__":
    main()     
