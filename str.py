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


# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it "
)

# Upload video
    choice = option_menu("Main Menu",["Home","Upload","About"],icons = ["house","cloud_upload","list-task"],menu_icon ="cast",default_index = 0,orientation = "horizontal")
    
    if choice == "Upload":
        st.subheader("Upload Your Video")

        video_file_path = st.file_uploader("accepting mp4,avi", type=["mp4", "avi"],server)

        if video_file_path is not None:
            path = video_file_path.name
            with open(path,mode='wb') as f: 
                f.write(video_file_path.read())         
                st.success("File Uploaded")
            cap = cv2.VideoCapture(path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
            
            if st.button("Caption Objects"):
                
                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()
    
                    if not ret:
                        break
    
                    
                    predict(frame, model)
    
                    # Display the resulting frame
                    
                cap.release()
                output.release()
                cv2.destroyAllWindows()
                
            key = st.text_input('Search key')
            key = key.lower()
            
            if key is not None:
            
                if st.button("Search for an object"):
                    
                    
                    # Start the video prediction loop
                    while cap.isOpened():
                        ret, frame = cap.read()
        
                        if not ret:
                            break
        
                        # Perform object detection
                        object_detection(key,frame, model)
                        
                    cap.release()
                    output.release()
                    #cv2.destroyAllWindows()

# Process uploaded image
''''if uploaded_image is not None:
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
        )'''

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
