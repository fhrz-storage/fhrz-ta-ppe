import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Upload the photo that we want to observe

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg', 'jpeg', 'png'])

# Check whether the photo is correct

if observe_photo is not None:
    try:
        image_detect = observe_photo.read()
        st.image(image_detect, caption='Uploaded photo for PPE detection', output_format='PNG')
    except AttributeError:
        st.header('Please upload an image first...')

# Detecting using our trained model

if st.button("Detect PPEs", type="primary"):
    detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
    try:
        with st.spinner("Detecting objects..."):
            results = detect.predict(image_detect, save=True)
            image_detect = st.image(results) # Display preds. Accepts all YOLO predict arguments
        st.image(results, output_format='PNG')
    except AttributeError:
        st.text("Error: Unable to detect PPEs.")