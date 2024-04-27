import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Upload the photo that we want to observe

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg', 'jpeg', 'png'])

# Check whether the photo is correct

if observe_photo is not None:
    try:
        image = Image.open(observe_photo)
        image_detect = st.image(image, caption='Uploaded photo for PPE detection')
    except AttributeError:
        st.header('Please upload an image first...')

# Detecting using our trained model

if st.button("Detect PPEs", type="primary"):
    if observe_photo is not None:
        file_extension = observe_photo.name.split('.')[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png']:
            detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
            try:
                results = detect.predict(observe_photo, save=True)
                image_detect = st.image(results) # Display preds. Accepts all YOLO predict arguments
            except AttributeError:
                st.text("Error: Unable to detect PPEs.")
        else:
            st.text("Error: Please upload an image in JPEG or PNG format.")
    else:
        st.text("Error: Please upload an image first.")
