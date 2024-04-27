import streamlit as st
from ultralytics import YOLO

# Upload the photo that we want to observe

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg','png'])

# Check whether the photo is correct

try:
    image_detect = st.image(observe_photo, caption='Uploaded photo for PPE detection')
except AttributeError:
    st.header('Please upload an image first...')

# Detecting using our trained model

if st.button("Detect PPEs", type="primary"):
    detect = YOLO('https://github.com/fhrz-storage/fhrz-ta-ppe/blob/a5dee7a4850b25a70d045f6ab28719355db4d68e/peripherals/weights/best.pt')
    try:
        results = detect.predict(source=observe_photo, show=True)
        image_detect = st.image(results) # Display preds. Accepts all YOLO predict arguments
    except AttributeError:
        st.text("Please enter the photos that you want to upload first")