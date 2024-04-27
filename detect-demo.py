import streamlit as st
from ultralytics import YOLO

# Upload the photo that we want to observe

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg','png'])

# Check whether the photo is correct

try:
    st.image(observe_photo, caption='Uploaded photo for PPE detection')
except AttributeError:
    st.header('Please upload an image first...')

# Detecting using our trained model

if st.button("Detect PPEs", type="primary"):
    detect = YOLO('/workspaces/fhrz-ta-ppe/peripherals/best.pt')
    try:
        results = detect.predict(source=observe_photo, save=False)
        st.image(results) # Display preds. Accepts all YOLO predict arguments
    except AttributeError:
        st.text("Please enter the photos that you want to upload first")