import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import cv2

# Upload the photo that we want to observe

uploaded_image = st.file_uploader("Upload a photo to observe", type=['jpg', 'jpeg', 'png'])

# Check whether the photo is correct

if uploaded_image is not None:
    # try:
    image_raw = uploaded_image.read()
    image_usable = Image.open(io.BytesIO(image_raw))
    st.image(image_usable, caption='Uploaded photo for PPE detection')

    # Detecting using our trained model

    if st.button("Detect PPEs", type="primary"):
        detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
        with st.spinner("Detecting objects..."):
            results = detect.predict(image_usable, save=True)
            result_image = cv2.imwrite('result_image.jpg', cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR))
            st.image(result_image)

    # except AttributeError:
    #     st.header('Please upload an image first...')
