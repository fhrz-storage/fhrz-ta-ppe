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
    image_repackaged = Image.open(io.BytesIO(image_raw))
    image_usable = image_repackaged.save("materials.png")
    st.image(image_usable)

    # Detecting using our trained model

    if st.button("Detect PPEs", type="primary"):
        detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
        with st.spinner("Detecting objects..."):
            results = detect.predict(f"image_usable")
            for x in results:
                st.image(x, caption="Image with object detected in it")

    # except AttributeError:
    #     st.header('Please upload an image first...')
