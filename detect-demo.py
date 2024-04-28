import streamlit as st
from PIL import Image
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
import numpy as np
import os
import cv2

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg'])
if uploaded_file is not None:
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()

    # Display the image
    # st.image(bytes_data)

    # Read file as bytes
    bytes_data = uploaded_file.getvalue()

    # Convert BytesIO to PIL Image
    pil_image = Image.open(uploaded_file)

    # Convert to numpy array
    # numpy_image = np.array(pil_image)

    st.image(pil_image)

    detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
    # with st.spinner("Detecting objects..."):
    results = detect.predict(pil_image)
    print(results)
    # pil_result = Image.open(f"runs/detect/predict9/{uploaded_file.name}")
    # st.image(pil_result)
