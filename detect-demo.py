import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io

# Upload the photo that we want to observe

uploaded_image = st.file_uploader("Upload a photo to observe", type=['jpg', 'jpeg', 'png'])

# Check whether the photo is correct

if uploaded_image is not None:
    # try:
    st.image(uploaded_image, caption='Uploaded photo for PPE detection')
    image_raw = Image.open(uploaded_image)
    image_raw = image_raw.save(f'data/images/{uploaded_image.name}')
    image_usable = f'data/images/{uploaded_image.name}'

    # Detecting using our trained model

    if st.button("Detect PPEs", type="primary"):
        detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
        with st.spinner("Detecting objects..."):
            results = detect.predict(image_usable, save=True, save_dir="https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/prediction_results/results.png")
        for x in results:
            st.image(results)

    # except AttributeError:
    #     st.header('Please upload an image first...')
