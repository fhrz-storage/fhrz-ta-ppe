import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import tempfile


# Upload the photo that we want to observe

uploaded_image = st.file_uploader("Upload a photo to observe", type=['jpg', 'jpeg', 'png'])


# Check whether the photo is correct

if uploaded_image is not None:
    # try:
    image_raw = uploaded_image.read()
    image_usable = Image.open(io.BytesIO(image_raw))
    st.image(image_usable, caption=f'{uploaded_image.name}')

    with tempfile.NamedTemporaryFile() as f:
        f.write(image_usable.read())
        f.flush()
        st.write(f.name)
    # Detecting using our trained model

    if st.button("Detect PPEs", type="primary"):
        detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
        with st.spinner("Detecting objects..."):
            results = detect.predict(f"{uploaded_image.name}")
            for x in results:
                st.image(x, caption="Image with object detected in it")

    # except AttributeError:
    #     st.header('Please upload an image first...')
