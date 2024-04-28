import streamlit as st
from PIL import Image
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
import io

# Upload the photo that we want to observe

uploaded_image = st.file_uploader("Upload a photo to observe", type=['jpg', 'jpeg', 'png'])

# Check whether the photo is correct

if uploaded_image is not None:
    # Save the uploaded image to a temporary file
    with NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file.write(uploaded_image.getvalue())
        temp_file.seek(0)
    
    st.image(f"prediction_results/{uploaded_image.name}")
    # try:
    # image_raw = uploaded_image.read()
    # image_usable = Image.open(io.BytesIO(image_raw))
    # st.image(image_usable)

    # Detecting using our trained model

    # if st.button("Detect PPEs", type="primary"):
    #     detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
    #     with st.spinner("Detecting objects..."):
    #         results = detect.predict("prediction_results/{uploaded_image.name}.jpg")
    #         for x in results:
    #             st.image(x, caption="Image with object detected in it")

    # except AttributeError:
    #     st.header('Please upload an image first...')
