import streamlit as st
from PIL import Image
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
import os

# Upload the photo that we want to observe

image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])

if image_file is not None:
    # Display file details
    file_details = {"FileName": image_file.name, "FileType": image_file.type}
    st.write(file_details)

    # Load and display the image
    img = Image.open(image_file)
    st.image(img, height=250, width=250)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved File: {uploadedfile.name} to tempDir")

save_uploadedfile(image_file)

if st.button("Detect PPEs", type="primary"):
    detect = YOLO('https://raw.githubusercontent.com/fhrz-storage/fhrz-ta-ppe/main/peripherals/weights/best.pt')
    with st.spinner("Detecting objects..."):
        results = detect.predict(img)
        for x in results:
            st.image(x, caption="Image with object detected in it")

    # except AttributeError:
    #     st.header('Please upload an image first...')
