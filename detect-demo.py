import streamlit as st

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg','png'])

try:
    st.image(observe_photo, caption='Uploaded photo for PPE detection')
except AttributeError:
    st.header('Please upload an image first...')
# else:
#     st.image(observe_photo, caption='Uploaded photo for PPE detection')

if st.button("Detect PPEs", type="primary"):
    from ultralytics import YOLO
    detect = YOLO('/workspaces/fhrz-ta-ppe/peripherals/best.pt')
    try:
        results = detect.predict(source=observe_photo, save=False)
        st.image(results) # Display preds. Accepts all YOLO predict arguments
    except AttributeError:
        st.text("Please enter the photos that you want to upload first")