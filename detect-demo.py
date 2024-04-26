import streamlit as st
import pandas as pd
from io import StringIO

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg','png'])

try:
    st.image(observe_photo, caption='Uploaded photo for PPE detection')
except AttributeError:
    st.header('Please upload an image first...')
else:
    st.image(observe_photo, caption='Uploaded photo for PPE detection')