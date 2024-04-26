import streamlit as st
import pandas as pd
from io import StringIO

observe_photo = st.file_uploader("Upload a photo to observe", type=['jpg','png'])
st.image(observe_photo, caption='Uploaded photo for PPE detection')
