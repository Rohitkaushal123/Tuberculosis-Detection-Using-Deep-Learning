import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tempfile
import requests

# Load model from Google Drive
@st.cache_resource
def load_remote_model():
    file_id = "1MKRlIex3PaoVSMYKqNehuAhquq-PiL-M"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        model = load_model(tmp_file.name)
    return model

model = load_remote_model()

# UI
st.set_page_config(page_title="TB X-ray Detector", layout="centered")
st.title("🩻 Tuberculosis Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error(f"❌ Tuberculosis Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Tuberculosis Detected (Confidence: {1 - prediction:.2f})")
