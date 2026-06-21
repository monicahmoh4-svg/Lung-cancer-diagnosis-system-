import streamlit as st
import requests
from PIL import Image
import io

st.title("Lung CT Denoising System")

file = st.file_uploader("Upload CT Image")

if file:
    files = {"file": file.getvalue()}
    res = requests.post("http://localhost:8000/denoise/", files=files)
    data = res.json()

    img_bytes = bytes.fromhex(data["image_bytes"])
    image = Image.open(io.BytesIO(img_bytes))

    st.image(image, caption="Denoised Image")
    st.write(data["metrics"])
