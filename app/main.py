from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import sys
import os

# FIX: add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import denoise

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/denoise/")
async def denoise_image(file: UploadFile = File(...)):

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Invalid image"}

    img = cv2.resize(img, (512,512)) / 255.0

    output = denoise(img)

    output_img = (output * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', output_img)

    return {
        "message": "success",
        "image_bytes": buffer.tobytes().hex()
    }
