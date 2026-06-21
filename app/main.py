from fastapi import FastAPI, UploadFile, File
import numpy as np, cv2
from processing.dicom import load_dicom
from app.pipeline import denoise

app = FastAPI()

@app.post("/denoise/")
async def denoise_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = load_dicom(file.file)
    except:
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)/255.0

    output = denoise(img)
    _, buffer = cv2.imencode('.png', (output*255).astype("uint8"))

    return {"image_bytes": buffer.tobytes().hex()}
