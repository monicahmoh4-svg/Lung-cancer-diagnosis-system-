import torch
import numpy as np
import sys
import os

# FIX: ensure root path is included
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dncnn import DnCNN
from processing.preprocessing import normalize
from processing.filters import anisotropic_gaussian
from processing.wavelet import decompose, reconstruct

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DnCNN().to(device)

try:
    model.load_state_dict(torch.load("models/weights/dncnn.pth", map_location=device))
except:
    pass

model.eval()

def process_band(band):
    tensor = torch.tensor(band).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze().cpu().numpy()

def denoise(img):
    img = normalize(img)
    img = anisotropic_gaussian(img)

    LL, (LH, HL, HH) = decompose(img)

    LH = process_band(LH)
    HL = process_band(HL)
    HH = process_band(HH)

    result = reconstruct((LL, (LH, HL, HH)))

    return np.clip(result, 0, 1)
