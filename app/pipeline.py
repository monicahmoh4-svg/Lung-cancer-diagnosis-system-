import torch
import numpy as np
import sys
import os

# ✅ CRITICAL FIX: force root directory into path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.dncnn import DnCNN
from processing.preprocessing import normalize
from processing.filters import anisotropic_gaussian
from processing.wavelet import decompose, reconstruct

device = "cpu"  # safer for Render

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
