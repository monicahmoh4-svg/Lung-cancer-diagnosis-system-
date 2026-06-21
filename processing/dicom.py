import pydicom
import numpy as np

def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(float)
    img = (img - img.min())/(img.max()-img.min())
    return img
