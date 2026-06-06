import io
import numpy as np
import cv2
from PIL import Image
import pydicom
import torch
from torchvision import transforms

IMG_SIZE = 224

# These are the EXACT same transforms used at inference time in training
INFERENCE_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image_bytes(file_bytes: bytes, filename: str) -> Image.Image:
    """
    Accept DICOM, PNG, or JPEG. Return a PIL Image in grayscale.
    
    DICOM is the native format for medical imaging equipment.
    We must handle it explicitly since PIL cannot read .dcm files.
    The pixel_array from pydicom gives raw 16-bit values which we
    normalise to uint8 (0-255) for consistent downstream processing.
    """
    ext = filename.lower().split(".")[-1]
    
    if ext in ("dcm", "dicom"):
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)
        
        # Handle photometric interpretation
        # Some DICOMs encode "white = background" (MONOCHROME1) vs
        # "black = background" (MONOCHROME2). Invert MONOCHROME1.
        if hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                arr = arr.max() - arr
        
        # Normalise to [0, 255]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
    
    return img


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """Apply CLAHE — same as training preprocessing."""
    arr = np.array(pil_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    return Image.fromarray(arr)


def preprocess_for_inference(file_bytes: bytes, filename: str) -> torch.Tensor:
    """Full pipeline: load → CLAHE → transforms → tensor (1, 3, 224, 224)"""
    img = load_image_bytes(file_bytes, filename)
    img = apply_clahe(img)
    tensor = INFERENCE_TF(img)           # (3, 224, 224)
    return tensor.unsqueeze(0)           # (1, 3, 224, 224)