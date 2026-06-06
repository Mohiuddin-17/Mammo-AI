import json
import torch
import timm
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_config = None


def load_model():
    global _model, _config
    
    config_path = WEIGHTS_DIR / "model_config.json"
    weights_path = WEIGHTS_DIR / "model_best.pt"
    
    with open(config_path) as f:
        _config = json.load(f)
    
    _model = timm.create_model(_config["model_name"], pretrained=False, num_classes=1)
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    # The checkpoint saved the full training state dict, extract just the model weights
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    _model.to(DEVICE)
    
    print(f"Model loaded: {_config['model_name']} on {DEVICE}")
    return _model, _config


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model, _config
