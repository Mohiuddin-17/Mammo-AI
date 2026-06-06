import gdown
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

MODEL_FILE  = WEIGHTS_DIR / "model_best.pt"
CONFIG_FILE = WEIGHTS_DIR / "model_config.json"

MODEL_ID  = "1tgHfUPbIUvngEOVV0LT48r903bdQdBQN"
CONFIG_ID = "1-mcZ1Quf2m5eVZLkZ3gLWsCEp1z6spuq"

def download_weights():
    if not CONFIG_FILE.exists():
        print("Downloading model_config.json...")
        gdown.download(f"https://drive.google.com/uc?id={CONFIG_ID}",
                      str(CONFIG_FILE), quiet=False)
        print("✅ model_config.json downloaded")
    else:
        print("✅ model_config.json already exists")

    if not MODEL_FILE.exists():
        print("Downloading model_best.pt (~200 MB, please wait)...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}",
                      str(MODEL_FILE), quiet=False)
        print("✅ model_best.pt downloaded")
    else:
        print("✅ model_best.pt already exists")

if __name__ == "__main__":
    download_weights()