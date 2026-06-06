### 6.4 `backend/app/schemas.py`


from pydantic import BaseModel

class PredictionResponse(BaseModel):
    classification: str      # "Malignant" or "Benign/Normal"
    confidence: float        # 0.0 – 1.0, probability of malignancy
    malignant_probability: float
    benign_probability: float
    warning: str             # Clinical disclaimer