from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import traceback

app = FastAPI()

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    traceback.print_exc()

class CampaignData(BaseModel):
    platform: int
    brand: int
    campaign_type: int
    spend: float
    impressions: int
    clicks: int
    orders: int
    day_of_week: int

@app.get("/")
async def home():
    return FileResponse("templates/index.html")

@app.post("/predict")
async def predict(data: CampaignData):
    ctr = round(data.clicks / data.impressions, 4)
    cpc = round(data.spend / data.clicks, 2)
    is_weekend = 1 if data.day_of_week >= 5 else 0

    features = np.array([[data.platform, data.brand, data.campaign_type,
                          data.spend, data.impressions, data.clicks,
                          data.orders, ctr, cpc, data.day_of_week, is_weekend]])

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100

    breakdown = []
    if ctr < 0.02:
        breakdown.append({"status": "warning", "text": "CTR is too low"})
    else:
        breakdown.append({"status": "good", "text": "CTR looks healthy"})

    if cpc > 500:
        breakdown.append({"status": "warning", "text": "CPC is too high"})
    else:
        breakdown.append({"status": "good", "text": "CPC is under control"})

    if data.orders < 20:
        breakdown.append({"status": "warning", "text": "Order count is very low"})
    else:
        breakdown.append({"status": "good", "text": "Orders are on track"})

    if data.spend > 30000:
        breakdown.append({"status": "warning", "text": "Spend is very high"})
    else:
        breakdown.append({"status": "good", "text": "Spend is within range"})

    return {
        "prediction": int(prediction),
        "confidence": round(confidence, 1),
        "label": "HIGH RISK 🔴" if prediction == 1 else "SAFE 🟢",
        "breakdown": breakdown
    }