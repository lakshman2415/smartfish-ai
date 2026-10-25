from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import DiagnosisRecord, SessionLocal
from inference import NotAFishError, detect_fish, predict
from water_quality import get_reading, save_reading


class WaterQualityInput(BaseModel):
    ph: float = Field(ge=0, le=14)
    temperature_c: float = Field(ge=-5, le=45)
    dissolved_oxygen_mg_l: float = Field(ge=0, le=20)
    ammonia_mg_l: float = Field(ge=0, le=10)

app = FastAPI(title="SmartFish API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect-fish")
async def detect_fish_endpoint(file: UploadFile = File(...), closeup: bool = Form(False)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    image_bytes = await file.read()
    return detect_fish(image_bytes, closeup=closeup)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), closeup: bool = Form(False)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    image_bytes = await file.read()
    try:
        result = predict(image_bytes, closeup=closeup)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except NotAFishError as e:
        raise HTTPException(status_code=422, detail=str(e))

    db = SessionLocal()
    try:
        db.add(DiagnosisRecord(
            filename=file.filename or "upload",
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            severity=result["severity"],
        ))
        db.commit()
    finally:
        db.close()

    return result


@app.get("/history")
def history(limit: int = 50):
    db = SessionLocal()
    try:
        records = (
            db.query(DiagnosisRecord)
            .order_by(DiagnosisRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "filename": r.filename,
                "predicted_class": r.predicted_class,
                "confidence": r.confidence,
                "severity": r.severity,
            }
            for r in records
        ]
    finally:
        db.close()


@app.get("/water-quality")
def water_quality():
    return get_reading()


@app.post("/water-quality")
def set_water_quality(reading: WaterQualityInput):
    return save_reading(
        ph=reading.ph,
        temperature_c=reading.temperature_c,
        dissolved_oxygen_mg_l=reading.dissolved_oxygen_mg_l,
        ammonia_mg_l=reading.ammonia_mg_l,
    )
