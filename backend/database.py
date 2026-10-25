import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = Path(__file__).resolve().parent / "smartfish.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class DiagnosisRecord(Base):
    __tablename__ = "diagnosis_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    filename = Column(String)
    predicted_class = Column(String)
    confidence = Column(Float)
    severity = Column(String)


class WaterQualityRecord(Base):
    __tablename__ = "water_quality_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    ph = Column(Float)
    temperature_c = Column(Float)
    dissolved_oxygen_mg_l = Column(Float)
    ammonia_mg_l = Column(Float)
    source = Column(String)  # "manual" or "simulated"


Base.metadata.create_all(bind=engine)
