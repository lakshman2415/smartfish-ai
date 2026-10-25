"""
Water-quality readings: user-editable, with a simulated fallback.

No physical pH/temperature/DO/ammonia probes are wired up, so there's nothing
to poll automatically. Readings are either entered manually (saved to
water_quality_history, GET always returns the latest one) or, until the user
first sets one, a plausible simulated value so the panel has something to
show. Swap in real probe reads later by writing into the same table.
"""

import random

from database import SessionLocal, WaterQualityRecord

PH_RANGE = (6.5, 8.5)
MIN_DISSOLVED_OXYGEN = 5.0
MAX_AMMONIA = 0.05


def _flags(ph: float, dissolved_oxygen_mg_l: float, ammonia_mg_l: float) -> dict:
    return {
        "ph_ok": PH_RANGE[0] <= ph <= PH_RANGE[1],
        "dissolved_oxygen_ok": dissolved_oxygen_mg_l >= MIN_DISSOLVED_OXYGEN,
        "ammonia_ok": ammonia_mg_l <= MAX_AMMONIA,
    }


def compute_status(ph: float, dissolved_oxygen_mg_l: float, ammonia_mg_l: float) -> str:
    flags = _flags(ph, dissolved_oxygen_mg_l, ammonia_mg_l)
    issues = []
    if not flags["ph_ok"]:
        issues.append("pH out of safe range")
    if not flags["dissolved_oxygen_ok"]:
        issues.append("low dissolved oxygen")
    if not flags["ammonia_ok"]:
        issues.append("elevated ammonia")
    return "Good" if not issues else "; ".join(issues)


def compute_message(ph: float, dissolved_oxygen_mg_l: float, ammonia_mg_l: float) -> str:
    flags = _flags(ph, dissolved_oxygen_mg_l, ammonia_mg_l)
    if all(flags.values()):
        return (
            f"All readings are within safe ranges - pH ({ph}), dissolved oxygen "
            f"({dissolved_oxygen_mg_l} mg/L) and ammonia ({ammonia_mg_l} mg/L) all look healthy."
        )

    parts = []
    if not flags["ph_ok"]:
        parts.append(f"pH is {ph}, outside the safe {PH_RANGE[0]}-{PH_RANGE[1]} range")
    if not flags["dissolved_oxygen_ok"]:
        parts.append(f"dissolved oxygen is {dissolved_oxygen_mg_l} mg/L, below the {MIN_DISSOLVED_OXYGEN} mg/L minimum")
    if not flags["ammonia_ok"]:
        parts.append(f"ammonia is {ammonia_mg_l} mg/L, above the {MAX_AMMONIA} mg/L limit")
    return "Needs attention: " + "; ".join(parts) + "."


def _build_result(ph, temperature_c, dissolved_oxygen_mg_l, ammonia_mg_l, source) -> dict:
    return {
        "ph": ph,
        "temperature_c": temperature_c,
        "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
        "ammonia_mg_l": ammonia_mg_l,
        "status": compute_status(ph, dissolved_oxygen_mg_l, ammonia_mg_l),
        "message": compute_message(ph, dissolved_oxygen_mg_l, ammonia_mg_l),
        "source": source,
        **_flags(ph, dissolved_oxygen_mg_l, ammonia_mg_l),
    }


def _simulated_reading() -> dict:
    ph = round(random.uniform(6.5, 8.0), 2)
    temperature_c = round(random.uniform(24.0, 30.0), 1)
    dissolved_oxygen = round(random.uniform(5.0, 8.0), 2)
    ammonia = round(random.uniform(0.0, 0.05), 3)
    return _build_result(ph, temperature_c, dissolved_oxygen, ammonia, "simulated")


def get_reading() -> dict:
    db = SessionLocal()
    try:
        latest = (
            db.query(WaterQualityRecord)
            .order_by(WaterQualityRecord.timestamp.desc())
            .first()
        )
        if latest is None:
            return _simulated_reading()
        return _build_result(
            latest.ph, latest.temperature_c, latest.dissolved_oxygen_mg_l,
            latest.ammonia_mg_l, latest.source,
        )
    finally:
        db.close()


def save_reading(ph: float, temperature_c: float, dissolved_oxygen_mg_l: float, ammonia_mg_l: float) -> dict:
    db = SessionLocal()
    try:
        db.add(WaterQualityRecord(
            ph=ph,
            temperature_c=temperature_c,
            dissolved_oxygen_mg_l=dissolved_oxygen_mg_l,
            ammonia_mg_l=ammonia_mg_l,
            source="manual",
        ))
        db.commit()
    finally:
        db.close()

    return _build_result(ph, temperature_c, dissolved_oxygen_mg_l, ammonia_mg_l, "manual")
