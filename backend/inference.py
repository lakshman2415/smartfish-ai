import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import CLASS_NAMES, MODELS_DIR, TREATMENT_RECOMMENDATIONS  # noqa: E402
from data import get_transforms  # noqa: E402
from model import HybridFishDiseaseModel, fish_confidence, localize_fish  # noqa: E402

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = get_transforms(train=False)
_model = None

FISH_GATE_THRESHOLD = 0.05  # low on purpose -- see fish_confidence() docstring

# fish_confidence() comes from a generic ImageNet classifier, which
# recognizes whole-fish photos but was never trained on close-up fish
# skin/gill/fin texture -- exactly the kind of tight crop this project's own
# disease dataset is full of, and exactly what a farmer photographing just
# the affected patch would upload. It's tempting to fall back to the disease
# classifier's own top-class confidence in that case, but that doesn't work:
# calibration against this project's own test images showed real low-gate
# photos scoring disease_conf in [0.93, 1.00] -- indistinguishable from
# synthetic non-fish images (flat color, gradients, smooth noise) which also
# scored up to 0.99. Softmax confidence is not a reliable "is this a fish"
# signal on its own; the model was never trained with a negative/background
# class to calibrate against. So instead of guessing, the frontend offers an
# explicit "this is a close-up" checkbox that bypasses the gate on the
# user's own assertion (see `closeup` below) rather than an automatic
# heuristic that provably can't tell the two cases apart.


class NotAFishError(Exception):
    pass


def _load_model() -> HybridFishDiseaseModel:
    global _model
    if _model is not None:
        return _model

    checkpoint_path = MODELS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained model at {checkpoint_path}. Train it first with "
            "notebooks/SmartFish_Training_Colab.ipynb, then copy the resulting "
            "models/best_model.pt into this project's models/ folder."
        )

    checkpoint = torch.load(checkpoint_path, map_location=_device)
    model = HybridFishDiseaseModel(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device).eval()
    _model = model
    return _model


def _severity(class_name: str, confidence: float) -> str:
    if class_name == "Healthy":
        return "None"
    if confidence < 0.5:
        return "Uncertain - retake photo with better lighting/focus"
    return TREATMENT_RECOMMENDATIONS[class_name]["severity_hint"]


def _prepare_crop(image_bytes: bytes):
    """Decode the upload, localize + crop the fish. Returns (cropped_rgb, x, y, w, h)
    with x/y/w/h in the ORIGINAL (uncropped) image's pixel space."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    x, y, w, h = localize_fish(image_bgr)
    cropped_rgb = image_rgb[y:y + h, x:x + w]
    if cropped_rgb.size == 0:
        cropped_rgb = image_rgb
        h, w = image_rgb.shape[:2]
        x, y = 0, 0
    return cropped_rgb, x, y, w, h


def detect_fish(image_bytes: bytes, closeup: bool = False) -> dict:
    """First step: locate the fish and check it's plausibly a fish, without
    running the (heavier) disease classifier. `closeup=True` skips the check
    entirely -- the user has explicitly said this is a tight crop of the
    affected area rather than a whole fish, which the gate can't judge (see
    the module-level comment above FISH_GATE_THRESHOLD)."""
    cropped_rgb, x, y, w, h = _prepare_crop(image_bytes)
    tensor = _transform(image=cropped_rgb)["image"].unsqueeze(0).to(_device)

    fish_conf = fish_confidence(tensor)
    return {
        "is_fish": closeup or fish_conf >= FISH_GATE_THRESHOLD,
        "fish_confidence": fish_conf,
        "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    }


def predict(image_bytes: bytes, closeup: bool = False) -> dict:
    model = _load_model()
    cropped_rgb, x, y, w, h = _prepare_crop(image_bytes)
    tensor = _transform(image=cropped_rgb)["image"].unsqueeze(0).to(_device)

    fish_conf = fish_confidence(tensor)
    if not closeup and fish_conf < FISH_GATE_THRESHOLD:
        raise NotAFishError(
            f"This doesn't look like a fish (fish-likeness {fish_conf:.0%}). "
            "Please upload a clear photo of a fish, or tick \"this is a "
            "close-up of the affected area\" if it's a tight crop."
        )

    with torch.no_grad():
        probs = model.predict_proba(tensor).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    rec = TREATMENT_RECOMMENDATIONS[predicted_class]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "severity": _severity(predicted_class, confidence),
        "class_probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "treatment": rec["treatment"],
        "prevention": rec["prevention"],
        "affected_region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    }
