# SmartFish - Automated Fish Disease Diagnosis

Hybrid deep learning system for freshwater fish disease diagnosis: upload or
capture a fish photo, get a predicted disease, confidence score, and a
treatment/prevention recommendation, on a dashboard with a water-quality
panel and diagnosis history.

## How this maps to the poster

| Poster stage | This project |
|---|---|
| RT-DETRv2 fish detection | `src/model.py: localize_fish()` - classical CV crop (Otsu threshold + largest contour). No bounding-box-annotated dataset was available to train a real detector; swap this for a fine-tuned RT-DETRv2 later if you annotate one (e.g. on Roboflow). |
| EfficientNet-B4 + ViT-B/16 hybrid fusion | Implemented as designed: `timm` pretrained backbones, projected to a shared space, fused with multi-head attention, classified with an FC head. See `src/model.py`. |
| 11 disease classes | Trained on **9 classes**, merged from two Kaggle datasets (subirbiswas19 + utpolkantidas): Bacterial Aeromoniasis, Bacterial Gill Disease, Bacterial Red Disease, Fungal Saprolegniasis, Parasitic Disease, Viral White Tail Disease, Ulcer Disease (from EUS), Tail_And_Fin_Rot (combined), Healthy. That covers 9 of the poster's 11 labels (Fin Rot + Tail Rot both satisfied jointly by one combined class). **Ich (White Spot Disease)** and **Dropsy** have no usable public dataset and are shown in the frontend's "Conditions This System Covers" legend as "in development," not live predictions. Extend `CLASS_NAMES`/`CLASS_KEYWORDS` in `src/config.py` and add real, verified images under `dataset/raw/<class>/` to close that gap. |
| "≥98.9% accuracy" target | Aspirational, not guaranteed - real accuracy depends on dataset size/quality. `src/evaluate.py` reports your actual numbers. |
| Water quality panel | Simulated in `backend/water_quality.py` (no physical sensors). Swap in real pH/DO/temp/ammonia probe reads later; the API contract won't need to change. |

## Project structure

```
src/                  dataset prep, model, training, evaluation (run these directly or via the notebook)
notebooks/            SmartFish_Training_Colab.ipynb - one-click training on a free Colab GPU
backend/              FastAPI app serving predictions, history, water quality
frontend/             React + Vite app (Home, Detection, Water Quality, Dashboard, Alerts)
dataset/              raw/ and split/ populate after you run dataset_prep.py (gitignored)
models/               best_model.pt lands here after training (gitignored)
results/              training curves, confusion matrix, ROC curves, metrics.json (gitignored)
```

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model (Google Colab - recommended)

This hybrid model (two pretrained backbones + attention fusion) is slow on
CPU. Train it on a free Colab GPU:

1. Upload the whole `SmartFish` folder to Google Drive at `MyDrive/SmartFish`.
2. Get a Kaggle API token: kaggle.com -> Settings -> Create New Token (downloads `kaggle.json`).
3. Open `notebooks/SmartFish_Training_Colab.ipynb` in Google Colab.
4. `Runtime > Change runtime type > T4 GPU`, then run all cells top to bottom.
   You'll be prompted to upload `kaggle.json` partway through.
5. Training writes `models/best_model.pt` (synced back into your Drive) and
   evaluation plots into `results/`.
6. Copy `models/best_model.pt` into this project's local `models/` folder so
   the backend can load it.

### Training locally instead (needs an NVIDIA GPU for reasonable speed)

```bash
python src/dataset_prep.py   # downloads + organizes + splits the dataset (needs kaggle.json in ~/.kaggle/)
python src/train.py
python src/evaluate.py
```

## 3. Run the backend

```bash
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8001 --reload
```

Endpoints: `POST /predict` (multipart image upload), `GET /history`,
`GET /water-quality`, `GET /health`. Until `models/best_model.pt` exists,
`/predict` returns a 503 explaining that training hasn't been done yet.

## 4. Run the frontend

```bash
cd frontend
npm install   # first time only
npm run dev
```

Open http://localhost:5500. It talks to the backend at
`http://localhost:8001` (change `API_BASE` in `frontend/src/api.js` if you
use a different port).

## Notes for your project report

- Data split: stratified 70/15/15 train/val/test, image-level (the public
  dataset has no per-fish/patient ID to split on, so true patient-level
  separation isn't possible with this source).
- Augmentation: horizontal flip, rotation, color jitter, Gaussian noise/blur
  on-the-fly per epoch (`src/data.py`), plus CutMix applied to ~30% of
  training batches (`src/train.py`) - not written to disk as extra files.
- Training: AdamW, lr 1e-4 with `ReduceLROnPlateau`, cross-entropy loss,
  batch size 32, up to 100 epochs with early stopping (patience 10).
- Metrics reported: accuracy, precision/recall/F1 (macro), per-class
  specificity, macro ROC-AUC, confusion matrix - all written to `results/`
  by `src/evaluate.py`.
