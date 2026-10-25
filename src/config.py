"""
Central config for the SmartFish hybrid disease classifier.

Dataset note
------------
The poster's target taxonomy has 11 classes (Bacterial/Fungal/Parasitic/Viral,
Fin Rot, Tail Rot, Gill Disease, Ich, Ulcer Disease, Dropsy, Healthy). No single
public dataset with clean labels covers all 11 with enough images per class.

This project merges two Kaggle datasets:

  "Freshwater Fish Disease Aquaculture in South Asia" (subirbiswas19)
    Bacterial diseases - Aeromoniasis, Bacterial gill disease,
    Bacterial Red disease, Fungal diseases Saprolegniasis, Healthy Fish,
    Parasitic diseases, Viral diseases White tail disease
    (~350 images/class)

  "Fresh Water Fish Disease Dataset" (utpolkantidas)
    Argulus, Bacterial Gill Rot, Broken Antennae and Rostrum, EUS, Healthy
    Fish, Red Spot, Tail and Fin Rot (6-31 images/class - small)

Red Spot / Bacterial Gill Rot / Argulus / Healthy Fish are folded into the
matching class above (bulks up volume, same underlying condition). EUS and
Tail and Fin Rot become two new classes (Ulcer_Disease, Tail_And_Fin_Rot) --
real disease categories from the poster's 11, but thin (9-23 images), so
expect noticeably weaker accuracy on those two specifically. Broken Antennae
and Rostrum is physical injury, not a disease in the poster's taxonomy, and
is left unmatched/skipped on purpose.

That gets to 9 of the poster's 11 classes. Fin Rot/Tail Rot as *separate*
classes and Dropsy still aren't represented in usable volume in public data
as of writing -- extend CLASS_NAMES/CLASS_KEYWORDS below and re-run
dataset_prep.py if you add more images under dataset/raw/<class_name>/.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
SPLIT_DATA_DIR = PROJECT_ROOT / "dataset" / "split"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = [
    "Bacterial_Aeromoniasis",
    "Bacterial_Gill_Disease",
    "Bacterial_Red_Disease",
    "Fungal_Saprolegniasis",
    "Parasitic_Disease",
    "Viral_White_Tail_Disease",
    "Ulcer_Disease",
    "Tail_And_Fin_Rot",
    "Healthy",
]
NUM_CLASSES = len(CLASS_NAMES)

# Lowercase keyword fragments used to auto-map the Kaggle downloads' actual
# folder names (which vary in punctuation/casing, and come from two source
# datasets) onto CLASS_NAMES above.
CLASS_KEYWORDS = {
    "Bacterial_Aeromoniasis": ["aeromon"],
    "Bacterial_Gill_Disease": ["gill"],
    "Bacterial_Red_Disease": ["red"],
    "Fungal_Saprolegniasis": ["fungal", "saprolegni"],
    "Parasitic_Disease": ["parasit", "argulus"],
    "Viral_White_Tail_Disease": ["viral", "white tail", "white_tail", "whitetail"],
    "Ulcer_Disease": ["eus", "epizootic ulcerative", "ulcer"],
    "Tail_And_Fin_Rot": ["tail and fin", "tail_and_fin", "tail rot", "fin rot"],
    "Healthy": ["healthy"],
}

TREATMENT_RECOMMENDATIONS = {
    "Bacterial_Aeromoniasis": {
        "severity_hint": "High",
        "treatment": "Isolate affected fish; antibiotic-medicated feed or bath "
        "(vet-prescribed, e.g. oxytetracycline) per local aquaculture guidance.",
        "prevention": "Improve water quality, reduce stocking density, avoid handling stress.",
    },
    "Bacterial_Gill_Disease": {
        "severity_hint": "High",
        "treatment": "Improve aeration and water exchange; salt bath or vet-approved "
        "antibacterial treatment.",
        "prevention": "Maintain low ammonia/nitrite, avoid overcrowding, quarantine new stock.",
    },
    "Bacterial_Red_Disease": {
        "severity_hint": "High",
        "treatment": "Vet-prescribed antibiotic treatment; isolate affected fish immediately.",
        "prevention": "Strict water quality control, disinfect equipment between ponds/tanks.",
    },
    "Fungal_Saprolegniasis": {
        "severity_hint": "Moderate",
        "treatment": "Salt bath or antifungal treatment (e.g. formalin/malachite green "
        "alternatives per local regulation); remove dead tissue.",
        "prevention": "Avoid skin injury, keep water clean and at stable temperature.",
    },
    "Parasitic_Disease": {
        "severity_hint": "Moderate",
        "treatment": "Anti-parasitic bath treatment; identify parasite type for targeted dosing.",
        "prevention": "Quarantine new fish, regular water testing, avoid overfeeding.",
    },
    "Viral_White_Tail_Disease": {
        "severity_hint": "Critical",
        "treatment": "No specific cure; supportive care, isolate and remove affected fish "
        "to limit spread, disinfect equipment.",
        "prevention": "Source disease-free broodstock, avoid stress, maintain biosecurity.",
    },
    "Ulcer_Disease": {
        "severity_hint": "Critical",
        "treatment": "No specific cure for EUS; supportive care, isolate affected fish, "
        "improve water quality immediately, consult a fish health specialist for outbreak management.",
        "prevention": "Avoid cold/low-pH stress periods, maintain good water quality, "
        "avoid overcrowding, quarantine new stock.",
    },
    "Tail_And_Fin_Rot": {
        "severity_hint": "Moderate",
        "treatment": "Improve water quality immediately; salt bath or vet-approved "
        "antibacterial/antifungal treatment; trim badly damaged tissue only if vet-advised.",
        "prevention": "Avoid overcrowding and fin nipping/injury, maintain clean water, "
        "avoid sudden temperature swings.",
    },
    "Healthy": {
        "severity_hint": "None",
        "treatment": "No treatment needed.",
        "prevention": "Maintain current water quality and feeding regimen.",
    },
}

# ---- data ----
IMG_SIZE = 224
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ---- training ----
BATCH_SIZE = 32
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# ---- backbones (timm model names) ----
CNN_BACKBONE = "efficientnet_b4"
VIT_BACKBONE = "vit_base_patch16_224"
FUSION_DIM = 512
