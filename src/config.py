"""
Configuration file for MovieLens 1M preprocessing
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
RATINGS_FILE = RAW_DATA_DIR / "ratings.dat"
MOVIES_FILE = RAW_DATA_DIR / "movies.dat"
USERS_FILE = RAW_DATA_DIR / "users.dat"

# Processed data paths
TRAIN_FILE = PROCESSED_DATA_DIR / "train.pt"
VAL_FILE = PROCESSED_DATA_DIR / "val.pt"
TEST_FILE = PROCESSED_DATA_DIR / "test.pt"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.pkl"
STATS_FILE = PROCESSED_DATA_DIR / "statistics.json"

# Preprocessing parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature engineering parameters
MIN_RATING_THRESHOLD = 20  # Minimum ratings per user/movie
GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

# Age groups mapping
AGE_MAPPING = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

# Occupation mapping
OCCUPATION_MAPPING = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}

print(f"✓ Configuration loaded")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  Raw data directory: {RAW_DATA_DIR}")