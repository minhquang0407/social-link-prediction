import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data_output"

CLEAN_DATA_DIR = DATA_DIR / "cleaned"
RAW_JSON_DIR = DATA_DIR / "raw/json"
RAW_PARQUET_DIR = DATA_DIR / "raw/parquet"
GRAPH_DIR = DATA_DIR / "graph"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = DATA_DIR / "models"
PREDICT_DIR = DATA_DIR / "predicting"
for d in [CLEAN_DATA_DIR,DATA_DIR, RAW_JSON_DIR, RAW_PARQUET_DIR, GRAPH_DIR, TRAINING_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_PARQUET_PATH = RAW_PARQUET_DIR / "raw_master.parquet"
EDGES_DATA_PATH = CLEAN_DATA_DIR / "edges_data.parquet"
NODES_DATA_PATH = CLEAN_DATA_DIR / "nodes_data.parquet"
GRAPH_PATH = GRAPH_DIR / "relationship_graph.pkl"
SEARCH_INDEX_PATH = GRAPH_DIR / "search_index.json"
ANALYTICS_PATH = GRAPH_DIR / "analytics.json"


PYG_DATA_PATH = TRAINING_DIR / "pyg_data.pt"
PYG_TRAINING_DATA_PATH = TRAINING_DIR / "pyg_training_data.pt"
MODEL_PATH = MODELS_DIR / "model.pt"
PREDICT_DATA_PATH = PREDICT_DIR/ "embeddings.pt"
METADATA_PATH = PREDICT_DIR / "metadata.pkl"
ADJACENCY_PATH = PREDICT_DIR / "adjacency.pkl.gz"
TRAINING_HISTORY_PATH = DATA_DIR / "training_history.json"

SPARQL_TIMEOUT = 300
DEFAULT_PAGE_SIZE = 5000
INPUT_DIM = 385
HIDDEN_DIM = 64
OUTPUT_DIM = 128
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 2048
FUZZY_THRESHOLD = 60.0

MAX_YEAR = 2025
MIN_YEAR = 1800
SEARCH_THRESHOLD = 60
