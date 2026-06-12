import os

class Config:
    # --- File Paths ---
    DATA_PATH = "data/ratings.csv"
    CKPT_DIR = "ckpt/"

    # --- Data & Graph Parameters ---
    TEST_SIZE = 0.2
    BATCH_SIZE = 8192
    SPLIT_TYPE = "temporal"   # "temporal" for time-sequential split, "random" for random split
    
    # --- Model Hyperparameters ---
    EMB_DIM = 64
    NUM_LAYERS = 3

    # --- Training Hyperparameters ---
    EPOCHS = 400
    LEARNING_RATE = 1e-3
    REG_WEIGHT = 1e-4
    EVAL_EPOCH = 20         # How often we calculate metrics
    SAVE_EPOCH = 50        # How often do we save model
    
    # --- Evaluate Hyperparameters ---
    K = 20                  # Top K

    # Ensure output directories exist when config is loaded
    os.makedirs(CKPT_DIR, exist_ok=True)