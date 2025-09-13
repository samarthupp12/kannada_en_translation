import torch

MODEL_NAME = "google/mt5-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SAMPLE_SIZE = 10000
VAL_SAMPLE_SIZE = 1000
MAX_LENGTH = 128
EPOCHS = 10
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
