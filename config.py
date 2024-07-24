import torch

ANNOTATION = "dataset/ann.json"
IMAGES = "dataset/images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
BATCH_SIZE = 16
WORKERS_NUMBER = 4
CLASSES_NUMBER = 10
LEARNING_RATE = 0.001
IMAGES_SIZE = [210, 60]