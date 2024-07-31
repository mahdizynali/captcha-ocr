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

category_id_to_label = {
    1: 0,  # zero
    2: 8,  # eight
    3: 3,  # three
    4: 4,  # four
    5: 6,  # six
    6: 9,  # nine
    7: 7,  # seven
    8: 5,  # five
    9: 2,  # two
    10: 1  # one
}