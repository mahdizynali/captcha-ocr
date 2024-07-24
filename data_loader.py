import torch
from config import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import cv2
import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import ImageDraw


class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img_path = os.path.join(self.root, path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        # print(f"Image shape after transform: {img.shape}")
        return img, anns

# transform augmentation
transform = transforms.Compose([
    transforms.Resize((IMAGES_SIZE[1], IMAGES_SIZE[0])),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),  # if use it , you must do the same for annotations
    # transforms.RandomRotation(30), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

coco_dataset = COCODataset(IMAGES, ANNOTATION, transform=transform)

dataloader = DataLoader(coco_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS_NUMBER)


#=========================================================================================

def draw_boxes(image, annotations):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        bbox = ann['bbox']

        x, y, width, height = bbox
        draw.rectangle([x, y, x + width, y + height], outline='red', width=2)
    return image

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, anns = dataset[i]
        img = TF.to_pil_image(img)
        img = draw_boxes(img, anns)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i}')
    plt.show()

show_images(coco_dataset, num_images=5)

#=========================================================================================