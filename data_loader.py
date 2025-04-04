import torch
from config import *
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import ImageDraw


class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')

        boxes, labels = [], []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            xmax, ymax = xmin + width, ymin + height
            boxes.append([xmin, ymin, xmax, ymax])

            category_id = ann['category_id']
            label = category_id_to_label.get(category_id, 0)  # Default to 0 if not found
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        img_id = torch.tensor([img_id])

        target = {"boxes": boxes, "labels": labels, "image_id": img_id}

        if self.transform:
            img = self.transform(img)

        return img, target


def draw_boxes(image, annotations):

    draw = ImageDraw.Draw(image)
    boxes = annotations.get("boxes", torch.empty((0, 4))).numpy()

    for bbox in boxes:
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)

    return image


def show_images(dataset, num_images=5):

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, anns = dataset[i]
        img = TF.to_pil_image(img).convert("RGB")

        img = draw_boxes(img, anns)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i}")

    plt.show()


def collate_fn(batch):
    """ Custom collate function to handle variable-sized tensors """
    images, targets = zip(*batch)
    images = torch.stack(images)

    max_boxes = 4  # Adjust based on dataset
    max_labels = 4  # Adjust based on dataset

    boxes, labels = [], []
    for t in targets:
        b = t["boxes"]
        l = t["labels"]

        # Truncate or pad bounding boxes
        if len(b) > max_boxes:
            b = b[:max_boxes]
        else:
            pad_b = torch.zeros((max_boxes - len(b), 4))
            b = torch.cat([b, pad_b], dim=0)

        # Truncate or pad labels
        if len(l) > max_labels:
            l = l[:max_labels]
        else:
            pad_l = torch.zeros((max_labels - len(l),), dtype=torch.int64)
            l = torch.cat([l, pad_l], dim=0)

        boxes.append(b)
        labels.append(l)

    targets = {
        "boxes": torch.stack(boxes),
        "labels": torch.stack(labels),
    }

    return images, targets


def create_dataset():

    transform = transforms.Compose([
        transforms.Resize((IMAGES_SIZE[1], IMAGES_SIZE[0])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = COCODataset(IMAGES, ANNOTATION, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS_NUMBER, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS_NUMBER, collate_fn=collate_fn)

    return train_loader, val_loader