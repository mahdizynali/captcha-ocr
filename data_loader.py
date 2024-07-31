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

        num_objs = len(anns)
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    

def draw_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        if 'boxes' in ann:
            boxes = ann['boxes'].numpy()
            for bbox in boxes:
                x_min, y_min, x_max, y_max = bbox
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
    return image

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, anns = dataset[i]
        img = TF.to_pil_image(img)
        if isinstance(anns, dict):
            anns = [anns]
        img = draw_boxes(img, anns)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i}')
    plt.show()


# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack(images)
#     return images, targets

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    
    boxes = [t['boxes'] for t in targets]
    labels = [t['labels'] for t in targets]
    max_boxes = max(len(b) for b in boxes)
    
    padded_boxes = [torch.cat([b, torch.zeros(max_boxes - len(b), 4)], dim=0) for b in boxes]
    padded_labels = [torch.cat([l, torch.zeros(max_boxes - len(l))], dim=0) for l in labels]
    
    masks = [torch.cat([torch.ones(len(b)), torch.zeros(max_boxes - len(b))], dim=0) for b in boxes]

    boxes_tensor = torch.stack(padded_boxes)
    labels_tensor = torch.stack(padded_labels)
    masks_tensor = torch.stack(masks)
    
    targets = {'boxes': boxes_tensor, 'labels': labels_tensor, 'masks': masks_tensor}
    return images, targets


def create_dataset():
    transform = transforms.Compose([
        transforms.Resize((IMAGES_SIZE[1], IMAGES_SIZE[0])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    coco_dataset = COCODataset(IMAGES, ANNOTATION, transform=transform)
    train_size = int(0.8 * len(coco_dataset))
    val_size = len(coco_dataset) - train_size
    train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])

    TRAIN_SET = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS_NUMBER, collate_fn=collate_fn)
    VALID_SET = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS_NUMBER, collate_fn=collate_fn)
    # show_images(TRAIN_SET.dataset, num_images=10)

    # for imgs, annotations in TRAIN_SET:
    #     imgs = imgs.to("cuda") 
    #     annotations = [{k: v.to("cuda") for k, v in t.items()} for t in annotations]
    #     print(annotations)
    
    return TRAIN_SET, VALID_SET
