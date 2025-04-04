import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os

from network import Captcha_OCR 
from data_loader import create_dataset
from config import *

model = Captcha_OCR().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
classification_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()

def train(model, train_loader, optimizer, classification_criterion, bbox_criterion, device='cuda'):
    model.train()
    running_loss, running_class_loss, running_bbox_loss = 0.0, 0.0, 0.0
    correct, total_chars = 0, 0  

    for images, targets in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        targets_labels = targets['labels'].to(device)
        targets_boxes = targets['boxes'].to(device)

        optimizer.zero_grad()

        bbox_pred, class_pred = model(images)
        class_pred = class_pred.view(images.shape[0], -1, model.num_classes)
        targets_labels = targets_labels.view(images.shape[0], -1)

        valid_labels_mask = targets_labels != 0  
        valid_class_labels = targets_labels[valid_labels_mask]
        valid_class_preds = class_pred.view(-1, model.num_classes)[valid_labels_mask.view(-1)]

        class_loss = classification_criterion(valid_class_preds, valid_class_labels)

        valid_boxes_mask = targets_boxes.sum(dim=2) != 0  
        valid_boxes = targets_boxes[valid_boxes_mask]
        valid_bbox_pred = bbox_pred[valid_boxes_mask]

        if valid_boxes.numel() > 0:
            bbox_loss = bbox_criterion(valid_bbox_pred, valid_boxes)
        else:
            bbox_loss = torch.tensor(0.0, device=device)  # Avoid NaN if no valid boxes

        loss = class_loss + bbox_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_class_loss += class_loss.item()
        running_bbox_loss += bbox_loss.item()

        _, predicted = torch.max(class_pred, 2)
        correct += (predicted == targets_labels).sum().item()
        total_chars += valid_labels_mask.sum().item()  # Count only non-padded chars

    avg_loss = running_loss / len(train_loader)
    avg_class_loss = running_class_loss / len(train_loader)
    avg_bbox_loss = running_bbox_loss / len(train_loader)
    accuracy = 100 * correct / total_chars if total_chars > 0 else 0.0

    print(f"Train Loss: {avg_loss:.4f}, Class Loss: {avg_class_loss:.4f}, BBox Loss: {avg_bbox_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, avg_class_loss, avg_bbox_loss, accuracy

def evaluate(model, val_loader, classification_criterion, bbox_criterion, device='cuda'):
    model.eval()
    running_loss, running_class_loss, running_bbox_loss = 0.0, 0.0, 0.0
    correct, total_chars = 0, 0  

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            targets_labels = targets['labels'].to(device)
            targets_boxes = targets['boxes'].to(device)

            bbox_pred, class_pred = model(images)
            class_pred = class_pred.view(images.shape[0], -1, model.num_classes)
            targets_labels = targets_labels.view(images.shape[0], -1)

            valid_labels_mask = targets_labels != 0  
            valid_class_labels = targets_labels[valid_labels_mask]
            valid_class_preds = class_pred.view(-1, model.num_classes)[valid_labels_mask.view(-1)]

            class_loss = classification_criterion(valid_class_preds, valid_class_labels)

            valid_boxes_mask = targets_boxes.sum(dim=2) != 0  
            valid_boxes = targets_boxes[valid_boxes_mask]
            valid_bbox_pred = bbox_pred[valid_boxes_mask]

            if valid_boxes.numel() > 0:
                bbox_loss = bbox_criterion(valid_bbox_pred, valid_boxes)
            else:
                bbox_loss = torch.tensor(0.0, device=device)

            loss = class_loss + bbox_loss

            running_loss += loss.item()
            running_class_loss += class_loss.item()
            running_bbox_loss += bbox_loss.item()

            _, predicted = torch.max(class_pred, 2)
            correct += (predicted == targets_labels).sum().item()
            total_chars += valid_labels_mask.sum().item()

    avg_loss = running_loss / len(val_loader)
    avg_class_loss = running_class_loss / len(val_loader)
    avg_bbox_loss = running_bbox_loss / len(val_loader)
    accuracy = 100 * correct / total_chars if total_chars > 0 else 0.0

    print(f"Validation Loss: {avg_loss:.4f}, Class Loss: {avg_class_loss:.4f}, BBox Loss: {avg_bbox_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, avg_class_loss, avg_bbox_loss, accuracy

def main():
    TRAIN_SET, VALID_SET = create_dataset()
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        train_loss, train_class_loss, train_bbox_loss, train_accuracy = train(
            model, TRAIN_SET, optimizer, classification_criterion, bbox_criterion, DEVICE
        )

        val_loss, val_class_loss, val_bbox_loss, val_accuracy = evaluate(
            model, VALID_SET, classification_criterion, bbox_criterion, DEVICE
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(SAVE_PATH, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best Model Saved at {best_model_path}")

    print("\nTraining Completed!")

if __name__ == "__main__":
    main()