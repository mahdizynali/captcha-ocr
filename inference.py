import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import matplotlib.pyplot as plt
from network import Captcha_OCR
from config import DEVICE, CLASSES_NUMBER
import os

idx_to_label = {v: k for k, v in {
    1: 0,  
    2: 8,  
    3: 3,  
    4: 4,  
    5: 6,  
    6: 9,  
    7: 7,  
    8: 5,  
    9: 2,  
    10: 1  
}.items()}

def load_model(model_path, device=DEVICE):
    model = Captcha_OCR(num_classes=CLASSES_NUMBER)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (210, 60))
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    return img.to(DEVICE)

def draw_boxes(image, boxes, labels):
    image = np.array(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        label = labels[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def infer(model, image_path):
    image_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        bbox_pred, class_pred = model(image_tensor)

    bbox_pred = bbox_pred.squeeze(0).cpu().numpy()
    class_pred = torch.argmax(class_pred, dim=2).squeeze(0).cpu().numpy()
    
    digit_labels = [str(idx_to_label.get(idx, "?")) for idx in class_pred]
    
    image = cv2.imread(image_path)
    image_with_boxes = draw_boxes(image, bbox_pred, digit_labels)

    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    model_path = "/home/mahdi/Desktop/captcha-ocr/saved_models/best_model.pth"
    test_image_path = "test.jpg"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    elif not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
    else:
        model = load_model(model_path)
        infer(model, test_image_path)
