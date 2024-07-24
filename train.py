from config import *
from data_loader import *
import torch.nn as nn
import torch.optim as optim
from network import Captcha_Model

model = Captcha_Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, annotations in train_loader:

        images = images.to(device) 
        boxes = [t['boxes'].to(device) for t in annotations]
        boxes = torch.cat(boxes, dim=0)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        print(outputs)
        loss = criterion(outputs, boxes)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Training loss: {epoch_loss:.4f}")
    return epoch_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:

            images = list(image.to(device) for image in images)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = corrects / total
    print(f"Validation loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    return epoch_loss, accuracy


def main():

    TRAIN_SET, VALID_SET =  create_dataset()
    
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss = train(model, TRAIN_SET, criterion, optimizer, DEVICE)
        val_loss, val_accuracy = evaluate(model, VALID_SET, criterion, DEVICE)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()



# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# import torchvision

# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     return model

# TRAIN_SET, VALID_SET =  create_dataset()  

# # 2 classes; Only target class or background
# num_classes = 10
# num_epochs = 10
# model = get_model_instance_segmentation(num_classes)

# # move model to the right device
# model.to(DEVICE)
    
# # parameters
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# len_dataloader = len(TRAIN_SET)

# for epoch in range(num_epochs):
#     model.train()
#     i = 0    
#     for imgs, annotations in TRAIN_SET:
#         i += 1
#         imgs = list(img.to(DEVICE) for img in imgs)
#         annotations = [{k: v.to(DEVICE) for k, v in t.items()} for t in annotations]
#         loss_dict = model(imgs, annotations)
#         losses = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')