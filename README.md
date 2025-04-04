## Captcha OCR - Deep Learning

A deep learning-based solution for detecting and classifying digits from CAPTCHA images. This repository provides scripts for dataset preprocessing,model training, evaluation, and inference, using a convolutional neural network (CNN) approach to perform Optical Character Recognition (OCR) on CAPTCHA images.
## Dataset
In this project im using coco dataset which i have provided from real captchas that you must organize in dataset folder
![alt text](https://raw.githubusercontent.com/mahdizynali/captcha-ocr/refs/heads/main/dataset/images/w_123.jpg)
![alt text](https://github.com/mahdizynali/captcha-ocr/blob/main/dataset/images/captcha_061.jpg) \
for downloading proper dataset you can use data2learn hub:
```
www.data2learn.ir
```
## Usage
In order to train your custom model, after providing coco dataset you have to configure config.py based on your desire traning hyperparameters and your custom labeld name like this one :
```python
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
```
Then just try to run trainer :
```
python3 train.py
```
### Inferencer
After reaching the best model of your training, you can test and inference it by adapt your model path in inferencer script and run it:
```
python3 inference.py
```
