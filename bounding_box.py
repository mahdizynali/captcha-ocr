# Authored By Mahdi Zeinali
# github : github.com/mahdizynali

import os
import json
import cv2 as cv

dir_base = os.path.dirname(os.path.realpath(__file__)) 
ann_path = dir_base + "/dataset/ann.json"
image_path = dir_base + "/dataset/images/"

with open(ann_path, "r") as ann:
    j_file = json.load(ann)
    for image in j_file["images"]:
        img_name = image["file_name"]
        img_id = image["id"]
        bboxes = []
        for annotation in j_file["annotations"]:
            if annotation["image_id"] == img_id:
                bbox = annotation["bbox"]
                bboxes.append(bbox)

        print(img_name, " | ", "   id : ", img_id, "|   boxes : ", bboxes)
        mat = cv.imread(image_path + img_name)
        for bbox in bboxes:
            image_box = cv.rectangle(mat, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (50, 205, 50), 1)
        cv.imshow("image", image_box)
        # print(image_box.shape)
        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()
            exit(0)