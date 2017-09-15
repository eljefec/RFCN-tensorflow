import random
import os
import numpy as np
import cv2
import tensorflow as tf
from . import BoxAwareRandZoom
import simple_parser
import explore_nexet

class NexetDataset:
    def __init__(self, path, annotation_csv, set="train", normalizeSize=True, randomZoom=True):
        self.path = path
        self.annotation_csv = annotation_csv
        self.normalizeSize=normalizeSize
        self.set=set
        self.randomZoom=randomZoom
        self.all_data, self.class_count, self.class_name_mapping, self.class_mapping = None, None, None, None
        self.dataset = None

    def init(self):
        self.dataset = explore_nexet.load_nexet(self.annotation_csv)

        print("Loaded "+str(len(self.dataset.all_data))+" images")

    def classCount(self):
        return len(self.dataset.classes_count)

    def getCaptions(self, categories):
        if categories is None:
            return None

        res = []
        if isinstance(categories, np.ndarray):
            categories = categories.tolist()

        for c in categories:
            res.append(self.dataset.class_mapping[c])

        return res

    def load(self):
        while True:
            #imgId=self.images[1]
            #imgId=self.images[3456]
            ds = self.dataset
            ex = ds.all_data[random.randint(0, len(ds.all_data)-1)]

            imgFile = os.path.join(self.path, ex.filename)
            img = cv2.imread(imgFile)

            if img is None:
                print("ERROR: Failed to load "+imgFile)
                continue

            sizeMul = 1.0
            padTop = 0
            padLeft = 0

            if len(ex.bboxes) <= 0:
                continue

            iBoxes=[{"x":b.x1,
                     "y":b.y1,
                     "w":b.w,
                     "h":b.h
                    } for b in ex.bboxes]

            if self.randomZoom:
                img, iBoxes = BoxAwareRandZoom.randZoom(img, iBoxes, keepOriginalRatio=False, keepOriginalSize=False, keepBoxes=True)

            if self.normalizeSize:
                sizeMul = 640.0 / min(img.shape[0], img.shape[1])
                img = cv2.resize(img, (int(img.shape[1]*sizeMul), int(img.shape[0]*sizeMul)))

            m = img.shape[1] % 32
            if m != 0:
                padLeft = int(m/2)
                img = img[:,padLeft : padLeft + img.shape[1] - m]

            m = img.shape[0] % 32
            if m != 0:
                m = img.shape[0] % 32
                padTop = int(m/2)
                img = img[padTop : padTop + img.shape[0] - m]

            if img.shape[0]<256 or img.shape[1]<256:
                print("Warning: Image too small, skipping: "+str(img.shape))
                continue

            boxes=[]
            categories=[]
            for i in range(len(ex.bboxes)):
                x1,y1,w,h = iBoxes[i]["x"],iBoxes[i]["y"],iBoxes[i]["w"],iBoxes[i]["h"]
                newBox=[int(x1*sizeMul) - padLeft, int(y1*sizeMul) - padTop, int((x1+w)*sizeMul) - padLeft, int((y1+h)*sizeMul) - padTop]
                newBox[0] = max(min(newBox[0], img.shape[1]),0)
                newBox[1] = max(min(newBox[1], img.shape[0]),0)
                newBox[2] = max(min(newBox[2], img.shape[1]),0)
                newBox[3] = max(min(newBox[3], img.shape[0]),0)

                # CocoDataset filtered out boxes smaller than 16x16.
                # if (newBox[2]-newBox[0]) >= 16 and (newBox[3]-newBox[1]) >= 16:
                boxes.append(newBox)
                categories.append(ds.class_name_mapping[ex.bboxes[i].class_name])

            if len(boxes)==0:
                print("Warning: No boxes on image. Skipping.")
                continue;

            boxes=np.array(boxes, dtype=np.float32)
            boxes=np.reshape(boxes, [-1,4])
            categories=np.array(categories, dtype=np.uint8)

            return img, boxes, categories

    def count(self):
        return len(self.dataset.all_data)
