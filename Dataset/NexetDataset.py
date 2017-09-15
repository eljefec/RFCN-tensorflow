import random
import numpy as np
import cv2
import tensorflow as tf
from . import BoxAwareRandZoom
import simple_parser

class NexetDataset:
    def __init__(self, path, annotation_csv, set="train", normalizeSize=True, randomZoom=True):
        self.path = path
        self.annotation_csv = annotation_csv
        self.normalizeSize=normalizeSize
        self.set=set
        self.randomZoom=randomZoom
        self.all_data, self.class_count, self.class_name_mapping, self.class_mapping = None, None, None, None

    def init(self):
        self.all_data, self.class_count, self.class_name_mapping, self.class_mapping = simple_parser.get_data(self.annotation_csv)

        print("Loaded "+str(len(self.all_data))+" images")

    def classCount(self):
        return self.class_count

    def getCaptions(self, categories):
        if categories is None:
            return None

        res = []
        if isinstance(categories, np.ndarray):
            categories = categories.tolist()

        for c in categories:
            res.append(self.class_mapping[c])

        return res

    def load(self):
        while True:
            #imgId=self.images[1]
            #imgId=self.images[3456]
            ex = self.all_data[random.randint(0, len(self.all_data)-1)]

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
            for box in ex.bboxes:
                x1,y1,w,h = box.x, box.y, box.w, box.h
                newBox=[int(x1*sizeMul) - padLeft, int(y1*sizeMul) - padTop, int((x1+w)*sizeMul) - padLeft, int((y1+h)*sizeMul) - padTop]
                newBox[0] = max(min(newBox[0], img.shape[1]),0)
                newBox[1] = max(min(newBox[1], img.shape[0]),0)
                newBox[2] = max(min(newBox[2], img.shape[1]),0)
                newBox[3] = max(min(newBox[3], img.shape[0]),0)

                # CocoDataset filtered out boxes smaller than 16x16.
                # if (newBox[2]-newBox[0]) >= 16 and (newBox[3]-newBox[1]) >= 16:
                boxes.append(newBox)
                categories.append(self.class_name_mapping[box.class_name])

            if len(boxes)==0:
                print("Warning: No boxes on image. Skipping.")
                continue;

            boxes=np.array(boxes, dtype=np.float32)
            boxes=np.reshape(boxes, [-1,4])
            categories=np.array(categories, dtype=np.uint8)

            return img, boxes, categories

    def count(self):
        return len(self.all_data)
