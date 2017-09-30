#!/usr/bin/python
#
# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from Dataset.BoxLoader import BoxLoader
from Dataset.NexetDataset import NexetDataset
from BoxInceptionResnet import BoxInceptionResnet
from Visualize import Visualize
from Utils import CheckpointLoader
from Utils import PreviewIO

def main():
    parser = argparse.ArgumentParser(description="RFCN tester")
    parser.add_argument('-gpu', type=str, default="0", help='Train on this GPU(s)')
    parser.add_argument('-n', type=str, help='Network checkpoint file')
    parser.add_argument('-i', type=str, help='Input file.')
    parser.add_argument('-o', type=str, default="", help='Write output here.')
    parser.add_argument('-p', type=int, default=1, help='Show preview')
    parser.add_argument('-threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('-delay', type=int, default=-1, help='Delay between frames in visualization. -1 for automatic, 0 for wait for keypress.')
    parser.add_argument('-dataset', type=str, default="/home/eljefec/data/nexet/train", help="Path to Nexet dataset")
    parser.add_argument('-annotation', type=str, default="/home/eljefec/data/nexet/train_boxes.simple.csv", help="Path to annotation csv")

    opt=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    test_rfcn(opt)

class Trim:
    def __init__(self, s, p):
        self.s = s
        self.p = p

class ZoomTrim:
    def __init__(self, zoom):
        self.zoom = zoom
        self.height_trim = None
        self.width_trim = None

def preprocessInput(img):
    def calcPad(size):
        m = size % 32
        p = int(m/2)
        s = size - m
        return s,p

    zoom = max(640.0 / img.shape[0], 640.0 / img.shape[1])
    img = cv2.resize(img, (int(zoom*img.shape[1]), int(zoom*img.shape[0])))

    zoomtrim = ZoomTrim(zoom)

    if img.shape[0] % 32 != 0:
        s,p = calcPad(img.shape[0])
        img = img[p:p+s]
        zoomtrim.height_trim = Trim(s, p)

    if img.shape[1] % 32 != 0:
        s,p = calcPad(img.shape[1])
        img = img[:,p:p+s]
        zoomtrim.width_trim = Trim(s, p)

    return img, zoomtrim

def unprocessBox(zoomtrim, box):
    if zoomtrim.width_trim:
        # Add p to each x value
        box[0] += zoomtrim.width_trim.p
        box[2] += zoomtrim.width_trim.p
    if zoomtrim.height_trim:
        # Add p to each y value
        box[1] += zoomtrim.height_trim.p
        box[3] += zoomtrim.height_trim.p
    # Undo zoom.
    box /= zoomtrim.zoom

class Box:
    def __init__(self, class_name, x1, y1, x2, y2, prob):
        self.class_name = class_name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.prob = prob

class RFCNTester:
    def __init__(self, save_dir, network_file, bbox_threshold,
                 dataset = '/home/eljefec/data/nexet/train',
                 annotation = '/home/eljefec/data/nexet/train_boxes.simple.csv'):
        dataset = BoxLoader()
        dataset.add(NexetDataset(dataset, annotation))
        print("Number of categories: "+str(dataset.categoryCount()))
        print(dataset.getCaptionMap())
        self.categories = dataset.getCaptionMap()

        self.image = tf.placeholder(tf.float32, [None, None, None, 3])
        self.net = BoxInceptionResnet(self.image, dataset.categoryCount(), name="boxnet")

        self.boxes, self.scores, self.classes = self.net.getBoxes(scoreThreshold=bbox_threshold)

        self.sess = tf.Session()

        if not CheckpointLoader.loadCheckpoint(self.sess, save_dir, network_file, ignoreVarsInFileNotInSess=True):
            print("Failed to load network.")
            sys.exit(-1)

    def predict(self, img, as_boxes = True):
        img, zoomtrim = preprocessInput(img)

        def clipCoord(xy):
            return np.minimum(np.maximum(np.array(xy,dtype=np.int32),0),[img.shape[1]-1, img.shape[0]-1]).tolist()

        rBoxes, rScores, rClasses = self.sess.run([self.boxes, self.scores, self.classes], feed_dict={self.image: np.expand_dims(img, 0)})

        pred_boxes = []
        for box in range(rBoxes.shape[0]):
            unprocessBox(zoomtrim, rBoxes[box])
            topleft = tuple(clipCoord(rBoxes[box][0:2]))
            bottomright = tuple(clipCoord(rBoxes[box][2:5]))
            class_name = self.categories[rClasses[box]]
            prob = rScores[box]
            pred_boxes.append(Box(class_name, topleft[0], topleft[1], bottomright[0], bottomright[1], prob))

        if as_boxes:
            return pred_boxes
        else:
            return rBoxes, rScores, rClasses

def test_rfcn(opt):
    model = RFCNTester('save/save', opt.n, opt.threshold, opt.dataset, opt.annotation)

    input = PreviewIO.PreviewInput(opt.i)
    output = PreviewIO.PreviewOutput(opt.o, input.getFps())

    palette = Visualize.Palette(len(model.categories))

    while True:
        img = input.get()
        if img is None:
            break

        rBoxes, rScores, rClasses = model.predict(img, as_boxes = False)

        res = Visualize.drawBoxes(img, rBoxes, rClasses, [model.categories[i] for i in rClasses.tolist()], palette, scores=rScores)

        output.put(input.getName(), res)

        if opt.p==1:
            cv2.imshow("result", res)
            if opt.o=="":
                cv2.waitKey(input.getDelay() if opt.delay <0 else opt.delay)
            else:
                cv2.waitKey(1)

if __name__ == '__main__':
    main()
