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

def preprocessInput(img):
    def calcPad(size):
        m = size % 32
        p = int(m/2)
        s = size - m
        return s,p

    zoom = max(640.0 / img.shape[0], 640.0 / img.shape[1])
    img = cv2.resize(img, (int(zoom*img.shape[1]), int(zoom*img.shape[0])))

    if img.shape[0] % 32 != 0:
        s,p = calcPad(img.shape[0])
        img = img[p:p+s]

    if img.shape[1] % 32 != 0:
        s,p = calcPad(img.shape[1])
        img = img[:,p:p+s]

    return img

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

    def predict(self, img):
        img = preprocessInput(img)

        rBoxes, rScores, rClasses = self.sess.run([self.boxes, self.scores, self.classes], feed_dict={self.image: np.expand_dims(img, 0)})

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

        img = preprocessInput(img)

        rBoxes, rScores, rClasses = model.predict(img)

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
