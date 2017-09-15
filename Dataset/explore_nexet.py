import cv2
import numpy as np
import os
import pickle
from simple_parser import get_data
from PIL import Image

class NexetData:
    def __init__(self, all_data, classes_count, class_name_mapping, class_mapping):
        self.all_data = all_data
        self.classes_count = classes_count
        self.class_name_mapping = class_name_mapping
        self.class_mapping = class_mapping

def pickle_nexet(simple_csv, picklename):
    all_data, classes_count, class_name_mapping, class_mapping = get_data(simple_csv)
    nexet_data = NexetData(all_data, classes_count, class_name_mapping, class_mapping)
    with open(picklename, 'wb') as f:
        pickle.dump(nexet_data, f)
    return nexet_data

def load_nexet(simple_csv):
    picklename = simple_csv + '.p'
    if os.path.exists(picklename):
        with open(picklename, 'rb') as f:
            return pickle.load(f)
    else:
        return pickle_nexet(simple_csv, picklename)

def generate_nexet(simple_csv):
    nexet = load_nexet(simple_csv)
    for ex in nexet.all_data:
        yield ex, nexet.classes_count, nexet.class_mapping

def draw_example(ex, area):
    img = cv2.imread(ex['filepath'])
    for bbox in ex['bboxes']:
        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color=(255,0,0))
        cv2.circle(img, (bbox['x1'], bbox['y1']), 5, (0,0,255), 1)
        cv2.putText(img, str(area), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.show()
    try:
        wait_for_input = input('Press enter for next: ')
    except SyntaxError:
        pass

def draw_nexet(simple_csv):
    for ex, cls_count, cls_map in generate_nexet(simple_csv):
        img = cv2.imread(ex['filepath'])
        for bbox in ex['bboxes']:
            cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color=(255,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img, 'RGB')
        pil_img.show()
        try:
            wait_for_input = input('Press enter for next: ')
        except SyntaxError:
            pass

def explore_nexet(simple_csv):
    areas = []
    for ex, cls_count, cls_map in generate_nexet(simple_csv):
        for bbox in ex['bboxes']:
            area = abs(bbox['x2'] - bbox['x1']) * abs(bbox['y2'] - bbox['y1'])
            areas.append(area)
            if area < 75 and area > 69 or area > 600000:
                draw_example(ex, area)

    hist = np.histogram(areas, bins=10, range=(0.0, 1000.0))
    mean = np.mean(areas)
    std = np.std(areas)
    min = np.min(areas)
    max = np.max(areas)
    print('stats', 'mean', mean, 'std', std, 'min', min, 'max', max)
    print('hist', hist)

def explore():
    explore_nexet('/home/eljefec/data/nexet/train_boxes.simple.csv')
    # draw_nexet('/home/eljefec/data/nexet/train_boxes.simple.csv')

if __name__ == '__main__':
    explore()
