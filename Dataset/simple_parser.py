import cv2
import numpy as np

class Box:
    def __init__(self, class_name, x1, y1, x2, y2):
        self.class_name = class_name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        assert(x1 <= x2)
        assert(y1 <= y2)
        self.w = x2 - x1
        self.h = y2 - y1

class Example:
    def __init__(self, filename, width, height, bboxes):
        self.filename = filename
        self.width = width
        self.height = height
        self.bboxes = bboxes

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_name_mapping = {}

    class_mapping = {}

    visualise = True

    with open(input_path,'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_name_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_name_mapping[class_name] = len(class_name_mapping)
                class_mapping[len(class_mapping)] = class_name

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0,6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


        all_data = []
        for key in all_imgs:
            img = all_imgs[key]
            bboxes = []
            for box in img['bboxes']:
                bboxes.append(Box(box['class'], box['x1'], box['y1'], box['x2'], box['y2']))
            ex = Example(img['filepath'], img['width'], img['height'], bboxes)
            all_data.append(ex)

        # make sure the bg class is last in the list
        if found_bg:
            if class_name_mapping['bg'] != len(class_name_mapping) - 1:
                key_to_switch = [key for key in class_name_mapping.keys() if class_name_mapping[key] == len(class_name_mapping)-1][0]
                val_to_switch = class_name_mapping['bg']
                class_name_mapping['bg'] = len(class_name_mapping) - 1
                class_name_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_name_mapping, class_mapping


