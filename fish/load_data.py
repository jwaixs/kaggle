import json
import os
import cv2
import random

import matplotlib.pyplot as plt

data_dir = '/data/kaggle/fisheries/'

classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft']
annotation_files = {
    c : os.path.join(data_dir, 'annotations/{}_labels.json'.format(c))
    for c in classes
}

image_info = dict()
for c in classes:
    with open(annotation_files[c], 'r') as f:
        j = json.load(f)
        for elm in j:
            image_info[elm['filename'].split('/')[-1]] = {
                'class' : c,
                'annotations' : elm['annotations']
            }

def show_image(img_name):
    image, annotation = load_image(img_name)
    plt.imshow(image)

    currentAxis = plt.gca()
    for a in annotation:
        print(a)
        x, y, w, h = a['x'], a['y'], a['width'], a['height']
        currentAxis.add_patch(plt.Rectangle((x, y), w, h,
                                        color = 'red', fill = False, lw = 1))

    plt.show()

def load_image(img_name):
    info = image_info[img_name]
    class_dir = info['class'].upper()
    image_file = os.path.join(data_dir, 'train/{}/{}'.format(class_dir, img_name))
    annotation = info['annotations']

    image = cv2.imread(image_file)

    return image, annotation

def show_random_images():
    for img_name in random.sample(image_info.keys(), 10):
        print('Load: {} {}'.format(img_name, image_info[img_name]['class']))
        show_image(img_name)
