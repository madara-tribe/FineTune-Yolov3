import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from natsort import natsorted
import json
import glob
from skimage.feature import hog
import xml.etree.ElementTree as ET
from os import getcwd


classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle']
def voc_annotation(anno_path, in_file):
    tree=ET.parse(os.path.join(anno_path, in_file))
    root = tree.getroot()
    save_name = 'VOC2012/JPEGImages/' + root.find('filename').text
    fname = 'VOC2012/JPEGImages/' + root.find('filename').text
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        return fname, save_name, b, cls_id

def save_voc_dataset():
    anno_path = "VOC2012/Annotations"
    fname = [path for path in os.listdir(anno_path)]
    sorted_file =[path for path in natsorted(fname)]

    df=[]
    for idx, f in enumerate(sorted_file):
        fname, save,  b, cls_id = convert_annotation(anno_path, f)
        #print(fname, save, b, cls)
        img = cv2.imread(fname)
        if idx<10:
            bbox = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=10)
            plt.imshow(bbox),plt.show()
        print(save, b[0], b[1], b[2], b[3], cls_id)
        df.append([save, b[0], b[1], b[2], b[3], cls_id])


    with open('yolov3_voc.txt', 'w+') as f:
      for d in df:
        f.write(','.join(map(str, d)) + '\n')



def save_original_dataset():
    classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle']
    anno_path = "dtc_train_annotations"
    filename = [path for path in os.listdir(anno_path)]
    sorted_file =[path for path in natsorted(filename)]

    save_dir= "dic_train/dtc_train1s"
    Name, bbox, timeofday =[], [], []
    for idx, file in enumerate(sorted_file):
      a = open(os.path.join(anno_path, file))
      Jso = json.load(a)
      name, _ = os.path.splitext(file)
      img_path = os.path.join(save_dir, name) + ".jpg"
      print(name, Jso['attributes']['timeofday'])
      for js in Jso['labels']:
        #print(img_path, js, Jso['attributes']['timeofday'])
        Name.append(img_path)
        bbox.append(js)
        timeofday.append(Jso['attributes']['timeofday'])
    print(len(Name), len(bbox), len(timeofday))



    with open('dic_image.txt', 'w+') as f:
        for d in df:
            f.write(','.join(map(str, d)) + '\n')
