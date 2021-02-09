# FineTuning yolov3 model's performance

# light image prediction（Morning and day）

![predict_2](https://user-images.githubusercontent.com/48679574/72659606-c1ddaf80-3a05-11ea-9a17-cee7adba2363.png)

![predict_3](https://user-images.githubusercontent.com/48679574/72659607-c2764600-3a05-11ea-851a-328bb7220df4.png)


# Dark image prediction（Night）
![predict_1](https://user-images.githubusercontent.com/48679574/72659622-ed609a00-3a05-11ea-87f7-ba317c34d56b.png)

![predict_4](https://user-images.githubusercontent.com/48679574/72659623-ee91c700-3a05-11ea-9d5f-a84ecaab2726.png)

# How to fine tune
```
1. Download YOLOv3 weights from yolo website
$ cd weight
$ wget https://pjreddie.com/media/files/yolov3.weights

2. python convert.py yolov3.cfg yolov3.weights yolo.h5

3. Instead of darknet53.weights, use yolo.h5 for train
DARKNET_WEIGHT_PATH = 'yolo.h5' (yolov3_train.py)
```

# Set up VOC images dataset and sample train
```
1 create VOC anno text file and dataset for sample train
$ ./setup_voc_anno/voc_setup.sh

2. train
pyhton3 train.py
```



# Data format
format: ```image_file_path, x_min,y_min,x_max,y_max,class_id```

<b>example</b>
```
image_path, x1, y1, x2, y2, class_id
path/to/img2.jpg. 120,300,250,600,2

```
# detail and logic

Its detail and logic are written bellow site

http://trafalbad.hatenadiary.jp/entry/2020/01/18/170842


