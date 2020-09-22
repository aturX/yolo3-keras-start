# yolo3-keras-start
a quickly start tool  for  yolo3

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```


环境：

1、IDE：pycharm

2、Python3.6

3、TensorFlow-GPU




下载预训练权重：https://pjreddie.com/media/files/yolov3.weights