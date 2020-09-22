"""
Retrain the YOLO model for your own dataset.
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data


class YoloModel():
    def __init__(self):
        self.JPGPaths = 'D://VOCdevkit/VOC2020/JPEGImages/train.jpg'
        self.TXTPaths = 'D://VOCdevkit/VOC2020/ImageSets/Main/train.txt'
        self.XMLPaths = 'D://VOCdevkit/VOC2020/Annotations/%s.xml'
        self.classes = ["大巴车", "公交车", "绿色渣土车", "红色渣土车", "灰色渣土车", "蓝色渣土车", "危险品罐车", "环卫车", "厢式货车", "水泥搅拌车", "工程车"]
        self.annotation_path = '2020_train.txt'
        self.log_dir = 'logs/000/'
        self.classes_path = 'dabache,gongjiaoche,greenzhatuche,redzhatuche,grayzhatuche,bluezhatuche,weixianpinche,huanweiche,xiangshihuoche,shuinijiaobanche,gongchengche'
        self.anchors_path = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
        self.weights_path = 'logs/000/ep115-loss10.940-val_loss11.115.h5'  # 放置最新的 h5 文件

        self.get_annotation()  # 转换 文件路径
        self.anchors = self.get_anchors()
        self.num_anchors = len(self.anchors)
        self.class_names = self.get_classes()
        self.num_classes = len(self.class_names)
        self.input_shape = (416, 416)
        self.val_split = 0.1    # 测试集训练集比例
        self.batch_size = 2     # note that more GPU memory is required after unfreezing the body


    def _main(self):

        model = self.create_model(freeze_body=2, weights_path=self.weights_path) # make sure you know what you freeze

        logging = TensorBoard(log_dir=self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*self.val_split)
        num_train = len(lines) - num_val

        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
            print('Unfreeze all of the layers.')

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, self.batch_size))
            model.fit_generator(self.data_generator_wrapper(lines[:num_train], self.batch_size, self.input_shape, self.anchors, self.num_classes),
                steps_per_epoch=max(1, num_train//self.batch_size),
                validation_data=self.data_generator_wrapper(lines[num_train:], self.batch_size, self.input_shape, self.anchors, self.num_classes),
                validation_steps=max(1, num_val//self.batch_size),
                epochs=200,
                initial_epoch=100,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            model.save_weights(self.log_dir + 'trained_weights_final.h5')

        # Further training if needed.


    def get_classes(self):
        '''loads the classes'''
        class_names = self.classes_path
        class_names =[str(x) for x in class_names.split(',')]
        return class_names

    def get_anchors(self):
        '''loads the anchors from a file'''

        anchors = self.anchors_path    #
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def create_model(self, load_pretrained=True, freeze_body=2, weights_path='model_data/yolo_weights.h5'):
        '''create the training model'''
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape  # multiple of 32, hw
        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            self.num_anchors//3, self.num_classes+5)) for l in range(3)]

        model_body = yolo_body(image_input, self.num_anchors//3, self.num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(self.num_anchors, self.num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers)-3)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model


    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i==0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


    def get_annotation(self):
        print("----------------START: create file paths -------------------")
        image_ids = open(self.TXTPaths).read().strip().split()
        list_file = open(self.annotation_path, 'w')
        for image_id in image_ids:
            list_file.write(self.JPGPaths)
            self.convert_annotation(image_id, list_file)
            list_file.write('\n')
        list_file.close()
        print("----------------END: create file paths -------------------")


    def convert_annotation(self, image_id, list_file):
        import xml.etree.ElementTree as ET


        in_file = open(self.XMLPaths % (image_id), encoding="utf8")
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':

    # 模型训练
    yoloModel = YoloModel()
    yoloModel._main()
