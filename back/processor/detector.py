# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np

from config import init
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, Shader
from core.utils import decode_cfg, load_weights


class Detector(object):
    def __init__(self):
        self.img_size = 416
        self.threshold = 0.4
        self.max_frame = 160
        self.init_model()
        self.names = init.XML.LABELS
        self.shader = Shader(init.TRAIN.NUM_CLASSES)

    def init_model(self):
        cfg = decode_cfg(init.TRAIN.YAML_PATH)

        from core.model.one_stage.yolov3 import YOLOv3_Tiny as Model

        _, model = Model(cfg)
        model.summary()
        self.model = model
        init_weight_path = init.EVAL.TEST_WEIGHT_PATH  # cfg['test']['init_weight_path']
        if init_weight_path:
            print('Load Weights File From:', init_weight_path)
            load_weights(model, init_weight_path)
        else:
            raise SystemExit('init_weight_path is Empty !')

    def detect(self, path):

        image = read_image(path)

        ms, bboxes, scores, classes = self.inference(image)
        image = draw_bboxes(image, bboxes, scores, classes, self.names, self.shader)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite('./tmp/draw/' + path.split('/')[-1], image)
        box = bboxes[0]
        w = box[2] - box[0]
        h = box[3] - box[1]
        obj = init.XML.LABELS[int(classes[0])]

        image_info = {obj: [str(int(w)) + 'x' + str(int(h)), str(round(scores[0],2))]}
        return image_info

    def inference(self, image):
        h, w = image.shape[:2]
        image = preprocess_image(image, (self.img_size, self.img_size)).astype(np.float32)
        images = np.expand_dims(image, axis=0)

        tic = time.time()
        bboxes, scores, classes, valid_detections = self.model.predict(images)
        toc = time.time()

        bboxes = bboxes[0][:valid_detections[0]]
        scores = scores[0][:valid_detections[0]]
        classes = classes[0][:valid_detections[0]]

        # bboxes *= image_size
        _, bboxes = postprocess_image(image, (w, h), bboxes)

        return (toc - tic) * 1000, bboxes, scores, classes
