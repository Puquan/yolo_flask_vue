# -*- coding: utf-8 -*-
import tensorflow as tf

from core.metrics import COCOEval
from core.callbacks.utils import local_eval

from config import init

class COCOEvalCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self,
                 save_path,
                 eval_model,
                 model_cfg,
                 only_save_weight=True,
                 verbose=1):
        super(COCOEvalCheckpoint, self).__init__()
        self.save_path = save_path
        self.eval_model = eval_model
        self.model_cfg = model_cfg

        self.only_save_weight = only_save_weight
        self.verbose = verbose

        self._image_size = self.model_cfg['test']['image_size'][0]
        self._best_AP = -float('inf')

        self.name_path = init.TRAIN.NAME_PATH       #self.model_cfg['yolo']['name_path']
        self.test_path = init.TRAIN.TEST_ANNO_PATH  #self.model_cfg['test']['anno_path']

    def on_epoch_end(self, epoch, logs=None):

        AP = local_eval(COCOEval, self.eval_model, self._image_size, self.test_path, self.name_path, self.verbose)
        print("The current mAP is %.4f" % AP)

        if AP > self._best_AP:
            if self.save_path is None:
                if self.verbose > 0:
                    print("mAP improved from {:.2%} to {:.2%}".format(self._best_AP, AP))
                    print(" ")
            else:
                save_path = self.save_path.format(mAP=AP)
                if self.verbose > 0:
                    print("mAP improved from {:.2%} to {:.2%}, saving model to {}".format(self._best_AP, AP, save_path))
                    print(" ")
                if self.only_save_weight:
                    self.eval_model.save_weights(save_path)
                else:
                    self.eval_model.save(save_path)
            self._best_AP = AP
        else:
            if self.verbose > 0:
                print("mAP not improved from {:.2%}".format(self._best_AP))
                print(" ")
