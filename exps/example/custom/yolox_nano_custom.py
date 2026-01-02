#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (640, 640)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define your dataset path
        self.data_dir = "datasets/COCO"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"

        # Number of classes: person and hand
        self.num_classes = 2

        # Training configuration
        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 10

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Override to use 'train' folder instead of default 'train2017'.
        Uses CustomCOCODataset with validation to prevent CUDA errors.
        """
        from yolox.data import CustomCOCODataset, TrainTransform

        return CustomCOCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",  # Use 'train' folder instead of default 'train2017'
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            num_classes=self.num_classes,  # Validate class IDs
        )

    def get_eval_dataset(self, **kwargs):
        """
        Get evaluation dataset.
        Override to use 'val' folder instead of default 'val2017'.
        Uses CustomCOCODataset with validation to prevent CUDA errors.
        """
        from yolox.data import CustomCOCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return CustomCOCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val" if not testdev else "test",  # Use 'val'/'test' folders instead of 'val2017'/'test2017'
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            num_classes=self.num_classes,  # Validate class IDs
        )

