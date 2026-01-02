#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Logger configuration
        self.logger = 'tensorboard'
        
        # Model configuration - using YOLOX-S as base
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Track best validation results
        self.best_val_results = {
            'best_ap_50_95': 0.0,
            'best_ap_50': 0.0,
            'best_epoch': 0,
            'all_results': []
        }
        
        # Disable multiscale training for faster training (fixed size 640x640)
        self.multiscale_range = 0
        
        # Dataset configuration
        self.data_dir = "datasets/COCO"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"  # Use val.json for testing if needed
        
        # Number of classes: 'hand' and 'person'
        self.num_classes = 2
        
        # Training configuration
        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 10
        self.print_interval = 10
        
        # Augmentation settings
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.enable_mixup = True
        
        # Testing configuration
        self.test_conf = 0.5
        self.nmsthre = 0.45

    def get_model(self):
        """Get YOLOX model with custom configuration."""
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act
            )
            self.model = YOLOX(backbone, head)
        
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get training dataset.
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
        Uses CustomCOCODataset with validation to prevent CUDA errors.
        """
        from yolox.data import CustomCOCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        
        return CustomCOCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val" if not testdev else "test",  # Use 'val'/'test' folders
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            num_classes=self.num_classes,  # Validate class IDs
        )
    
    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        """
        Override eval to save best validation results as JSON.
        """
        import json
        import os
        
        # Call parent eval method
        results = super(Exp, self).eval(model, evaluator, is_distributed, half, return_outputs)
        
        from yolox.utils import is_main_process
        if is_main_process():
                ap50_95, ap50, summary = results[:3] if len(results) >= 3 else (0, 0, "")
                
                # Get current epoch from trainer if available
                # This will be set by the trainer
                current_epoch = getattr(self, '_current_epoch', 0)
                
                # Record this evaluation result
                result_entry = {
                    'epoch': current_epoch,
                    'ap_50_95': float(ap50_95),
                    'ap_50': float(ap50),
                }
                self.best_val_results['all_results'].append(result_entry)
                
                # Update best results if this is better
                if ap50_95 > self.best_val_results['best_ap_50_95']:
                    self.best_val_results['best_ap_50_95'] = float(ap50_95)
                    self.best_val_results['best_ap_50'] = float(ap50)
                    self.best_val_results['best_epoch'] = current_epoch
                    
                    # Save best validation results to JSON
                    output_dir = getattr(self, 'output_dir', './YOLOX_outputs')
                    exp_name = getattr(self, 'exp_name', 'yolox_custom_hand')
                    results_dir = os.path.join(output_dir, exp_name)
                    os.makedirs(results_dir, exist_ok=True)
                    
                    results_file = os.path.join(results_dir, 'best_validation_results.json')
                    with open(results_file, 'w') as f:
                        json.dump(self.best_val_results, f, indent=4)
                    
                    from loguru import logger
                    logger.info(f"Best validation results saved to {results_file}")
        
        return results

