#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Custom COCO dataset class with class ID validation to prevent CUDA device-side assert errors.
"""

import os
import cv2
import numpy as np
import copy
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .coco import remove_useless_info
from .datasets_wrapper import CacheDataset, cache_read_img


class CustomCOCODataset(CacheDataset):
    """
    Custom COCO dataset class with validation for class IDs.
    This prevents CUDA device-side assert errors by ensuring class indices are in range.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
        num_classes=None,
    ):
        """
        COCO dataset initialization with class ID validation.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            num_classes (int): Expected number of classes. If provided, validates class IDs.
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.num_classes = num_classes

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        
        # Get all category IDs from the annotation file
        all_cat_ids = sorted(self.coco.getCatIds())
        
        # Get category IDs that are actually used in annotations
        # This prevents issues when annotation files define more categories than are used
        used_cat_ids = set()
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                used_cat_ids.add(ann['category_id'])
        
        # Only use categories that are actually present in annotations
        self.class_ids = sorted(used_cat_ids)
        self.cats = self.coco.loadCats(self.class_ids)
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        
        # Validate class IDs
        self._validate_class_ids()
        
        self.annotations = self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def _validate_class_ids(self):
        """Validate that class IDs are in the expected range."""
        if self.num_classes is None:
            return
        
        # Check if we have the right number of classes
        if len(self.class_ids) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, but found {len(self.class_ids)} "
                f"category IDs in annotation file: {self.class_ids}"
            )
        
        # Check if class IDs are in valid range
        # COCO format typically uses 1-based indexing (1, 2, ...)
        # But we need to ensure they map to indices 0, 1, ..., num_classes-1
        min_id = min(self.class_ids)
        max_id = max(self.class_ids)
        
        # Accept either 0-based (0, 1) or 1-based (1, 2) indexing
        if not ((min_id == 0 and max_id == self.num_classes - 1) or 
                (min_id == 1 and max_id == self.num_classes)):
            raise ValueError(
                f"Category IDs {self.class_ids} are not in valid range for {self.num_classes} classes.\n"
                f"Expected IDs: [0, {self.num_classes-1}] (0-based) or [1, {self.num_classes}] (1-based).\n"
                f"Found IDs: {self.class_ids}\n"
                f"Please update your annotation file to use category IDs 1 and 2 (or 0 and 1)."
            )
        
        print(f"✓ Validated {len(self.class_ids)} classes with IDs: {self.class_ids}")

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            try:
                cls = self.class_ids.index(obj["category_id"])
            except ValueError:
                raise ValueError(
                    f"Category ID {obj['category_id']} not found in class_ids {self.class_ids}. "
                    f"This will cause CUDA device-side assert errors. "
                    f"Please check your annotation file."
                )
            
            # Double-check that class index is in valid range
            if self.num_classes is not None and cls >= self.num_classes:
                raise ValueError(
                    f"Class index {cls} is out of range for {self.num_classes} classes. "
                    f"Category ID: {obj['category_id']}, Class IDs: {self.class_ids}"
                )
            
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

