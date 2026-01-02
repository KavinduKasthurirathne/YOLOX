#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import os
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.output_dir = None  # Will be set by trainer if needed

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        
        # Generate confusion matrix if in main process
        if is_main_process() and self.output_dir is not None:
            try:
                self.generate_confusion_matrix(data_list)
            except Exception as e:
                logger.warning(f"Failed to generate confusion matrix: {e}")

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # Ensure required fields exist for pycocotools (they may have been removed by remove_useless_info)
            if 'info' not in cocoGt.dataset:
                cocoGt.dataset['info'] = {}
            if 'licenses' not in cocoGt.dataset:
                cocoGt.dataset['licenses'] = []
            
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
                # Try to instantiate to check if C++ compilation works
                # If it fails, fall back to standard COCOeval
                try:
                    cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
                except Exception:
                    # C++ compilation failed (e.g., Windows without compiler)
                    from pycocotools.cocoeval import COCOeval
                    cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
                    logger.warning("Fast COCOeval compilation failed, using standard COCOeval.")
            except ImportError:
                # Fast COCOeval not available at all
                from pycocotools.cocoeval import COCOeval
                cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
                logger.warning("Fast COCOeval not available, using standard COCOeval.")
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
    
    def generate_confusion_matrix(self, predictions):
        """
        Generate and save confusion matrix from predictions and ground truth.
        
        Args:
            predictions: List of prediction dictionaries in COCO format
        """
        try:
            from pycocotools.coco import COCO
            
            cocoGt = self.dataloader.dataset.coco
            cat_ids = sorted(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in cat_ids]
            
            # Create mapping from COCO category IDs to class indices
            cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
            
            # Initialize confusion matrix (add 1 for background/false positives)
            num_classes = len(cat_ids)
            confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
            
            # Group predictions by image_id
            pred_by_image = defaultdict(list)
            for pred in predictions:
                pred_by_image[pred['image_id']].append(pred)
            
            # For each image, match predictions to ground truth
            for img_id, preds in pred_by_image.items():
                # Get ground truth annotations for this image
                ann_ids = cocoGt.getAnnIds(imgIds=[img_id])
                anns = cocoGt.loadAnns(ann_ids)
                
                # Convert predictions to numpy arrays
                pred_boxes = []
                pred_classes = []
                pred_scores = []
                for pred in preds:
                    bbox = pred['bbox']  # [x, y, w, h]
                    # Convert to [x1, y1, x2, y2]
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                    pred_boxes.append([x1, y1, x2, y2])
                    pred_classes.append(cat_id_to_idx.get(pred['category_id'], -1))
                    pred_scores.append(pred['score'])
                
                if len(pred_boxes) == 0:
                    # No predictions - all ground truth are false negatives
                    for ann in anns:
                        gt_class = cat_id_to_idx.get(ann['category_id'], -1)
                        if gt_class >= 0:
                            confusion_matrix[gt_class, num_classes] += 1  # FN
                    continue
                
                pred_boxes = np.array(pred_boxes)
                pred_classes = np.array(pred_classes)
                pred_scores = np.array(pred_scores)
                
                # Convert ground truth to numpy arrays
                gt_boxes = []
                gt_classes = []
                for ann in anns:
                    bbox = ann['bbox']  # [x, y, w, h]
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_class = cat_id_to_idx.get(ann['category_id'], -1)
                    gt_classes.append(gt_class)
                
                if len(gt_boxes) == 0:
                    # No ground truth - all predictions are false positives
                    for pred_cls in pred_classes:
                        if pred_cls >= 0:
                            confusion_matrix[num_classes, pred_cls] += 1  # FP
                    continue
                
                gt_boxes = np.array(gt_boxes)
                gt_classes = np.array(gt_classes)
                
                # Compute IoU matrix
                ious = self._compute_iou_matrix(pred_boxes, gt_boxes)
                
                # Match predictions to ground truth using IoU threshold of 0.5
                iou_threshold = 0.5
                matched_gt = set()
                matched_pred = set()
                
                # Sort by IoU (highest first)
                matches = []
                for i in range(len(pred_boxes)):
                    for j in range(len(gt_boxes)):
                        if ious[i, j] >= iou_threshold:
                            matches.append((ious[i, j], i, j))
                
                matches.sort(reverse=True)
                
                for iou, pred_idx, gt_idx in matches:
                    if pred_idx not in matched_pred and gt_idx not in matched_gt:
                        pred_cls = pred_classes[pred_idx]
                        gt_cls = gt_classes[gt_idx]
                        if pred_cls >= 0 and gt_cls >= 0:
                            confusion_matrix[gt_cls, pred_cls] += 1  # TP or misclassification
                        matched_pred.add(pred_idx)
                        matched_gt.add(gt_idx)
                
                # Unmatched predictions are false positives
                for pred_idx in range(len(pred_boxes)):
                    if pred_idx not in matched_pred:
                        pred_cls = pred_classes[pred_idx]
                        if pred_cls >= 0:
                            confusion_matrix[num_classes, pred_cls] += 1  # FP
                
                # Unmatched ground truth are false negatives
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx not in matched_gt:
                        gt_cls = gt_classes[gt_idx]
                        if gt_cls >= 0:
                            confusion_matrix[gt_cls, num_classes] += 1  # FN
            
            # Plot confusion matrix
            self._plot_confusion_matrix(confusion_matrix, cat_names, num_classes)
            
        except Exception as e:
            logger.warning(f"Error generating confusion matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes."""
        # boxes: [N, 4] in format [x1, y1, x2, y2]
        def box_area(boxes):
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        inter = np.prod(np.clip(rb - lt, 0, None), axis=2)  # [N, M]
        union = area1[:, None] + area2 - inter
        
        iou = inter / (union + 1e-6)
        return iou
    
    def _plot_confusion_matrix(self, confusion_matrix, class_names, num_classes):
        """Plot and save confusion matrix."""
        # Add "Background" to class names for the last row/column
        display_names = class_names + ['Background']
        
        # Normalize confusion matrix for better visualization
        cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        # Create figure
        plt.figure(figsize=(max(10, num_classes), max(10, num_classes)))
        
        # Plot using seaborn for better visualization
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=display_names,
            yticklabels=display_names,
            cbar_kws={'label': 'Normalized Count'},
            square=True,
            linewidths=0.5
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        if self.output_dir and os.path.exists(self.output_dir):
            cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to {cm_path}")
        else:
            plt.close()
            if self.output_dir:
                logger.warning(f"Output directory does not exist: {self.output_dir}")
