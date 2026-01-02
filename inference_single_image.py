#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX Single Image Inference Script
Run inference on a single image using trained YOLOX model weights.
"""

import argparse
import os
import time
from loguru import logger

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.tools.demo import Predictor


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Single Image Inference")
    parser.add_argument(
        "-f",
        "--exp_file",
        type=str,
        default="exps/yolox_custom_hand.py",
        help="Path to experiment file",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save output image (optional)",
    )
    parser.add_argument(
        "--device",
        default="gpu" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device to run inference on (cpu or gpu)",
    )
    parser.add_argument(
        "--conf",
        default=0.3,
        type=float,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--nms",
        default=0.3,
        type=float,
        help="NMS threshold",
    )
    parser.add_argument(
        "--tsize",
        default=None,
        type=int,
        help="Test image size (will use exp.test_size if not provided)",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for faster inference",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="Use legacy preprocessing",
    )
    return parser


def main():
    args = make_parser().parse_args()
    
    # Load experiment configuration
    logger.info("Loading experiment configuration from: {}".format(args.exp_file))
    exp = get_exp(args.exp_file, None)
    
    # Update test configuration if provided
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    
    # Get model
    logger.info("Building model...")
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    
    # Set device
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # Convert to FP16
    model.eval()
    
    # Load checkpoint
    logger.info("Loading checkpoint from: {}".format(args.ckpt))
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError("Checkpoint file not found: {}".format(args.ckpt))
    
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # Handle different checkpoint formats
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    logger.info("Checkpoint loaded successfully.")
    
    # Fuse model if requested
    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)
    
    # Create predictor
    # Use COCO_CLASSES which should be defined for your 2 classes: 'person' and 'hand'
    # COCO_CLASSES is imported from yolox.data.datasets and should match your dataset
    predictor = Predictor(
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device=args.device,
        fp16=args.fp16,
        legacy=args.legacy,
    )
    
    # Run inference
    logger.info("Running inference on image: {}".format(args.image))
    if not os.path.exists(args.image):
        raise FileNotFoundError("Image file not found: {}".format(args.image))
    
    outputs, img_info = predictor.inference(args.image)
    
    # Visualize results
    result_image = predictor.visual(outputs[0], img_info, cls_conf=args.conf)
    
    # Save or display result
    if args.output:
        logger.info("Saving result to: {}".format(args.output))
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        cv2.imwrite(args.output, result_image)
        logger.info("Result saved successfully!")
    else:
        # Display image
        cv2.namedWindow("YOLOX Detection Result", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLOX Detection Result", result_image)
        logger.info("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print detection summary
    if outputs[0] is not None:
        num_detections = len(outputs[0])
        logger.info("Found {} detection(s)".format(num_detections))
        for i, det in enumerate(outputs[0]):
            cls_id = int(det[6])
            conf = float(det[4] * det[5])
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            logger.info("  Detection {}: {} (confidence: {:.2f})".format(i+1, cls_name, conf))
    else:
        logger.info("No detections found.")


if __name__ == "__main__":
    main()

