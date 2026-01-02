#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX Video Inference Script
Run inference on video files (mp4, mov, avi, etc.) using trained YOLOX model weights.
"""

import argparse
import os
import time
from loguru import logger

import cv2
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.tools.demo import Predictor


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Video Inference")
    parser.add_argument(
        "-f",
        "--exp_file",
        type=str,
        default=None,
        help="Path to experiment file (default: YOLOX/exps/yolox_custom_hand.py or exps/yolox_custom_hand.py)",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth file). If not provided, uses best_ckpt.pth from output directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input video file (mp4, mov, avi, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save output video (optional, if not provided, displays video)",
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
    parser.add_argument(
        "--fps",
        default=None,
        type=float,
        help="Output video FPS (uses input video FPS if not provided)",
    )
    return parser


def main():
    args = make_parser().parse_args()
    
    # Set default exp file if not provided
    if args.exp_file is None:
        # Try different possible locations
        possible_paths = [
            "YOLOX/exps/yolox_custom_hand.py",
            "exps/yolox_custom_hand.py",
            os.path.join(os.path.dirname(__file__), "exps", "yolox_custom_hand.py"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.exp_file = os.path.abspath(path)
                break
        if args.exp_file is None:
            raise FileNotFoundError(
                "Could not find experiment file. Please specify with -f/--exp_file. "
                "Tried: {}".format(", ".join(possible_paths))
            )
    
    # Resolve exp file path - make it absolute if relative
    if not os.path.isabs(args.exp_file):
        # Try relative to current directory first
        if os.path.exists(args.exp_file):
            args.exp_file = os.path.abspath(args.exp_file)
        # Try relative to YOLOX directory
        elif os.path.exists(os.path.join("YOLOX", args.exp_file)):
            args.exp_file = os.path.abspath(os.path.join("YOLOX", args.exp_file))
        # Try as-is in case we're already in YOLOX directory
        else:
            args.exp_file = os.path.abspath(args.exp_file)
    
    # Load experiment configuration
    logger.info("Loading experiment configuration from: {}".format(args.exp_file))
    if not os.path.exists(args.exp_file):
        raise FileNotFoundError("Experiment file not found: {}".format(args.exp_file))
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
    
    # Determine checkpoint path - use best_ckpt.pth if not provided
    if args.ckpt is None:
        # Try to find best_ckpt.pth in output directory
        output_dir = os.path.join(exp.output_dir, exp.exp_name)
        ckpt_path = os.path.join(output_dir, "best_ckpt.pth")
        # Try relative to YOLOX directory if not found
        if not os.path.exists(ckpt_path) and os.path.exists(os.path.join("YOLOX", ckpt_path)):
            ckpt_path = os.path.join("YOLOX", ckpt_path)
        if os.path.exists(ckpt_path):
            args.ckpt = os.path.abspath(ckpt_path)
            logger.info("Using best checkpoint from: {}".format(args.ckpt))
        else:
            raise FileNotFoundError(
                "Checkpoint not provided and best_ckpt.pth not found in: {}. "
                "Please provide checkpoint path with -c/--ckpt".format(output_dir)
            )
    else:
        # Resolve checkpoint path
        if not os.path.isabs(args.ckpt):
            if os.path.exists(args.ckpt):
                args.ckpt = os.path.abspath(args.ckpt)
            elif os.path.exists(os.path.join("YOLOX", args.ckpt)):
                args.ckpt = os.path.abspath(os.path.join("YOLOX", args.ckpt))
            else:
                args.ckpt = os.path.abspath(args.ckpt)
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError("Checkpoint file not found: {}".format(args.ckpt))
    
    # Load checkpoint
    logger.info("Loading checkpoint from: {}".format(args.ckpt))
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
    
    # Get class names (use COCO_CLASSES, but limit to num_classes)
    class_names = COCO_CLASSES[:exp.num_classes] if exp.num_classes <= len(COCO_CLASSES) else COCO_CLASSES
    
    # Create predictor
    predictor = Predictor(
        model,
        exp,
        cls_names=class_names,
        trt_file=None,
        decoder=None,
        device=args.device,
        fp16=args.fp16,
        legacy=args.legacy,
    )
    
    # Resolve input video path
    if not os.path.isabs(args.input):
        # Try relative to current directory first
        if os.path.exists(args.input):
            args.input = os.path.abspath(args.input)
        # Try relative to YOLOX directory
        elif os.path.exists(os.path.join("YOLOX", args.input)):
            args.input = os.path.abspath(os.path.join("YOLOX", args.input))
        else:
            args.input = os.path.abspath(args.input)
    
    # Open input video
    logger.info("Opening input video: {}".format(args.input))
    if not os.path.exists(args.input):
        raise FileNotFoundError("Video file not found: {}".format(args.input))
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError("Could not open video file: {}".format(args.input))
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info("Video properties: {}x{} @ {:.2f} FPS, {} frames".format(
        width, height, fps, total_frames
    ))
    
    # Setup output video writer if saving
    vid_writer = None
    if args.output:
        output_fps = args.fps if args.fps is not None else fps
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(args.output, fourcc, output_fps, (width, height))
        logger.info("Saving output video to: {} @ {:.2f} FPS".format(args.output, output_fps))
    
    # Process video
    frame_count = 0
    total_time = 0.0
    start_time = time.time()
    
    logger.info("Starting video inference...")
    logger.info("Press 'q' to quit if displaying video")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            inference_start = time.time()
            outputs, img_info = predictor.inference(frame)
            inference_time = time.time() - inference_start
            total_time += inference_time
            
            # Visualize results with bounding boxes and class labels
            result_frame = predictor.visual(outputs[0], img_info, cls_conf=args.conf)
            
            # Add FPS overlay
            fps_text = "FPS: {:.1f}".format(1.0 / inference_time if inference_time > 0 else 0)
            frame_text = "Frame: {}/{}".format(frame_count, total_frames)
            cv2.putText(result_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, frame_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Count detections
            if outputs[0] is not None:
                num_detections = len(outputs[0])
                det_text = "Detections: {}".format(num_detections)
                cv2.putText(result_frame, det_text, (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save or display
            if vid_writer:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("YOLOX Video Inference", cv2.WINDOW_NORMAL)
                cv2.imshow("YOLOX Video Inference", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Stopped by user")
                    break
            
            # Log progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                logger.info("Processed {}/{} frames ({:.1f}%), Avg FPS: {:.2f}".format(
                    frame_count, total_frames, progress, avg_fps
                ))
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if vid_writer:
            vid_writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        avg_inference_time = total_time / frame_count if frame_count > 0 else 0
        
        logger.info("=" * 50)
        logger.info("Video inference completed!")
        logger.info("Total frames processed: {}".format(frame_count))
        logger.info("Total time: {:.2f}s".format(elapsed))
        logger.info("Average FPS: {:.2f}".format(avg_fps))
        logger.info("Average inference time per frame: {:.4f}s".format(avg_inference_time))
        if args.output:
            logger.info("Output video saved to: {}".format(args.output))
        logger.info("=" * 50)


if __name__ == "__main__":
    main()

