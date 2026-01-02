#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Standalone script to parse train_log.txt and generate training visualization plots.
Usage: python plot_results.py [--log_file path/to/train_log.txt] [--output_dir path/to/output]
"""

import argparse
import os
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def parse_train_log(log_file_path):
    """
    Parse train_log.txt to extract Loss and mAP values.
    
    Args:
        log_file_path: Path to train_log.txt file
        
    Returns:
        dict: Dictionary containing parsed metrics
    """
    metrics = {
        'epochs': [],
        'total_loss': [],
        'iou_loss': [],
        'l1_loss': [],
        'conf_loss': [],
        'cls_loss': [],
        'mAP_50_95': [],
        'mAP_50': [],
        'lr': []
    }
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found: {log_file_path}")
        return metrics
    
    current_epoch = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract epoch number from lines like "epoch: 1/50"
            epoch_match = re.search(r'epoch:\s*(\d+)/(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Extract loss values from lines like "total_loss: 2.5, iou_loss: 1.2, ..."
            if 'total_loss' in line.lower() or 'iou_loss' in line.lower():
                # Try to extract all loss components
                loss_patterns = {
                    'total_loss': r'total_loss:\s*([\d.]+)',
                    'iou_loss': r'iou_loss:\s*([\d.]+)',
                    'l1_loss': r'l1_loss:\s*([\d.]+)',
                    'conf_loss': r'conf_loss:\s*([\d.]+)',
                    'cls_loss': r'cls_loss:\s*([\d.]+)',
                }
                
                for key, pattern in loss_patterns.items():
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and current_epoch is not None:
                        value = float(match.group(1))
                        # Only record once per epoch (use the last value in the epoch)
                        if len(metrics['epochs']) == 0 or metrics['epochs'][-1] != current_epoch:
                            metrics['epochs'].append(current_epoch)
                            for k in ['total_loss', 'iou_loss', 'l1_loss', 'conf_loss', 'cls_loss']:
                                if k not in metrics or len(metrics[k]) < len(metrics['epochs']):
                                    metrics[k].append(None)
                        
                        idx = metrics['epochs'].index(current_epoch)
                        metrics[key][idx] = value
            
            # Extract learning rate from lines like "lr: 1.234e-05"
            lr_match = re.search(r'lr:\s*([\d.eE+-]+)', line)
            if lr_match and current_epoch is not None:
                lr_value = float(lr_match.group(1))
                if len(metrics['epochs']) == 0 or metrics['epochs'][-1] != current_epoch:
                    metrics['epochs'].append(current_epoch)
                    metrics['lr'].append(lr_value)
                else:
                    idx = metrics['epochs'].index(current_epoch)
                    if idx < len(metrics['lr']):
                        metrics['lr'][idx] = lr_value
                    else:
                        metrics['lr'].append(lr_value)
            
            # Extract mAP values from evaluation lines
            # Look for patterns like "COCOAP50: 0.45" or "AP50: 0.45" or "mAP@0.5: 0.45"
            map_50_match = re.search(r'(?:COCOAP50|AP50|mAP@0\.5):\s*([\d.]+)', line, re.IGNORECASE)
            map_50_95_match = re.search(r'(?:COCOAP50_95|AP50:95|mAP@0\.5:0\.95):\s*([\d.]+)', line, re.IGNORECASE)
            
            if map_50_match or map_50_95_match:
                if current_epoch is not None:
                    # Ensure epoch is in the list
                    if len(metrics['epochs']) == 0 or metrics['epochs'][-1] != current_epoch:
                        metrics['epochs'].append(current_epoch)
                    
                    idx = metrics['epochs'].index(current_epoch)
                    
                    # Pad lists if needed
                    while len(metrics['mAP_50']) < len(metrics['epochs']):
                        metrics['mAP_50'].append(None)
                    while len(metrics['mAP_50_95']) < len(metrics['epochs']):
                        metrics['mAP_50_95'].append(None)
                    
                    if map_50_match:
                        metrics['mAP_50'][idx] = float(map_50_match.group(1))
                    if map_50_95_match:
                        metrics['mAP_50_95'][idx] = float(map_50_95_match.group(1))
            
            # Alternative: Look for summary lines with AP values
            # Format: "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.XXX"
            ap_summary_match = re.search(r'Average Precision.*IoU=0\.50:0\.95.*=\s*([\d.]+)', line)
            if ap_summary_match and current_epoch is not None:
                if len(metrics['epochs']) == 0 or metrics['epochs'][-1] != current_epoch:
                    metrics['epochs'].append(current_epoch)
                
                idx = metrics['epochs'].index(current_epoch)
                while len(metrics['mAP_50_95']) < len(metrics['epochs']):
                    metrics['mAP_50_95'].append(None)
                metrics['mAP_50_95'][idx] = float(ap_summary_match.group(1))
            
            ap_50_summary_match = re.search(r'Average Precision.*IoU=0\.50.*=\s*([\d.]+)', line)
            if ap_50_summary_match and current_epoch is not None:
                if len(metrics['epochs']) == 0 or metrics['epochs'][-1] != current_epoch:
                    metrics['epochs'].append(current_epoch)
                
                idx = metrics['epochs'].index(current_epoch)
                while len(metrics['mAP_50']) < len(metrics['epochs']):
                    metrics['mAP_50'].append(None)
                metrics['mAP_50'][idx] = float(ap_50_summary_match.group(1))
    
    return metrics


def plot_training_results(metrics, output_dir):
    """
    Generate Loss vs Epoch and mAP vs Epoch plots.
    
    Args:
        metrics: Dictionary containing parsed metrics
        output_dir: Directory to save the plots
    """
    if not metrics['epochs']:
        print("Warning: No epoch data found in log file.")
        return
    
    epochs = metrics['epochs']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Results from train_log.txt', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss vs Epoch
    ax1 = axes[0]
    
    # Filter out None values for plotting
    if metrics['total_loss']:
        valid_loss = [(e, v) for e, v in zip(epochs, metrics['total_loss']) if v is not None]
        if valid_loss:
            ep, loss = zip(*valid_loss)
            ax1.plot(ep, loss, 'b-', label='Total Loss', linewidth=2, marker='o', markersize=4)
    
    if metrics['iou_loss']:
        valid_loss = [(e, v) for e, v in zip(epochs, metrics['iou_loss']) if v is not None]
        if valid_loss:
            ep, loss = zip(*valid_loss)
            ax1.plot(ep, loss, 'r--', label='IoU Loss', linewidth=1.5, marker='s', markersize=3)
    
    if metrics['l1_loss']:
        valid_loss = [(e, v) for e, v in zip(epochs, metrics['l1_loss']) if v is not None]
        if valid_loss:
            ep, loss = zip(*valid_loss)
            ax1.plot(ep, loss, 'g--', label='L1 Loss', linewidth=1.5, marker='^', markersize=3)
    
    if metrics['conf_loss']:
        valid_loss = [(e, v) for e, v in zip(epochs, metrics['conf_loss']) if v is not None]
        if valid_loss:
            ep, loss = zip(*valid_loss)
            ax1.plot(ep, loss, 'm--', label='Conf Loss', linewidth=1.5, marker='d', markersize=3)
    
    if metrics['cls_loss']:
        valid_loss = [(e, v) for e, v in zip(epochs, metrics['cls_loss']) if v is not None]
        if valid_loss:
            ep, loss = zip(*valid_loss)
            ax1.plot(ep, loss, 'c--', label='Cls Loss', linewidth=1.5, marker='v', markersize=3)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: mAP vs Epoch
    ax2 = axes[1]
    
    if metrics['mAP_50_95']:
        valid_map = [(e, v) for e, v in zip(epochs, metrics['mAP_50_95']) if v is not None]
        if valid_map:
            ep, map_vals = zip(*valid_map)
            ax2.plot(ep, map_vals, 'b-o', label='mAP@0.5:0.95', linewidth=2, markersize=6)
    
    if metrics['mAP_50']:
        valid_map = [(e, v) for e, v in zip(epochs, metrics['mAP_50']) if v is not None]
        if valid_map:
            ep, map_vals = zip(*valid_map)
            ax2.plot(ep, map_vals, 'r-s', label='mAP@0.5', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.set_title('mAP vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_results_from_log.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")
    
    # Print summary
    print("\n=== Parsed Metrics Summary ===")
    print(f"Total epochs found: {len(epochs)}")
    if metrics['total_loss']:
        valid_losses = [v for v in metrics['total_loss'] if v is not None]
        if valid_losses:
            print(f"Total Loss - Min: {min(valid_losses):.4f}, Max: {max(valid_losses):.4f}, Last: {valid_losses[-1]:.4f}")
    if metrics['mAP_50_95']:
        valid_maps = [v for v in metrics['mAP_50_95'] if v is not None]
        if valid_maps:
            print(f"mAP@0.5:0.95 - Min: {min(valid_maps):.4f}, Max: {max(valid_maps):.4f}, Last: {valid_maps[-1]:.4f}")
    if metrics['mAP_50']:
        valid_maps = [v for v in metrics['mAP_50'] if v is not None]
        if valid_maps:
            print(f"mAP@0.5 - Min: {min(valid_maps):.4f}, Max: {max(valid_maps):.4f}, Last: {valid_maps[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Parse train_log.txt and generate training plots')
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to train_log.txt file. If not specified, will search in YOLOX_outputs directories.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output plots. Default: same directory as log file.'
    )
    
    args = parser.parse_args()
    
    # Find log file if not specified
    if args.log_file is None:
        # Search in common output directories
        search_dirs = [
            './YOLOX_outputs',
            '../YOLOX_outputs',
            './YOLOX/YOLOX_outputs',
        ]
        
        log_file = None
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    if 'train_log.txt' in files:
                        log_file = os.path.join(root, 'train_log.txt')
                        break
                if log_file:
                    break
        
        if log_file is None:
            print("Error: train_log.txt not found. Please specify --log_file path.")
            return
        
        args.log_file = log_file
        print(f"Found log file: {args.log_file}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.log_file)
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    metrics = parse_train_log(args.log_file)
    
    # Generate plots
    plot_training_results(metrics, args.output_dir)


if __name__ == '__main__':
    main()

