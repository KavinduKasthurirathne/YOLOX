#!/usr/bin/env python3
"""
Script to clean and re-index COCO JSON annotation files.
Consolidates 3 classes into 2: 'hand' (Class 0) and 'person' (Class 1).

Mapping:
- category_id 2 (hand) → category_id 0 (hand)
- category_id 3 (person) → category_id 1 (person)
- category_id 1 (hand-person) → category_id 1 (person)
"""

import json
import os
from pathlib import Path


def clean_coco_annotations(json_file_path):
    """
    Clean and re-index a COCO JSON annotation file.
    
    Args:
        json_file_path: Path to the COCO JSON file to process
    """
    print(f"Processing: {json_file_path}")
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track statistics
    stats = {
        'original': {1: 0, 2: 0, 3: 0},
        'updated': {0: 0, 1: 0}
    }
    
    # Update annotations
    if 'annotations' in data:
        for annotation in data['annotations']:
            old_category_id = annotation['category_id']
            
            # Count original categories
            if old_category_id in stats['original']:
                stats['original'][old_category_id] += 1
            
            # Map category IDs
            if old_category_id == 2:  # hand → 0
                annotation['category_id'] = 0
                stats['updated'][0] += 1
            elif old_category_id == 3:  # person → 1
                annotation['category_id'] = 1
                stats['updated'][1] += 1
            elif old_category_id == 1:  # hand-person → 1 (person)
                annotation['category_id'] = 1
                stats['updated'][1] += 1
            else:
                print(f"Warning: Found unexpected category_id {old_category_id} in annotation {annotation.get('id', 'unknown')}")
    
    # Update categories list
    data['categories'] = [
        {"id": 0, "name": "hand", "supercategory": "none"},
        {"id": 1, "name": "person", "supercategory": "none"}
    ]
    
    # Write back to file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # Print statistics
    print(f"  Original categories: {stats['original']}")
    print(f"  Updated categories: {stats['updated']}")
    print(f"  Total annotations processed: {sum(stats['updated'].values())}")
    print(f"  ✓ Successfully updated: {json_file_path}\n")


def main():
    """Main function to process all three COCO JSON files."""
    # Get the script directory and construct paths
    script_dir = Path(__file__).parent
    annotations_dir = script_dir / "datasets" / "COCO" / "annotations"
    
    # Files to process
    files_to_process = [
        "instances_train.json",
        "instances_val.json",
        "instances_test.json"
    ]
    
    print("=" * 60)
    print("COCO Annotation Cleaner")
    print("=" * 60)
    print(f"Annotations directory: {annotations_dir}\n")
    
    # Process each file
    for filename in files_to_process:
        file_path = annotations_dir / filename
        
        if not file_path.exists():
            print(f"⚠ Warning: File not found: {file_path}")
            print(f"  Skipping...\n")
            continue
        
        try:
            clean_coco_annotations(file_path)
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}\n")
    
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

