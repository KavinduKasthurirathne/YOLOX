#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Script to check and validate COCO annotation files for class ID issues.
This helps diagnose CUDA device-side assert errors.
"""

import json
import os
from collections import Counter

def check_annotations(json_file):
    """Check category IDs in annotation file."""
    print(f"\n{'='*60}")
    print(f"Checking: {json_file}")
    print(f"{'='*60}")
    
    if not os.path.exists(json_file):
        print(f"ERROR: File not found: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get all category IDs
    categories = data.get('categories', [])
    category_ids = [cat['id'] for cat in categories]
    category_names = {cat['id']: cat['name'] for cat in categories}
    
    print(f"\nCategories found ({len(categories)}):")
    for cat_id, cat_name in sorted(category_names.items()):
        print(f"  ID {cat_id}: {cat_name}")
    
    # Check annotations
    annotations = data.get('annotations', [])
    if annotations:
        used_category_ids = [ann['category_id'] for ann in annotations]
        category_counts = Counter(used_category_ids)
        
        print(f"\nCategory IDs used in annotations:")
        for cat_id, count in sorted(category_counts.items()):
            cat_name = category_names.get(cat_id, "UNKNOWN")
            print(f"  ID {cat_id} ({cat_name}): {count} instances")
        
        # Check for issues
        min_id = min(used_category_ids)
        max_id = max(used_category_ids)
        unique_ids = sorted(set(used_category_ids))
        
        print(f"\nCategory ID range: {min_id} to {max_id}")
        print(f"Unique category IDs: {unique_ids}")
        
        # Expected range for 2 classes
        if min_id < 1 or max_id > 2:
            print(f"\n⚠️  WARNING: Category IDs are not in range [1, 2]!")
            print(f"   Expected IDs: 1, 2 (for 2 classes)")
            print(f"   Found IDs: {unique_ids}")
            print(f"   This will cause CUDA device-side assert errors!")
            return False
        elif len(unique_ids) != 2:
            print(f"\n⚠️  WARNING: Found {len(unique_ids)} unique category IDs, expected 2!")
            return False
        else:
            print(f"\n[OK] Category IDs are correct for 2 classes!")
            return True
    else:
        print("\n⚠️  WARNING: No annotations found in file!")
        return False

def main():
    """Main function to check all annotation files."""
    base_dir = "datasets/COCO/annotations"
    
    files = [
        "instances_train.json",
        "instances_val.json",
        "instances_test.json"
    ]
    
    results = {}
    for filename in files:
        json_file = os.path.join(base_dir, filename)
        results[filename] = check_annotations(json_file)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_ok = True
    for filename, result in results.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"{filename}: {status}")
        if not result:
            all_ok = False
    
    if not all_ok:
        print(f"\n{'='*60}")
        print("FIX REQUIRED:")
        print(f"{'='*60}")
        print("Your annotation files have category IDs that don't match")
        print("the expected range for 2 classes (should be 1 and 2).")
        print("\nYou need to either:")
        print("1. Update your annotation files to use category IDs 1 and 2")
        print("2. Or create a custom dataset class that remaps the IDs")
        print("\nTo fix, ensure your categories in the JSON files have:")
        print("  - First category: id=1, name='person'")
        print("  - Second category: id=2, name='hand'")
    else:
        print("\n[OK] All annotation files are correct!")

if __name__ == "__main__":
    main()

