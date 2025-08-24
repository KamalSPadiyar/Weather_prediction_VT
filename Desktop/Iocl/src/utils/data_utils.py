"""
Data utilities for PPE detection system
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import yaml
import random
import shutil


def validate_annotations(dataset_path: Path) -> bool:
    """Validate YOLO format annotations"""
    logger = logging.getLogger(__name__)
    
    try:
        annotations_dir = dataset_path / "labels"
        images_dir = dataset_path / "images"
        
        if not annotations_dir.exists():
            logger.error(f"Annotations directory not found: {annotations_dir}")
            return False
        
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return False
        
        # Check each annotation file
        annotation_files = list(annotations_dir.glob("*.txt"))
        logger.info(f"Found {len(annotation_files)} annotation files")
        
        for ann_file in annotation_files:
            image_file = images_dir / f"{ann_file.stem}.jpg"
            if not image_file.exists():
                image_file = images_dir / f"{ann_file.stem}.png"
            
            if not image_file.exists():
                logger.warning(f"No corresponding image for annotation: {ann_file}")
                continue
            
            # Validate annotation format
            with open(ann_file, 'r') as f:
                lines = f.readlines()
                
            for line_no, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    logger.error(f"Invalid annotation format in {ann_file}:{line_no}")
                    return False
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Check if values are in valid range [0, 1]
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        logger.error(f"Invalid bbox coordinates in {ann_file}:{line_no}")
                        return False
                        
                except ValueError:
                    logger.error(f"Invalid number format in {ann_file}:{line_no}")
                    return False
        
        logger.info("Annotation validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating annotations: {e}")
        return False


def prepare_dataset(source_dir: Path, output_dir: Path, 
                   train_ratio: float = 0.7, val_ratio: float = 0.2) -> bool:
    """Prepare dataset in YOLO format with train/val/test splits"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory structure
        output_dir = Path(output_dir)
        for split in ['train', 'val', 'test']:
            (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(source_dir.glob(f"**/*{ext}")))
            image_files.extend(list(source_dir.glob(f"**/*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} images")
        
        # Filter images that have corresponding annotations
        valid_pairs = []
        for img_file in image_files:
            ann_file = img_file.parent / 'labels' / f"{img_file.stem}.txt"
            if not ann_file.exists():
                ann_file = source_dir / 'labels' / f"{img_file.stem}.txt"
            
            if ann_file.exists():
                valid_pairs.append((img_file, ann_file))
        
        logger.info(f"Found {len(valid_pairs)} valid image-annotation pairs")
        
        if len(valid_pairs) == 0:
            logger.error("No valid image-annotation pairs found")
            return False
        
        # Shuffle and split
        random.shuffle(valid_pairs)
        
        n_total = len(valid_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': valid_pairs[:n_train],
            'val': valid_pairs[n_train:n_train + n_val],
            'test': valid_pairs[n_train + n_val:]
        }
        
        # Copy files to appropriate directories
        for split_name, pairs in splits.items():
            logger.info(f"Creating {split_name} split with {len(pairs)} samples")
            
            for img_file, ann_file in pairs:
                # Copy image
                dst_img = output_dir / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy annotation
                dst_ann = output_dir / 'labels' / split_name / ann_file.name
                shutil.copy2(ann_file, dst_ann)
        
        logger.info("Dataset preparation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return False


def augment_dataset(dataset_path: Path, augmentation_factor: int = 2) -> bool:
    """Apply data augmentation to increase dataset size"""
    logger = logging.getLogger(__name__)
    
    try:
        import albumentations as A
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(p=0.2),
            A.ColorJitter(p=0.2)
        ], bbox_params=A.BboxParams(format='yolo'))
        
        for split in ['train']:  # Usually only augment training data
            images_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for img_file in image_files:
                ann_file = labels_dir / f"{img_file.stem}.txt"
                if not ann_file.exists():
                    continue
                
                # Load image
                image = cv2.imread(str(img_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load annotations
                bboxes = []
                class_labels = []
                
                with open(ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
                
                # Apply augmentation
                for i in range(augmentation_factor):
                    try:
                        augmented = transform(image=image, bboxes=bboxes)
                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        
                        # Save augmented image
                        aug_img_name = f"{img_file.stem}_aug_{i}{img_file.suffix}"
                        aug_img_path = images_dir / aug_img_name
                        
                        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(aug_img_path), aug_image_bgr)
                        
                        # Save augmented annotations
                        aug_ann_path = labels_dir / f"{img_file.stem}_aug_{i}.txt"
                        with open(aug_ann_path, 'w') as f:
                            for bbox, class_id in zip(aug_bboxes, class_labels):
                                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    
                    except Exception as e:
                        logger.warning(f"Failed to augment {img_file}: {e}")
                        continue
        
        logger.info("Dataset augmentation completed")
        return True
        
    except ImportError:
        logger.warning("Albumentations not installed, skipping augmentation")
        return False
    except Exception as e:
        logger.error(f"Error during augmentation: {e}")
        return False


def create_sample_dataset():
    """Create a sample dataset for testing purposes"""
    logger = logging.getLogger(__name__)
    
    try:
        # This would create synthetic or download sample data
        # For now, just create the directory structure
        
        sample_dir = Path("data/datasets/sample_ppe")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (sample_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (sample_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Create dataset config
        config = {
            'path': str(sample_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': 6,
            'names': ['helmet', 'safety_vest', 'gloves', 'safety_boots', 'safety_goggles', 'person']
        }
        
        with open(sample_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Sample dataset structure created at: {sample_dir}")
        logger.info("Add your images and annotations to complete the dataset")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample dataset: {e}")
        return False


def analyze_dataset(dataset_path: Path) -> Dict:
    """Analyze dataset statistics"""
    logger = logging.getLogger(__name__)
    
    try:
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'splits': {}
        }
        
        class_names = ['helmet', 'safety_vest', 'gloves', 'safety_boots', 'safety_goggles', 'person']
        
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if not split_dir.exists():
                continue
            
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            split_stats = {
                'images': len(images),
                'annotations': 0,
                'class_counts': {name: 0 for name in class_names}
            }
            
            for img_file in images:
                ann_file = labels_dir / f"{img_file.stem}.txt"
                if ann_file.exists():
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()
                        split_stats['annotations'] += len(lines)
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    split_stats['class_counts'][class_name] += 1
            
            stats['splits'][split] = split_stats
            stats['total_images'] += split_stats['images']
            stats['total_annotations'] += split_stats['annotations']
            
            for class_name, count in split_stats['class_counts'].items():
                if class_name not in stats['class_distribution']:
                    stats['class_distribution'][class_name] = 0
                stats['class_distribution'][class_name] += count
        
        logger.info("Dataset Analysis:")
        logger.info(f"Total Images: {stats['total_images']}")
        logger.info(f"Total Annotations: {stats['total_annotations']}")
        logger.info("Class Distribution:")
        for class_name, count in stats['class_distribution'].items():
            logger.info(f"  {class_name}: {count}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample dataset
    create_sample_dataset()
    
    # Analyze existing dataset if available
    dataset_path = Path("data/datasets/sample_ppe")
    if dataset_path.exists():
        analyze_dataset(dataset_path)
