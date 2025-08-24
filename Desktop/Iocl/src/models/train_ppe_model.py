"""
Model Training Module
Script for training custom PPE detection models
"""

import argparse
import yaml
import logging
import torch
from ultralytics import YOLO
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.data_utils import prepare_dataset, validate_annotations


def setup_logging():
    """Setup logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def create_dataset_config(data_path: str, class_names: list) -> str:
    """Create YOLO dataset configuration file"""
    
    dataset_config = {
        'path': str(Path(data_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    config_path = Path(data_path) / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    return str(config_path)


def train_ppe_model(args):
    """Train PPE detection model using YOLOv8"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PPE model training")
    
    # PPE classes
    ppe_classes = [
        'helmet',
        'safety_vest', 
        'gloves',
        'safety_boots',
        'safety_goggles',
        'person'
    ]
    
    try:
        # Prepare dataset
        logger.info("Preparing dataset...")
        dataset_path = Path(args.data)
        
        # Create dataset config
        config_path = create_dataset_config(args.data, ppe_classes)
        logger.info(f"Dataset config created: {config_path}")
        
        # Validate dataset
        if args.validate:
            logger.info("Validating annotations...")
            validate_annotations(dataset_path)
        
        # Initialize model
        logger.info(f"Loading base model: {args.model}")
        model = YOLO(args.model)
        
        # Training parameters
        train_params = {
            'data': config_path,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'name': args.name,
            'save_period': args.save_period,
            'device': args.device,
            'workers': args.workers,
            'project': args.project,
            'patience': args.patience,
            'save': True,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': args.resume,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'augment': False
        }
        
        # Log training parameters
        logger.info("Training parameters:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        # Start training
        logger.info("Starting training...")
        results = model.train(**train_params)
        
        # Save final model
        output_path = Path(args.project) / args.name / 'weights' / 'best.pt'
        final_model_path = f"data/models/ppe_detection_{args.name}.pt"
        
        # Copy best model to data/models directory
        import shutil
        Path("data/models").mkdir(parents=True, exist_ok=True)
        shutil.copy(output_path, final_model_path)
        
        logger.info(f"Training completed! Best model saved to: {final_model_path}")
        logger.info(f"Training results: {results}")
        
        # Validate final model
        if args.validate_final:
            logger.info("Validating final model...")
            model = YOLO(final_model_path)
            val_results = model.val(data=config_path)
            logger.info(f"Validation results: {val_results}")
        
        return final_model_path
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train PPE Detection Model')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Base model to use (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes')
    
    # Output arguments
    parser.add_argument('--name', type=str, default='ppe_training',
                       help='Training run name')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    
    # Training arguments
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save checkpoint every n epochs')
    parser.add_argument('--resume', type=str, default=False,
                       help='Resume training from checkpoint')
    
    # Validation arguments
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset before training')
    parser.add_argument('--validate-final', action='store_true',
                       help='Validate final model after training')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    # Start training
    try:
        model_path = train_ppe_model(args)
        print(f"\n✅ Training completed successfully!")
        print(f"📍 Model saved to: {model_path}")
        print(f"🔧 Update config/settings.yaml with the new model path")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
