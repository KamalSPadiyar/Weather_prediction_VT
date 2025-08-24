# Data Directory Structure

This directory contains datasets, trained models, and sample data for the PPE detection system.

## Subdirectories:

### datasets/
Training and validation datasets for PPE detection:
- `images/` - Training images with PPE annotations
- `annotations/` - YOLO format annotation files
- `splits/` - Train/validation/test split information

### models/
Trained model files:
- `ppe_detection.pt` - Main PPE detection model (YOLOv8)
- `backup/` - Model backup versions
- `configs/` - Model configuration files

### samples/
Sample videos and images for testing:
- `videos/` - Sample CCTV footage
- `images/` - Sample images for testing
- `outputs/` - Processed output videos

## Setup Instructions:

1. **Download Pre-trained Models:**
   ```bash
   # Download YOLOv8 base model
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

2. **Prepare Training Data:**
   - Collect CCTV footage with workers wearing/not wearing PPE
   - Annotate images using tools like LabelImg or Roboflow
   - Organize annotations in YOLO format

3. **Model Training:**
   ```bash
   # Train custom PPE detection model
   python src/models/train_ppe_model.py --data datasets/ppe_dataset.yaml --epochs 100
   ```

## Data Sources:

- Internal IOCL CCTV footage (with proper privacy compliance)
- Open-source construction safety datasets
- Synthetic data generation for edge cases

## File Formats:

- **Images:** JPG, PNG (recommended: 640x640 for YOLO)
- **Videos:** MP4, AVI
- **Annotations:** YOLO format (.txt files)
- **Models:** PyTorch (.pt files)

## Storage Requirements:

- Estimated storage: 10-50 GB depending on dataset size
- SSD recommended for faster training/inference
- Regular backups of trained models recommended
