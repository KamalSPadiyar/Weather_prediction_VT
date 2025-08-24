#!/usr/bin/env python3
"""
Quick start script for IOCL PPE Detection System
Sets up the environment and runs basic tests
"""

import sys
import subprocess
import logging
from pathlib import Path
import yaml


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_python_version():
    """Check Python version compatibility"""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required dependencies"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data/datasets",
        "data/models", 
        "data/samples/videos",
        "data/samples/images",
        "data/samples/outputs",
        "logs",
        "src/dashboard/templates",
        "src/dashboard/static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_base_model():
    """Download base YOLO model"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Downloading base YOLO model...")
        from ultralytics import YOLO
        
        # Download YOLOv8 nano model
        model = YOLO('yolov8n.pt')
        logger.info("Base model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download base model: {e}")
        return False


def test_imports():
    """Test if all required modules can be imported"""
    logger = logging.getLogger(__name__)
    
    required_modules = [
        'cv2', 'torch', 'ultralytics', 'flask', 'numpy', 'pandas'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError:
            failed_imports.append(module)
            logger.error(f"✗ {module}")
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    
    logger.info("All required modules imported successfully")
    return True


def test_camera():
    """Test camera access"""
    logger = logging.getLogger(__name__)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.warning("No camera detected or camera access denied")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            logger.info("✓ Camera test successful")
            return True
        else:
            logger.warning("Camera detected but failed to read frame")
            return False
            
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
        return False


def create_sample_config():
    """Create sample configuration if not exists"""
    logger = logging.getLogger(__name__)
    
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        logger.info("Configuration file already exists")
        return True
    
    try:
        # Load and verify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration file validated")
        return True
        
    except Exception as e:
        logger.error(f"Error with configuration: {e}")
        return False


def run_basic_test():
    """Run basic system test"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test configuration loading
        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Test detector initialization
        sys.path.append('src')
        from detection.ppe_detector import PPEDetector
        
        detector = PPEDetector(config)
        logger.info("✓ PPE Detector initialized successfully")
        
        # Test with dummy image
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect_ppe(dummy_image)
        
        logger.info("✓ Basic detection test successful")
        logger.info(f"Detection result: {result['compliance']['message']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic test failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("IOCL PPE Detection System - Quick Start Complete!")
    print("="*60)
    print("\n📋 USAGE INSTRUCTIONS:")
    print("\n1. Real-time camera detection:")
    print("   python main.py --source camera --display")
    print("\n2. Process video file:")
    print("   python main.py --source video --input path/to/video.mp4 --output output.mp4")
    print("\n3. Start web dashboard:")
    print("   python main.py --source dashboard")
    print("   Then open: http://localhost:5000")
    print("\n4. Train custom model:")
    print("   python src/models/train_ppe_model.py --data data/datasets/your_dataset")
    print("\n📁 IMPORTANT FILES:")
    print("   • config/settings.yaml - Configuration settings")
    print("   • data/models/ - Trained model files") 
    print("   • data/datasets/ - Training datasets")
    print("   • logs/ - System logs")
    print("\n⚙️  CONFIGURATION:")
    print("   Edit config/settings.yaml to:")
    print("   • Adjust detection thresholds")
    print("   • Configure camera sources")
    print("   • Set up email alerts")
    print("   • Customize PPE requirements")
    print("\n🔍 NEXT STEPS:")
    print("   1. Add your CCTV camera URLs to config/settings.yaml")
    print("   2. Collect training data for your specific environment") 
    print("   3. Train a custom PPE detection model")
    print("   4. Set up email/webhook alerts")
    print("   5. Deploy the web dashboard for monitoring")
    print("\n" + "="*60)


def main():
    """Main setup function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 IOCL PPE Detection System - Quick Start")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_success = install_dependencies()
    if not install_success:
        logger.error("Setup failed during dependency installation")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        logger.error("Setup failed during import test")
        logger.info("Try running: pip install -r requirements.txt")
        sys.exit(1)
    
    # Download base model
    if not download_base_model():
        logger.warning("Failed to download base model - you can download it manually later")
    
    # Test camera (optional)
    test_camera()
    
    # Verify configuration
    if not create_sample_config():
        sys.exit(1)
    
    # Run basic test
    if not run_basic_test():
        logger.error("Basic system test failed")
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()
    
    logger.info("🎉 Setup completed successfully!")


if __name__ == "__main__":
    main()
