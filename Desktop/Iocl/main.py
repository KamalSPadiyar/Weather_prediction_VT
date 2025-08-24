#!/usr/bin/env python3
"""
IOCL PPE Detection System - Main Entry Point
Detects Personal Protective Equipment compliance from video feeds
"""

import argparse
import cv2
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detection.ppe_detector import PPEDetector
from video.video_processor import VideoProcessor
from alerts.alert_manager import AlertManager
from dashboard.app import create_app


def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file_path']),
            logging.StreamHandler()
        ]
    )


def load_config(config_path="config/settings.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='IOCL PPE Detection System')
    parser.add_argument('--source', choices=['camera', 'video', 'rtsp', 'dashboard'], 
                       default='camera', help='Input source type')
    parser.add_argument('--input', type=str, help='Input file path or camera index')
    parser.add_argument('--output', type=str, help='Output file path for processed video')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Configuration file path')
    parser.add_argument('--device', type=int, default=0, help='Camera device index')
    parser.add_argument('--display', action='store_true', help='Display detection results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting IOCL PPE Detection System")
    
    if args.source == 'dashboard':
        # Start web dashboard
        logger.info("Starting web dashboard")
        app = create_app(config)
        app.run(
            host=config['dashboard']['host'],
            port=config['dashboard']['port'],
            debug=config['dashboard']['debug']
        )
        return
    
    # Initialize components
    try:
        detector = PPEDetector(config)
        alert_manager = AlertManager(config)
        video_processor = VideoProcessor(config, detector, alert_manager)
        
        # Determine input source
        if args.source == 'camera':
            source = args.device if args.input is None else int(args.input)
        elif args.source == 'video':
            if args.input is None:
                logger.error("Video file path required for video source")
                sys.exit(1)
            source = args.input
        elif args.source == 'rtsp':
            if args.input is None:
                logger.error("RTSP URL required for RTSP source")
                sys.exit(1)
            source = args.input
        else:
            logger.error(f"Unsupported source type: {args.source}")
            sys.exit(1)
        
        # Process video
        logger.info(f"Processing {args.source} source: {source}")
        video_processor.process_video(
            source=source,
            output_path=args.output,
            display=args.display
        )
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)
    finally:
        logger.info("PPE Detection System stopped")


if __name__ == "__main__":
    main()
