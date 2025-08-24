"""
IOCL PPE Detection System
Main source package
"""

__version__ = "1.0.0"
__author__ = "IOCL Development Team"
__description__ = "Personal Protective Equipment Detection System using Computer Vision"

from .detection import PPEDetector
from .video import VideoProcessor, StreamManager
from .alerts import AlertManager
from .dashboard import create_app

__all__ = [
    'PPEDetector',
    'VideoProcessor', 
    'StreamManager',
    'AlertManager',
    'create_app'
]
