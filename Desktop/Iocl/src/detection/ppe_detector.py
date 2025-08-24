"""
PPE Detector Module
Core detection logic using YOLO for PPE classification
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class PPEDetector:
    """PPE Detection using YOLO model"""
    
    def __init__(self, config: Dict):
        """Initialize PPE detector with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PPE class mapping
        self.ppe_classes = {
            0: 'helmet',
            1: 'safety_vest', 
            2: 'gloves',
            3: 'safety_boots',
            4: 'safety_goggles',
            5: 'person'  # Person detection for context
        }
        
        # Load model
        self.model = self._load_model()
        
        # Detection parameters
        self.confidence_threshold = config['model']['confidence_threshold']
        self.iou_threshold = config['model']['iou_threshold']
        self.input_size = tuple(config['model']['input_size'])
        
        self.logger.info("PPE Detector initialized successfully")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        try:
            model_path = self.config['model']['weights_path']
            
            # Check if custom trained model exists
            if Path(model_path).exists():
                self.logger.info(f"Loading custom PPE model: {model_path}")
                model = YOLO(model_path)
            else:
                # Use pre-trained YOLO model and fine-tune for PPE
                self.logger.info("Loading pre-trained YOLO model")
                model = YOLO(self.config['model']['name'] + '.pt')
                
                # Note: In production, you would load a custom-trained PPE model
                self.logger.warning("Using general object detection model. "
                                  "Train custom PPE model for better accuracy.")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def detect_ppe(self, frame: np.ndarray) -> Dict:
        """
        Detect PPE in a single frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Resize frame if needed
            original_shape = frame.shape[:2]
            if frame.shape[:2] != self.input_size:
                frame_resized = cv2.resize(frame, self.input_size)
            else:
                frame_resized = frame.copy()
            
            # Run inference
            results = self.model(frame_resized, conf=self.confidence_threshold, 
                               iou=self.iou_threshold, verbose=False)
            
            # Process results
            detections = self._process_results(results[0], original_shape)
            
            # Analyze PPE compliance
            compliance_status = self._analyze_compliance(detections)
            
            return {
                'detections': detections,
                'compliance': compliance_status,
                'frame_shape': original_shape,
                'detection_count': len(detections)
            }
            
        except Exception as e:
            self.logger.error(f"Error during PPE detection: {e}")
            return {
                'detections': [],
                'compliance': {'compliant': False, 'missing_ppe': [], 'message': 'Detection error'},
                'frame_shape': frame.shape[:2],
                'detection_count': 0
            }
    
    def _process_results(self, results, original_shape: Tuple[int, int]) -> List[Dict]:
        """Process YOLO detection results"""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Scale boxes back to original image size
            scale_x = original_shape[1] / self.input_size[0]
            scale_y = original_shape[0] / self.input_size[1]
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                # Scale coordinates
                x1, y1, x2, y2 = box
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)  
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Map class ID to PPE type (this would be different for custom PPE model)
                ppe_type = self._map_class_to_ppe(cls_id)
                
                if ppe_type:
                    detections.append({
                        'id': i,
                        'ppe_type': ppe_type,
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                    })
        
        return detections
    
    def _map_class_to_ppe(self, class_id: int) -> Optional[str]:
        """Map YOLO class ID to PPE type"""
        # Note: This mapping would be different for a custom-trained PPE model
        # For now, using general object detection classes as placeholders
        
        # General YOLO classes that might be relevant:
        yolo_to_ppe = {
            0: 'person',  # Person detection
            # For custom PPE model, you would have:
            # 0: 'helmet',
            # 1: 'safety_vest',
            # 2: 'gloves',
            # etc.
        }
        
        return yolo_to_ppe.get(class_id)
    
    def _analyze_compliance(self, detections: List[Dict]) -> Dict:
        """Analyze PPE compliance based on detections"""
        required_ppe = self.config['compliance']['required_ppe']
        detected_ppe = set([det['ppe_type'] for det in detections if det['ppe_type'] != 'person'])
        
        # Check if person is detected
        person_detected = any(det['ppe_type'] == 'person' for det in detections)
        
        if not person_detected:
            return {
                'compliant': True,
                'missing_ppe': [],
                'message': 'No person detected in frame'
            }
        
        # Check required PPE
        missing_ppe = []
        for ppe_type, required in required_ppe.items():
            if required and ppe_type not in detected_ppe:
                missing_ppe.append(ppe_type)
        
        compliant = len(missing_ppe) == 0
        
        return {
            'compliant': compliant,
            'detected_ppe': list(detected_ppe),
            'missing_ppe': missing_ppe,
            'message': 'PPE compliant' if compliant else f'Missing: {", ".join(missing_ppe)}'
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       compliance: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        frame_copy = frame.copy()
        
        # Color coding
        colors = {
            'helmet': (0, 255, 0),      # Green
            'safety_vest': (255, 165, 0), # Orange
            'gloves': (255, 0, 255),     # Magenta
            'safety_boots': (0, 255, 255), # Cyan
            'safety_goggles': (255, 255, 0), # Yellow
            'person': (0, 0, 255)        # Red
        }
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            ppe_type = detection['ppe_type']
            confidence = detection['confidence']
            
            color = colors.get(ppe_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{ppe_type}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw compliance status
        status_color = (0, 255, 0) if compliance['compliant'] else (0, 0, 255)
        status_text = f"Status: {compliance['message']}"
        
        cv2.rectangle(frame_copy, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.putText(frame_copy, status_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame_copy
