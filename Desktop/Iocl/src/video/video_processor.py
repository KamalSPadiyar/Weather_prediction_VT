"""
Video Processing Module
Handles video input/output and frame processing for PPE detection
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Union, Dict, Any
from pathlib import Path
import threading
import queue


class VideoProcessor:
    """Video processing for PPE detection"""
    
    def __init__(self, config: Dict, detector, alert_manager):
        """Initialize video processor"""
        self.config = config
        self.detector = detector
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        
        # Video settings
        self.frame_skip = config['video']['frame_skip']
        self.resize_width = config['video']['resize_width']
        self.resize_height = config['video']['resize_height']
        self.buffer_size = config['video']['buffer_size']
        
        # Processing state
        self.is_processing = False
        self.frame_count = 0
        self.detection_results = []
        
        self.logger.info("Video Processor initialized")
    
    def process_video(self, source: Union[str, int], output_path: Optional[str] = None, 
                     display: bool = False) -> bool:
        """
        Process video from source and detect PPE
        
        Args:
            source: Video source (file path, camera index, or RTSP URL)
            output_path: Optional output video path
            display: Whether to display processed frames
            
        Returns:
            Success status
        """
        try:
            # Open video source
            cap = self._open_video_source(source)
            if cap is None:
                return False
            
            # Setup video writer if output path provided
            video_writer = None
            if output_path:
                video_writer = self._setup_video_writer(cap, output_path)
            
            self.is_processing = True
            self.frame_count = 0
            fps_counter = FPSCounter()
            
            self.logger.info("Starting video processing...")
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and Path(source).exists():
                        # End of video file
                        break
                    else:
                        # Camera/stream issue, try to reconnect
                        self.logger.warning("Failed to read frame, attempting reconnection...")
                        cap.release()
                        time.sleep(1)
                        cap = self._open_video_source(source)
                        if cap is None:
                            break
                        continue
                
                self.frame_count += 1
                
                # Skip frames if configured
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Resize frame
                frame = self._resize_frame(frame)
                
                # Detect PPE
                detection_result = self.detector.detect_ppe(frame)
                
                # Process alerts
                self.alert_manager.process_detection(detection_result, self.frame_count)
                
                # Draw detections
                processed_frame = self.detector.draw_detections(
                    frame, detection_result['detections'], detection_result['compliance']
                )
                
                # Add FPS counter
                fps = fps_counter.update()
                self._draw_fps(processed_frame, fps)
                
                # Display frame
                if display:
                    cv2.imshow('PPE Detection - IOCL', processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                
                # Write to output video
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Store results for analysis
                self._store_detection_result(detection_result)
            
            self.logger.info(f"Video processing completed. Processed {self.frame_count} frames")
            
        except KeyboardInterrupt:
            self.logger.info("Video processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during video processing: {e}")
            return False
        finally:
            # Cleanup
            self.is_processing = False
            if 'cap' in locals():
                cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        return True
    
    def _open_video_source(self, source: Union[str, int]) -> Optional[cv2.VideoCapture]:
        """Open video source (camera, file, or stream)"""
        try:
            self.logger.info(f"Opening video source: {source}")
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open video source: {source}")
                return None
            
            # Set camera properties if it's a camera
            if isinstance(source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test read
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read test frame from source")
                cap.release()
                return None
            
            self.logger.info(f"Video source opened successfully. Frame size: {frame.shape}")
            return cap
            
        except Exception as e:
            self.logger.error(f"Error opening video source: {e}")
            return None
    
    def _setup_video_writer(self, cap: cv2.VideoCapture, output_path: str) -> Optional[cv2.VideoWriter]:
        """Setup video writer for output"""
        try:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = self.resize_width
            height = self.resize_height
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                self.logger.error(f"Failed to create video writer: {output_path}")
                return None
            
            self.logger.info(f"Video writer created: {output_path}")
            return video_writer
            
        except Exception as e:
            self.logger.error(f"Error setting up video writer: {e}")
            return None
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to configured dimensions"""
        if frame.shape[1] != self.resize_width or frame.shape[0] != self.resize_height:
            return cv2.resize(frame, (self.resize_width, self.resize_height))
        return frame
    
    def _draw_fps(self, frame: np.ndarray, fps: float):
        """Draw FPS counter on frame"""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _store_detection_result(self, result: Dict):
        """Store detection result for analysis"""
        # Keep only recent results to manage memory
        if len(self.detection_results) >= self.buffer_size:
            self.detection_results.pop(0)
        
        result['timestamp'] = time.time()
        result['frame_number'] = self.frame_count
        self.detection_results.append(result)
    
    def get_recent_detections(self, count: int = 10) -> list:
        """Get recent detection results"""
        return self.detection_results[-count:]
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        self.logger.info("Video processing stop requested")


class FPSCounter:
    """Simple FPS counter"""
    
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        current_time = time.time()
        
        if len(self.frame_times) >= self.buffer_size:
            self.frame_times.pop(0)
        
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return 0.0


class StreamManager:
    """Manage multiple video streams"""
    
    def __init__(self, config: Dict, detector, alert_manager):
        self.config = config
        self.detector = detector
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        
        self.streams = {}
        self.stream_threads = {}
        self.is_running = False
    
    def add_stream(self, stream_id: str, source: Union[str, int]) -> bool:
        """Add a new video stream"""
        try:
            processor = VideoProcessor(self.config, self.detector, self.alert_manager)
            self.streams[stream_id] = {
                'processor': processor,
                'source': source,
                'status': 'inactive'
            }
            self.logger.info(f"Stream added: {stream_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding stream {stream_id}: {e}")
            return False
    
    def start_stream(self, stream_id: str) -> bool:
        """Start processing a specific stream"""
        if stream_id not in self.streams:
            self.logger.error(f"Stream not found: {stream_id}")
            return False
        
        try:
            stream_info = self.streams[stream_id]
            processor = stream_info['processor']
            source = stream_info['source']
            
            # Start processing in separate thread
            thread = threading.Thread(
                target=processor.process_video,
                args=(source,),
                daemon=True
            )
            thread.start()
            
            self.stream_threads[stream_id] = thread
            stream_info['status'] = 'active'
            
            self.logger.info(f"Stream started: {stream_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting stream {stream_id}: {e}")
            return False
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop processing a specific stream"""
        if stream_id not in self.streams:
            return False
        
        try:
            self.streams[stream_id]['processor'].stop_processing()
            self.streams[stream_id]['status'] = 'inactive'
            
            if stream_id in self.stream_threads:
                del self.stream_threads[stream_id]
            
            self.logger.info(f"Stream stopped: {stream_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping stream {stream_id}: {e}")
            return False
    
    def get_stream_status(self) -> Dict:
        """Get status of all streams"""
        status = {}
        for stream_id, stream_info in self.streams.items():
            status[stream_id] = {
                'status': stream_info['status'],
                'source': stream_info['source'],
                'recent_detections': stream_info['processor'].get_recent_detections(5)
            }
        return status
