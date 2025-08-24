import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pygame
import matplotlib.pyplot as plt
import requests
from urllib.parse import urlparse
from sklearn.cluster import KMeans
import math

class EnhancedFlareMonitoringSystem:
    def __init__(self, config_file='flare_config.json'):
        """
        Initialize the Enhanced Flare Monitoring System with improved detection
        """
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.alarm_active = False
        self.calibration_factor = self.config['calibration']['pixels_per_meter']
        self.threshold_height = self.config['safety']['max_height_meters']
        
        # Enhanced detection parameters
        self.flare_detection_confidence = 0.0
        self.frame_history = []
        self.motion_threshold = self.config['detection']['motion_threshold']
        
        # Initialize pygame for audio alerts
        pygame.mixer.init()
        
    def load_config(self, config_file):
        """Load configuration from JSON file with enhanced detection parameters"""
        default_config = {
            "safety": {
                "max_height_meters": 10.0,
                "warning_height_meters": 8.0
            },
            "calibration": {
                "pixels_per_meter": 10.0,
                "reference_object_height_meters": 5.0
            },
            "detection": {
                "flame_color_lower": [0, 50, 50],
                "flame_color_upper": [30, 255, 255],
                "min_contour_area": 1000,
                "flare_confidence_threshold": 0.7,
                "motion_threshold": 30,
                "brightness_threshold": 150,
                "aspect_ratio_range": [0.3, 3.0],
                "temporal_consistency_frames": 5,
                "flare_shape_circularity": 0.3,
                "intensity_variation_threshold": 50,
                "background_subtraction_enabled": True
            },
            "alerts": {
                "email_enabled": True,
                "email_smtp": "smtp.gmail.com",
                "email_port": 587,
                "sender_email": "alert@refinery.com",
                "sender_password": "your_password",
                "recipients": ["safety@refinery.com", "operator@refinery.com"],
                "audio_enabled": True,
                "audio_file": "alarm.wav"
            },
            "monitoring": {
                "check_interval_seconds": 2,
                "log_file": "flare_monitor.log"
            },
            "camera": {
                "mobile_ip_cam_url": "",
                "rtsp_url": "",
                "http_stream_url": "",
                "connection_timeout": 10
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['monitoring']['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_flare_like_shape(self, contour):
        """
        Analyze contour to determine if it has flare-like characteristics
        """
        # Calculate basic shape properties
        area = cv2.contourArea(contour)
        if area < self.config['detection']['min_contour_area']:
            return False, 0.0
        
        # Calculate perimeter and circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate solidity (area/convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Flare characteristics scoring
        score = 0.0
        
        # Flares are typically elongated vertically (flame-like)
        if 0.3 <= aspect_ratio <= 1.5:  # Vertical elongation preferred
            score += 0.3
        elif aspect_ratio > 1.5:  # Too wide, less likely to be a flare
            score += 0.1
        
        # Flares have moderate circularity (not too circular, not too irregular)
        if 0.2 <= circularity <= 0.8:
            score += 0.2
        
        # Flares have good solidity (not too hollow)
        if solidity > 0.7:
            score += 0.25
        
        # Size check - flares should be reasonably large
        if area > 2000:
            score += 0.25
        
        return score > 0.5, score
    
    def analyze_brightness_distribution(self, roi, flame_mask):
        """
        Analyze brightness distribution in the flame region
        """
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to get only flame pixels
        flame_pixels = gray_roi[flame_mask > 0]
        
        if len(flame_pixels) == 0:
            return False, 0.0
        
        # Calculate brightness statistics
        mean_brightness = np.mean(flame_pixels)
        std_brightness = np.std(flame_pixels)
        max_brightness = np.max(flame_pixels)
        
        # Flares should have high brightness and variation
        brightness_score = 0.0
        
        # High average brightness
        if mean_brightness > self.config['detection']['brightness_threshold']:
            brightness_score += 0.4
        
        # Good brightness variation (flickering effect)
        if std_brightness > self.config['detection']['intensity_variation_threshold']:
            brightness_score += 0.3
        
        # Very bright pixels present
        if max_brightness > 200:
            brightness_score += 0.3
        
        return brightness_score > 0.6, brightness_score
    
    def detect_motion_in_flame_region(self, current_mask, previous_mask):
        """
        Detect motion/flickering in flame regions
        """
        if previous_mask is None:
            return False, 0.0
        
        # Calculate difference between current and previous masks
        diff = cv2.absdiff(current_mask, previous_mask)
        motion_pixels = np.sum(diff > self.motion_threshold)
        total_pixels = np.sum(current_mask > 0)
        
        if total_pixels == 0:
            return False, 0.0
        
        motion_ratio = motion_pixels / total_pixels
        
        # Flares typically have some motion/flickering
        motion_score = min(motion_ratio * 2, 1.0)  # Normalize to 0-1
        
        return motion_score > 0.3, motion_score
    
    def advanced_flame_detection(self, image, previous_mask=None):
        """
        Advanced flame detection with multiple validation criteria
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for flame colors (orange/red/yellow)
        lower_flame = np.array(self.config['detection']['flame_color_lower'])
        upper_flame = np.array(self.config['detection']['flame_color_upper'])
        
        # Create mask for flame colors
        mask1 = cv2.inRange(hsv, lower_flame, upper_flame)
        
        # Additional mask for red flames
        lower_red = np.array([160, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine masks
        flame_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_CLOSE, kernel)
        flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(flame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [], flame_mask, 0.0, "No flame-colored regions detected"
        
        # Analyze each contour
        valid_flare_contours = []
        confidence_scores = []
        analysis_details = []
        
        for i, contour in enumerate(contours):
            # Check if contour has flare-like shape
            is_flare_shape, shape_score = self.is_flare_like_shape(contour)
            
            if not is_flare_shape:
                analysis_details.append(f"Contour {i}: Failed shape test (score: {shape_score:.2f})")
                continue
            
            # Get ROI for this contour
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            roi_mask = flame_mask[y:y+h, x:x+w]
            
            # Analyze brightness distribution
            is_bright_enough, brightness_score = self.analyze_brightness_distribution(roi, roi_mask)
            
            # Analyze motion (if previous mask available)
            has_motion, motion_score = False, 0.0
            if previous_mask is not None:
                prev_roi_mask = previous_mask[y:y+h, x:x+w] if y+h <= previous_mask.shape[0] and x+w <= previous_mask.shape[1] else None
                if prev_roi_mask is not None:
                    has_motion, motion_score = self.detect_motion_in_flame_region(roi_mask, prev_roi_mask)
            
            # Calculate overall confidence
            total_confidence = (shape_score * 0.4 + brightness_score * 0.4 + motion_score * 0.2)
            
            # Apply threshold
            if total_confidence >= self.config['detection']['flare_confidence_threshold']:
                valid_flare_contours.append(contour)
                confidence_scores.append(total_confidence)
                analysis_details.append(
                    f"Contour {i}: VALID FLARE (confidence: {total_confidence:.2f}, "
                    f"shape: {shape_score:.2f}, brightness: {brightness_score:.2f}, motion: {motion_score:.2f})"
                )
            else:
                analysis_details.append(
                    f"Contour {i}: Failed confidence test (confidence: {total_confidence:.2f}, "
                    f"shape: {shape_score:.2f}, brightness: {brightness_score:.2f}, motion: {motion_score:.2f})"
                )
        
        # Calculate overall detection confidence
        overall_confidence = max(confidence_scores) if confidence_scores else 0.0
        
        # Create detailed analysis report
        analysis_report = f"Flare Detection Analysis:\n"
        analysis_report += f"Total contours found: {len(contours)}\n"
        analysis_report += f"Valid flare contours: {len(valid_flare_contours)}\n"
        analysis_report += f"Overall confidence: {overall_confidence:.2f}\n"
        analysis_report += "Detailed analysis:\n" + "\n".join(analysis_details)
        
        return valid_flare_contours, flame_mask, overall_confidence, analysis_report
    
    def is_actually_a_flare(self, image, flame_contours, confidence_score):
        """
        Final validation to determine if detected objects are actually flares
        """
        if not flame_contours:
            return False, "No flame contours detected"
        
        if confidence_score < self.config['detection']['flare_confidence_threshold']:
            return False, f"Confidence too low: {confidence_score:.2f} < {self.config['detection']['flare_confidence_threshold']}"
        
        # Additional contextual checks
        # Check if the flare is in a reasonable location (typically should be elevated)
        image_height = image.shape[0]
        
        for contour in flame_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Flare should typically be in upper portion of image
            if y + h/2 > image_height * 0.8:  # If center is in bottom 20% of image
                return False, "Flare position too low in image (possible false positive)"
        
        return True, f"Valid flare detected with confidence: {confidence_score:.2f}"
    
    def calculate_flame_height(self, flame_contours):
        """
        Calculate the height of the flame in meters (only for valid flares)
        """
        if not flame_contours:
            return 0.0
        
        # Find the largest contour (main flame)
        largest_contour = max(flame_contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert pixel height to meters using calibration factor
        height_meters = h / self.calibration_factor
        
        return height_meters
    
    def process_image(self, image_path):
        """
        Process a single image with enhanced flare detection
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None
            
            # Advanced flame detection
            flame_contours, flame_mask, confidence, analysis = self.advanced_flame_detection(image)
            
            # Validate if it's actually a flare
            is_flare, validation_msg = self.is_actually_a_flare(image, flame_contours, confidence)
            
            height_meters = 0.0
            if is_flare:
                height_meters = self.calculate_flame_height(flame_contours)
            
            # Create visualization
            result_image = self.create_enhanced_visualization(
                image, flame_contours, height_meters, confidence, is_flare, validation_msg
            )
            
            # Check thresholds only if it's actually a flare
            alarm_triggered = False
            warning = False
            
            if is_flare:
                if height_meters > self.threshold_height:
                    self.trigger_alarm(height_meters, image_path)
                    alarm_triggered = True
                elif height_meters > self.config['safety']['warning_height_meters']:
                    self.logger.warning(f"Flame height warning: {height_meters:.2f}m")
                    warning = True
            
            return {
                'height_meters': height_meters,
                'is_flare': is_flare,
                'confidence': confidence,
                'validation_message': validation_msg,
                'analysis_details': analysis,
                'alarm_triggered': alarm_triggered,
                'warning': warning,
                'result_image': result_image,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def create_enhanced_visualization(self, image, flame_contours, height_meters, confidence, is_flare, validation_msg):
        """
        Create enhanced visualization with detailed flare analysis
        """
        result = image.copy()
        
        # Draw flame contours with different colors based on validity
        if is_flare:
            cv2.drawContours(result, flame_contours, -1, (0, 255, 0), 3)  # Green for valid flares
        else:
            cv2.drawContours(result, flame_contours, -1, (0, 0, 255), 2)  # Red for invalid detections
        
        # Determine status color
        if is_flare:
            if height_meters > self.threshold_height:
                status_color = (0, 0, 255)  # Red for alarm
                status_text = "ALARM"
            elif height_meters > self.config['safety']['warning_height_meters']:
                status_color = (0, 165, 255)  # Orange for warning
                status_text = "WARNING"
            else:
                status_color = (0, 255, 0)  # Green for normal
                status_text = "NORMAL"
        else:
            status_color = (128, 128, 128)  # Gray for no flare
            status_text = "NO FLARE"
        
        # Add flare detection status
        cv2.putText(result, f"Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add confidence score
        cv2.putText(result, f"Confidence: {confidence:.2f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add height measurement (only if it's a valid flare)
        if is_flare:
            cv2.putText(result, f"Height: {height_meters:.1f}m", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Add threshold information
            cv2.putText(result, f"Threshold: {self.threshold_height:.1f}m", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(result, "Not a flare", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        
        # Add validation message (truncated for display)
        validation_display = validation_msg[:50] + "..." if len(validation_msg) > 50 else validation_msg
        cv2.putText(result, validation_display, 
                   (10, result.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, 
                   (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add alarm status
        if self.alarm_active:
            cv2.putText(result, "ALARM ACTIVE!", 
                       (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return result
    
    def live_monitoring(self, camera_source=None):
        """
        Enhanced live monitoring with improved flare detection
        """
        if camera_source is None:
            camera_source = 0  # Default camera
        
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            self.logger.error("Failed to open camera")
            return
        
        self.logger.info(f"Starting enhanced live flare monitoring")
        
        # Initialize background subtractor for motion detection
        if self.config['detection']['background_subtraction_enabled']:
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        previous_mask = None
        frame_count = 0
        flare_detection_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Advanced flame detection
            flame_contours, flame_mask, confidence, analysis = self.advanced_flame_detection(frame, previous_mask)
            
            # Update previous mask
            previous_mask = flame_mask.copy()
            
            # Validate if it's actually a flare
            is_flare, validation_msg = self.is_actually_a_flare(frame, flame_contours, confidence)
            
            # Update detection history for temporal consistency
            flare_detection_history.append(is_flare)
            if len(flare_detection_history) > self.config['detection']['temporal_consistency_frames']:
                flare_detection_history.pop(0)
            
            # Apply temporal consistency check
            recent_detections = sum(flare_detection_history)
            temporal_confidence = recent_detections / len(flare_detection_history)
            
            # Final flare determination
            final_is_flare = is_flare and temporal_confidence >= 0.6
            
            height_meters = 0.0
            if final_is_flare:
                height_meters = self.calculate_flame_height(flame_contours)
            
            # Create visualization
            result_frame = self.create_enhanced_visualization(
                frame, flame_contours, height_meters, confidence, final_is_flare, validation_msg
            )
            
            # Add temporal consistency info
            cv2.putText(result_frame, f"Temporal: {temporal_confidence:.2f}", 
                       (result_frame.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Check thresholds only if it's actually a flare
            if final_is_flare and height_meters > self.threshold_height:
                self.trigger_alarm(height_meters)
            
            # Display result
            cv2.imshow('Enhanced Flare Monitoring', result_frame)
            
            # Log detailed analysis every 30 frames
            if frame_count % 30 == 0:
                self.logger.info(f"Frame {frame_count}: Flare={final_is_flare}, Confidence={confidence:.2f}, Height={height_meters:.2f}m")
                if not final_is_flare and confidence > 0.3:
                    self.logger.info(f"Detection details: {validation_msg}")
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Save screenshot with analysis
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_name = f"flare_analysis_{timestamp_str}.jpg"
                cv2.imwrite(screenshot_name, result_frame)
                
                # Save detailed analysis
                analysis_file = f"analysis_{timestamp_str}.txt"
                with open(analysis_file, 'w') as f:
                    f.write(f"Flare Analysis Report - {datetime.now()}\n")
                    f.write("="*50 + "\n")
                    f.write(f"Is Flare: {final_is_flare}\n")
                    f.write(f"Confidence: {confidence:.2f}\n")
                    f.write(f"Height: {height_meters:.2f}m\n")
                    f.write(f"Validation: {validation_msg}\n")
                    f.write(f"Temporal Confidence: {temporal_confidence:.2f}\n")
                    f.write("\nDetailed Analysis:\n")
                    f.write(analysis)
                
                print(f"Screenshot and analysis saved: {screenshot_name}, {analysis_file}")
            
            time.sleep(self.config['monitoring']['check_interval_seconds'])
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Keep all other methods from the original class...
    def detect_available_cameras(self):
        """Detect all available camera sources"""
        available_cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append({
                        'type': 'local',
                        'index': i,
                        'name': f'Camera {i}',
                        'source': i
                    })
                cap.release()
        
        if self.config['camera']['mobile_ip_cam_url']:
            available_cameras.append({
                'type': 'ip_camera',
                'name': 'Mobile IP Camera',
                'source': self.config['camera']['mobile_ip_cam_url']
            })
        
        return available_cameras
    
    def trigger_alarm(self, height_meters, image_path=None):
        """Trigger alarm when flame height exceeds threshold"""
        if self.alarm_active:
            return
        
        self.alarm_active = True
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alarm_message = f"CRITICAL ALERT: Flare height exceeded threshold!\n"
        alarm_message += f"Current height: {height_meters:.2f} meters\n"
        alarm_message += f"Threshold: {self.threshold_height:.2f} meters\n"
        alarm_message += f"Time: {timestamp}\n"
        
        self.logger.critical(alarm_message)
        
        # Send email alert
        if self.config['alerts']['email_enabled']:
            self.send_email_alert(alarm_message, image_path)
        
        # Play audio alert
        if self.config['alerts']['audio_enabled']:
            self.play_audio_alert()
        
        # Start alarm reset timer
        threading.Timer(30.0, self.reset_alarm).start()
    
    def send_email_alert(self, message, image_path=None):
        """Send email alert to configured recipients"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['alerts']['sender_email']
            msg['To'] = ', '.join(self.config['alerts']['recipients'])
            msg['Subject'] = "CRITICAL FLARE HEIGHT ALERT - Immediate Action Required"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.config['alerts']['email_smtp'], 
                                self.config['alerts']['email_port'])
            server.starttls()
            server.login(self.config['alerts']['sender_email'], 
                        self.config['alerts']['sender_password'])
            
            text = msg.as_string()
            server.sendmail(self.config['alerts']['sender_email'], 
                          self.config['alerts']['recipients'], text)
            server.quit()
            
            self.logger.info("Email alert sent successfully")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def play_audio_alert(self):
        """Play audio alert sound"""
        try:
            if os.path.exists(self.config['alerts']['audio_file']):
                pygame.mixer.music.load(self.config['alerts']['audio_file'])
                pygame.mixer.music.play(-1)
            else:
                for _ in range(5):
                    print('\a')
                    time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Failed to play audio alert: {e}")
    
    def reset_alarm(self):
        """Reset alarm state"""
        self.alarm_active = False
        pygame.mixer.music.stop()
        self.logger.info("Alarm reset")


def main():
    """Main function for the enhanced flare monitoring system"""
    monitor = EnhancedFlareMonitoringSystem()
    
    print("Enhanced Flare Height Monitoring System")
    print("=" * 50)
    print("Features:")
    print("- Advanced flare detection with confidence scoring")
    print("- Shape, brightness, and motion analysis")
    print("- Temporal consistency checking")
    print("- False positive reduction")
    print("=" * 50)
    print("1. Process single image")
    print("2. Start live monitoring")
    print("3. Test camera connection")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = monitor.process_image(image_path)
                if result:
                    print(f"\nAnalysis Results:")
                    print(f"Is Flare: {result['is_flare']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print(f"Validation: {result['validation_message']}")
                    if result['is_flare']:
                        print(f"Flame height: {result['height_meters']:.2f} meters")
                        if result['alarm_triggered']:
                            print("ALARM: Height exceeds threshold!")
                        elif result['warning']:
                            print("WARNING: Height approaching threshold!")
                    
                    # Save result image
                    output_path = f"result_{os.path.basename(image_path)}"
                    cv2.imwrite(output_path, result['result_image'])
                    print(f"Result saved to: {output_path}")
                    
                    # Save detailed analysis
                    analysis_path = f"analysis_{os.path.basename(image_path)}.txt"
                    with open(analysis_path, 'w') as f:
                        f.write(result['analysis_details'])
                    print(f"Detailed analysis saved to: {analysis_path}")
            else:
                print("Image file not found!")
        
        elif choice == '2':
            camera_source = input("Enter camera source (index or URL, or press Enter for default): ").strip()
            if camera_source == "":
                camera_source = 0
            elif camera_source.isdigit():
                camera_source = int(camera_source)
            
            print("Starting live monitoring... Press 'q' to quit, 's' to save screenshot with analysis")
            monitor.live_monitoring(camera_source)
        
        elif choice == '3':
            print("Detecting available cameras...")
            cameras = monitor.detect_available_cameras()
            if cameras:
                print("Available cameras:")
                for cam in cameras:
                    print(f"- {cam['name']} (Type: {cam['type']}, Source: {cam['source']})")
            else:
                print("No cameras detected")
        
        elif choice == '4':
            print("Exiting Enhanced Flare Monitoring System")
            break
        
        else:
            print("Invalid choice! Please select 1-4.")


class FlareAnalysisReporter:
    """
    Additional class for generating detailed reports and analytics
    """
    def __init__(self, monitor_system):
        self.monitor = monitor_system
        self.detection_log = []
    
    def log_detection(self, result):
        """Log detection result for analysis"""
        self.detection_log.append(result)
    
    def generate_daily_report(self, date_str=None):
        """Generate daily flare activity report"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        daily_detections = [
            detection for detection in self.detection_log
            if detection['timestamp'].startswith(date_str)
        ]
        
        report = f"Daily Flare Activity Report - {date_str}\n"
        report += "=" * 50 + "\n"
        report += f"Total detections analyzed: {len(daily_detections)}\n"
        
        valid_flares = [d for d in daily_detections if d['is_flare']]
        report += f"Valid flare detections: {len(valid_flares)}\n"
        
        if valid_flares:
            heights = [d['height_meters'] for d in valid_flares]
            report += f"Average flare height: {np.mean(heights):.2f}m\n"
            report += f"Maximum flare height: {max(heights):.2f}m\n"
            report += f"Minimum flare height: {min(heights):.2f}m\n"
            
            alarms = [d for d in valid_flares if d['alarm_triggered']]
            warnings = [d for d in valid_flares if d['warning']]
            report += f"Alarm events: {len(alarms)}\n"
            report += f"Warning events: {len(warnings)}\n"
        
        return report
    
    def plot_flare_height_trends(self, hours=24):
        """Plot flare height trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_detections = [
            d for d in self.detection_log
            if datetime.fromisoformat(d['timestamp']) > cutoff_time and d['is_flare']
        ]
        
        if not recent_detections:
            print("No flare detections in the specified time period")
            return
        
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in recent_detections]
        heights = [d['height_meters'] for d in recent_detections]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, heights, 'o-', linewidth=2, markersize=6)
        plt.axhline(y=self.monitor.threshold_height, color='r', linestyle='--', 
                   label=f'Alarm Threshold ({self.monitor.threshold_height}m)')
        plt.axhline(y=self.monitor.config['safety']['warning_height_meters'], 
                   color='orange', linestyle='--', 
                   label=f'Warning Threshold ({self.monitor.config["safety"]["warning_height_meters"]}m)')
        
        plt.xlabel('Time')
        plt.ylabel('Flare Height (meters)')
        plt.title(f'Flare Height Trends - Last {hours} Hours')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'flare_trends_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename)
        print(f"Trend plot saved as: {filename}")
        plt.show()


class FlareDatasetBuilder:
    """
    Class for building training datasets for improved flare detection
    """
    def __init__(self):
        self.positive_samples = []  # Images with confirmed flares
        self.negative_samples = []  # Images without flares
    
    def add_positive_sample(self, image_path, flare_region):
        """Add a positive sample (confirmed flare) to the dataset"""
        self.positive_samples.append({
            'image_path': image_path,
            'flare_region': flare_region,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_negative_sample(self, image_path):
        """Add a negative sample (no flare) to the dataset"""
        self.negative_samples.append({
            'image_path': image_path,
            'timestamp': datetime.now().isoformat()
        })
    
    def extract_features(self, image_path):
        """Extract features from an image for training"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract various features
        features = {}
        
        # Color histogram features
        features['hsv_hist'] = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        features['lab_hist'] = cv2.calcHist([lab], [0, 1, 2], None, [50, 60, 60], [0, 256, 0, 256, 0, 256])
        
        # Texture features using LBP (simplified)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def save_dataset(self, filename):
        """Save the dataset for future training"""
        dataset = {
            'positive_samples': self.positive_samples,
            'negative_samples': self.negative_samples,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved: {len(self.positive_samples)} positive, {len(self.negative_samples)} negative samples")


class FlareCalibrationTool:
    """
    Tool for calibrating the flare monitoring system
    """
    def __init__(self, monitor_system):
        self.monitor = monitor_system
        self.reference_points = []
    
    def interactive_calibration(self, image_path):
        """Interactive calibration using reference objects"""
        image = cv2.imread(image_path)
        if image is None:
            print("Could not load calibration image")
            return
        
        print("Calibration Instructions:")
        print("1. Click on the bottom of a known-height reference object")
        print("2. Click on the top of the same reference object")
        print("3. Press 'c' to complete calibration")
        print("Press ESC to cancel")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.reference_points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Calibration', image)
                print(f"Point {len(self.reference_points)}: ({x}, {y})")
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        cv2.imshow('Calibration', image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c') and len(self.reference_points) >= 2:
                # Calculate calibration factor
                point1, point2 = self.reference_points[-2:]
                pixel_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                
                known_height = float(input("Enter the known height of the reference object (in meters): "))
                calibration_factor = pixel_distance / known_height
                
                print(f"Calibration completed!")
                print(f"Pixels per meter: {calibration_factor:.2f}")
                
                # Update monitor configuration
                self.monitor.calibration_factor = calibration_factor
                self.monitor.config['calibration']['pixels_per_meter'] = calibration_factor
                
                # Save updated config
                with open('flare_config.json', 'w') as f:
                    json.dump(self.monitor.config, f, indent=4)
                
                break
        
        cv2.destroyAllWindows()


def create_sample_config():
    """Create a sample configuration file with detailed explanations"""
    config = {
        "safety": {
            "max_height_meters": 12.0,
            "warning_height_meters": 10.0
        },
        "calibration": {
            "pixels_per_meter": 15.0,
            "reference_object_height_meters": 5.0
        },
        "detection": {
            "flame_color_lower": [5, 50, 50],
            "flame_color_upper": [35, 255, 255],
            "min_contour_area": 1500,
            "flare_confidence_threshold": 0.65,
            "motion_threshold": 25,
            "brightness_threshold": 140,
            "aspect_ratio_range": [0.2, 2.5],
            "temporal_consistency_frames": 7,
            "flare_shape_circularity": 0.25,
            "intensity_variation_threshold": 45,
            "background_subtraction_enabled": True
        },
        "alerts": {
            "email_enabled": False,
            "email_smtp": "smtp.gmail.com",
            "email_port": 587,
            "sender_email": "flare.monitor@yourcompany.com",
            "sender_password": "your_app_password",
            "recipients": ["safety@yourcompany.com", "operations@yourcompany.com"],
            "audio_enabled": True,
            "audio_file": "alarm.wav"
        },
        "monitoring": {
            "check_interval_seconds": 1.5,
            "log_file": "enhanced_flare_monitor.log"
        },
        "camera": {
            "mobile_ip_cam_url": "http://192.168.1.100:8080/video",
            "rtsp_url": "rtsp://username:password@ip:port/stream",
            "http_stream_url": "http://ip:port/mjpg/video.mjpg",
            "connection_timeout": 15
        }
    }
    
    with open('flare_config_sample.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Sample configuration saved as 'flare_config_sample.json'")
    print("Edit this file with your specific settings and rename to 'flare_config.json'")


if __name__ == "__main__":
    # Check if config file exists, create sample if not
    if not os.path.exists('flare_config.json'):
        print("Configuration file not found. Creating sample configuration...")
        create_sample_config()
        print("Please edit 'flare_config_sample.json' and rename it to 'flare_config.json'")
        print("Then run the program again.")
    else:
        main()