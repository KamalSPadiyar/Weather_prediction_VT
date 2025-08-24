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

class FlareMonitoringSystem:
    def __init__(self, config_file='flare_config.json'):
        """
        Initialize the Flare Monitoring System
        """
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.alarm_active = False
        self.calibration_factor = self.config['calibration']['pixels_per_meter']
        self.threshold_height = self.config['safety']['max_height_meters']
        
        # Initialize pygame for audio alerts
        pygame.mixer.init()
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
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
                "min_contour_area": 1000
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
            # Create default config file
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
    
    def detect_available_cameras(self):
        """
        Detect all available camera sources
        """
        available_cameras = []
        
        # Check local cameras (0-10)
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
        
        # Check if mobile IP camera URL is configured
        if self.config['camera']['mobile_ip_cam_url']:
            available_cameras.append({
                'type': 'ip_camera',
                'name': 'Mobile IP Camera',
                'source': self.config['camera']['mobile_ip_cam_url']
            })
        
        # Check if RTSP URL is configured
        if self.config['camera']['rtsp_url']:
            available_cameras.append({
                'type': 'rtsp',
                'name': 'RTSP Stream',
                'source': self.config['camera']['rtsp_url']
            })
        
        # Check if HTTP stream URL is configured
        if self.config['camera']['http_stream_url']:
            available_cameras.append({
                'type': 'http_stream',
                'name': 'HTTP Stream',
                'source': self.config['camera']['http_stream_url']
            })
        
        return available_cameras
    
    def setup_mobile_camera_connection(self):
        """
        Setup mobile camera connection using various methods
        """
        print("\nMobile Camera Setup Options:")
        print("="*50)
        print("1. IP Webcam App (Android/iOS)")
        print("2. DroidCam (Android/iOS)")  
        print("3. RTSP Stream")
        print("4. HTTP MJPEG Stream")
        print("5. Manual URL Entry")
        print("6. Use USB connected phone")
        
        choice = input("\nSelect mobile camera method (1-6): ").strip()
        
        if choice == '1':
            print("\nIP Webcam App Setup:")
            print("1. Install 'IP Webcam' app on your phone")
            print("2. Open the app and tap 'Start Server'")
            print("3. Note the IP address shown (e.g., http://192.168.1.100:8080)")
            ip_url = input("Enter the IP Webcam URL: ").strip()
            if ip_url:
                # Add video endpoint for IP Webcam
                if not ip_url.endswith('/'):
                    ip_url += '/'
                video_url = ip_url + "video"
                self.config['camera']['mobile_ip_cam_url'] = video_url
                return video_url
        
        elif choice == '2':
            print("\nDroidCam Setup:")
            print("1. Install 'DroidCam' app on your phone")
            print("2. Install DroidCam client on your computer")
            print("3. Connect via WiFi or USB")
            print("4. DroidCam creates a virtual camera device")
            print("5. Use camera index when prompted")
            return None
        
        elif choice == '3':
            print("\nRTSP Stream Setup:")
            print("Use apps like 'RTSP Camera' or 'Live Reporter'")
            rtsp_url = input("Enter RTSP URL (rtsp://ip:port/stream): ").strip()
            if rtsp_url:
                self.config['camera']['rtsp_url'] = rtsp_url
                return rtsp_url
        
        elif choice == '4':
            print("\nHTTP MJPEG Stream Setup:")
            http_url = input("Enter HTTP stream URL: ").strip()
            if http_url:
                self.config['camera']['http_stream_url'] = http_url
                return http_url
        
        elif choice == '5':
            manual_url = input("Enter custom camera URL: ").strip()
            if manual_url:
                return manual_url
        
        elif choice == '6':
            print("\nUSB Connection:")
            print("1. Enable USB Debugging on your phone")
            print("2. Connect phone via USB")
            print("3. Select 'File Transfer' or 'PTP' mode")
            print("4. Phone may appear as a camera device")
            return None
        
        return None
    
    def test_camera_connection(self, source):
        """
        Test camera connection before starting monitoring
        """
        print(f"Testing camera connection: {source}")
        
        try:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            
            if not cap.isOpened():
                print("Failed to open camera")
                return False
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                cap.release()
                return False
            
            print(f"Camera connected successfully!")
            print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            # Show test frame for 3 seconds
            cv2.imshow('Camera Test', frame)
            print("Showing test frame for 3 seconds...")
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            
            cap.release()
            return True
            
        except Exception as e:
            print(f"Camera test failed: {e}")
            return False
    
    def create_optimized_capture(self, source):
        """
        Create optimized video capture for mobile cameras
        """
        cap = cv2.VideoCapture(source)
        
        # Optimize capture settings for mobile cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        cap.set(cv2.CAP_PROP_FPS, 30)        # Set FPS
        
        # Try to set resolution (mobile cameras often support multiple resolutions)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        return cap
    
    def detect_flame(self, image):
        """
        Detect flame in the image using color-based segmentation
        Returns the flame contour and flame mask
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
        
        # Filter contours by area
        min_area = self.config['detection']['min_contour_area']
        flame_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return flame_contours, flame_mask
    
    def calculate_flame_height(self, flame_contours):
        """
        Calculate the height of the flame in meters
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
    
    def calibrate_system(self, image, reference_height_pixels):
        """
        Calibrate the system using a reference object
        """
        reference_height_meters = self.config['calibration']['reference_object_height_meters']
        self.calibration_factor = reference_height_pixels / reference_height_meters
        
        # Update config
        self.config['calibration']['pixels_per_meter'] = self.calibration_factor
        
        self.logger.info(f"System calibrated: {self.calibration_factor:.2f} pixels per meter")
    
    def trigger_alarm(self, height_meters, image_path=None):
        """
        Trigger alarm when flame height exceeds threshold
        """
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
        """
        Send email alert to configured recipients
        """
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
        """
        Play audio alert sound
        """
        try:
            if os.path.exists(self.config['alerts']['audio_file']):
                pygame.mixer.music.load(self.config['alerts']['audio_file'])
                pygame.mixer.music.play(-1)  # Loop indefinitely
            else:
                # Generate beep sound if no audio file
                for _ in range(5):
                    print('\a')  # System beep
                    time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Failed to play audio alert: {e}")
    
    def reset_alarm(self):
        """Reset alarm state"""
        self.alarm_active = False
        pygame.mixer.music.stop()
        self.logger.info("Alarm reset")
    
    def process_image(self, image_path):
        """
        Process a single image and check for flame height violations
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None
            
            # Detect flame
            flame_contours, flame_mask = self.detect_flame(image)
            
            # Calculate height
            height_meters = self.calculate_flame_height(flame_contours)
            
            # Create visualization
            result_image = self.create_visualization(image, flame_contours, height_meters)
            
            # Check thresholds
            if height_meters > self.threshold_height:
                self.trigger_alarm(height_meters, image_path)
            elif height_meters > self.config['safety']['warning_height_meters']:
                self.logger.warning(f"Flame height warning: {height_meters:.2f}m")
            
            return {
                'height_meters': height_meters,
                'alarm_triggered': height_meters > self.threshold_height,
                'warning': height_meters > self.config['safety']['warning_height_meters'],
                'result_image': result_image,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def create_visualization(self, image, flame_contours, height_meters):
        """
        Create visualization with flame detection and measurements
        """
        result = image.copy()
        
        # Draw flame contours
        cv2.drawContours(result, flame_contours, -1, (0, 255, 0), 2)
        
        # Add text information
        status_color = (0, 255, 0)  # Green
        if height_meters > self.threshold_height:
            status_color = (0, 0, 255)  # Red
        elif height_meters > self.config['safety']['warning_height_meters']:
            status_color = (0, 165, 255)  # Orange
        
        # Add height measurement
        cv2.putText(result, f"Height: {height_meters:.1f}m", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add threshold information
        cv2.putText(result, f"Threshold: {self.threshold_height:.1f}m", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, 
                   (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add alarm status
        if self.alarm_active:
            cv2.putText(result, "ALARM ACTIVE!", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return result
    
    def live_monitoring(self, camera_source=None):
        """
        Start live monitoring from camera feed (enhanced for mobile cameras)
        """
        if camera_source is None:
            # Show available cameras and let user choose
            available_cameras = self.detect_available_cameras()
            
            if not available_cameras:
                print("No cameras detected. Setting up mobile camera...")
                camera_source = self.setup_mobile_camera_connection()
                if camera_source is None:
                    camera_source = 0  # Default to first camera
            else:
                print("\nAvailable cameras:")
                for i, cam in enumerate(available_cameras):
                    print(f"{i + 1}. {cam['name']} ({cam['type']})")
                
                choice = input(f"\nSelect camera (1-{len(available_cameras)}) or 0 for manual setup: ").strip()
                
                if choice == '0':
                    camera_source = self.setup_mobile_camera_connection()
                    if camera_source is None:
                        camera_source = 0
                elif choice.isdigit() and 1 <= int(choice) <= len(available_cameras):
                    selected_cam = available_cameras[int(choice) - 1]
                    camera_source = selected_cam['source']
                else:
                    camera_source = 0
        
        # Test camera connection first
        if not self.test_camera_connection(camera_source):
            print("Camera connection failed. Please check your setup.")
            return
        
        # Create optimized capture
        cap = self.create_optimized_capture(camera_source)
        
        if not cap.isOpened():
            self.logger.error("Failed to open camera")
            return
        
        self.logger.info(f"Starting live flare monitoring with camera: {camera_source}")
        
        # Performance monitoring
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process frame
            flame_contours, flame_mask = self.detect_flame(frame)
            height_meters = self.calculate_flame_height(flame_contours)
            
            # Create visualization
            result_frame = self.create_visualization(frame, flame_contours, height_meters)
            
            # Add FPS counter
            if frame_count > 10:  # Calculate FPS after some frames
                fps = frame_count / (time.time() - start_time)
                cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                           (result_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Check thresholds
            if height_meters > self.threshold_height:
                self.trigger_alarm(height_meters)
            
            # Display result
            cv2.imshow('Live Flare Monitoring - Mobile Camera', result_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Save screenshot
                screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_name, result_frame)
                print(f"Screenshot saved: {screenshot_name}")
            elif key == ord('c'):  # Calibrate
                print("Click and drag to select reference object for calibration")
                # Add interactive calibration here
            
            # Wait for next check interval
            time.sleep(self.config['monitoring']['check_interval_seconds'])
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show performance stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nMonitoring completed:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
    
    def generate_report(self, results, output_file='flare_report.html'):
        """
        Generate HTML report of monitoring results
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flare Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .alert {{ background-color: #ffcccc; padding: 10px; margin: 10px 0; }}
                .warning {{ background-color: #fff2cc; padding: 10px; margin: 10px 0; }}
                .normal {{ background-color: #ccffcc; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Flare Height Monitoring Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Threshold: {self.threshold_height:.1f} meters</p>
            </div>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Measurements</td><td>{len(results)}</td></tr>
                <tr><td>Alarms Triggered</td><td>{sum(1 for r in results if r.get('height_meters', 0) > self.threshold_height)}</td></tr>
                <tr><td>Warnings</td><td>{sum(1 for r in results if self.config['safety']['warning_height_meters'] < r.get('height_meters', 0) <= self.threshold_height)}</td></tr>
                <tr><td>Maximum Height</td><td>{max([r.get('height_meters', 0) for r in results], default=0):.2f} meters</td></tr>
            </table>
            
            <h2>Detailed Results</h2>
            <table>
                <tr><th>Timestamp</th><th>Height (m)</th><th>Status</th></tr>
        """
        
        for result in results:
            height = result.get('height_meters', 0)
            status = "ALARM" if height > self.threshold_height else "WARNING" if height > self.config['safety']['warning_height_meters'] else "NORMAL"
            status_class = "alert" if status == "ALARM" else "warning" if status == "WARNING" else "normal"
            
            html_content += f"""
                <tr class="{status_class}">
                    <td>{result.get('timestamp', 'N/A')}</td>
                    <td>{height:.2f}</td>
                    <td>{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Report generated: {output_file}")


def main():
    """
    Main function to demonstrate the flare monitoring system
    """
    # Initialize the monitoring system
    monitor = FlareMonitoringSystem()
    
    print("Enhanced Flare Height Monitoring System with Mobile Camera Support")
    print("=" * 70)
    print("1. Process single image")
    print("2. Process video file")
    print("3. Start live monitoring (Auto-detect cameras)")
    print("4. Setup mobile camera connection")
    print("5. Test camera connection")
    print("6. Calibrate system")
    print("7. Exit")
    
    while True:
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = monitor.process_image(image_path)
                if result:
                    print(f"Flame height: {result['height_meters']:.2f} meters")
                    if result['alarm_triggered']:
                        print("ALARM: Height exceeds threshold!")
                    elif result['warning']:
                        print("WARNING: Height approaching threshold!")
                    
                    # Save result image
                    output_path = f"result_{os.path.basename(image_path)}"
                    cv2.imwrite(output_path, result['result_image'])
                    print(f"Result saved to: {output_path}")
            else:
                print("Image file not found!")
        
        elif choice == '2':
            video_path = input("Enter video path: ").strip()
            if os.path.exists(video_path):
                output_path = f"result_{os.path.basename(video_path)}"
                results = monitor.process_video(video_path, output_path)
                print(f"Processed {len(results)} frames")
                monitor.generate_report(results)
            else:
                print("Video file not found!")
        
        elif choice == '3':
            monitor.live_monitoring()
        
        elif choice == '4':
            camera_url = monitor.setup_mobile_camera_connection()
            if camera_url:
                print(f"Mobile camera URL configured: {camera_url}")
                # Save configuration
                with open('flare_config.json', 'w') as f:
                    json.dump(monitor.config, f, indent=4)
                print("Configuration saved!")
        
        elif choice == '5':
            source = input("Enter camera source (index, URL, or 'auto' for detection): ").strip()
            if source == 'auto':
                cameras = monitor.detect_available_cameras()
                for cam in cameras:
                    print(f"Testing {cam['name']}...")
                    monitor.test_camera_connection(cam['source'])
            else:
                if source.isdigit():
                    source = int(source)
                monitor.test_camera_connection(source)
        
        elif choice == '6':
            image_path = input("Enter calibration image path: ").strip()
            if os.path.exists(image_path):
                ref_height = input("Enter reference object height in pixels: ").strip()
                if ref_height.isdigit():
                    image = cv2.imread(image_path)
                    monitor.calibrate_system(image, int(ref_height))
                    print("System calibrated successfully!")
                else:
                    print("Invalid height value!")
            else:
                print("Calibration image not found!")
        
        elif choice == '7':
            print("Exiting...")
            break
        
        else:
            print(f"Invalid choice! Please select 1-7.")


if __name__ == "__main__":
    main()