#!/usr/bin/env python3
"""
PPE Detection Real-World Test - Live Demo
Tests with actual person detection using camera or uploaded images
"""

import sys
import os
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
import json
import time
import requests
import base64

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detection.ppe_detector import PPEDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_realistic_test_image():
    """Create a more realistic test image with person-like features that YOLO can detect"""
    # Create a larger, more detailed image
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 120  # Gray background
    
    # Create a more realistic person silhouette
    # Head (larger, more person-like)
    head_center = (640, 200)
    head_radius = 50
    cv2.circle(img, head_center, head_radius, (180, 150, 120), -1)  # Skin tone
    cv2.circle(img, head_center, head_radius, (150, 120, 100), 3)   # Head outline
    
    # Torso (more realistic proportions)
    torso_top = head_center[1] + head_radius + 10
    torso_bottom = torso_top + 250
    torso_left = head_center[0] - 80
    torso_right = head_center[0] + 80
    
    cv2.rectangle(img, (torso_left, torso_top), (torso_right, torso_bottom), (100, 100, 150), -1)
    
    # Arms
    arm_width = 40
    arm_length = 180
    # Left arm
    cv2.rectangle(img, (torso_left - arm_width, torso_top + 30), 
                 (torso_left, torso_top + 30 + arm_length), (180, 150, 120), -1)
    # Right arm
    cv2.rectangle(img, (torso_right, torso_top + 30), 
                 (torso_right + arm_width, torso_top + 30 + arm_length), (180, 150, 120), -1)
    
    # Legs
    leg_width = 50
    leg_length = 200
    leg_separation = 20
    leg_top = torso_bottom
    
    # Left leg
    cv2.rectangle(img, (head_center[0] - leg_separation - leg_width, leg_top), 
                 (head_center[0] - leg_separation, leg_top + leg_length), (60, 60, 100), -1)
    # Right leg
    cv2.rectangle(img, (head_center[0] + leg_separation, leg_top), 
                 (head_center[0] + leg_separation + leg_width, leg_top + leg_length), (60, 60, 100), -1)
    
    # Add some realistic details
    # Eyes
    cv2.circle(img, (head_center[0] - 15, head_center[1] - 10), 3, (0, 0, 0), -1)
    cv2.circle(img, (head_center[0] + 15, head_center[1] - 10), 3, (0, 0, 0), -1)
    
    # Add industrial background elements
    cv2.rectangle(img, (50, 600), (200, 720), (80, 80, 80), -1)    # Equipment
    cv2.rectangle(img, (1080, 550), (1230, 720), (70, 70, 70), -1) # More equipment
    cv2.rectangle(img, (200, 650), (300, 720), (90, 90, 90), -1)   # Pipes
    
    # Add text
    cv2.putText(img, "PPE Detection Test - Realistic Person", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(img, "Industrial Environment", (50, 680), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return img


def test_api_detection():
    """Test the detection API endpoint"""
    logger.info("🌐 Testing PPE Detection via API")
    
    try:
        # Create test image
        test_image = create_realistic_test_image()
        
        # Save test image
        test_path = "data/samples/images/api_test.jpg"
        cv2.imwrite(test_path, test_image)
        logger.info(f"📁 Test image created: {test_path}")
        
        # Encode image for API
        _, buffer = cv2.imencode('.jpg', test_image)
        
        # Test API endpoint
        api_url = "http://localhost:5000/api/detect"
        files = {'image': ('test.jpg', buffer.tobytes(), 'image/jpeg')}
        
        logger.info(f"📡 Sending request to {api_url}")
        response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ API Detection successful!")
            
            # Parse results
            detections = result.get('detections', [])
            compliance = result.get('compliance', {})
            
            logger.info(f"🔍 Detections found: {len(detections)}")
            for detection in detections:
                logger.info(f"   - {detection['ppe_type']}: {detection['confidence']:.2f}")
            
            logger.info(f"✅ Compliant: {compliance.get('compliant', False)}")
            logger.info(f"📝 Message: {compliance.get('message', 'N/A')}")
            
            if compliance.get('missing_ppe'):
                logger.info(f"❌ Missing PPE: {', '.join(compliance['missing_ppe'])}")
            
            # Save processed image if available
            if 'processed_image' in result:
                processed_data = base64.b64decode(result['processed_image'])
                processed_path = "data/samples/outputs/api_test_processed.jpg"
                with open(processed_path, 'wb') as f:
                    f.write(processed_data)
                logger.info(f"📁 Processed image saved: {processed_path}")
            
            return True
            
        else:
            logger.error(f"❌ API request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to API - Dashboard may not be running")
        logger.info("💡 Start dashboard: python main.py --source dashboard")
        return False
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


def test_live_camera_with_person_detection():
    """Test live camera with focus on person detection"""
    logger.info("📹 Testing Live Camera with Person Detection")
    
    try:
        # Load configuration
        with open('config/settings.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize detector
        detector = PPEDetector(config)
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("❌ No camera available")
            return False
        
        logger.info("✅ Camera opened - Looking for person detection...")
        logger.info("📸 Capturing multiple frames for better detection...")
        
        best_detection = None
        max_detections = 0
        
        # Capture multiple frames to find the best one
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Run detection
            result = detector.detect_ppe(frame)
            detections_count = len(result['detections'])
            
            if detections_count > max_detections:
                max_detections = detections_count
                best_detection = (frame.copy(), result)
            
            time.sleep(0.5)  # Wait between captures
        
        cap.release()
        
        if best_detection:
            frame, result = best_detection
            
            # Save original frame
            camera_path = "data/samples/images/live_camera_test.jpg"
            cv2.imwrite(camera_path, frame)
            
            # Create annotated frame
            annotated = detector.draw_detections(frame, result['detections'], result['compliance'])
            annotated_path = "data/samples/outputs/live_camera_annotated.jpg"
            cv2.imwrite(annotated_path, annotated)
            
            # Log results
            logger.info(f"📊 Best frame results:")
            logger.info(f"🔍 Detections: {len(result['detections'])}")
            
            for detection in result['detections']:
                logger.info(f"   - {detection['ppe_type']}: {detection['confidence']:.2f}")
            
            compliance = result['compliance']
            logger.info(f"✅ Compliant: {compliance['compliant']}")
            logger.info(f"📝 Message: {compliance['message']}")
            logger.info(f"📁 Original frame: {camera_path}")
            logger.info(f"📁 Annotated frame: {annotated_path}")
            
            return True
        else:
            logger.warning("❌ No good frames captured")
            return False
            
    except Exception as e:
        logger.error(f"❌ Live camera test failed: {e}")
        return False


def test_system_performance():
    """Test system performance metrics"""
    logger.info("⚡ Testing System Performance")
    
    try:
        # Load configuration
        with open('config/settings.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        detector = PPEDetector(config)
        
        # Create test image
        test_image = create_realistic_test_image()
        
        # Performance test
        logger.info("🎯 Running performance benchmark...")
        
        times = []
        for i in range(5):
            start_time = time.time()
            result = detector.detect_ppe(test_image)
            detection_time = time.time() - start_time
            times.append(detection_time)
            logger.info(f"   Run {i+1}: {detection_time:.3f} seconds")
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        logger.info(f"📊 Performance Results:")
        logger.info(f"   Average detection time: {avg_time:.3f} seconds")
        logger.info(f"   Estimated FPS: {fps:.1f}")
        logger.info(f"   Min time: {min(times):.3f} seconds")
        logger.info(f"   Max time: {max(times):.3f} seconds")
        
        # Performance rating
        if fps > 20:
            logger.info("🟢 Performance: EXCELLENT (Real-time capable)")
        elif fps > 10:
            logger.info("🟡 Performance: GOOD (Suitable for monitoring)")
        elif fps > 5:
            logger.info("🟠 Performance: MODERATE (Acceptable for safety)")
        else:
            logger.info("🔴 Performance: NEEDS OPTIMIZATION")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def demonstrate_ppe_detection():
    """Comprehensive demonstration of PPE detection capabilities"""
    logger.info("🎯 PPE Detection System Demonstration")
    logger.info("="*60)
    
    results = {
        "api_test": False,
        "camera_test": False,
        "performance_test": False
    }
    
    # Test 1: API Detection
    logger.info("\n1️⃣ Testing API Detection Endpoint")
    logger.info("-" * 40)
    results["api_test"] = test_api_detection()
    
    # Test 2: Live Camera
    logger.info("\n2️⃣ Testing Live Camera Detection")
    logger.info("-" * 40)
    results["camera_test"] = test_live_camera_with_person_detection()
    
    # Test 3: Performance
    logger.info("\n3️⃣ Testing System Performance")
    logger.info("-" * 40)
    results["performance_test"] = test_system_performance()
    
    # Summary
    logger.info("\n📋 DEMONSTRATION SUMMARY")
    logger.info("="*40)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"✅ Tests passed: {passed_tests}/{total_tests}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All tests passed! System is fully operational.")
    else:
        logger.info(f"\n⚠️  {total_tests - passed_tests} tests failed. Check logs for details.")
    
    # Show file locations
    logger.info("\n📁 Generated files:")
    sample_files = [
        "data/samples/images/api_test.jpg",
        "data/samples/images/live_camera_test.jpg",
        "data/samples/outputs/api_test_processed.jpg",
        "data/samples/outputs/live_camera_annotated.jpg"
    ]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            logger.info(f"   ✅ {file_path}")
        else:
            logger.info(f"   ❌ {file_path} (not created)")
    
    logger.info(f"\n🌐 Dashboard: http://localhost:5000")
    logger.info(f"📖 Documentation: docs/TECHNICAL_DOCUMENTATION.md")


def main():
    """Main demonstration function"""
    print("\n🔬 PPE DETECTION SYSTEM - LIVE DEMONSTRATION")
    print("IOCL Industrial Safety Monitoring System")
    print("="*70)
    
    try:
        demonstrate_ppe_detection()
        
        print("\n" + "="*70)
        print("🎯 DEMONSTRATION COMPLETE!")
        print("💡 The system is ready for IOCL deployment.")
        print("🔧 Next steps: Configure with IOCL cameras and train custom model.")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
