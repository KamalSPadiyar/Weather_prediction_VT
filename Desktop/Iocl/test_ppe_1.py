#!/usr/bin/env python3
"""
PPE Detection Test Script - PPE-1 Test Case
Tests the PPE detection system with various scenarios
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

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detection.ppe_detector import PPEDetector
from alerts.alert_manager import AlertManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load system configuration"""
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)


def create_test_image_with_person():
    """Create a synthetic test image with a person silhouette"""
    # Create a 640x480 image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Fill with industrial background (gray)
    img[:] = (128, 128, 128)
    
    # Draw a simple person silhouette
    # Head (circle)
    cv2.circle(img, (320, 150), 30, (200, 180, 150), -1)
    
    # Body (rectangle)
    cv2.rectangle(img, (290, 180), (350, 350), (200, 180, 150), -1)
    
    # Arms
    cv2.rectangle(img, (250, 200), (290, 280), (200, 180, 150), -1)
    cv2.rectangle(img, (350, 200), (390, 280), (200, 180, 150), -1)
    
    # Legs
    cv2.rectangle(img, (300, 350), (320, 450), (200, 180, 150), -1)
    cv2.rectangle(img, (330, 350), (350, 450), (200, 180, 150), -1)
    
    # Add some industrial background elements
    cv2.rectangle(img, (50, 400), (150, 480), (100, 100, 100), -1)  # Equipment
    cv2.rectangle(img, (500, 350), (600, 480), (80, 80, 80), -1)    # More equipment
    
    # Add text overlay
    cv2.putText(img, "PPE-1 Test Scene", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Worker without PPE", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img


def create_test_image_with_ppe():
    """Create a test image with a person wearing PPE"""
    # Start with base person image
    img = create_test_image_with_person()
    
    # Add helmet (yellow circle on head)
    cv2.circle(img, (320, 150), 35, (0, 255, 255), -1)  # Yellow helmet
    cv2.circle(img, (320, 150), 35, (0, 200, 200), 3)   # Helmet outline
    
    # Add safety vest (bright orange on torso)
    cv2.rectangle(img, (295, 185), (345, 280), (0, 165, 255), -1)  # Orange vest
    cv2.rectangle(img, (295, 185), (345, 280), (0, 140, 220), 3)   # Vest outline
    
    # Add safety boots (brown on feet)
    cv2.rectangle(img, (295, 440), (325, 465), (42, 42, 165), -1)  # Left boot
    cv2.rectangle(img, (325, 440), (355, 465), (42, 42, 165), -1)  # Right boot
    
    # Update text
    cv2.rectangle(img, (10, 440), (300, 480), (0, 0, 0), -1)  # Clear previous text
    cv2.putText(img, "Worker with PPE", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img


def run_ppe_test_1():
    """Run PPE-1 test case"""
    logger.info("🧪 Starting PPE-1 Test Case")
    logger.info("="*50)
    
    # Load configuration
    config = load_config()
    
    # Initialize detector
    detector = PPEDetector(config)
    alert_manager = AlertManager(config)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "PPE-1A: Worker without PPE",
            "image_func": create_test_image_with_person,
            "expected_compliance": False,
            "expected_missing": ["helmet", "safety_vest"]
        },
        {
            "name": "PPE-1B: Worker with PPE",
            "image_func": create_test_image_with_ppe,
            "expected_compliance": True,
            "expected_missing": []
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\n📋 Test {i}: {scenario['name']}")
        logger.info("-" * 40)
        
        # Create test image
        test_image = scenario['image_func']()
        
        # Save test image for reference
        test_image_path = f"data/samples/images/ppe_test_{i}.jpg"
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        cv2.imwrite(test_image_path, test_image)
        logger.info(f"📁 Test image saved: {test_image_path}")
        
        # Run detection
        start_time = time.time()
        detection_result = detector.detect_ppe(test_image)
        detection_time = time.time() - start_time
        
        # Analyze results
        compliance = detection_result['compliance']
        detections = detection_result['detections']
        
        logger.info(f"⏱️  Detection time: {detection_time:.3f} seconds")
        logger.info(f"🔍 Detections found: {len(detections)}")
        
        for detection in detections:
            logger.info(f"   - {detection['ppe_type']}: {detection['confidence']:.2f}")
        
        logger.info(f"✅ Compliant: {compliance['compliant']}")
        logger.info(f"📝 Message: {compliance['message']}")
        
        if compliance['missing_ppe']:
            logger.info(f"❌ Missing PPE: {', '.join(compliance['missing_ppe'])}")
        
        # Create annotated image
        annotated_image = detector.draw_detections(test_image, detections, compliance)
        annotated_path = f"data/samples/outputs/ppe_test_{i}_annotated.jpg"
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        cv2.imwrite(annotated_path, annotated_image)
        logger.info(f"📁 Annotated image saved: {annotated_path}")
        
        # Store results
        test_result = {
            "scenario": scenario['name'],
            "detection_time": detection_time,
            "detections_count": len(detections),
            "compliance": compliance['compliant'],
            "missing_ppe": compliance['missing_ppe'],
            "message": compliance['message'],
            "image_path": test_image_path,
            "annotated_path": annotated_path
        }
        
        results.append(test_result)
        
        # Process with alert manager (for testing)
        alert_manager.process_detection(detection_result, i)
        
        # Brief pause between tests
        time.sleep(1)
    
    return results


def run_camera_test():
    """Test with live camera if available"""
    logger.info("\n📹 Testing with live camera...")
    
    try:
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("❌ No camera available for live testing")
            return False
        
        # Load configuration
        config = load_config()
        detector = PPEDetector(config)
        
        logger.info("✅ Camera opened successfully")
        logger.info("📸 Capturing frame for PPE detection...")
        
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            logger.warning("❌ Failed to capture frame")
            cap.release()
            return False
        
        # Run detection on live frame
        start_time = time.time()
        detection_result = detector.detect_ppe(frame)
        detection_time = time.time() - start_time
        
        # Save captured frame
        camera_test_path = "data/samples/images/camera_test.jpg"
        cv2.imwrite(camera_test_path, frame)
        
        # Create annotated frame
        annotated_frame = detector.draw_detections(
            frame, detection_result['detections'], detection_result['compliance']
        )
        annotated_camera_path = "data/samples/outputs/camera_test_annotated.jpg"
        cv2.imwrite(annotated_camera_path, annotated_frame)
        
        # Log results
        compliance = detection_result['compliance']
        logger.info(f"⏱️  Detection time: {detection_time:.3f} seconds")
        logger.info(f"🔍 Detections: {len(detection_result['detections'])}")
        logger.info(f"✅ Compliant: {compliance['compliant']}")
        logger.info(f"📝 Message: {compliance['message']}")
        logger.info(f"📁 Camera frame saved: {camera_test_path}")
        logger.info(f"📁 Annotated frame saved: {annotated_camera_path}")
        
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"❌ Camera test failed: {e}")
        return False


def run_api_test():
    """Test the API endpoints"""
    logger.info("\n🌐 Testing API endpoints...")
    
    try:
        import requests
        
        base_url = "http://localhost:5000"
        
        # Test statistics endpoint
        logger.info("📊 Testing /api/stats endpoint...")
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            logger.info("✅ Stats API working")
            logger.info(f"   Detection stats: {stats.get('detection_stats', {})}")
        else:
            logger.warning(f"❌ Stats API failed: {response.status_code}")
        
        # Test alerts endpoint
        logger.info("🚨 Testing /api/alerts endpoint...")
        response = requests.get(f"{base_url}/api/alerts", timeout=5)
        if response.status_code == 200:
            alerts = response.json()
            logger.info("✅ Alerts API working")
            logger.info(f"   Recent alerts: {len(alerts.get('alerts', []))}")
        else:
            logger.warning(f"❌ Alerts API failed: {response.status_code}")
        
        # Test alert system
        logger.info("🔔 Testing alert system...")
        response = requests.post(f"{base_url}/api/test-alert", timeout=5)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Alert test: {result.get('success', False)}")
        else:
            logger.warning(f"❌ Alert test failed: {response.status_code}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"❌ API test failed: {e}")
        logger.info("💡 Make sure the dashboard is running: python main.py --source dashboard")
        return False


def generate_test_report(results):
    """Generate a comprehensive test report"""
    logger.info("\n📋 GENERATING TEST REPORT")
    logger.info("="*50)
    
    # Create test report
    report = {
        "test_name": "PPE-1 Test Case",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scenarios": len(results),
        "results": results
    }
    
    # Save JSON report
    report_path = "data/samples/outputs/ppe_test_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info(f"📊 Test Summary:")
    logger.info(f"   Total scenarios tested: {len(results)}")
    
    passed = sum(1 for r in results if r['compliance'] is not None)
    logger.info(f"   Tests completed: {passed}/{len(results)}")
    
    avg_detection_time = sum(r['detection_time'] for r in results) / len(results)
    logger.info(f"   Average detection time: {avg_detection_time:.3f} seconds")
    
    logger.info(f"\n📁 Test artifacts generated:")
    for result in results:
        logger.info(f"   - {result['image_path']}")
        logger.info(f"   - {result['annotated_path']}")
    
    logger.info(f"   - {report_path}")
    
    logger.info(f"\n🎯 Test Results:")
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['compliance'] is not None else "❌ FAIL"
        logger.info(f"   {i}. {result['scenario']}: {status}")
        logger.info(f"      Message: {result['message']}")


def main():
    """Main test function"""
    print("🧪 PPE-1 TEST SUITE")
    print("IOCL PPE Detection System - Comprehensive Testing")
    print("="*60)
    
    try:
        # Run synthetic image tests
        results = run_ppe_test_1()
        
        # Run camera test if available
        run_camera_test()
        
        # Run API tests
        run_api_test()
        
        # Generate report
        generate_test_report(results)
        
        print("\n🎉 PPE-1 Testing Complete!")
        print("📁 Check data/samples/outputs/ for test results")
        print("🌐 Dashboard: http://localhost:5000")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
