#!/usr/bin/env python3
"""
Test script for IOCL PPE Detection System
Runs comprehensive tests to verify system functionality
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import yaml
import numpy as np
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestPPEDetectionSystem(unittest.TestCase):
    """Test cases for PPE Detection System"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_config = {
            'model': {
                'name': 'yolov8n',
                'weights_path': 'data/models/test_model.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'input_size': [640, 640]
            },
            'video': {
                'frame_skip': 1,
                'resize_width': 640,
                'resize_height': 480,
                'buffer_size': 30
            },
            'alerts': {
                'enabled': False,  # Disable for testing
                'alert_threshold': 3,
                'notification_methods': []
            },
            'compliance': {
                'required_ppe': {
                    'helmet': True,
                    'safety_vest': True,
                    'gloves': False,
                    'safety_boots': True,
                    'safety_goggles': False
                }
            },
            'database': {
                'path': 'test_detection_logs.db'
            },
            'logging': {
                'level': 'ERROR',
                'file_path': 'test.log'
            },
            'dashboard': {
                'host': '127.0.0.1',
                'port': 5001,
                'debug': False
            }
        }
    
    def setUp(self):
        """Set up each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, 'test.db')
        self.test_config['database']['path'] = self.test_db_path
    
    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up test database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_config_loading(self):
        """Test configuration loading"""
        config_path = "config/settings.yaml"
        self.assertTrue(os.path.exists(config_path), "Configuration file should exist")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertIsInstance(config, dict, "Configuration should be a dictionary")
        self.assertIn('model', config, "Configuration should have model section")
        self.assertIn('video', config, "Configuration should have video section")
    
    def test_ppe_detector_initialization(self):
        """Test PPE detector initialization"""
        try:
            from detection.ppe_detector import PPEDetector
            detector = PPEDetector(self.test_config)
            self.assertIsNotNone(detector, "Detector should initialize successfully")
        except ImportError:
            self.skipTest("PPE detector module not available")
    
    def test_ppe_detection_with_dummy_image(self):
        """Test PPE detection with dummy image"""
        try:
            from detection.ppe_detector import PPEDetector
            
            detector = PPEDetector(self.test_config)
            
            # Create dummy image
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Run detection
            result = detector.detect_ppe(dummy_image)
            
            # Verify result structure
            self.assertIsInstance(result, dict, "Result should be a dictionary")
            self.assertIn('detections', result, "Result should have detections")
            self.assertIn('compliance', result, "Result should have compliance info")
            self.assertIsInstance(result['detections'], list, "Detections should be a list")
            
        except ImportError:
            self.skipTest("Required modules not available")
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization"""
        try:
            from alerts.alert_manager import AlertManager
            
            alert_manager = AlertManager(self.test_config)
            self.assertIsNotNone(alert_manager, "Alert manager should initialize")
            
        except ImportError:
            self.skipTest("Alert manager module not available")
    
    def test_video_processor_initialization(self):
        """Test video processor initialization"""
        try:
            from detection.ppe_detector import PPEDetector
            from alerts.alert_manager import AlertManager
            from video.video_processor import VideoProcessor
            
            detector = PPEDetector(self.test_config)
            alert_manager = AlertManager(self.test_config)
            video_processor = VideoProcessor(self.test_config, detector, alert_manager)
            
            self.assertIsNotNone(video_processor, "Video processor should initialize")
            
        except ImportError:
            self.skipTest("Required modules not available")
    
    def test_database_creation(self):
        """Test database creation and operations"""
        try:
            from alerts.alert_manager import AlertManager
            
            alert_manager = AlertManager(self.test_config)
            
            # Check if database file was created
            self.assertTrue(os.path.exists(self.test_db_path), "Database should be created")
            
            # Test getting stats (should not fail even with empty database)
            stats = alert_manager.get_detection_stats(24)
            self.assertIsInstance(stats, dict, "Stats should be a dictionary")
            
        except ImportError:
            self.skipTest("Alert manager module not available")
    
    def test_dashboard_app_creation(self):
        """Test dashboard app creation"""
        try:
            from dashboard.app import create_app
            
            app = create_app(self.test_config)
            self.assertIsNotNone(app, "Dashboard app should be created")
            
        except ImportError:
            self.skipTest("Dashboard module not available")
    
    def test_main_script_imports(self):
        """Test that main script can import all required modules"""
        try:
            # Test importing main application components
            import main
            self.assertTrue(True, "Main script imports successfully")
            
        except ImportError as e:
            self.fail(f"Main script import failed: {e}")
    
    def test_requirements_coverage(self):
        """Test that all required packages are available"""
        required_packages = [
            'yaml', 'numpy', 'pathlib', 'logging', 'sqlite3',
            'json', 'datetime', 'threading', 'queue', 'time'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        self.assertEqual(len(missing_packages), 0, 
                        f"Missing required packages: {missing_packages}")
    
    def test_directory_structure(self):
        """Test that required directories exist or can be created"""
        required_dirs = [
            'src/detection',
            'src/video', 
            'src/alerts',
            'src/dashboard',
            'config',
            'data'
        ]
        
        for directory in required_dirs:
            self.assertTrue(os.path.exists(directory), 
                          f"Required directory should exist: {directory}")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test with valid config
        self.assertIsInstance(self.test_config['model']['confidence_threshold'], float)
        self.assertGreater(self.test_config['model']['confidence_threshold'], 0)
        self.assertLess(self.test_config['model']['confidence_threshold'], 1)
        
        # Test required PPE configuration
        required_ppe = self.test_config['compliance']['required_ppe']
        self.assertIsInstance(required_ppe, dict)
        self.assertIn('helmet', required_ppe)
        self.assertIn('safety_vest', required_ppe)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration tests"""
        self.config_path = "config/settings.yaml"
    
    def test_end_to_end_configuration(self):
        """Test end-to-end system with real configuration"""
        if not os.path.exists(self.config_path):
            self.skipTest("Configuration file not found")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test that all required sections exist
            required_sections = ['model', 'video', 'alerts', 'compliance']
            for section in required_sections:
                self.assertIn(section, config, f"Config should have {section} section")
            
        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")


def run_performance_test():
    """Run basic performance test"""
    print("\n" + "="*50)
    print("PERFORMANCE TEST")
    print("="*50)
    
    try:
        import time
        import psutil
        
        # Monitor system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory_info.percent}%")
        print(f"Available Memory: {memory_info.available / (1024**3):.1f} GB")
        
        # Test detection speed with dummy data
        from detection.ppe_detector import PPEDetector
        
        config_path = "config/settings.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            detector = PPEDetector(config)
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Time detection
            start_time = time.time()
            num_detections = 10
            
            for _ in range(num_detections):
                result = detector.detect_ppe(dummy_image)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_detections
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"Average Detection Time: {avg_time:.3f} seconds")
            print(f"Estimated FPS: {fps:.1f}")
            
            if fps > 15:
                print("✅ Performance: GOOD (Real-time capable)")
            elif fps > 5:
                print("⚠️  Performance: MODERATE (May need optimization)")
            else:
                print("❌ Performance: POOR (Optimization required)")
        
    except Exception as e:
        print(f"Performance test failed: {e}")


def main():
    """Run all tests"""
    print("🧪 IOCL PPE Detection System - Test Suite")
    print("="*50)
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPPEDetectionSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, error in result.errors:
            print(f"  {test}: {error}")
    
    if result.skipped:
        print("\n⏭️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"  {test}: {reason}")
    
    # Run performance test if basic tests pass
    if len(result.failures) == 0 and len(result.errors) == 0:
        run_performance_test()
    
    # Overall result
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
