# IOCL Flare Detection System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive **Flare Height Monitoring System** designed for IOCL (Indian Oil Corporation Limited) refineries to detect, monitor, and alert on flare stack heights in real-time. This system uses computer vision techniques to analyze video feeds and ensure safety compliance by monitoring flare heights against predefined thresholds.

## 🔥 Features

### Core Functionality
- **Real-time Flare Detection**: Advanced computer vision algorithms for flame detection
- **Height Measurement**: Accurate pixel-to-meter conversion with calibration support
- **Multi-Camera Support**: Works with webcams, IP cameras, RTSP streams, and mobile cameras
- **Safety Monitoring**: Configurable warning and critical height thresholds
- **Automated Alerts**: Email notifications and audio alarms for threshold violations

### Advanced Capabilities
- **Mobile Camera Integration**: Support for IP Webcam, DroidCam, and other mobile camera apps
- **Background Subtraction**: Enhanced detection using motion analysis
- **Temporal Consistency**: Multi-frame analysis for improved accuracy
- **Shape Analysis**: Circularity and aspect ratio validation
- **Intensity Variation**: Dynamic flame behavior analysis
- **Performance Monitoring**: FPS tracking and system optimization

### Monitoring & Reporting
- **Live Visualization**: Real-time display with overlay information
- **HTML Reports**: Comprehensive monitoring reports with statistics
- **Logging System**: Detailed event logging with timestamps
- **Screenshot Capture**: Manual and automatic image capture
- **Configuration Management**: JSON-based settings with hot-reload

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- Windows/Linux/macOS
- Webcam or IP camera access
- Minimum 4GB RAM (8GB recommended)
- OpenCV-compatible system

### Python Dependencies
```txt
opencv-python>=4.5.0
numpy>=1.19.0
pygame>=2.0.0
matplotlib>=3.3.0
requests>=2.25.0
smtplib (built-in)
threading (built-in)
json (built-in)
logging (built-in)
datetime (built-in)
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KamalSPadiyar/IOCL_FLARE_DETECTION.git
cd IOCL_FLARE_DETECTION
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration Setup
The system will automatically create a default configuration file (`flare_config.json`) on first run. You can customize the settings according to your requirements.

## 🔧 Configuration

### Basic Configuration (`flare_config.json`)
```json
{
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
        "min_contour_area": 1500
    },
    "alerts": {
        "email_enabled": true,
        "sender_email": "your-email@domain.com",
        "recipients": ["safety@company.com"]
    }
}
```

### Email Alert Setup
1. Update `sender_email` and `sender_password` in the configuration
2. Add recipient email addresses to the `recipients` list
3. For Gmail, use app-specific passwords instead of your regular password

### Mobile Camera Setup
The system supports multiple mobile camera options:

1. **IP Webcam App** (Android/iOS)
   - Install "IP Webcam" from app store
   - Start server and note the IP address
   - Use format: `http://192.168.1.100:8080/video`

2. **DroidCam** (Android/iOS)
   - Install DroidCam app and desktop client
   - Creates virtual camera device

3. **RTSP Streams**
   - Use RTSP-compatible camera apps
   - Format: `rtsp://ip:port/stream`

## 🎯 Usage

### Running the System
```bash
python main.py
```

### Menu Options
1. **Process Single Image**: Analyze a static image for flare detection
2. **Process Video File**: Batch process video files
3. **Live Monitoring**: Real-time camera monitoring (recommended)
4. **Mobile Camera Setup**: Configure mobile camera connections
5. **Test Camera**: Verify camera connectivity
6. **System Calibration**: Set up pixel-to-meter conversion
7. **Exit**: Close the application

### Live Monitoring Controls
- **Q**: Quit monitoring
- **S**: Save screenshot
- **C**: Start calibration mode

### Sample Usage Examples

#### Single Image Processing
```python
from flare_height_monitoring_system import FlareMonitoringSystem

monitor = FlareMonitoringSystem()
result = monitor.process_image("sample.jpg")
print(f"Detected flare height: {result['height_meters']:.2f} meters")
```

#### Live Monitoring with Custom Camera
```python
monitor = FlareMonitoringSystem()
monitor.live_monitoring(camera_source="http://192.168.1.100:8080/video")
```

## 📊 System Architecture

### Core Components
1. **FlareMonitoringSystem**: Main class handling all operations
2. **Image Processing Pipeline**: HSV color space conversion, morphological operations
3. **Detection Algorithm**: Contour analysis, shape validation
4. **Measurement System**: Calibrated pixel-to-meter conversion
5. **Alert System**: Multi-channel notification system
6. **Configuration Manager**: JSON-based settings management

### Detection Pipeline
```
Input Image/Frame
       ↓
HSV Color Conversion
       ↓
Color Mask Generation
       ↓
Morphological Operations
       ↓
Contour Detection
       ↓
Shape & Size Filtering
       ↓
Height Calculation
       ↓
Threshold Comparison
       ↓
Alert Generation (if needed)
```

## 🎨 Sample Output

### Detection Visualization
The system provides real-time visualization with:
- **Green contours**: Detected flame boundaries
- **Height display**: Current measurement in meters
- **Status indicators**: Color-coded alerts (Green/Orange/Red)
- **Timestamp**: Current monitoring time
- **FPS counter**: System performance metrics

### Generated Reports
- **HTML Dashboard**: Comprehensive monitoring statistics
- **Log Files**: Detailed event history
- **Screenshots**: Captured moments for analysis

## ⚙️ Calibration

### System Calibration Process
1. Place a reference object of known height in the camera view
2. Run calibration mode from the menu
3. Measure the object's height in pixels
4. System automatically calculates pixels-per-meter ratio

### Calibration Tips
- Use objects with clear, measurable dimensions
- Ensure consistent lighting conditions
- Place reference object at the same distance as the flare stack
- Recalibrate when camera position changes

## 🚨 Safety Features

### Multi-Level Alerts
- **Warning Level**: Configurable warning threshold
- **Critical Level**: Immediate alarm activation
- **Escalation**: Automatic notification to multiple recipients

### Alert Mechanisms
1. **Visual Alerts**: On-screen notifications and color coding
2. **Audio Alerts**: System beeps and custom sound files
3. **Email Notifications**: Instant email alerts to safety teams
4. **Log Recording**: Permanent record of all safety events

## 📱 Mobile Camera Integration

### Supported Mobile Apps
- **IP Webcam** (Free, Android/iOS)
- **DroidCam** (Free/Paid, Android/iOS)
- **RTSP Camera** (Various apps available)
- **HTTP MJPEG** streamers

### Network Configuration
- Ensure mobile device and computer are on the same network
- Configure firewall settings if needed
- Use static IP addresses for reliable connections

## 🔍 Troubleshooting

### Common Issues

#### Camera Connection Failed
- Verify camera permissions
- Check network connectivity for IP cameras
- Test camera with other applications
- Restart camera application/device

#### Poor Detection Accuracy
- Adjust color thresholds in configuration
- Improve lighting conditions
- Recalibrate the system
- Clean camera lens

#### Email Alerts Not Working
- Verify email credentials
- Check internet connectivity
- Use app-specific passwords for Gmail
- Review firewall/antivirus settings

#### Performance Issues
- Lower camera resolution
- Increase check interval
- Close unnecessary applications
- Upgrade hardware if needed

## 📈 Performance Optimization

### System Optimization Tips
1. **Camera Settings**: Use appropriate resolution (720p recommended)
2. **Buffer Management**: Minimize capture buffer size for real-time processing
3. **Frame Rate**: Optimize FPS based on hardware capabilities
4. **Memory Management**: Regular cleanup of temporary data
5. **CPU Usage**: Adjust detection parameters for performance balance

## 🤝 Contributing

### Development Guidelines
1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Update documentation as needed
5. Submit pull request with detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Add proper docstrings to functions
- Include error handling
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Kamal Singh Padiyar** - *Lead Developer* - [GitHub Profile](https://github.com/KamalSPadiyar)

## 🙏 Acknowledgments

- Indian Oil Corporation Limited (IOCL) for project requirements
- OpenCV community for computer vision libraries
- Python community for excellent documentation and support

## 📞 Support

For technical support or questions:
- **Email**: kamalsinghpadiyar1919@gmail.com
- **GitHub Issues**: [Create an Issue](https://github.com/KamalSPadiyar/IOCL_FLARE_DETECTION/issues)

## 🔮 Future Enhancements

### Planned Features
- Machine learning-based detection improvements
- Cloud integration for remote monitoring
- Mobile app for on-the-go monitoring
- Integration with industrial control systems
- Advanced analytics and predictive maintenance
- Multi-site monitoring dashboard

---

**⚠️ Safety Notice**: This system is designed to assist in safety monitoring but should not be the sole method for critical safety decisions. Always follow proper industrial safety protocols and have backup monitoring systems in place.

**🏭 Industrial Use**: This system is specifically designed for refinery and industrial applications where flare stack monitoring is critical for safety and environmental compliance.
