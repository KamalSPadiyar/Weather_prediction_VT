# IOCL PPE Detection System - Technical Documentation

## System Overview

The IOCL PPE Detection System is a computer vision-based safety monitoring solution designed to automatically detect Personal Protective Equipment (PPE) compliance from CCTV/video feeds in industrial environments.

## Architecture

### Core Components

1. **Detection Engine** (`src/detection/`)
   - PPE detection using YOLOv8 object detection model
   - Real-time classification of safety equipment
   - Confidence scoring and threshold management

2. **Video Processing** (`src/video/`)
   - Multi-source video input handling (cameras, files, RTSP streams)
   - Frame preprocessing and optimization
   - Stream management for multiple concurrent sources

3. **Alert Management** (`src/alerts/`)
   - Real-time violation detection and alerting
   - Multiple notification channels (email, webhook, dashboard)
   - Alert filtering and cooldown management
   - SQLite database for logging and analytics

4. **Web Dashboard** (`src/dashboard/`)
   - Real-time monitoring interface
   - Live stream viewing and statistics
   - Alert management and reporting
   - REST API for integration

## PPE Categories Detected

| PPE Type | Description | Critical Level |
|----------|-------------|----------------|
| Helmet/Hard Hat | Head protection | Critical |
| Safety Vest | High-visibility clothing | Critical |
| Safety Gloves | Hand protection | Medium |
| Safety Boots | Foot protection | High |
| Safety Goggles | Eye protection | Medium |

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for real-time processing)
- Minimum 8GB RAM
- 10GB free disk space

### Quick Installation

```bash
# Clone or extract the project
cd iocl-ppe-detection

# Run the setup script
python setup.py

# Start the system
python main.py --source dashboard
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download YOLO base model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create necessary directories
mkdir -p data/{models,datasets,samples} logs

# Test the installation
python main.py --source camera --display
```

## Configuration

### Main Configuration File: `config/settings.yaml`

#### Model Settings
```yaml
model:
  name: "yolov8n"  # Model size: n, s, m, l, x
  weights_path: "data/models/ppe_detection.pt"
  confidence_threshold: 0.5  # Detection confidence
  iou_threshold: 0.45       # Non-maximum suppression
```

#### PPE Requirements
```yaml
compliance:
  required_ppe:
    helmet: true          # Always required
    safety_vest: true     # Always required
    gloves: false         # Optional by default
    safety_boots: true    # Required for most areas
    safety_goggles: false # Area-specific
```

#### Camera Configuration
```yaml
camera:
  default_source: 0  # Default camera index
  rtsp_urls:
    - "rtsp://camera1.iocl.local:554/stream"
    - "rtsp://camera2.iocl.local:554/stream"
```

#### Alert Settings
```yaml
alerts:
  enabled: true
  alert_threshold: 3  # Consecutive violations before alert
  notification_methods:
    - email
    - webhook
    - dashboard
  
  email:
    smtp_server: "smtp.gmail.com"
    sender_email: "alerts@iocl.com"
    recipients:
      - "safety@iocl.com"
```

## Usage Examples

### Real-time Camera Monitoring
```bash
# Single camera
python main.py --source camera --device 0 --display

# Multiple cameras (requires stream configuration)
python main.py --source dashboard
```

### Video File Processing
```bash
# Process single video
python main.py --source video --input footage.mp4 --output processed.mp4

# Batch processing
for video in videos/*.mp4; do
    python main.py --source video --input "$video" --output "processed_$video"
done
```

### RTSP Stream Processing
```bash
# Live RTSP stream
python main.py --source rtsp --input "rtsp://camera.ip:554/stream" --display
```

### Web Dashboard
```bash
# Start dashboard server
python main.py --source dashboard

# Access via browser: http://localhost:5000
```

## Model Training

### Dataset Preparation

1. **Collect Training Data**
   ```bash
   # Create dataset structure
   mkdir -p data/datasets/ppe_training/{images,labels}/{train,val,test}
   ```

2. **Annotation Format** (YOLO format)
   ```
   # Each line in .txt file:
   class_id x_center y_center width height
   
   # Example annotation file:
   0 0.5 0.3 0.2 0.4  # helmet at center-top
   1 0.5 0.7 0.3 0.5  # safety_vest at center-bottom
   ```

3. **Class Mapping**
   ```
   0: helmet
   1: safety_vest
   2: gloves
   3: safety_boots
   4: safety_goggles
   5: person
   ```

### Training Process

```bash
# Basic training
python src/models/train_ppe_model.py \
    --data data/datasets/ppe_training \
    --epochs 100 \
    --batch 16 \
    --name ppe_v1

# Advanced training with validation
python src/models/train_ppe_model.py \
    --data data/datasets/ppe_training \
    --epochs 200 \
    --batch 32 \
    --model yolov8s.pt \
    --imgsz 640 \
    --device 0 \
    --validate \
    --validate-final
```

### Model Evaluation

```bash
# Validate trained model
python -c "
from ultralytics import YOLO
model = YOLO('data/models/ppe_detection.pt')
results = model.val(data='data/datasets/ppe_training/dataset.yaml')
print(f'mAP50: {results.box.map50}')
print(f'mAP50-95: {results.box.map}')
"
```

## API Reference

### REST API Endpoints

#### Detection API
```http
POST /api/detect
Content-Type: multipart/form-data

# Upload image for PPE detection
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/api/detect
```

#### Statistics API
```http
GET /api/stats?hours=24

# Response:
{
  "detection_stats": {
    "total_detections": 1250,
    "compliant_detections": 1100,
    "violation_count": 150,
    "compliance_rate": 88.0
  },
  "stream_status": {...}
}
```

#### Alert Management
```http
GET /api/alerts?hours=12

# Get recent alerts
POST /api/test-alert

# Test alert system
```

### Python API

```python
import yaml
from src.detection.ppe_detector import PPEDetector
from src.alerts.alert_manager import AlertManager

# Load configuration
with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize detector
detector = PPEDetector(config)

# Detect PPE in image
import cv2
image = cv2.imread('test_image.jpg')
result = detector.detect_ppe(image)

print(f"Compliance: {result['compliance']['compliant']}")
print(f"Missing PPE: {result['compliance']['missing_ppe']}")
```

## Performance Optimization

### Hardware Recommendations

- **CPU**: Multi-core processor (Intel i7/AMD Ryzen 7 or better)
- **GPU**: NVIDIA RTX 3060 or better for real-time processing
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: SSD for model and data storage

### Software Optimization

1. **GPU Acceleration**
   ```yaml
   performance:
     use_gpu: true
     gpu_memory_fraction: 0.8
   ```

2. **Frame Processing**
   ```yaml
   video:
     frame_skip: 2  # Process every 2nd frame
     resize_width: 640
     resize_height: 480
   ```

3. **Concurrent Streams**
   ```yaml
   performance:
     max_concurrent_streams: 4
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or image size
   # Or use CPU processing
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Camera Access Issues**
   ```bash
   # Check camera permissions
   ls -la /dev/video*
   
   # Test camera
   python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
   ```

3. **Model Loading Errors**
   ```bash
   # Verify model file
   python -c "
   from ultralytics import YOLO
   model = YOLO('data/models/ppe_detection.pt')
   print('Model loaded successfully')
   "
   ```

### Logging and Debugging

1. **Enable Debug Logging**
   ```yaml
   logging:
     level: "DEBUG"
   ```

2. **Check Log Files**
   ```bash
   tail -f logs/ppe_detection.log
   ```

3. **Performance Monitoring**
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1
   
   # Monitor CPU/Memory
   htop
   ```

## Security Considerations

### Data Privacy
- All video processing is performed locally
- No video data is transmitted to external servers
- Personal identifiable information should be anonymized

### Network Security
- Use HTTPS for web dashboard in production
- Implement proper authentication for API access
- Secure RTSP stream credentials

### Access Control
- Restrict dashboard access to authorized personnel
- Use role-based permissions for different user types
- Audit log access and modifications

## Deployment

### Production Deployment

1. **Server Setup**
   ```bash
   # Install system dependencies
   sudo apt update
   sudo apt install python3.8 python3-pip git
   
   # Install NVIDIA drivers for GPU support
   sudo apt install nvidia-driver-470
   ```

2. **Service Configuration**
   ```bash
   # Create systemd service
   sudo nano /etc/systemd/system/ppe-detection.service
   
   # Enable and start service
   sudo systemctl enable ppe-detection
   sudo systemctl start ppe-detection
   ```

3. **Reverse Proxy (nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Docker Deployment

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "main.py", "--source", "dashboard"]
```

## Monitoring and Maintenance

### Health Checks
- Monitor system resource usage
- Check detection accuracy regularly
- Verify alert delivery systems

### Regular Maintenance
- Update models with new training data
- Review and tune detection thresholds
- Archive old detection logs
- Update system dependencies

### Performance Metrics
- Detection accuracy (mAP scores)
- Processing speed (FPS)
- System uptime and reliability
- Alert response times

## Support and Contact

For technical support, documentation updates, or feature requests:

- **Technical Lead**: [Contact Information]
- **Project Repository**: [Internal Git Repository]
- **Documentation**: [Internal Wiki/Confluence]
- **Bug Reports**: [Issue Tracking System]

## Changelog

### Version 1.0.0 (Initial Release)
- Core PPE detection functionality
- Multi-stream video processing
- Web-based monitoring dashboard
- Email and webhook alerts
- SQLite logging and analytics
- REST API for integration

### Future Enhancements
- Advanced analytics and reporting
- Mobile app for alerts
- Integration with existing IOCL systems
- Advanced AI models for specific PPE types
- Real-time dashboard with live video feeds
