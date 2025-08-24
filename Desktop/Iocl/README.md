# PPE Detection System for IOCL

A computer vision system for detecting Personal Protective Equipment (PPE) compliance from CCTV/video feeds.

## Features

- **Real-time PPE Detection**: Analyze live video streams for PPE compliance
- **Multi-PPE Classification**: Detect helmets, safety vests, gloves, boots, and goggles
- **Alert System**: Generate alerts for non-compliance incidents
- **Web Dashboard**: Monitor detection results and statistics
- **Report Generation**: Generate compliance reports and analytics

## PPE Categories Detected

1. **Hard Hats/Helmets** - Head protection
2. **Safety Vests/Jackets** - High-visibility clothing
3. **Protective Gloves** - Hand protection
4. **Safety Boots** - Foot protection
5. **Safety Goggles** - Eye protection

## Project Structure

```
iocl-ppe-detection/
├── src/
│   ├── models/           # ML models and training
│   ├── detection/        # Core detection logic
│   ├── video/           # Video processing
│   ├── alerts/          # Alert system
│   └── dashboard/       # Web interface
├── data/
│   ├── datasets/        # Training datasets
│   ├── models/          # Trained model files
│   └── samples/         # Sample videos/images
├── config/              # Configuration files
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download pre-trained models (instructions in docs/)
4. Configure settings in `config/settings.yaml`

## Usage

### Real-time Detection
```bash
python src/main.py --source camera --device 0
```

### Video File Analysis
```bash
python src/main.py --source video --input path/to/video.mp4
```

### Web Dashboard
```bash
python src/dashboard/app.py
```

## Requirements

- Python 3.8+
- OpenCV 4.5+
- PyTorch/TensorFlow
- Flask (for web interface)
- NumPy, Pandas
- See `requirements.txt` for complete list

## Model Training

The system uses YOLO-based object detection models. Training instructions and datasets are available in the `docs/` directory.

## API Documentation

RESTful API endpoints for integration with existing IOCL systems:
- `/detect` - Single image/frame detection
- `/stream` - Real-time stream processing
- `/reports` - Generate compliance reports

## Contributing

Please read `CONTRIBUTING.md` for development guidelines and coding standards.

## License

This project is proprietary software developed for IOCL company.

## Support

For technical support and questions, contact the development team.
