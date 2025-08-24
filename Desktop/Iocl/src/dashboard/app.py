"""
Web Dashboard for PPE Detection System
Flask-based web interface for monitoring and configuration
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import logging
import threading
import time
from datetime import datetime, timedelta
import base64
import cv2
import numpy as np
from typing import Dict, Any
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detection.ppe_detector import PPEDetector
from src.video.video_processor import StreamManager
from src.alerts.alert_manager import AlertManager


class DashboardApp:
    """Main dashboard application"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ppe-detection-secret-key'
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.detector = PPEDetector(config)
        self.alert_manager = AlertManager(config)
        self.stream_manager = StreamManager(config, self.detector, self.alert_manager)
        
        # Dashboard state
        self.dashboard_stats = {}
        self.active_streams = {}
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_handlers()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("Dashboard application initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html', config=self.config)
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get system statistics"""
            try:
                stats = self.alert_manager.get_detection_stats(24)
                stream_status = self.stream_manager.get_stream_status()
                
                return jsonify({
                    'detection_stats': stats,
                    'stream_status': stream_status,
                    'system_status': 'active',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error getting stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get recent alerts"""
            try:
                hours = request.args.get('hours', 24, type=int)
                alerts = self.alert_manager.get_recent_alerts(hours)
                return jsonify({'alerts': alerts})
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/streams', methods=['GET', 'POST'])
        def manage_streams():
            """Manage video streams"""
            if request.method == 'GET':
                return jsonify(self.stream_manager.get_stream_status())
            
            elif request.method == 'POST':
                try:
                    data = request.get_json()
                    action = data.get('action')
                    stream_id = data.get('stream_id')
                    
                    if action == 'start':
                        source = data.get('source')
                        self.stream_manager.add_stream(stream_id, source)
                        success = self.stream_manager.start_stream(stream_id)
                        return jsonify({'success': success})
                    
                    elif action == 'stop':
                        success = self.stream_manager.stop_stream(stream_id)
                        return jsonify({'success': success})
                    
                    else:
                        return jsonify({'error': 'Invalid action'}), 400
                        
                except Exception as e:
                    self.logger.error(f"Error managing streams: {e}")
                    return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/detect', methods=['POST'])
        def detect_image():
            """Detect PPE in uploaded image"""
            try:
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No image selected'}), 400
                
                # Read image
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Invalid image format'}), 400
                
                # Detect PPE
                result = self.detector.detect_ppe(image)
                
                # Draw detections
                processed_image = self.detector.draw_detections(
                    image, result['detections'], result['compliance']
                )
                
                # Encode processed image
                _, buffer = cv2.imencode('.jpg', processed_image)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    'detections': result['detections'],
                    'compliance': result['compliance'],
                    'processed_image': processed_image_base64
                })
                
            except Exception as e:
                self.logger.error(f"Error in image detection: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/test-alert', methods=['POST'])
        def test_alert():
            """Test alert system"""
            try:
                success = self.alert_manager.test_alert_system()
                return jsonify({'success': success})
            except Exception as e:
                self.logger.error(f"Error testing alerts: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info("Dashboard client connected")
            emit('status', {'message': 'Connected to PPE Detection System'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Dashboard client disconnected")
        
        @self.socketio.on('request_stats')
        def handle_stats_request():
            """Handle request for statistics"""
            try:
                stats = self.alert_manager.get_detection_stats(24)
                emit('stats_update', stats)
            except Exception as e:
                self.logger.error(f"Error sending stats: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks for real-time updates"""
        
        def update_dashboard():
            """Periodically update dashboard with latest data"""
            while True:
                try:
                    # Get latest stats
                    stats = self.alert_manager.get_detection_stats(1)  # Last hour
                    stream_status = self.stream_manager.get_stream_status()
                    
                    # Emit to all connected clients
                    self.socketio.emit('stats_update', {
                        'detection_stats': stats,
                        'stream_status': stream_status,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Get recent alerts
                    recent_alerts = self.alert_manager.get_recent_alerts(1)
                    if recent_alerts:
                        self.socketio.emit('new_alerts', {'alerts': recent_alerts})
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in dashboard update: {e}")
                    time.sleep(10)
        
        # Start background thread
        update_thread = threading.Thread(target=update_dashboard, daemon=True)
        update_thread.start()
        
        self.logger.info("Background tasks started")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard application"""
        self.logger.info(f"Starting dashboard on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def create_app(config: Dict) -> Flask:
    """Create and configure Flask application"""
    dashboard = DashboardApp(config)
    return dashboard.app


# Template files would be created separately
def create_dashboard_templates():
    """Create HTML templates for the dashboard"""
    
    # Create templates directory
    templates_dir = "src/dashboard/templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    # Main dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IOCL PPE Detection System</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .status-card { margin-bottom: 20px; }
        .alert-item { border-left: 4px solid #dc3545; }
        .compliant { color: #28a745; }
        .non-compliant { color: #dc3545; }
        .live-indicator { 
            display: inline-block; 
            width: 10px; 
            height: 10px; 
            background: #28a745; 
            border-radius: 50%; 
            animation: pulse 2s infinite; 
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <span class="live-indicator"></span>
                IOCL PPE Detection System
            </span>
            <span class="navbar-text" id="connection-status">Connected</span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <!-- Statistics Cards -->
            <div class="col-md-3">
                <div class="card status-card">
                    <div class="card-body">
                        <h5 class="card-title">Compliance Rate</h5>
                        <h2 class="text-success" id="compliance-rate">--</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card status-card">
                    <div class="card-body">
                        <h5 class="card-title">Total Detections</h5>
                        <h2 class="text-info" id="total-detections">--</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card status-card">
                    <div class="card-body">
                        <h5 class="card-title">Violations</h5>
                        <h2 class="text-warning" id="violations">--</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card status-card">
                    <div class="card-body">
                        <h5 class="card-title">Active Alerts</h5>
                        <h2 class="text-danger" id="active-alerts">--</h2>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- Live Stream Status -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Live Streams</h5>
                    </div>
                    <div class="card-body" id="stream-status">
                        <div class="text-muted">No active streams</div>
                    </div>
                </div>
            </div>

            <!-- Recent Alerts -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Alerts</h5>
                    </div>
                    <div class="card-body" id="recent-alerts" style="max-height: 300px; overflow-y: auto;">
                        <div class="text-muted">No recent alerts</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <!-- Image Upload for Testing -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Test PPE Detection</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-form" enctype="multipart/form-data">
                            <div class="mb-3">
                                <input type="file" class="form-control" id="image-input" accept="image/*">
                            </div>
                            <button type="submit" class="btn btn-primary">Detect PPE</button>
                        </form>
                        <div id="detection-result" class="mt-3"></div>
                    </div>
                </div>
            </div>

            <!-- System Controls -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>System Controls</h5>
                    </div>
                    <div class="card-body">
                        <button class="btn btn-success me-2" onclick="testAlerts()">Test Alerts</button>
                        <button class="btn btn-info me-2" onclick="refreshStats()">Refresh Stats</button>
                        <button class="btn btn-secondary" onclick="downloadReport()">Download Report</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = 'Connected';
            document.getElementById('connection-status').className = 'navbar-text text-success';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('connection-status').className = 'navbar-text text-danger';
        });
        
        socket.on('stats_update', function(data) {
            updateDashboard(data);
        });
        
        socket.on('new_alerts', function(data) {
            updateAlerts(data.alerts);
        });
        
        // Update dashboard with new data
        function updateDashboard(data) {
            const stats = data.detection_stats || {};
            
            document.getElementById('compliance-rate').textContent = (stats.compliance_rate || 0) + '%';
            document.getElementById('total-detections').textContent = stats.total_detections || 0;
            document.getElementById('violations').textContent = stats.violation_count || 0;
            document.getElementById('active-alerts').textContent = stats.total_alerts || 0;
            
            // Update stream status
            const streamStatus = data.stream_status || {};
            updateStreamStatus(streamStatus);
        }
        
        function updateStreamStatus(streams) {
            const container = document.getElementById('stream-status');
            if (Object.keys(streams).length === 0) {
                container.innerHTML = '<div class="text-muted">No active streams</div>';
                return;
            }
            
            let html = '';
            for (const [streamId, stream] of Object.entries(streams)) {
                const statusClass = stream.status === 'active' ? 'text-success' : 'text-secondary';
                html += `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span>${streamId}</span>
                        <span class="${statusClass}">${stream.status}</span>
                    </div>
                `;
            }
            container.innerHTML = html;
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('recent-alerts');
            if (!alerts || alerts.length === 0) {
                return;
            }
            
            let html = '';
            alerts.slice(0, 5).forEach(alert => {
                const alertData = JSON.parse(alert.missing_ppe || '[]');
                html += `
                    <div class="alert-item p-2 mb-2 border rounded">
                        <div class="d-flex justify-content-between">
                            <strong>${alert.alert_type}</strong>
                            <small>${new Date(alert.timestamp).toLocaleTimeString()}</small>
                        </div>
                        <div class="text-muted small">${alert.message}</div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }
        
        // Image upload for testing
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('image-input');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            fetch('/api/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayDetectionResult(data);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error processing image');
            });
        });
        
        function displayDetectionResult(data) {
            const container = document.getElementById('detection-result');
            
            if (data.error) {
                container.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                return;
            }
            
            const compliance = data.compliance;
            const complianceClass = compliance.compliant ? 'alert-success' : 'alert-danger';
            
            container.innerHTML = `
                <div class="alert ${complianceClass}">
                    <strong>Result:</strong> ${compliance.message}
                </div>
                <div class="mt-2">
                    <strong>Detections:</strong> ${data.detections.length}
                </div>
                ${data.processed_image ? `<img src="data:image/jpeg;base64,${data.processed_image}" class="img-fluid mt-2" style="max-height: 300px;">` : ''}
            `;
        }
        
        // System control functions
        function testAlerts() {
            fetch('/api/test-alert', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                alert(data.success ? 'Test alert sent successfully' : 'Error sending test alert');
            });
        }
        
        function refreshStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                updateDashboard(data);
            });
        }
        
        function downloadReport() {
            // Implementation for downloading reports
            alert('Report download feature coming soon');
        }
        
        // Load initial data
        refreshStats();
        
        // Request periodic updates
        setInterval(() => {
            socket.emit('request_stats');
        }, 10000); // Every 10 seconds
    </script>
</body>
</html>
    """
    
    with open(f"{templates_dir}/dashboard.html", "w") as f:
        f.write(dashboard_html)


if __name__ == "__main__":
    # Create templates if running directly
    create_dashboard_templates()
