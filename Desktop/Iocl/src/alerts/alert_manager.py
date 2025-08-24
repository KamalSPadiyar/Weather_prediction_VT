"""
Alert Management System
Handles notifications and alerts for PPE compliance violations
"""

import logging
import smtplib
import json
import sqlite3
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
import threading
import queue
import time


class AlertManager:
    """Manages alerts for PPE compliance violations"""
    
    def __init__(self, config: Dict):
        """Initialize alert manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.alerts_config = config['alerts']
        self.is_enabled = self.alerts_config['enabled']
        self.alert_threshold = self.alerts_config['alert_threshold']
        
        # Alert tracking
        self.violation_buffer = []
        self.last_alert_time = {}
        self.alert_cooldown = 60  # seconds between similar alerts
        
        # Database setup
        self.db_path = config['database']['path']
        self._setup_database()
        
        # Alert queue for async processing
        self.alert_queue = queue.Queue()
        self.alert_thread = None
        
        if self.is_enabled:
            self._start_alert_processor()
        
        self.logger.info("Alert Manager initialized")
    
    def _setup_database(self):
        """Setup SQLite database for logging alerts"""
        try:
            # Create directory if it doesn't exist
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    alert_type TEXT,
                    message TEXT,
                    frame_number INTEGER,
                    missing_ppe TEXT,
                    confidence_scores TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create detection logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    frame_number INTEGER,
                    person_detected BOOLEAN,
                    compliance_status BOOLEAN,
                    detected_ppe TEXT,
                    missing_ppe TEXT,
                    detection_details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
    
    def process_detection(self, detection_result: Dict, frame_number: int):
        """Process detection result and trigger alerts if needed"""
        if not self.is_enabled:
            return
        
        try:
            # Log detection to database
            self._log_detection(detection_result, frame_number)
            
            compliance = detection_result['compliance']
            
            if not compliance['compliant']:
                # Add to violation buffer
                self.violation_buffer.append({
                    'timestamp': datetime.now(),
                    'frame_number': frame_number,
                    'missing_ppe': compliance['missing_ppe'],
                    'message': compliance['message']
                })
                
                # Keep buffer size manageable
                if len(self.violation_buffer) > self.alert_threshold * 2:
                    self.violation_buffer = self.violation_buffer[-self.alert_threshold:]
                
                # Check if alert should be triggered
                if self._should_trigger_alert():
                    self._trigger_alert(detection_result, frame_number)
            else:
                # Clear buffer on compliance
                self.violation_buffer.clear()
        
        except Exception as e:
            self.logger.error(f"Error processing detection for alerts: {e}")
    
    def _should_trigger_alert(self) -> bool:
        """Determine if an alert should be triggered"""
        if len(self.violation_buffer) < self.alert_threshold:
            return False
        
        # Check recent violations (last N frames)
        recent_violations = [
            v for v in self.violation_buffer 
            if (datetime.now() - v['timestamp']).seconds < 10
        ]
        
        if len(recent_violations) >= self.alert_threshold:
            # Check cooldown
            violation_type = str(sorted(recent_violations[-1]['missing_ppe']))
            current_time = datetime.now()
            
            if violation_type in self.last_alert_time:
                time_diff = (current_time - self.last_alert_time[violation_type]).seconds
                if time_diff < self.alert_cooldown:
                    return False
            
            self.last_alert_time[violation_type] = current_time
            return True
        
        return False
    
    def _trigger_alert(self, detection_result: Dict, frame_number: int):
        """Trigger alert for PPE violation"""
        try:
            compliance = detection_result['compliance']
            
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'type': 'PPE_VIOLATION',
                'frame_number': frame_number,
                'missing_ppe': compliance['missing_ppe'],
                'message': compliance['message'],
                'severity': self._determine_severity(compliance['missing_ppe']),
                'location': 'Camera_01',  # Would be dynamic in real system
                'detection_count': detection_result['detection_count']
            }
            
            # Add to alert queue for async processing
            self.alert_queue.put(alert_data)
            
            # Log alert to database
            self._log_alert(alert_data)
            
            self.logger.warning(f"PPE violation alert triggered: {alert_data['message']}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def _determine_severity(self, missing_ppe: List[str]) -> str:
        """Determine alert severity based on missing PPE"""
        critical_ppe = {'helmet', 'safety_vest'}
        high_risk_ppe = {'safety_boots', 'safety_goggles'}
        
        missing_set = set(missing_ppe)
        
        if missing_set.intersection(critical_ppe):
            return 'CRITICAL'
        elif missing_set.intersection(high_risk_ppe):
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _start_alert_processor(self):
        """Start background thread for processing alerts"""
        def process_alerts():
            while True:
                try:
                    alert_data = self.alert_queue.get(timeout=1)
                    self._send_notifications(alert_data)
                    self.alert_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in alert processor: {e}")
        
        self.alert_thread = threading.Thread(target=process_alerts, daemon=True)
        self.alert_thread.start()
        
        self.logger.info("Alert processor started")
    
    def _send_notifications(self, alert_data: Dict):
        """Send notifications via configured methods"""
        methods = self.alerts_config['notification_methods']
        
        for method in methods:
            try:
                if method == 'email':
                    self._send_email_alert(alert_data)
                elif method == 'webhook':
                    self._send_webhook_alert(alert_data)
                elif method == 'dashboard':
                    self._send_dashboard_alert(alert_data)
            except Exception as e:
                self.logger.error(f"Error sending {method} notification: {e}")
    
    def _send_email_alert(self, alert_data: Dict):
        """Send email alert"""
        if 'email' not in self.alerts_config:
            return
        
        email_config = self.alerts_config['email']
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = f"PPE Violation Alert - {alert_data['severity']}"
        
        # Email body
        body = f"""
        PPE Violation Detected - IOCL Safety System
        
        Time: {alert_data['timestamp']}
        Location: {alert_data['location']}
        Severity: {alert_data['severity']}
        
        Violation Details:
        - Missing PPE: {', '.join(alert_data['missing_ppe'])}
        - Message: {alert_data['message']}
        - Frame Number: {alert_data['frame_number']}
        
        Please take immediate action to ensure worker safety.
        
        This is an automated alert from the IOCL PPE Detection System.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        # Note: In production, use proper authentication
        # server.login(email_config['username'], email_config['password'])
        server.send_message(msg)
        server.quit()
        
        self.logger.info(f"Email alert sent for {alert_data['type']}")
    
    def _send_webhook_alert(self, alert_data: Dict):
        """Send webhook alert to external system"""
        # Example webhook implementation
        webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"  # Configure as needed
        
        payload = {
            "text": f"🚨 PPE Violation Alert - {alert_data['severity']}",
            "attachments": [
                {
                    "color": "danger" if alert_data['severity'] == 'CRITICAL' else "warning",
                    "fields": [
                        {
                            "title": "Location",
                            "value": alert_data['location'],
                            "short": True
                        },
                        {
                            "title": "Missing PPE",
                            "value": ', '.join(alert_data['missing_ppe']),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert_data['timestamp'],
                            "short": False
                        }
                    ]
                }
            ]
        }
        
        # Uncomment for actual webhook sending
        # response = requests.post(webhook_url, json=payload, timeout=10)
        # response.raise_for_status()
        
        self.logger.info(f"Webhook alert sent for {alert_data['type']}")
    
    def _send_dashboard_alert(self, alert_data: Dict):
        """Send alert to dashboard (via WebSocket or similar)"""
        # This would integrate with the web dashboard
        # For now, just log the alert
        self.logger.info(f"Dashboard alert: {alert_data}")
    
    def _log_detection(self, detection_result: Dict, frame_number: int):
        """Log detection result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            compliance = detection_result['compliance']
            
            cursor.execute('''
                INSERT INTO detection_logs 
                (timestamp, frame_number, person_detected, compliance_status, 
                 detected_ppe, missing_ppe, detection_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                frame_number,
                len(detection_result['detections']) > 0,
                compliance['compliant'],
                json.dumps(compliance.get('detected_ppe', [])),
                json.dumps(compliance['missing_ppe']),
                json.dumps(detection_result['detections'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging detection: {e}")
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (timestamp, alert_type, message, frame_number, missing_ppe, confidence_scores)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert_data['timestamp'],
                alert_data['type'],
                alert_data['message'],
                alert_data['frame_number'],
                json.dumps(alert_data['missing_ppe']),
                json.dumps({})  # Placeholder for confidence scores
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            columns = [description[0] for description in cursor.description]
            alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def get_detection_stats(self, hours: int = 24) -> Dict:
        """Get detection statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Total detections
            cursor.execute('SELECT COUNT(*) FROM detection_logs WHERE timestamp > ?', (cutoff_time,))
            total_detections = cursor.fetchone()[0]
            
            # Compliant detections
            cursor.execute('SELECT COUNT(*) FROM detection_logs WHERE timestamp > ? AND compliance_status = 1', (cutoff_time,))
            compliant_detections = cursor.fetchone()[0]
            
            # Total alerts
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE timestamp > ?', (cutoff_time,))
            total_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            compliance_rate = (compliant_detections / total_detections * 100) if total_detections > 0 else 0
            
            return {
                'total_detections': total_detections,
                'compliant_detections': compliant_detections,
                'violation_count': total_detections - compliant_detections,
                'compliance_rate': round(compliance_rate, 2),
                'total_alerts': total_alerts,
                'period_hours': hours
            }
            
        except Exception as e:
            self.logger.error(f"Error getting detection stats: {e}")
            return {}
    
    def test_alert_system(self) -> bool:
        """Test the alert system"""
        try:
            test_alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'TEST_ALERT',
                'frame_number': 0,
                'missing_ppe': ['helmet'],
                'message': 'Test alert from PPE detection system',
                'severity': 'LOW',
                'location': 'Test_Location',
                'detection_count': 1
            }
            
            self.alert_queue.put(test_alert)
            self.logger.info("Test alert sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing alert system: {e}")
            return False
