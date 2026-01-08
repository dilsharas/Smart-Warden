#!/usr/bin/env python3
"""
Training Pipeline Monitoring and Alerting System.
Implements real-time training progress tracking, visualization, and automated alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for training monitoring."""
    # Monitoring intervals
    metrics_update_interval: float = 1.0  # seconds
    resource_update_interval: float = 5.0  # seconds
    alert_check_interval: float = 10.0  # seconds
    
    # Alert thresholds
    max_training_time_hours: float = 24.0
    min_accuracy_threshold: float = 0.5
    max_memory_usage_gb: float = 16.0
    max_cpu_usage_percent: float = 95.0
    stagnation_patience: int = 50  # epochs without improvement
    
    # Visualization
    enable_live_plots: bool = True
    plot_update_interval: float = 30.0  # seconds
    max_plot_points: int = 1000
    
    # Alerting
    enable_email_alerts: bool = False
    email_config: Dict = None
    enable_slack_alerts: bool = False
    slack_webhook_url: str = None
    
    # Logging
    log_to_file: bool = True
    log_file_path: str = "training_monitor.log"
    
    # Data retention
    max_history_size: int = 10000
    save_monitoring_data: bool = True
    monitoring_data_path: str = "monitoring_data.json"

@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    epoch: int
    timestamp: float
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    batch_time: float
    epoch_time: float

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage_gb: float
    gpu_usage: List[float]
    gpu_memory_gb: List[float]
    disk_usage_gb: float
    network_io: Dict[str, float]

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    timestamp: float
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    metric_name: str
    metric_value: Any
    threshold: Any

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts = {}
        self.alert_history = []
        
    def check_and_send_alert(self, alert: Alert):
        """Check if alert should be sent and send it."""
        # Avoid duplicate alerts
        if alert.alert_id in self.active_alerts:
            return
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"ALERT: {alert.title} - {alert.message}")
        
        # Send email alert
        if self.config.enable_email_alerts and self.config.email_config:
            self._send_email_alert(alert)
        
        # Send Slack alert
        if self.config.enable_slack_alerts and self.config.slack_webhook_url:
            self._send_slack_alert(alert)
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        try:
            email_config = self.config.email_config
            
            msg = MimeMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"Training Alert: {alert.title}"
            
            body = f"""
            Training Alert Notification
            
            Severity: {alert.severity.upper()}
            Title: {alert.title}
            Message: {alert.message}
            Metric: {alert.metric_name}
            Value: {alert.metric_value}
            Threshold: {alert.threshold}
            Time: {datetime.fromtimestamp(alert.timestamp)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls', True):
                server.starttls()
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert."""
        try:
            import requests
            
            color_map = {
                'info': '#36a64f',
                'warning': '#ff9500',
                'critical': '#ff0000'
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, '#36a64f'),
                    "title": f"Training Alert: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": str(alert.metric_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True}
                    ],
                    "timestamp": alert.timestamp
                }]
            }
            
            response = requests.post(self.config.slack_webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class TrainingMonitor:
    """
    Real-time training pipeline monitoring system.
    """
    
    def __init__(self, config: MonitoringConfig = None):
        """
        Initialize the training monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.start_time = None
        
        # Data storage
        self.training_metrics_history = deque(maxlen=self.config.max_history_size)
        self.resource_metrics_history = deque(maxlen=self.config.max_history_size)
        
        # Current metrics
        self.current_training_metrics = None
        self.current_resource_metrics = None
        
        # Performance tracking
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.training_start_time = None
        
        # Visualization
        self.live_plot_fig = None
        self.live_plot_axes = None
        
        logger.info("Initialized TrainingMonitor")
    
    def start_monitoring(self, training_start_callback: Optional[Callable] = None):
        """Start monitoring training process."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.training_start_time = time.time()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Initialize live plots if enabled
        if self.config.enable_live_plots:
            self._initialize_live_plots()
        
        logger.info("🚀 Training monitoring started")
        
        if training_start_callback:
            training_start_callback()
    
    def stop_monitoring(self):
        """Stop monitoring training process."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Save monitoring data
        if self.config.save_monitoring_data:
            self._save_monitoring_data()
        
        logger.info("⏹️ Training monitoring stopped")
    
    def update_training_metrics(self, metrics: TrainingMetrics):
        """Update training metrics."""
        metrics.timestamp = time.time()
        self.current_training_metrics = metrics
        self.training_metrics_history.append(metrics)
        
        # Check for improvements
        if metrics.val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = metrics.val_accuracy
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Log metrics
        logger.info(f"Epoch {metrics.epoch}: "
                   f"Train Loss: {metrics.train_loss:.4f}, "
                   f"Val Loss: {metrics.val_loss:.4f}, "
                   f"Val Acc: {metrics.val_accuracy:.4f}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_resource_update = 0
        last_alert_check = 0
        last_plot_update = 0
        
        while self.is_monitoring:
            current_time = time.time()
            
            # Update resource metrics
            if current_time - last_resource_update >= self.config.resource_update_interval:
                self._update_resource_metrics()
                last_resource_update = current_time
            
            # Check alerts
            if current_time - last_alert_check >= self.config.alert_check_interval:
                self._check_alerts()
                last_alert_check = current_time
            
            # Update live plots
            if (self.config.enable_live_plots and 
                current_time - last_plot_update >= self.config.plot_update_interval):
                self._update_live_plots()
                last_plot_update = current_time
            
            time.sleep(self.config.metrics_update_interval)
    
    def _update_resource_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU and memory
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_usage_gb = memory.used / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_gb = disk.used / (1024**3)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
            # GPU metrics (if available)
            gpu_usage = []
            gpu_memory_gb = []
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_usage.append(gpu.load * 100)
                    gpu_memory_gb.append(gpu.memoryUsed / 1024)
            except ImportError:
                pass
            
            # Create resource metrics
            resource_metrics = ResourceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage_gb=memory_usage_gb,
                gpu_usage=gpu_usage,
                gpu_memory_gb=gpu_memory_gb,
                disk_usage_gb=disk_usage_gb,
                network_io=network_io
            )
            
            self.current_resource_metrics = resource_metrics
            self.resource_metrics_history.append(resource_metrics)
            
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        current_time = time.time()
        
        # Training time alert
        if self.training_start_time:
            training_hours = (current_time - self.training_start_time) / 3600
            if training_hours > self.config.max_training_time_hours:
                alert = Alert(
                    alert_id="training_time_exceeded",
                    timestamp=current_time,
                    severity="warning",
                    title="Training Time Exceeded",
                    message=f"Training has been running for {training_hours:.1f} hours",
                    metric_name="training_time_hours",
                    metric_value=training_hours,
                    threshold=self.config.max_training_time_hours
                )
                self.alert_manager.check_and_send_alert(alert)
        
        # Accuracy alert
        if (self.current_training_metrics and 
            self.current_training_metrics.val_accuracy < self.config.min_accuracy_threshold):
            alert = Alert(
                alert_id="low_accuracy",
                timestamp=current_time,
                severity="warning",
                title="Low Validation Accuracy",
                message=f"Validation accuracy is {self.current_training_metrics.val_accuracy:.3f}",
                metric_name="val_accuracy",
                metric_value=self.current_training_metrics.val_accuracy,
                threshold=self.config.min_accuracy_threshold
            )
            self.alert_manager.check_and_send_alert(alert)
        
        # Memory usage alert
        if (self.current_resource_metrics and 
            self.current_resource_metrics.memory_usage_gb > self.config.max_memory_usage_gb):
            alert = Alert(
                alert_id="high_memory_usage",
                timestamp=current_time,
                severity="critical",
                title="High Memory Usage",
                message=f"Memory usage is {self.current_resource_metrics.memory_usage_gb:.1f} GB",
                metric_name="memory_usage_gb",
                metric_value=self.current_resource_metrics.memory_usage_gb,
                threshold=self.config.max_memory_usage_gb
            )
            self.alert_manager.check_and_send_alert(alert)
        
        # CPU usage alert
        if (self.current_resource_metrics and 
            self.current_resource_metrics.cpu_usage > self.config.max_cpu_usage_percent):
            alert = Alert(
                alert_id="high_cpu_usage",
                timestamp=current_time,
                severity="warning",
                title="High CPU Usage",
                message=f"CPU usage is {self.current_resource_metrics.cpu_usage:.1f}%",
                metric_name="cpu_usage",
                metric_value=self.current_resource_metrics.cpu_usage,
                threshold=self.config.max_cpu_usage_percent
            )
            self.alert_manager.check_and_send_alert(alert)
        
        # Training stagnation alert
        if self.epochs_without_improvement > self.config.stagnation_patience:
            alert = Alert(
                alert_id="training_stagnation",
                timestamp=current_time,
                severity="warning",
                title="Training Stagnation",
                message=f"No improvement for {self.epochs_without_improvement} epochs",
                metric_name="epochs_without_improvement",
                metric_value=self.epochs_without_improvement,
                threshold=self.config.stagnation_patience
            )
            self.alert_manager.check_and_send_alert(alert)
    
    def _initialize_live_plots(self):
        """Initialize live plotting."""
        plt.ion()  # Turn on interactive mode
        
        self.live_plot_fig, self.live_plot_axes = plt.subplots(2, 2, figsize=(15, 10))
        self.live_plot_fig.suptitle('Training Monitor - Live Metrics')
        
        # Configure subplots
        self.live_plot_axes[0, 0].set_title('Training & Validation Loss')
        self.live_plot_axes[0, 0].set_xlabel('Epoch')
        self.live_plot_axes[0, 0].set_ylabel('Loss')
        
        self.live_plot_axes[0, 1].set_title('Training & Validation Accuracy')
        self.live_plot_axes[0, 1].set_xlabel('Epoch')
        self.live_plot_axes[0, 1].set_ylabel('Accuracy')
        
        self.live_plot_axes[1, 0].set_title('System Resources')
        self.live_plot_axes[1, 0].set_xlabel('Time')
        self.live_plot_axes[1, 0].set_ylabel('Usage')
        
        self.live_plot_axes[1, 1].set_title('Training Progress')
        self.live_plot_axes[1, 1].set_xlabel('Epoch')
        self.live_plot_axes[1, 1].set_ylabel('Time (s)')
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _update_live_plots(self):
        """Update live plots with current data."""
        if not self.live_plot_fig or not self.training_metrics_history:
            return
        
        try:
            # Clear all axes
            for ax in self.live_plot_axes.flat:
                ax.clear()
            
            # Get recent data
            recent_training = list(self.training_metrics_history)[-self.config.max_plot_points:]
            recent_resources = list(self.resource_metrics_history)[-self.config.max_plot_points:]
            
            if recent_training:
                epochs = [m.epoch for m in recent_training]
                train_losses = [m.train_loss for m in recent_training]
                val_losses = [m.val_loss for m in recent_training]
                train_accs = [m.train_accuracy for m in recent_training]
                val_accs = [m.val_accuracy for m in recent_training]
                epoch_times = [m.epoch_time for m in recent_training]
                
                # Plot 1: Loss curves
                self.live_plot_axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
                self.live_plot_axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
                self.live_plot_axes[0, 0].set_title('Training & Validation Loss')
                self.live_plot_axes[0, 0].set_xlabel('Epoch')
                self.live_plot_axes[0, 0].set_ylabel('Loss')
                self.live_plot_axes[0, 0].legend()
                self.live_plot_axes[0, 0].grid(True)
                
                # Plot 2: Accuracy curves
                self.live_plot_axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc')
                self.live_plot_axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Acc')
                self.live_plot_axes[0, 1].set_title('Training & Validation Accuracy')
                self.live_plot_axes[0, 1].set_xlabel('Epoch')
                self.live_plot_axes[0, 1].set_ylabel('Accuracy')
                self.live_plot_axes[0, 1].legend()
                self.live_plot_axes[0, 1].grid(True)
                
                # Plot 4: Training time
                self.live_plot_axes[1, 1].plot(epochs, epoch_times, 'g-', label='Epoch Time')
                self.live_plot_axes[1, 1].set_title('Training Progress')
                self.live_plot_axes[1, 1].set_xlabel('Epoch')
                self.live_plot_axes[1, 1].set_ylabel('Time (s)')
                self.live_plot_axes[1, 1].legend()
                self.live_plot_axes[1, 1].grid(True)
            
            if recent_resources:
                timestamps = [(m.timestamp - self.start_time) / 60 for m in recent_resources]  # Minutes
                cpu_usage = [m.cpu_usage for m in recent_resources]
                memory_usage = [m.memory_usage_gb for m in recent_resources]
                
                # Plot 3: System resources
                ax3 = self.live_plot_axes[1, 0]
                ax3_twin = ax3.twinx()
                
                line1 = ax3.plot(timestamps, cpu_usage, 'b-', label='CPU %')
                line2 = ax3_twin.plot(timestamps, memory_usage, 'r-', label='Memory GB')
                
                ax3.set_title('System Resources')
                ax3.set_xlabel('Time (minutes)')
                ax3.set_ylabel('CPU Usage (%)', color='b')
                ax3_twin.set_ylabel('Memory Usage (GB)', color='r')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='upper left')
                
                ax3.grid(True)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            logger.error(f"Error updating live plots: {e}")
    
    def _save_monitoring_data(self):
        """Save monitoring data to file."""
        try:
            data = {
                'config': asdict(self.config),
                'training_metrics': [asdict(m) for m in self.training_metrics_history],
                'resource_metrics': [asdict(m) for m in self.resource_metrics_history],
                'alert_history': [asdict(a) for a in self.alert_manager.alert_history],
                'monitoring_duration': time.time() - self.start_time if self.start_time else 0
            }
            
            with open(self.config.monitoring_data_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Monitoring data saved to {self.config.monitoring_data_path}")
            
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def get_monitoring_summary(self) -> Dict:
        """Get comprehensive monitoring summary."""
        current_time = time.time()
        
        summary = {
            'monitoring_duration': current_time - self.start_time if self.start_time else 0,
            'total_epochs': len(self.training_metrics_history),
            'best_val_accuracy': self.best_val_accuracy,
            'epochs_without_improvement': self.epochs_without_improvement,
            'active_alerts': len(self.alert_manager.active_alerts),
            'total_alerts': len(self.alert_manager.alert_history)
        }
        
        if self.current_training_metrics:
            summary['current_training_metrics'] = asdict(self.current_training_metrics)
        
        if self.current_resource_metrics:
            summary['current_resource_metrics'] = asdict(self.current_resource_metrics)
        
        # Performance statistics
        if self.training_metrics_history:
            recent_metrics = list(self.training_metrics_history)[-10:]  # Last 10 epochs
            summary['recent_performance'] = {
                'avg_epoch_time': np.mean([m.epoch_time for m in recent_metrics]),
                'avg_val_accuracy': np.mean([m.val_accuracy for m in recent_metrics]),
                'accuracy_trend': 'improving' if len(recent_metrics) > 1 and 
                                recent_metrics[-1].val_accuracy > recent_metrics[0].val_accuracy else 'declining'
            }
        
        return summary


def main():
    """Example usage of TrainingMonitor."""
    # Configure monitoring
    config = MonitoringConfig(
        metrics_update_interval=1.0,
        resource_update_interval=2.0,
        enable_live_plots=True,
        max_training_time_hours=1.0,  # 1 hour for demo
        min_accuracy_threshold=0.7,
        stagnation_patience=5
    )
    
    # Initialize monitor
    monitor = TrainingMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate training process
    print("🚀 Starting simulated training with monitoring...")
    
    try:
        for epoch in range(20):
            # Simulate training metrics
            train_loss = 1.0 - (epoch * 0.04) + np.random.normal(0, 0.05)
            val_loss = 1.1 - (epoch * 0.035) + np.random.normal(0, 0.08)
            train_acc = 0.5 + (epoch * 0.02) + np.random.normal(0, 0.02)
            val_acc = 0.45 + (epoch * 0.025) + np.random.normal(0, 0.03)
            
            # Clamp values
            train_loss = max(0.1, train_loss)
            val_loss = max(0.1, val_loss)
            train_acc = min(0.95, max(0.4, train_acc))
            val_acc = min(0.95, max(0.4, val_acc))
            
            metrics = TrainingMetrics(
                epoch=epoch,
                timestamp=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                learning_rate=0.001 * (0.95 ** epoch),
                batch_time=0.5 + np.random.normal(0, 0.1),
                epoch_time=30 + np.random.normal(0, 5)
            )
            
            monitor.update_training_metrics(metrics)
            
            # Simulate epoch duration
            time.sleep(2)
            
            print(f"Epoch {epoch}: Val Acc = {val_acc:.3f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Get summary
        summary = monitor.get_monitoring_summary()
        
        print("\n📊 Monitoring Summary:")
        print(f"Duration: {summary['monitoring_duration']:.1f}s")
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Best validation accuracy: {summary['best_val_accuracy']:.3f}")
        print(f"Active alerts: {summary['active_alerts']}")
        print(f"Total alerts: {summary['total_alerts']}")
        
        if 'recent_performance' in summary:
            perf = summary['recent_performance']
            print(f"Average epoch time: {perf['avg_epoch_time']:.1f}s")
            print(f"Recent accuracy trend: {perf['accuracy_trend']}")
        
        print("\n✅ Training monitoring demonstration complete!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()