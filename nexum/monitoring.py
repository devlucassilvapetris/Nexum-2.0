"""
Nexum v2.0 Monitoring System
Comprehensive logging and monitoring for all system components
"""

import logging
import time
import json
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import os

import psutil


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    timestamp: float
    level: LogLevel
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: float


@dataclass
class SystemAlert:
    id: str
    severity: str
    component: str
    message: str
    timestamp: float
    resolved: bool = False
    details: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] = self.counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric"""
        with self.lock:
            key = self._make_key(name, labels)
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)
            
            # Keep only last 1000 values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timer metric"""
        with self.lock:
            key = self._make_key(name, labels)
            if key not in self.timers:
                self.timers[key] = []
            self.timers[key].append(duration)
            
            # Keep only last 1000 values
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key with labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> List[Metric]:
        """Get all current metrics"""
        metrics = []
        timestamp = time.time()
        
        # Counters
        for key, value in self.counters.items():
            metrics.append(Metric(
                name=key,
                type=MetricType.COUNTER,
                value=value,
                labels={},
                timestamp=timestamp
            ))
        
        # Gauges
        for key, value in self.gauges.items():
            metrics.append(Metric(
                name=key,
                type=MetricType.GAUGE,
                value=value,
                labels={},
                timestamp=timestamp
            ))
        
        # Histograms (use average)
        for key, values in self.histograms.items():
            if values:
                avg_value = sum(values) / len(values)
                metrics.append(Metric(
                    name=key + "_avg",
                    type=MetricType.GAUGE,
                    value=avg_value,
                    labels={},
                    timestamp=timestamp
                ))
        
        # Timers (use average)
        for key, values in self.timers.items():
            if values:
                avg_value = sum(values) / len(values)
                metrics.append(Metric(
                    name=key + "_avg",
                    type=MetricType.GAUGE,
                    value=avg_value,
                    labels={},
                    timestamp=timestamp
                ))
        
        return metrics


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, SystemAlert] = {}
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def add_alert_rule(self, name: str, condition: str, severity: str, 
                      message_template: str, threshold: float = None):
        """Add alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message_template": message_template,
            "threshold": threshold
        }
        self.alert_rules.append(rule)
    
    def check_alerts(self, metrics: List[Metric], logs: List[LogEntry]):
        """Check alert conditions"""
        with self.lock:
            for rule in self.alert_rules:
                self._check_rule(rule, metrics, logs)
    
    def _check_rule(self, rule: Dict[str, Any], metrics: List[Metric], logs: List[LogEntry]):
        """Check individual alert rule"""
        condition = rule["condition"]
        threshold = rule.get("threshold")
        
        # Find relevant metrics
        relevant_metrics = [m for m in metrics if rule["name"] in m.name]
        
        for metric in relevant_metrics:
            triggered = False
            
            if condition == "greater_than" and threshold:
                triggered = metric.value > threshold
            elif condition == "less_than" and threshold:
                triggered = metric.value < threshold
            elif condition == "equals" and threshold:
                triggered = abs(metric.value - threshold) < 0.001
            elif condition == "error_logs":
                # Check for error logs in last minute
                recent_errors = [l for l in logs 
                               if l.level == LogLevel.ERROR and 
                               time.time() - l.timestamp < 60]
                triggered = len(recent_errors) > 0
            
            if triggered:
                alert_id = f"{rule['name']}_{metric.name}"
                
                if alert_id not in self.alerts:
                    # Create new alert
                    message = rule["message_template"].format(
                        metric_name=metric.name,
                        metric_value=metric.value,
                        threshold=threshold
                    )
                    
                    alert = SystemAlert(
                        id=alert_id,
                        severity=rule["severity"],
                        component=metric.name,
                        message=message,
                        timestamp=time.time(),
                        details={"metric": asdict(metric), "rule": rule}
                    )
                    
                    self.alerts[alert_id] = alert
                    
                    # Trigger callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logging.error(f"Alert callback error: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active alerts"""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def register_callback(self, callback: Callable[[SystemAlert], None]):
        """Register alert callback"""
        self.alert_callbacks.append(callback)


class SystemMonitor:
    """
    Comprehensive system monitoring for Nexum v2.0
    Handles logging, metrics collection, and alerting
    """
    
    def __init__(self, log_file: str = "nexum.log", max_file_size_mb: int = 100):
        self.log_file = log_file
        self.max_file_size = max_file_size * 1024 * 1024
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Log storage
        self.logs: List[LogEntry] = []
        self.max_logs = 10000
        self.log_callbacks: List[Callable[[LogEntry], None]] = []
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_interval = 1.0  # seconds
        
        # Setup logging
        self._setup_logging()
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_logging(self):
        """Setup Python logging"""
        # Create logger
        self.logger = logging.getLogger("nexum")
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # CPU usage alert
        self.alert_manager.add_alert_rule(
            "high_cpu",
            "greater_than",
            "WARNING",
            "High CPU usage: {metric_value:.1f}%",
            80.0
        )
        
        # Memory usage alert
        self.alert_manager.add_alert_rule(
            "high_memory",
            "greater_than",
            "WARNING",
            "High memory usage: {metric_value:.1f}%",
            85.0
        )
        
        # Error logs alert
        self.alert_manager.add_alert_rule(
            "error_logs",
            "error_logs",
            "ERROR",
            "System errors detected"
        )
        
        # Disk space alert
        self.alert_manager.add_alert_rule(
            "low_disk_space",
            "less_than",
            "CRITICAL",
            "Low disk space: {metric_value:.1f}%",
            10.0
        )
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.info("monitoring", "System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.info("monitoring", "System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alerts
                metrics = self.metrics_collector.get_metrics()
                self.alert_manager.check_alerts(metrics, self.logs)
                
                # Cleanup old logs
                self._cleanup_old_logs()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.error("monitoring", f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_available_gb", 
                                         memory.available / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_free_percent = (disk.free / disk.total) * 100
            self.metrics_collector.set_gauge("system_disk_free_percent", disk_free_percent)
            
            # Process metrics
            process = psutil.Process()
            self.metrics_collector.set_gauge("nexum_memory_mb", 
                                         process.memory_info().rss / (1024**2))
            self.metrics_collector.set_gauge("nexum_cpu_percent", 
                                         process.cpu_percent())
            
        except Exception as e:
            self.error("monitoring", f"Metrics collection error: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up old log entries"""
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def debug(self, component: str, message: str, details: Dict[str, Any] = None):
        """Log debug message"""
        self._log(LogLevel.DEBUG, component, message, details)
    
    def info(self, component: str, message: str, details: Dict[str, Any] = None):
        """Log info message"""
        self._log(LogLevel.INFO, component, message, details)
    
    def warning(self, component: str, message: str, details: Dict[str, Any] = None):
        """Log warning message"""
        self._log(LogLevel.WARNING, component, message, details)
    
    def error(self, component: str, message: str, details: Dict[str, Any] = None):
        """Log error message"""
        self._log(LogLevel.ERROR, component, message, details)
    
    def critical(self, component: str, message: str, details: Dict[str, Any] = None):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, component, message, details)
    
    def _log(self, level: LogLevel, component: str, message: str, 
              details: Dict[str, Any] = None):
        """Internal logging method"""
        timestamp = time.time()
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=timestamp,
            level=level,
            component=component,
            message=message,
            details=details
        )
        
        # Add to storage
        self.logs.append(log_entry)
        
        # Log to Python logger
        log_method = getattr(self.logger, level.value.lower())
        log_method(f"[{component}] {message}")
        
        # Update metrics
        self.metrics_collector.increment_counter(f"logs_total", 
                                            labels={"level": level.value, 
                                                   "component": component})
        
        # Trigger callbacks
        for callback in self.log_callbacks:
            try:
                callback(log_entry)
            except Exception as e:
                self.logger.error(f"Log callback error: {e}")
    
    def timer(self, name: str):
        """Create a timer context manager"""
        return TimerContext(self, name)
    
    def get_logs(self, level: LogLevel = None, component: str = None, 
                 limit: int = None) -> List[LogEntry]:
        """Get filtered logs"""
        filtered_logs = self.logs
        
        if level:
            filtered_logs = [l for l in filtered_logs if l.level == level]
        
        if component:
            filtered_logs = [l for l in filtered_logs if l.component == component]
        
        if limit:
            filtered_logs = filtered_logs[-limit:]
        
        return filtered_logs
    
    def get_metrics(self) -> List[Metric]:
        """Get all metrics"""
        return self.metrics_collector.get_metrics()
    
    def get_alerts(self, active_only: bool = True) -> List[SystemAlert]:
        """Get alerts"""
        if active_only:
            return self.alert_manager.get_active_alerts()
        else:
            return list(self.alert_manager.alerts.values())
    
    def register_log_callback(self, callback: Callable[[LogEntry], None]):
        """Register log callback"""
        self.log_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Register alert callback"""
        self.alert_manager.register_callback(callback)
    
    def export_logs(self, filename: str, format: str = "json"):
        """Export logs to file"""
        logs_data = [asdict(log) for log in self.logs]
        
        if format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(logs_data, f, indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            with open(filename, 'w', newline='') as f:
                if logs_data:
                    writer = csv.DictWriter(f, fieldnames=logs_data[0].keys())
                    writer.writeheader()
                    writer.writerows(logs_data)
    
    def export_metrics(self, filename: str, format: str = "json"):
        """Export metrics to file"""
        metrics_data = [asdict(metric) for metric in self.get_metrics()]
        
        if format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        elif format.lower() == "prometheus":
            # Export in Prometheus format
            with open(filename, 'w') as f:
                for metric in metrics_data:
                    f.write(f"{metric['name']} {metric['value']}\n")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        recent_logs = self.get_logs(limit=100)
        recent_errors = [l for l in recent_logs if l.level == LogLevel.ERROR]
        recent_warnings = [l for l in recent_logs if l.level == LogLevel.WARNING]
        
        active_alerts = self.get_alerts(active_only=True)
        
        metrics = self.get_metrics()
        cpu_metric = next((m for m in metrics if "cpu" in m.name), None)
        memory_metric = next((m for m in metrics if "memory" in m.name), None)
        
        return {
            "status": "healthy" if len(active_alerts) == 0 else "degraded",
            "timestamp": time.time(),
            "recent_errors": len(recent_errors),
            "recent_warnings": len(recent_warnings),
            "active_alerts": len(active_alerts),
            "cpu_usage": cpu_metric.value if cpu_metric else None,
            "memory_usage": memory_metric.value if memory_metric else None,
            "total_logs": len(self.logs),
            "uptime": time.time() - (self.logs[0].timestamp if self.logs else time.time())
        }


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: SystemMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.metrics_collector.record_timer(self.name, duration)


# Global monitor instance
_global_monitor: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor


def init_monitoring(log_file: str = "nexum.log", 
                  max_file_size_mb: int = 100) -> SystemMonitor:
    """Initialize global monitoring"""
    global _global_monitor
    _global_monitor = SystemMonitor(log_file, max_file_size_mb)
    _global_monitor.start_monitoring()
    return _global_monitor


# Decorators for easy monitoring
def monitor_function(name: str = None):
    """Decorator to monitor function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            with monitor.timer(func_name):
                try:
                    result = func(*args, **kwargs)
                    monitor.debug("function", f"Function {func_name} completed successfully")
                    return result
                except Exception as e:
                    monitor.error("function", f"Function {func_name} failed: {e}")
                    raise
        
        return wrapper
    return decorator


def log_performance(component: str, operation: str):
    """Decorator to log performance metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                monitor.error(component, f"Operation {operation} failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                monitor.metrics_collector.record_timer(
                    f"{component}_{operation}_duration", 
                    duration
                )
                monitor.metrics_collector.increment_counter(
                    f"{component}_{operation}_calls",
                    labels={"success": str(success)}
                )
                
                if success:
                    monitor.info(component, 
                              f"Operation {operation} completed in {duration:.3f}s")
            
            return result
        return wrapper
    return decorator
