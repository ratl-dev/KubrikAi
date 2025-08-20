"""
Metrics Logger - Comprehensive logging and metrics collection for 
the intelligent routing system.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, Counter
import threading
from datetime import datetime, timedelta

try:
    from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events to log"""
    QUERY_ROUTED = "query_routed"
    VALIDATION_COMPLETED = "validation_completed" 
    SECURITY_CHECK = "security_check"
    QUERY_EXECUTED = "query_executed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class MetricEvent:
    """A single metric event"""
    event_type: EventType
    timestamp: float
    user_id: str
    database_id: Optional[str] = None
    query_type: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricsLogger:
    """
    Collects and manages metrics for the intelligent routing system.
    Supports both local logging and Prometheus metrics.
    """
    
    def __init__(self, enable_prometheus: bool = True):
        """
        Initialize the metrics logger.
        
        Args:
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.events: List[MetricEvent] = []
        self.lock = threading.RLock()
        
        # In-memory aggregated metrics
        self.query_counts = defaultdict(int)
        self.database_usage = defaultdict(int)
        self.routing_decisions = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.performance_stats = defaultdict(list)
        
        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics collectors."""
        try:
            self.prom_query_counter = PrometheusCounter(
                'kubrik_queries_total',
                'Total number of queries processed',
                ['user_id', 'database_id', 'query_type', 'status']
            )
            
            self.prom_routing_duration = Histogram(
                'kubrik_routing_duration_seconds',
                'Time spent on routing decisions',
                ['database_id', 'query_type']
            )
            
            self.prom_validation_duration = Histogram(
                'kubrik_validation_duration_seconds', 
                'Time spent on query validation',
                ['database_id', 'validation_level']
            )
            
            self.prom_active_queries = Gauge(
                'kubrik_active_queries',
                'Number of currently active queries',
                ['database_id']
            )
            
            self.prom_database_usage = PrometheusCounter(
                'kubrik_database_usage_total',
                'Total database usage by database',
                ['database_id', 'database_type']
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Prometheus metrics: {e}")
            self.enable_prometheus = False
    
    def log_event(self, event: MetricEvent) -> None:
        """
        Log a metric event.
        
        Args:
            event: Event to log
        """
        with self.lock:
            self.events.append(event)
            
            # Update aggregated metrics
            self._update_aggregated_metrics(event)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self._update_prometheus_metrics(event)
            
            # Log to standard logger
            self._log_to_standard_logger(event)
    
    def _update_aggregated_metrics(self, event: MetricEvent) -> None:
        """Update in-memory aggregated metrics."""
        # Query counts by type
        if event.query_type:
            self.query_counts[event.query_type] += 1
        
        # Database usage
        if event.database_id:
            self.database_usage[event.database_id] += 1
        
        # Routing decisions
        if event.event_type == EventType.QUERY_ROUTED and event.database_id:
            self.routing_decisions[event.database_id] += 1
        
        # Error counts
        if not event.success and event.error_message:
            error_key = event.error_message.split(':')[0]  # First part of error
            self.error_counts[error_key] += 1
        
        # Performance stats
        if event.duration_ms is not None:
            perf_key = f"{event.event_type.value}_{event.database_id or 'unknown'}"
            self.performance_stats[perf_key].append(event.duration_ms)
            
            # Keep only last 1000 measurements
            if len(self.performance_stats[perf_key]) > 1000:
                self.performance_stats[perf_key] = self.performance_stats[perf_key][-1000:]
    
    def _update_prometheus_metrics(self, event: MetricEvent) -> None:
        """Update Prometheus metrics."""
        try:
            # Query counter
            status = 'success' if event.success else 'error'
            self.prom_query_counter.labels(
                user_id=event.user_id,
                database_id=event.database_id or 'unknown',
                query_type=event.query_type or 'unknown',
                status=status
            ).inc()
            
            # Duration histograms
            if event.duration_ms is not None:
                duration_seconds = event.duration_ms / 1000.0
                
                if event.event_type == EventType.QUERY_ROUTED:
                    self.prom_routing_duration.labels(
                        database_id=event.database_id or 'unknown',
                        query_type=event.query_type or 'unknown'
                    ).observe(duration_seconds)
                
                elif event.event_type == EventType.VALIDATION_COMPLETED:
                    validation_level = event.metadata.get('validation_level', 'unknown') if event.metadata else 'unknown'
                    self.prom_validation_duration.labels(
                        database_id=event.database_id or 'unknown',
                        validation_level=validation_level
                    ).observe(duration_seconds)
            
            # Database usage
            if event.database_id and event.event_type == EventType.QUERY_EXECUTED:
                database_type = event.metadata.get('database_type', 'unknown') if event.metadata else 'unknown'
                self.prom_database_usage.labels(
                    database_id=event.database_id,
                    database_type=database_type
                ).inc()
                
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _log_to_standard_logger(self, event: MetricEvent) -> None:
        """Log event to standard Python logger."""
        log_data = {
            'event_type': event.event_type.value,
            'timestamp': event.timestamp,
            'user_id': event.user_id,
            'database_id': event.database_id,
            'query_type': event.query_type,
            'duration_ms': event.duration_ms,
            'success': event.success,
            'error_message': event.error_message
        }
        
        if event.metadata:
            log_data.update(event.metadata)
        
        if event.success:
            logger.info(f"METRIC: {json.dumps(log_data)}")
        else:
            logger.error(f"METRIC_ERROR: {json.dumps(log_data)}")
    
    def log_query_routed(self, user_id: str, database_id: str, query_type: str,
                        routing_time_ms: float, confidence_score: float,
                        similarity_scores: List[tuple]) -> None:
        """Log a query routing decision."""
        event = MetricEvent(
            event_type=EventType.QUERY_ROUTED,
            timestamp=time.time(),
            user_id=user_id,
            database_id=database_id,
            query_type=query_type,
            duration_ms=routing_time_ms,
            success=True,
            metadata={
                'confidence_score': confidence_score,
                'similarity_scores': similarity_scores,
                'routing_time_ms': routing_time_ms
            }
        )
        self.log_event(event)
    
    def log_validation_completed(self, user_id: str, database_id: str,
                                validation_time_ms: float, is_valid: bool,
                                validation_level: str, errors: List[str]) -> None:
        """Log a query validation completion."""
        event = MetricEvent(
            event_type=EventType.VALIDATION_COMPLETED,
            timestamp=time.time(),
            user_id=user_id,
            database_id=database_id,
            duration_ms=validation_time_ms,
            success=is_valid,
            error_message='; '.join(errors) if errors else None,
            metadata={
                'validation_level': validation_level,
                'error_count': len(errors)
            }
        )
        self.log_event(event)
    
    def log_security_check(self, user_id: str, check_type: str,
                          success: bool, error_message: Optional[str] = None) -> None:
        """Log a security check result."""
        event = MetricEvent(
            event_type=EventType.SECURITY_CHECK,
            timestamp=time.time(),
            user_id=user_id,
            success=success,
            error_message=error_message,
            metadata={'check_type': check_type}
        )
        self.log_event(event)
    
    def log_query_executed(self, user_id: str, database_id: str, query_type: str,
                          execution_time_ms: float, rows_returned: int,
                          success: bool, error_message: Optional[str] = None) -> None:
        """Log a query execution completion."""
        event = MetricEvent(
            event_type=EventType.QUERY_EXECUTED,
            timestamp=time.time(),
            user_id=user_id,
            database_id=database_id,
            query_type=query_type,
            duration_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            metadata={
                'rows_returned': rows_returned,
                'execution_time_ms': execution_time_ms
            }
        )
        self.log_event(event)
    
    def log_error(self, user_id: str, error_type: str, error_message: str,
                 database_id: Optional[str] = None) -> None:
        """Log an error event."""
        event = MetricEvent(
            event_type=EventType.ERROR_OCCURRED,
            timestamp=time.time(),
            user_id=user_id,
            database_id=database_id,
            success=False,
            error_message=error_message,
            metadata={'error_type': error_type}
        )
        self.log_event(event)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing metrics summary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            # Filter events by time
            recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
            
            # Calculate summary statistics
            total_queries = len([e for e in recent_events 
                               if e.event_type == EventType.QUERY_ROUTED])
            
            successful_queries = len([e for e in recent_events 
                                    if e.event_type == EventType.QUERY_EXECUTED and e.success])
            
            failed_queries = len([e for e in recent_events 
                                if e.event_type == EventType.QUERY_EXECUTED and not e.success])
            
            # Database usage stats
            db_usage = Counter([e.database_id for e in recent_events 
                              if e.database_id and e.event_type == EventType.QUERY_EXECUTED])
            
            # Query type distribution
            query_types = Counter([e.query_type for e in recent_events 
                                 if e.query_type and e.event_type == EventType.QUERY_ROUTED])
            
            # Performance statistics
            routing_times = [e.duration_ms for e in recent_events 
                           if e.event_type == EventType.QUERY_ROUTED and e.duration_ms]
            
            execution_times = [e.duration_ms for e in recent_events 
                             if e.event_type == EventType.QUERY_EXECUTED and e.duration_ms]
            
            return {
                'time_period_hours': hours,
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'failed_queries': failed_queries,
                'success_rate': successful_queries / max(total_queries, 1),
                'database_usage': dict(db_usage.most_common(10)),
                'query_type_distribution': dict(query_types),
                'average_routing_time_ms': sum(routing_times) / max(len(routing_times), 1),
                'average_execution_time_ms': sum(execution_times) / max(len(execution_times), 1),
                'total_events': len(recent_events)
            }
    
    def get_database_stats(self, database_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific database."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            db_events = [e for e in self.events 
                        if e.database_id == database_id and e.timestamp >= cutoff_time]
            
            executions = [e for e in db_events if e.event_type == EventType.QUERY_EXECUTED]
            routing = [e for e in db_events if e.event_type == EventType.QUERY_ROUTED]
            
            execution_times = [e.duration_ms for e in executions if e.duration_ms]
            routing_times = [e.duration_ms for e in routing if e.duration_ms]
            
            return {
                'database_id': database_id,
                'time_period_hours': hours,
                'total_executions': len(executions),
                'successful_executions': len([e for e in executions if e.success]),
                'failed_executions': len([e for e in executions if not e.success]),
                'times_selected': len(routing),
                'average_execution_time_ms': sum(execution_times) / max(len(execution_times), 1),
                'average_routing_time_ms': sum(routing_times) / max(len(routing_times), 1)
            }
    
    def cleanup_old_events(self, max_age_hours: int = 168) -> int:  # Default 1 week
        """
        Clean up old events to prevent memory growth.
        
        Args:
            max_age_hours: Maximum age of events to keep
            
        Returns:
            Number of events removed
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self.lock:
            initial_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp >= cutoff_time]
            removed_count = initial_count - len(self.events)
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old metric events")
            
            return removed_count