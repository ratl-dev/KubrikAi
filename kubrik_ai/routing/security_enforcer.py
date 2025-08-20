"""
Security Enforcer - Enforces read-only constraints, query limits, 
and other security policies for SQL execution.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SecurityViolation(Exception):
    """Exception raised when security policy is violated"""
    pass


@dataclass
class QueryLimits:
    """Query execution limits"""
    max_execution_time_seconds: int = 300
    max_result_rows: int = 10000
    max_memory_mb: int = 512
    max_concurrent_queries: int = 5


@dataclass
class UserContext:
    """User context for security enforcement"""
    user_id: str
    roles: List[str]
    permissions: List[str]
    query_limits: QueryLimits


class SecurityEnforcer:
    """
    Enforces security policies including read-only constraints,
    query limits, rate limiting, and access controls.
    """
    
    def __init__(self):
        """Initialize the security enforcer."""
        self.active_queries: Dict[str, Dict[str, Any]] = {}
        self.query_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
    def enforce_read_only(self, query: str) -> None:
        """
        Enforce read-only constraint on SQL query.
        
        Args:
            query: SQL query to check
            
        Raises:
            SecurityViolation: If query contains write operations
        """
        query_upper = query.upper().strip()
        
        # List of forbidden statements for read-only mode
        write_statements = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'MERGE', 'UPSERT'
        ]
        
        # Administrative statements
        admin_statements = [
            'GRANT', 'REVOKE', 'SET', 'RESET', 'SHOW', 'EXPLAIN',
            'ANALYZE', 'VACUUM', 'REINDEX'
        ]
        
        # Check for write operations
        for statement in write_statements:
            if query_upper.startswith(statement):
                raise SecurityViolation(
                    f"Write operation '{statement}' not allowed in read-only mode"
                )
        
        # Allow SELECT, WITH, and some administrative commands
        allowed_starts = ['SELECT', 'WITH', 'EXPLAIN', 'SHOW']
        if not any(query_upper.startswith(cmd) for cmd in allowed_starts):
            # Check if it's a function call or expression
            if not ('(' in query_upper and ')' in query_upper):
                raise SecurityViolation(
                    f"Query type not allowed in read-only mode: {query_upper[:20]}..."
                )
    
    def check_query_limits(self, user_context: UserContext, 
                          estimated_rows: Optional[int] = None,
                          estimated_time: Optional[float] = None) -> None:
        """
        Check if query respects configured limits.
        
        Args:
            user_context: User context with limits
            estimated_rows: Estimated number of result rows
            estimated_time: Estimated execution time in seconds
            
        Raises:
            SecurityViolation: If query exceeds limits
        """
        limits = user_context.query_limits
        
        # Check estimated result rows
        if estimated_rows and estimated_rows > limits.max_result_rows:
            raise SecurityViolation(
                f"Query estimated to return {estimated_rows} rows, "
                f"exceeds limit of {limits.max_result_rows}"
            )
        
        # Check estimated execution time
        if estimated_time and estimated_time > limits.max_execution_time_seconds:
            raise SecurityViolation(
                f"Query estimated to take {estimated_time:.1f}s, "
                f"exceeds limit of {limits.max_execution_time_seconds}s"
            )
        
        # Check concurrent query limit
        with self.lock:
            user_active_queries = sum(
                1 for q in self.active_queries.values()
                if q.get('user_id') == user_context.user_id
            )
            
            if user_active_queries >= limits.max_concurrent_queries:
                raise SecurityViolation(
                    f"User has {user_active_queries} active queries, "
                    f"exceeds limit of {limits.max_concurrent_queries}"
                )
    
    def check_rate_limits(self, user_id: str, queries_per_minute: int = 30) -> None:
        """
        Check if user has exceeded rate limits.
        
        Args:
            user_id: User identifier
            queries_per_minute: Maximum queries per minute
            
        Raises:
            SecurityViolation: If rate limit exceeded
        """
        current_time = time.time()
        
        with self.lock:
            # Initialize rate limiting for user if not exists
            if user_id not in self.rate_limits:
                self.rate_limits[user_id] = {
                    'query_times': deque(maxlen=queries_per_minute),
                    'window_start': current_time
                }
            
            rate_data = self.rate_limits[user_id]
            query_times = rate_data['query_times']
            
            # Remove queries older than 1 minute
            minute_ago = current_time - 60
            while query_times and query_times[0] < minute_ago:
                query_times.popleft()
            
            # Check if rate limit exceeded
            if len(query_times) >= queries_per_minute:
                raise SecurityViolation(
                    f"Rate limit exceeded: {len(query_times)} queries in the last minute, "
                    f"limit is {queries_per_minute}"
                )
            
            # Add current query
            query_times.append(current_time)
    
    def check_table_access(self, user_context: UserContext, tables: List[str]) -> None:
        """
        Check if user has access to specified tables.
        
        Args:
            user_context: User context with permissions
            tables: List of table names being accessed
            
        Raises:
            SecurityViolation: If user lacks table access
        """
        # Extract table permissions from user context
        table_permissions = [
            perm for perm in user_context.permissions
            if perm.startswith('table:')
        ]
        
        allowed_tables = set()
        for perm in table_permissions:
            # Format: "table:database.schema.table" or "table:*"
            if perm == 'table:*':
                # User has access to all tables
                return
            
            table_name = perm.replace('table:', '')
            allowed_tables.add(table_name.lower())
        
        # Check each table access
        for table in tables:
            table_lower = table.lower()
            
            # Check exact match
            if table_lower not in allowed_tables:
                # Check wildcard patterns
                has_access = False
                for allowed in allowed_tables:
                    if '*' in allowed:
                        # Simple wildcard matching
                        pattern = allowed.replace('*', '.*')
                        import re
                        if re.match(pattern, table_lower):
                            has_access = True
                            break
                
                if not has_access:
                    raise SecurityViolation(
                        f"Access denied to table: {table}"
                    )
    
    def register_query_start(self, query_id: str, user_context: UserContext, 
                           query: str) -> None:
        """
        Register the start of a query execution.
        
        Args:
            query_id: Unique query identifier
            user_context: User context
            query: SQL query being executed
        """
        with self.lock:
            self.active_queries[query_id] = {
                'user_id': user_context.user_id,
                'query': query,
                'start_time': time.time(),
                'max_duration': user_context.query_limits.max_execution_time_seconds
            }
            
            # Add to user history
            self.query_history[user_context.user_id].append({
                'query_id': query_id,
                'query': query[:200],  # Truncate for storage
                'start_time': time.time()
            })
    
    def register_query_end(self, query_id: str) -> None:
        """
        Register the end of a query execution.
        
        Args:
            query_id: Unique query identifier
        """
        with self.lock:
            if query_id in self.active_queries:
                query_info = self.active_queries.pop(query_id)
                execution_time = time.time() - query_info['start_time']
                
                logger.info(
                    f"Query {query_id} completed in {execution_time:.2f}s "
                    f"for user {query_info['user_id']}"
                )
    
    def check_query_timeout(self, query_id: str) -> bool:
        """
        Check if a query has exceeded its timeout.
        
        Args:
            query_id: Query identifier to check
            
        Returns:
            True if query has timed out, False otherwise
        """
        with self.lock:
            if query_id not in self.active_queries:
                return False
            
            query_info = self.active_queries[query_id]
            execution_time = time.time() - query_info['start_time']
            
            return execution_time > query_info['max_duration']
    
    def kill_query(self, query_id: str, reason: str = "Timeout") -> bool:
        """
        Kill a running query.
        
        Args:
            query_id: Query identifier to kill
            reason: Reason for killing the query
            
        Returns:
            True if query was killed, False if not found
        """
        with self.lock:
            if query_id in self.active_queries:
                query_info = self.active_queries.pop(query_id)
                logger.warning(
                    f"Killed query {query_id} for user {query_info['user_id']}: {reason}"
                )
                return True
            return False
    
    def get_active_queries(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of active queries, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of active query information
        """
        with self.lock:
            queries = []
            for query_id, info in self.active_queries.items():
                if user_id is None or info['user_id'] == user_id:
                    queries.append({
                        'query_id': query_id,
                        'user_id': info['user_id'],
                        'query': info['query'][:100],  # Truncated
                        'running_time': time.time() - info['start_time']
                    })
            return queries
    
    def get_user_query_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get query history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of queries to return
            
        Returns:
            List of historical query information
        """
        with self.lock:
            history = list(self.query_history[user_id])
            return history[-limit:] if limit > 0 else history
    
    def cleanup_expired_data(self, max_age_hours: int = 24) -> None:
        """
        Clean up expired query data and rate limiting information.
        
        Args:
            max_age_hours: Maximum age of data to keep in hours
        """
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        with self.lock:
            # Clean up rate limiting data
            for user_id in list(self.rate_limits.keys()):
                rate_data = self.rate_limits[user_id]
                query_times = rate_data['query_times']
                
                # Remove old entries
                while query_times and query_times[0] < cutoff_time:
                    query_times.popleft()
                
                # Remove empty rate limit entries
                if not query_times:
                    del self.rate_limits[user_id]
            
            # Note: query_history uses deque with maxlen, so it auto-manages size