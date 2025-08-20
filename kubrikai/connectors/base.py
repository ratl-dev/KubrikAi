"""
Base database connector interface for KubrikAI.

Defines the interface for database connections and provides common functionality
for query execution, schema introspection, and connection management.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "prefer"
    connection_timeout: int = 30
    query_timeout: int = 300
    max_connections: int = 10


@dataclass
class QueryResult:
    """Result of a database query execution."""
    success: bool
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    columns: List[str]
    error: Optional[str] = None
    warnings: List[str] = None


@dataclass
class SchemaInfo:
    """Database schema information."""
    tables: List[Dict[str, Any]]
    views: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]


class DatabaseConnector(ABC):
    """
    Abstract base class for database connectors.
    
    Defines the interface that all database-specific connectors must implement
    for consistent interaction across different database systems.
    """
    
    def __init__(self, config: ConnectionConfig):
        """Initialize the database connector."""
        self.config = config
        self.connection_pool = None
        self.is_connected = False
        self.last_activity = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> QueryResult:
        """
        Execute a SQL query.
        
        Args:
            sql: SQL query to execute
            params: Optional query parameters
            timeout: Optional query timeout in seconds
        
        Returns:
            Query execution result
        """
        pass
    
    @abstractmethod
    async def get_schema_info(self) -> SchemaInfo:
        """
        Retrieve database schema information.
        
        Returns:
            Schema information including tables, views, etc.
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection is working, False otherwise
        """
        pass
    
    @abstractmethod
    async def explain_query(self, sql: str) -> Dict[str, Any]:
        """
        Get query execution plan.
        
        Args:
            sql: SQL query to explain
        
        Returns:
            Execution plan information
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status information
        """
        try:
            start_time = datetime.now()
            
            # Test basic connectivity
            is_connected = await self.test_connection()
            
            if not is_connected:
                return {
                    "status": "unhealthy",
                    "error": "Connection failed",
                    "response_time_ms": 0
                }
            
            # Test simple query
            result = await self.execute_query("SELECT 1 as test")
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy" if result.success else "degraded",
                "response_time_ms": response_time,
                "last_activity": self.last_activity.isoformat() if self.last_activity else None,
                "error": result.error if not result.success else None
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": 0
            }
    
    def _update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Detailed table information
        """
        # This is a generic implementation that can be overridden
        try:
            # Get column information
            columns_sql = f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            
            columns_result = await self.execute_query(columns_sql)
            
            if not columns_result.success:
                return {"error": columns_result.error}
            
            return {
                "table_name": table_name,
                "columns": columns_result.rows,
                "column_count": len(columns_result.rows)
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            return {"error": str(e)}
    
    async def estimate_table_size(self, table_name: str) -> Dict[str, Any]:
        """
        Estimate table size and row count.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Table size estimation
        """
        try:
            # Generic row count query
            count_sql = f"SELECT COUNT(*) as row_count FROM {table_name}"
            result = await self.execute_query(count_sql)
            
            if result.success and result.rows:
                return {
                    "table_name": table_name,
                    "estimated_rows": result.rows[0]["row_count"],
                    "estimation_method": "count_query"
                }
            else:
                return {
                    "table_name": table_name,
                    "estimated_rows": 0,
                    "error": result.error
                }
                
        except Exception as e:
            logger.error(f"Failed to estimate size for {table_name}: {str(e)}")
            return {
                "table_name": table_name,
                "estimated_rows": 0,
                "error": str(e)
            }