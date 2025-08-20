"""
PostgreSQL database connector for KubrikAI.

Implements PostgreSQL-specific functionality for connection management,
query execution, and schema introspection.
"""

import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime

from .base import DatabaseConnector, ConnectionConfig, QueryResult, SchemaInfo

logger = logging.getLogger(__name__)


class PostgreSQLConnector(DatabaseConnector):
    """
    PostgreSQL-specific database connector.
    
    Handles PostgreSQL connections, query execution, and schema introspection
    with optimizations for PostgreSQL-specific features.
    """
    
    def __init__(self, config: ConnectionConfig):
        """Initialize PostgreSQL connector."""
        super().__init__(config)
        self.connection_pool = None
        
    async def connect(self) -> bool:
        """Establish connection pool to PostgreSQL."""
        try:
            # Build connection string
            dsn = f"postgresql://{self.config.username}:{self.config.password}@" \
                  f"{self.config.host}:{self.config.port}/{self.config.database}"
            
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=self.config.max_connections,
                command_timeout=self.config.query_timeout,
                server_settings={
                    'application_name': 'KubrikAI',
                    'search_path': 'public'
                }
            )
            
            self.is_connected = True
            self._update_activity()
            
            logger.info(f"Connected to PostgreSQL: {self.config.host}:{self.config.port}/{self.config.database}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None
            self.is_connected = False
            logger.info("Disconnected from PostgreSQL")
    
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> QueryResult:
        """Execute SQL query on PostgreSQL."""
        if not self.connection_pool:
            return QueryResult(
                success=False,
                rows=[],
                row_count=0,
                execution_time_ms=0,
                columns=[],
                error="Not connected to database"
            )
        
        start_time = time.time()
        
        try:
            async with self.connection_pool.acquire() as connection:
                # Set query timeout
                query_timeout = timeout or self.config.query_timeout
                
                # Execute query
                if params:
                    # Convert named parameters to positional
                    param_values = list(params.values())
                    param_sql = sql
                    for i, key in enumerate(params.keys(), 1):
                        param_sql = param_sql.replace(f":{key}", f"${i}")
                    
                    result = await asyncio.wait_for(
                        connection.fetch(param_sql, *param_values),
                        timeout=query_timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        connection.fetch(sql),
                        timeout=query_timeout
                    )
                
                # Convert result to dict format
                rows = [dict(row) for row in result]
                columns = list(result[0].keys()) if result else []
                
                execution_time = (time.time() - start_time) * 1000
                self._update_activity()
                
                return QueryResult(
                    success=True,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=execution_time,
                    columns=columns,
                    warnings=[]
                )
                
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Query timeout after {query_timeout} seconds"
            logger.warning(error_msg)
            
            return QueryResult(
                success=False,
                rows=[],
                row_count=0,
                execution_time_ms=execution_time,
                columns=[],
                error=error_msg
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Query execution failed: {error_msg}")
            
            return QueryResult(
                success=False,
                rows=[],
                row_count=0,
                execution_time_ms=execution_time,
                columns=[],
                error=error_msg
            )
    
    async def get_schema_info(self) -> SchemaInfo:
        """Retrieve PostgreSQL schema information."""
        try:
            # Get tables
            tables_sql = """
                SELECT 
                    table_name,
                    table_type,
                    table_schema
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_name
            """
            
            tables_result = await self.execute_query(tables_sql)
            tables = tables_result.rows if tables_result.success else []
            
            # Get views
            views_sql = """
                SELECT 
                    table_name as view_name,
                    view_definition,
                    table_schema
                FROM information_schema.views
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_name
            """
            
            views_result = await self.execute_query(views_sql)
            views = views_result.rows if views_result.success else []
            
            # Get functions
            functions_sql = """
                SELECT 
                    routine_name as function_name,
                    routine_type,
                    routine_schema
                FROM information_schema.routines
                WHERE routine_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY routine_name
            """
            
            functions_result = await self.execute_query(functions_sql)
            functions = functions_result.rows if functions_result.success else []
            
            # Get indexes
            indexes_sql = """
                SELECT 
                    indexname as index_name,
                    tablename as table_name,
                    indexdef as index_definition
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY indexname
            """
            
            indexes_result = await self.execute_query(indexes_sql)
            indexes = indexes_result.rows if indexes_result.success else []
            
            # Get constraints
            constraints_sql = """
                SELECT 
                    constraint_name,
                    table_name,
                    constraint_type
                FROM information_schema.table_constraints
                WHERE table_schema = 'public'
                ORDER BY constraint_name
            """
            
            constraints_result = await self.execute_query(constraints_sql)
            constraints = constraints_result.rows if constraints_result.success else []
            
            return SchemaInfo(
                tables=tables,
                views=views,
                functions=functions,
                indexes=indexes,
                constraints=constraints
            )
            
        except Exception as e:
            logger.error(f"Failed to get schema info: {str(e)}")
            return SchemaInfo(
                tables=[],
                views=[],
                functions=[],
                indexes=[],
                constraints=[]
            )
    
    async def test_connection(self) -> bool:
        """Test PostgreSQL connectivity."""
        try:
            result = await self.execute_query("SELECT 1")
            return result.success
        except Exception:
            return False
    
    async def explain_query(self, sql: str) -> Dict[str, Any]:
        """Get PostgreSQL query execution plan."""
        try:
            explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE FALSE) {sql}"
            result = await self.execute_query(explain_sql)
            
            if result.success and result.rows:
                return {
                    "plan": result.rows[0],
                    "estimated_cost": self._extract_cost_from_plan(result.rows[0]),
                    "plan_type": "postgresql_explain"
                }
            else:
                return {
                    "error": result.error or "Failed to get execution plan"
                }
                
        except Exception as e:
            logger.error(f"EXPLAIN query failed: {str(e)}")
            return {"error": str(e)}
    
    def _extract_cost_from_plan(self, plan_data: Dict[str, Any]) -> float:
        """Extract estimated cost from PostgreSQL execution plan."""
        try:
            # PostgreSQL EXPLAIN format varies, try to extract cost
            if isinstance(plan_data, dict):
                if "Total Cost" in plan_data:
                    return float(plan_data["Total Cost"])
                elif "Plan" in plan_data:
                    plan = plan_data["Plan"]
                    if "Total Cost" in plan:
                        return float(plan["Total Cost"])
            
            return 0.0
        except Exception:
            return 0.0
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed PostgreSQL table information."""
        try:
            # Get column information
            columns_sql = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = 'public'
                ORDER BY ordinal_position
            """
            
            columns_result = await self.execute_query(columns_sql, {"table_name": table_name})
            
            # Get primary key information
            pk_sql = """
                SELECT column_name
                FROM information_schema.key_column_usage k
                JOIN information_schema.table_constraints t ON k.constraint_name = t.constraint_name
                WHERE t.constraint_type = 'PRIMARY KEY' 
                AND t.table_name = $1 AND t.table_schema = 'public'
                ORDER BY k.ordinal_position
            """
            
            pk_result = await self.execute_query(pk_sql, {"table_name": table_name})
            
            # Get foreign key information
            fk_sql = """
                SELECT 
                    kcu.column_name,
                    ccu.table_name AS referenced_table,
                    ccu.column_name AS referenced_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = $1 AND tc.table_schema = 'public'
            """
            
            fk_result = await self.execute_query(fk_sql, {"table_name": table_name})
            
            # Get table statistics
            stats_sql = """
                SELECT 
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples,
                    last_vacuum,
                    last_analyze
                FROM pg_stat_user_tables
                WHERE relname = $1
            """
            
            stats_result = await self.execute_query(stats_sql, {"table_name": table_name})
            
            return {
                "table_name": table_name,
                "columns": columns_result.rows if columns_result.success else [],
                "primary_keys": [row["column_name"] for row in pk_result.rows] if pk_result.success else [],
                "foreign_keys": fk_result.rows if fk_result.success else [],
                "statistics": stats_result.rows[0] if stats_result.success and stats_result.rows else {},
                "column_count": len(columns_result.rows) if columns_result.success else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            return {"error": str(e)}
    
    async def estimate_table_size(self, table_name: str) -> Dict[str, Any]:
        """Estimate PostgreSQL table size using system catalogs."""
        try:
            # Use PostgreSQL system catalogs for more accurate estimates
            size_sql = """
                SELECT 
                    pg_size_pretty(pg_total_relation_size($1)) as total_size,
                    pg_size_pretty(pg_relation_size($1)) as table_size,
                    reltuples::bigint as estimated_rows
                FROM pg_class
                WHERE relname = $1
            """
            
            result = await self.execute_query(size_sql, {"table_name": table_name})
            
            if result.success and result.rows:
                return {
                    "table_name": table_name,
                    "estimated_rows": result.rows[0]["estimated_rows"],
                    "total_size": result.rows[0]["total_size"],
                    "table_size": result.rows[0]["table_size"],
                    "estimation_method": "pg_class_statistics"
                }
            else:
                # Fallback to count query
                return await super().estimate_table_size(table_name)
                
        except Exception as e:
            logger.error(f"Failed to estimate size for {table_name}: {str(e)}")
            return await super().estimate_table_size(table_name)