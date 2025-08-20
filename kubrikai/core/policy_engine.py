"""
Policy and governance engine for KubrikAI.

Handles access control, data governance, compliance, and security policies
for database routing and query execution.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import re
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Possible policy actions."""
    ALLOW = "allow"
    DENY = "deny"
    RESTRICT = "restrict"
    AUDIT = "audit"


@dataclass
class DatabasePolicy:
    """Policy configuration for a database."""
    db_id: str
    read_only: bool = True
    max_rows: Optional[int] = 1000
    max_execution_time: Optional[int] = 30  # seconds
    allowed_tables: Optional[Set[str]] = None
    blocked_tables: Optional[Set[str]] = None
    allowed_columns: Optional[Dict[str, Set[str]]] = None
    blocked_columns: Optional[Dict[str, Set[str]]] = None
    require_where_clause: bool = False
    allowed_functions: Optional[Set[str]] = None
    blocked_functions: Optional[Set[str]] = None
    domain_restrictions: Optional[List[str]] = None
    freshness_requirement: Optional[int] = None  # hours
    
    def __post_init__(self):
        """Initialize sets if None."""
        if self.allowed_tables is None:
            self.allowed_tables = set()
        if self.blocked_tables is None:
            self.blocked_tables = set()
        if self.allowed_columns is None:
            self.allowed_columns = {}
        if self.blocked_columns is None:
            self.blocked_columns = {}
        if self.allowed_functions is None:
            self.allowed_functions = set()
        if self.blocked_functions is None:
            self.blocked_functions = set()


@dataclass
class QueryContext:
    """Context information for query execution."""
    user_id: str
    session_id: str
    query: str
    db_id: str
    timestamp: datetime
    user_roles: List[str]
    request_metadata: Dict[str, Any]


@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    policy_type: str
    severity: str
    message: str
    suggested_action: PolicyAction
    details: Dict[str, Any]


class PolicyEngine:
    """
    Core policy and governance engine.
    
    Enforces data access policies, compliance requirements, and security rules
    for database routing and query execution.
    """
    
    def __init__(self):
        """Initialize the policy engine."""
        self.database_policies: Dict[str, DatabasePolicy] = {}
        self.global_policies: Dict[str, Any] = {
            "default_read_only": True,
            "default_max_rows": 1000,
            "default_max_execution_time": 30,
            "audit_all_queries": True,
            "block_drop_statements": True,
            "block_truncate_statements": True,
            "block_delete_statements": True,
            "block_update_statements": True,
            "block_insert_statements": True,
            "require_limit_clause": True
        }
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info("Initialized PolicyEngine with default security policies")
    
    def register_database_policy(self, policy: DatabasePolicy) -> None:
        """Register a policy for a specific database."""
        self.database_policies[policy.db_id] = policy
        logger.info(f"Registered policy for database {policy.db_id}")
    
    def evaluate_database_access(
        self, 
        db_id: str, 
        context: QueryContext
    ) -> List[PolicyViolation]:
        """
        Evaluate if access to a database is allowed based on policies.
        
        Args:
            db_id: Database identifier
            context: Query execution context
        
        Returns:
            List of policy violations (empty if access allowed)
        """
        violations = []
        
        # Check if database has specific policies
        policy = self.database_policies.get(db_id)
        if not policy:
            # Use default restrictive policy
            policy = DatabasePolicy(
                db_id=db_id,
                read_only=True,
                max_rows=self.global_policies["default_max_rows"],
                max_execution_time=self.global_policies["default_max_execution_time"]
            )
        
        # Check domain restrictions
        if policy.domain_restrictions:
            user_domain = context.request_metadata.get("domain")
            if user_domain not in policy.domain_restrictions:
                violations.append(PolicyViolation(
                    policy_type="domain_access",
                    severity="high",
                    message=f"User domain '{user_domain}' not allowed for database {db_id}",
                    suggested_action=PolicyAction.DENY,
                    details={"allowed_domains": policy.domain_restrictions}
                ))
        
        # Check freshness requirements
        if policy.freshness_requirement:
            db_last_updated = context.request_metadata.get("db_last_updated")
            if db_last_updated:
                hours_old = (context.timestamp - db_last_updated).total_seconds() / 3600
                if hours_old > policy.freshness_requirement:
                    violations.append(PolicyViolation(
                        policy_type="data_freshness",
                        severity="medium",
                        message=f"Database {db_id} data is {hours_old:.1f} hours old, exceeds {policy.freshness_requirement}h limit",
                        suggested_action=PolicyAction.RESTRICT,
                        details={"hours_old": hours_old, "max_hours": policy.freshness_requirement}
                    ))
        
        return violations
    
    def validate_sql_query(
        self, 
        sql: str, 
        db_id: str, 
        context: QueryContext
    ) -> List[PolicyViolation]:
        """
        Validate SQL query against security and governance policies.
        
        Args:
            sql: SQL query to validate
            db_id: Target database identifier
            context: Query execution context
        
        Returns:
            List of policy violations
        """
        violations = []
        sql_lower = sql.lower().strip()
        
        # Get database policy
        policy = self.database_policies.get(db_id, DatabasePolicy(db_id=db_id))
        
        # Check read-only enforcement
        if policy.read_only or self.global_policies["default_read_only"]:
            write_operations = ["insert", "update", "delete", "drop", "create", "alter", "truncate"]
            for op in write_operations:
                if re.search(rf'\b{op}\b', sql_lower):
                    violations.append(PolicyViolation(
                        policy_type="read_only_violation",
                        severity="critical",
                        message=f"Write operation '{op}' not allowed in read-only mode",
                        suggested_action=PolicyAction.DENY,
                        details={"operation": op, "sql": sql}
                    ))
        
        # Check for dangerous operations
        dangerous_patterns = [
            (r'\bdrop\s+table\b', "DROP TABLE"),
            (r'\btruncate\s+table\b', "TRUNCATE TABLE"),
            (r'\bdelete\s+from\s+\w+\s*(?:;|$)', "DELETE without WHERE"),
            (r'\bupdate\s+\w+\s+set\s+.*?(?:;|$)(?!.*where)', "UPDATE without WHERE")
        ]
        
        for pattern, operation in dangerous_patterns:
            if re.search(pattern, sql_lower):
                violations.append(PolicyViolation(
                    policy_type="dangerous_operation",
                    severity="critical",
                    message=f"Dangerous operation detected: {operation}",
                    suggested_action=PolicyAction.DENY,
                    details={"operation": operation, "pattern": pattern}
                ))
        
        # Check LIMIT clause requirement
        if self.global_policies["require_limit_clause"] and policy.max_rows:
            if re.search(r'\bselect\b', sql_lower) and not re.search(r'\blimit\b', sql_lower):
                violations.append(PolicyViolation(
                    policy_type="missing_limit",
                    severity="medium",
                    message="SELECT queries must include LIMIT clause",
                    suggested_action=PolicyAction.RESTRICT,
                    details={"max_rows": policy.max_rows}
                ))
        
        # Check table access
        if policy.blocked_tables:
            for table in policy.blocked_tables:
                if re.search(rf'\b{table}\b', sql_lower):
                    violations.append(PolicyViolation(
                        policy_type="table_access_denied",
                        severity="high",
                        message=f"Access to table '{table}' is not allowed",
                        suggested_action=PolicyAction.DENY,
                        details={"blocked_table": table}
                    ))
        
        # Check function usage
        if policy.blocked_functions:
            for func in policy.blocked_functions:
                if re.search(rf'\b{func}\s*\(', sql_lower):
                    violations.append(PolicyViolation(
                        policy_type="function_blocked",
                        severity="medium",
                        message=f"Function '{func}' is not allowed",
                        suggested_action=PolicyAction.DENY,
                        details={"blocked_function": func}
                    ))
        
        return violations
    
    def enforce_query_limits(self, sql: str, db_id: str) -> str:
        """
        Enforce query limits by modifying the SQL.
        
        Args:
            sql: Original SQL query
            db_id: Target database identifier
        
        Returns:
            Modified SQL with limits enforced
        """
        policy = self.database_policies.get(db_id, DatabasePolicy(db_id=db_id))
        
        sql_lower = sql.lower().strip()
        
        # Add LIMIT clause if missing for SELECT queries
        if re.search(r'\bselect\b', sql_lower) and not re.search(r'\blimit\b', sql_lower):
            if policy.max_rows:
                sql = sql.rstrip(';') + f" LIMIT {policy.max_rows}"
        
        # Modify existing LIMIT if it exceeds max_rows
        if policy.max_rows:
            limit_match = re.search(r'\blimit\s+(\d+)', sql_lower)
            if limit_match:
                current_limit = int(limit_match.group(1))
                if current_limit > policy.max_rows:
                    sql = re.sub(r'\blimit\s+\d+', f'LIMIT {policy.max_rows}', sql, flags=re.IGNORECASE)
        
        return sql
    
    def log_query_execution(
        self, 
        context: QueryContext, 
        sql: str, 
        violations: List[PolicyViolation],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log query execution for audit purposes.
        
        Args:
            context: Query execution context
            sql: Executed SQL query
            violations: Policy violations found
            execution_result: Query execution results and metadata
        """
        log_entry = {
            "timestamp": context.timestamp.isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "db_id": context.db_id,
            "query": context.query,
            "sql": sql,
            "violations": [
                {
                    "type": v.policy_type,
                    "severity": v.severity,
                    "message": v.message,
                    "action": v.suggested_action.value
                }
                for v in violations
            ],
            "execution_result": execution_result or {},
            "user_roles": context.user_roles
        }
        
        self.audit_log.append(log_entry)
        
        # Log to system logger
        if violations:
            logger.warning(f"Policy violations for user {context.user_id}: {len(violations)} violations")
        else:
            logger.info(f"Query executed successfully for user {context.user_id}")
    
    def get_database_policy(self, db_id: str) -> Optional[DatabasePolicy]:
        """Get policy for a specific database."""
        return self.database_policies.get(db_id)
    
    def update_global_policy(self, key: str, value: Any) -> None:
        """Update a global policy setting."""
        self.global_policies[key] = value
        logger.info(f"Updated global policy {key} = {value}")
    
    def get_audit_log(
        self, 
        user_id: Optional[str] = None,
        db_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries with optional filtering.
        
        Args:
            user_id: Filter by user ID
            db_id: Filter by database ID  
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
        
        Returns:
            Filtered list of audit log entries
        """
        filtered_log = self.audit_log
        
        if user_id:
            filtered_log = [entry for entry in filtered_log if entry["user_id"] == user_id]
        
        if db_id:
            filtered_log = [entry for entry in filtered_log if entry["db_id"] == db_id]
        
        if start_time:
            start_str = start_time.isoformat()
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] >= start_str]
        
        if end_time:
            end_str = end_time.isoformat()
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] <= end_str]
        
        return filtered_log