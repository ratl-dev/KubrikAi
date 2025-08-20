"""
SQL validation and analysis using SQLGlot.

Provides SQL parsing, validation, and security analysis for query routing
and execution safety.
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
import sqlglot
from sqlglot import parse_one, transpile
from sqlglot.errors import ParseError, TokenError
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    error: Optional[str]
    warnings: List[str]
    parsed_query: Optional[Dict[str, Any]]
    tables_accessed: Set[str]
    operations: Set[str]
    has_subqueries: bool
    estimated_complexity: int


class SQLValidator:
    """
    SQL validation and analysis engine using SQLGlot.
    
    Provides comprehensive SQL parsing, validation, and security analysis
    to ensure safe query execution.
    """
    
    def __init__(self):
        """Initialize the SQL validator."""
        self.dialect_mapping = {
            "postgresql": "postgres",
            "mysql": "mysql", 
            "bigquery": "bigquery",
            "snowflake": "snowflake",
            "sqlite": "sqlite",
            "oracle": "oracle",
            "mssql": "tsql"
        }
        
        # Security patterns to detect
        self.dangerous_patterns = [
            (r";\s*(drop|truncate|delete|update|insert)", "Multiple statements with write operations"),
            (r"union\s+select", "UNION-based injection pattern"),
            (r"(exec|execute)\s*\(", "Dynamic SQL execution"),
            (r"xp_cmdshell", "Command execution function"),
            (r"sp_executesql", "Dynamic SQL execution"),
            (r"information_schema", "System catalog access"),
            (r"pg_catalog", "PostgreSQL system catalog"),
            (r"mysql\.", "MySQL system database"),
            (r"sys\.", "System database access"),
            (r"master\.", "Master database access")
        ]
        
        logger.info("Initialized SQLValidator with security patterns")
    
    def validate_sql(self, sql: str, db_type: str = "postgresql") -> ValidationResult:
        """
        Validate SQL query for syntax, security, and compliance.
        
        Args:
            sql: SQL query to validate
            db_type: Database type/dialect
        
        Returns:
            Validation result with analysis
        """
        warnings = []
        tables_accessed = set()
        operations = set()
        has_subqueries = False
        complexity = 0
        
        try:
            # Clean and normalize SQL
            cleaned_sql = self._clean_sql(sql)
            
            # Check for dangerous patterns first
            security_issues = self._check_security_patterns(cleaned_sql)
            if security_issues:
                return ValidationResult(
                    is_valid=False,
                    error=f"Security violation: {'; '.join(security_issues)}",
                    warnings=warnings,
                    parsed_query=None,
                    tables_accessed=tables_accessed,
                    operations=operations,
                    has_subqueries=has_subqueries,
                    estimated_complexity=complexity
                )
            
            # Parse with SQLGlot
            dialect = self.dialect_mapping.get(db_type.lower(), "postgres")
            
            try:
                parsed = parse_one(cleaned_sql, dialect=dialect)
            except (ParseError, TokenError) as e:
                return ValidationResult(
                    is_valid=False,
                    error=f"SQL parsing error: {str(e)}",
                    warnings=warnings,
                    parsed_query=None,
                    tables_accessed=tables_accessed,
                    operations=operations,
                    has_subqueries=has_subqueries,
                    estimated_complexity=complexity
                )
            
            # Analyze parsed query
            analysis = self._analyze_parsed_query(parsed)
            tables_accessed = analysis["tables"]
            operations = analysis["operations"]
            has_subqueries = analysis["has_subqueries"]
            complexity = analysis["complexity"]
            warnings.extend(analysis["warnings"])
            
            # Validate operations
            validation_warnings = self._validate_operations(operations, tables_accessed)
            warnings.extend(validation_warnings)
            
            # Check for read-only compliance
            write_ops = {"INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"}
            if any(op in write_ops for op in operations):
                warnings.append("Query contains write operations")
            
            return ValidationResult(
                is_valid=True,
                error=None,
                warnings=warnings,
                parsed_query=self._extract_query_info(parsed),
                tables_accessed=tables_accessed,
                operations=operations,
                has_subqueries=has_subqueries,
                estimated_complexity=complexity
            )
            
        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error=f"Validation error: {str(e)}",
                warnings=warnings,
                parsed_query=None,
                tables_accessed=tables_accessed,
                operations=operations,
                has_subqueries=has_subqueries,
                estimated_complexity=complexity
            )
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL query."""
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        
        return sql
    
    def _check_security_patterns(self, sql: str) -> List[str]:
        """Check for dangerous SQL patterns."""
        issues = []
        sql_lower = sql.lower()
        
        for pattern, description in self.dangerous_patterns:
            if re.search(pattern, sql_lower):
                issues.append(description)
        
        # Check for potential SQL injection patterns
        injection_patterns = [
            r"'\s*(or|and)\s*'",  # Basic injection
            r"'\s*;\s*--",        # Comment injection
            r"'\s*union\s*select", # Union injection
            r"0x[0-9a-f]+",       # Hex injection
            r"char\s*\(",         # Char function injection
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_lower):
                issues.append("Potential SQL injection pattern detected")
                break
        
        return issues
    
    def _analyze_parsed_query(self, parsed) -> Dict[str, Any]:
        """Analyze parsed SQL query for detailed information."""
        tables = set()
        operations = set()
        warnings = []
        complexity = 0
        has_subqueries = False
        
        # Extract operations
        if hasattr(parsed, 'key'):
            operations.add(parsed.key.upper())
        
        # Walk the AST to find tables and operations
        for node in parsed.walk():
            node_type = type(node).__name__
            
            # Track table references
            if node_type == "Table":
                if hasattr(node, 'name'):
                    tables.add(str(node.name))
            
            # Track subqueries
            if node_type in ["Subquery", "CTE"]:
                has_subqueries = True
                complexity += 2
            
            # Track joins
            if node_type == "Join":
                complexity += 1
            
            # Track aggregations
            if node_type in ["Count", "Sum", "Avg", "Max", "Min"]:
                complexity += 1
            
            # Track window functions
            if node_type == "Window":
                complexity += 2
            
            # Track complex conditions
            if node_type in ["And", "Or"]:
                complexity += 1
        
        # Check for complex patterns
        if complexity > 10:
            warnings.append("High query complexity detected")
        
        if len(tables) > 5:
            warnings.append("Query accesses many tables")
        
        return {
            "tables": tables,
            "operations": operations,
            "warnings": warnings,
            "complexity": complexity,
            "has_subqueries": has_subqueries
        }
    
    def _validate_operations(self, operations: Set[str], tables: Set[str]) -> List[str]:
        """Validate SQL operations and provide warnings."""
        warnings = []
        
        # Check for potentially expensive operations
        expensive_ops = {"SELECT", "JOIN", "GROUP BY", "ORDER BY", "WINDOW"}
        if any(op in expensive_ops for op in operations) and len(tables) > 3:
            warnings.append("Complex query with multiple tables may be expensive")
        
        # Check for missing LIMIT
        if "SELECT" in operations and "LIMIT" not in operations:
            warnings.append("SELECT query without LIMIT clause")
        
        # Check for cartesian products
        if "JOIN" in operations and len(tables) > 2:
            warnings.append("Multiple joins may create cartesian products")
        
        return warnings
    
    def _extract_query_info(self, parsed) -> Dict[str, Any]:
        """Extract detailed information from parsed query."""
        info = {
            "query_type": type(parsed).__name__,
            "select_count": 0,
            "join_count": 0,
            "where_conditions": 0,
            "group_by_columns": 0,
            "order_by_columns": 0
        }
        
        for node in parsed.walk():
            node_type = type(node).__name__
            
            if node_type == "Select":
                info["select_count"] += 1
            elif node_type == "Join":
                info["join_count"] += 1
            elif node_type == "Where":
                info["where_conditions"] += 1
            elif node_type == "Group":
                info["group_by_columns"] += len(node.expressions) if hasattr(node, 'expressions') else 1
            elif node_type == "Order":
                info["order_by_columns"] += len(node.expressions) if hasattr(node, 'expressions') else 1
        
        return info
    
    def transpile_sql(self, sql: str, source_dialect: str, target_dialect: str) -> Optional[str]:
        """
        Transpile SQL from one dialect to another.
        
        Args:
            sql: Source SQL query
            source_dialect: Source database dialect
            target_dialect: Target database dialect
        
        Returns:
            Transpiled SQL or None if failed
        """
        try:
            source = self.dialect_mapping.get(source_dialect.lower(), source_dialect)
            target = self.dialect_mapping.get(target_dialect.lower(), target_dialect)
            
            transpiled = transpile(sql, read=source, write=target)
            return transpiled[0] if transpiled else None
            
        except Exception as e:
            logger.error(f"SQL transpilation failed: {str(e)}")
            return None
    
    def estimate_query_cost(self, sql: str, table_sizes: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Estimate query execution cost based on operations and table sizes.
        
        Args:
            sql: SQL query to analyze
            table_sizes: Optional mapping of table names to row counts
        
        Returns:
            Cost estimation details
        """
        validation = self.validate_sql(sql)
        
        if not validation.is_valid:
            return {"estimated_cost": 0, "error": validation.error}
        
        base_cost = 1
        complexity_multiplier = 1
        
        # Cost factors
        if "JOIN" in validation.operations:
            join_cost = len(validation.tables_accessed) ** 2
            base_cost += join_cost
        
        if validation.has_subqueries:
            base_cost *= 2
        
        if "GROUP BY" in validation.operations:
            base_cost *= 1.5
        
        if "ORDER BY" in validation.operations:
            base_cost *= 1.3
        
        # Table size factor
        if table_sizes:
            total_rows = sum(
                table_sizes.get(table, 1000) 
                for table in validation.tables_accessed
            )
            complexity_multiplier = min(total_rows / 10000, 100)  # Cap at 100x
        
        estimated_cost = base_cost * complexity_multiplier
        
        return {
            "estimated_cost": estimated_cost,
            "base_cost": base_cost,
            "complexity_multiplier": complexity_multiplier,
            "tables_accessed": list(validation.tables_accessed),
            "operations": list(validation.operations),
            "complexity_score": validation.estimated_complexity
        }
    
    def suggest_optimizations(self, sql: str) -> List[str]:
        """
        Suggest query optimizations based on analysis.
        
        Args:
            sql: SQL query to analyze
        
        Returns:
            List of optimization suggestions
        """
        validation = self.validate_sql(sql)
        suggestions = []
        
        if not validation.is_valid:
            return [f"Fix SQL errors first: {validation.error}"]
        
        # Missing LIMIT
        if "SELECT" in validation.operations and "LIMIT" not in validation.operations:
            suggestions.append("Add LIMIT clause to restrict result set size")
        
        # Multiple joins without indexes
        if validation.parsed_query and validation.parsed_query.get("join_count", 0) > 2:
            suggestions.append("Consider adding indexes on join columns for better performance")
        
        # Complex WHERE conditions
        if validation.parsed_query and validation.parsed_query.get("where_conditions", 0) > 3:
            suggestions.append("Complex WHERE conditions may benefit from query restructuring")
        
        # High complexity
        if validation.estimated_complexity > 10:
            suggestions.append("Consider breaking complex query into smaller parts")
        
        # Many tables
        if len(validation.tables_accessed) > 5:
            suggestions.append("Query accesses many tables - consider if all are necessary")
        
        # Subqueries
        if validation.has_subqueries:
            suggestions.append("Consider converting subqueries to JOINs for better performance")
        
        return suggestions