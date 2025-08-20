"""
SQL Validator - Validates SQL queries using SQLGlot parsing and EXPLAIN analysis
to ensure queries are safe, correct, and performant before execution.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlglot
import re
import time

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"      # Syntax only
    STANDARD = "standard"  # Syntax + safety checks
    STRICT = "strict"    # Full validation including performance


@dataclass
class ValidationResult:
    """Result of SQL validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    estimated_cost: Optional[float] = None
    execution_plan: Optional[Dict[str, Any]] = None
    rewritten_query: Optional[str] = None
    validation_time_ms: float = 0.0


class SQLValidator:
    """
    Validates SQL queries using SQLGlot for parsing and optional database
    connection for EXPLAIN analysis.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize the SQL validator.
        
        Args:
            validation_level: Level of validation to perform
        """
        self.validation_level = validation_level
        self.forbidden_keywords = {
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
        }
        self.max_query_complexity = 100  # Arbitrary complexity score limit
        
    def parse_sql(self, query: str, dialect: str = "postgres") -> Tuple[bool, List[str], Any]:
        """
        Parse SQL using SQLGlot to check syntax and structure.
        
        Args:
            query: SQL query to parse
            dialect: SQL dialect to use for parsing
            
        Returns:
            Tuple of (is_valid, errors, parsed_ast)
        """
        errors = []
        parsed_ast = None
        
        try:
            # Clean the query
            query = query.strip()
            if not query:
                errors.append("Query is empty")
                return False, errors, None
            
            # Parse with SQLGlot
            parsed_ast = sqlglot.parse_one(query, dialect=dialect)
            
            if parsed_ast is None:
                errors.append("Failed to parse SQL query")
                return False, errors, None
                
        except Exception as e:
            errors.append(f"SQL parsing error: {str(e)}")
            return False, errors, None
            
        return True, errors, parsed_ast
    
    def check_read_only(self, parsed_ast: Any) -> List[str]:
        """
        Check if query is read-only (no data modification).
        
        Args:
            parsed_ast: Parsed SQL AST from SQLGlot
            
        Returns:
            List of errors if query is not read-only
        """
        errors = []
        
        if parsed_ast is None:
            return ["Cannot check read-only: query not parsed"]
        
        # Check the type of the statement
        statement_type = type(parsed_ast).__name__.upper()
        
        if statement_type not in ['SELECT', 'WITH']:
            errors.append(f"Non-read-only statement detected: {statement_type}")
        
        # Check for forbidden keywords in the query string
        query_upper = str(parsed_ast).upper()
        for keyword in self.forbidden_keywords:
            if re.search(r'\b' + keyword + r'\b', query_upper):
                errors.append(f"Forbidden keyword detected: {keyword}")
        
        return errors
    
    def check_query_complexity(self, parsed_ast: Any) -> List[str]:
        """
        Check query complexity to prevent resource-intensive queries.
        
        Args:
            parsed_ast: Parsed SQL AST from SQLGlot
            
        Returns:
            List of warnings for complex queries
        """
        warnings = []
        
        if parsed_ast is None:
            return ["Cannot check complexity: query not parsed"]
        
        complexity_score = 0
        query_str = str(parsed_ast).upper()
        
        # Count complexity factors
        complexity_score += query_str.count('JOIN') * 10
        complexity_score += query_str.count('UNION') * 15
        complexity_score += query_str.count('SUBQUERY') * 20
        complexity_score += query_str.count('WINDOW') * 25
        complexity_score += query_str.count('RECURSIVE') * 50
        complexity_score += len(re.findall(r'\bSELECT\b', query_str)) * 5
        
        if complexity_score > self.max_query_complexity:
            warnings.append(
                f"High query complexity detected (score: {complexity_score}). "
                f"Consider simplifying the query."
            )
        
        return warnings
    
    def check_security_issues(self, query: str, parsed_ast: Any) -> List[str]:
        """
        Check for potential security issues in the query.
        
        Args:
            query: Original query string
            parsed_ast: Parsed SQL AST
            
        Returns:
            List of security-related errors
        """
        errors = []
        
        # Check for SQL injection patterns
        injection_patterns = [
            r"';\s*DROP\s+TABLE",
            r"';\s*DELETE\s+FROM",
            r"UNION\s+SELECT.*FROM\s+INFORMATION_SCHEMA",
            r"';--",
            r"'\s+OR\s+'1'\s*=\s*'1",
        ]
        
        query_upper = query.upper()
        for pattern in injection_patterns:
            if re.search(pattern, query_upper):
                errors.append(f"Potential SQL injection pattern detected: {pattern}")
        
        # Check for attempts to access system tables
        system_tables = [
            'INFORMATION_SCHEMA', 'PG_CATALOG', 'MYSQL.USER', 
            'SYS.', 'SYSTEM.', '__SCHEMA__'
        ]
        
        for table in system_tables:
            if table in query_upper:
                errors.append(f"Access to system table detected: {table}")
        
        return errors
    
    def optimize_query(self, parsed_ast: Any, dialect: str = "postgres") -> Optional[str]:
        """
        Attempt to optimize the query using SQLGlot transformations.
        
        Args:
            parsed_ast: Parsed SQL AST
            dialect: Target SQL dialect
            
        Returns:
            Optimized query string or None if optimization failed
        """
        try:
            if parsed_ast is None:
                return None
            
            # Apply basic optimizations
            optimized = sqlglot.optimizer.optimize(parsed_ast, dialect=dialect)
            
            return str(optimized)
            
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return None
    
    def validate_with_explain(self, query: str, connection, max_cost: float = 1000.0) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate query using database EXPLAIN to check execution plan and cost.
        
        Args:
            query: SQL query to validate
            connection: Database connection object
            max_cost: Maximum allowed query cost
            
        Returns:
            Tuple of (is_valid, execution_plan, errors)
        """
        errors = []
        execution_plan = {}
        
        try:
            # Run EXPLAIN without executing the query
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE FALSE) {query}"
            
            cursor = connection.cursor()
            cursor.execute(explain_query)
            result = cursor.fetchone()
            
            if result:
                execution_plan = result[0] if isinstance(result[0], dict) else {}
                
                # Extract cost information (PostgreSQL format)
                if 'Plan' in execution_plan:
                    total_cost = execution_plan['Plan'].get('Total Cost', 0)
                    
                    if total_cost > max_cost:
                        errors.append(
                            f"Query cost too high: {total_cost:.2f} > {max_cost}"
                        )
                        return False, execution_plan, errors
            
            cursor.close()
            
        except Exception as e:
            errors.append(f"EXPLAIN validation failed: {str(e)}")
            return False, {}, errors
        
        return True, execution_plan, errors
    
    def validate_query(self, query: str, dialect: str = "postgres", 
                      connection=None) -> ValidationResult:
        """
        Perform comprehensive SQL validation.
        
        Args:
            query: SQL query to validate
            dialect: SQL dialect for parsing
            connection: Optional database connection for EXPLAIN analysis
            
        Returns:
            Validation result with errors, warnings, and optimization info
        """
        start_time = time.time()
        errors = []
        warnings = []
        execution_plan = None
        rewritten_query = None
        estimated_cost = None
        
        logger.info(f"Validating query with {self.validation_level.value} level")
        
        # Step 1: Parse SQL
        is_parsed, parse_errors, parsed_ast = self.parse_sql(query, dialect)
        errors.extend(parse_errors)
        
        if not is_parsed:
            validation_time = (time.time() - start_time) * 1000
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                validation_time_ms=validation_time
            )
        
        # Step 2: Check read-only constraints
        readonly_errors = self.check_read_only(parsed_ast)
        errors.extend(readonly_errors)
        
        # Step 3: Security checks (STANDARD and STRICT levels)
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            security_errors = self.check_security_issues(query, parsed_ast)
            errors.extend(security_errors)
        
        # Step 4: Complexity checks
        complexity_warnings = self.check_query_complexity(parsed_ast)
        warnings.extend(complexity_warnings)
        
        # Step 5: Query optimization (STRICT level)
        if self.validation_level == ValidationLevel.STRICT:
            rewritten_query = self.optimize_query(parsed_ast, dialect)
        
        # Step 6: EXPLAIN analysis (if connection provided)
        if connection and self.validation_level == ValidationLevel.STRICT:
            explain_valid, execution_plan, explain_errors = self.validate_with_explain(
                query, connection
            )
            errors.extend(explain_errors)
            
            # Extract estimated cost
            if execution_plan and 'Plan' in execution_plan:
                estimated_cost = execution_plan['Plan'].get('Total Cost')
        
        validation_time = (time.time() - start_time) * 1000
        is_valid = len(errors) == 0
        
        logger.info(f"Validation completed in {validation_time:.2f}ms, valid: {is_valid}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            estimated_cost=estimated_cost,
            execution_plan=execution_plan,
            rewritten_query=rewritten_query,
            validation_time_ms=validation_time
        )
    
    def set_max_complexity(self, max_complexity: int) -> None:
        """Set maximum allowed query complexity score."""
        self.max_query_complexity = max_complexity
        
    def add_forbidden_keyword(self, keyword: str) -> None:
        """Add a keyword to the forbidden list."""
        self.forbidden_keywords.add(keyword.upper())
        
    def remove_forbidden_keyword(self, keyword: str) -> None:
        """Remove a keyword from the forbidden list."""
        self.forbidden_keywords.discard(keyword.upper())