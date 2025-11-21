# KubrikAI Agents and Sub-Agents Architecture Guide

## 📍 Where is the Code for Agents and Sub-Agents?

This document explains the agent architecture in KubrikAI and where to find the relevant code.

## 🎯 Current Implementation Status

### What's Described in Documentation
The README.md describes a **Mixture of Experts (MoE) architecture** with specialized expert agents:
- Schema Expert
- Optimization Expert
- Policy Expert
- Syntax Expert

**Documented Location**: `kubrikai/experts/` directory (as shown in README.md)

### What Actually Exists
The current implementation uses a **router-based architecture** instead of discrete expert agents. The functionality is distributed across core engine modules:

**Actual Locations**:
```
kubrikai/
├── core/
│   ├── router.py           # Main intelligent routing orchestrator
│   ├── schema_engine.py    # Schema understanding and similarity (Schema Expert functionality)
│   ├── policy_engine.py    # Access control and governance (Policy Expert functionality)
│   ├── sql_validator.py    # SQL validation and security (part of Syntax Expert)
│   └── mock_embeddings.py  # Embedding generation support
├── connectors/
│   ├── base.py            # Database connector abstraction
│   └── postgresql.py      # PostgreSQL-specific implementation (Syntax Expert functionality)
└── cli/
    └── main.py            # Command-line interface
```

## 🏗️ Agent Architecture Mapping

### 1. Schema Expert → `schema_engine.py`
**Location**: `kubrikai/core/schema_engine.py`

**Functionality**:
- Database schema representation and embedding
- Semantic similarity calculation
- Query intent analysis
- Schema compatibility checking
- Domain detection

**Key Classes**:
- `SchemaEngine`: Main orchestrator
- `DatabaseSchema`: Schema representation
- `QueryAnalysis`: Query understanding results

### 2. Policy Expert → `policy_engine.py`
**Location**: `kubrikai/core/policy_engine.py`

**Functionality**:
- Database access control
- Query policy validation
- Read-only enforcement
- Resource limits (row limits, execution time)
- Audit logging

**Key Classes**:
- `PolicyEngine`: Main orchestrator
- `DatabasePolicy`: Policy configuration
- `QueryContext`: Query execution context
- `PolicyViolation`: Violation representation

### 3. Syntax Expert → `sql_validator.py` + `connectors/`
**Location**: 
- `kubrikai/core/sql_validator.py`
- `kubrikai/connectors/`

**Functionality**:
- SQL syntax validation using SQLGlot
- Security pattern detection (SQL injection, dangerous operations)
- Query complexity analysis
- Database-specific SQL generation
- Multi-dialect support

**Key Classes**:
- `SQLValidator`: SQL validation engine
- `ValidationResult`: Validation results
- Database connectors: PostgreSQL, MySQL, etc. (dialect-specific)

### 4. Optimization Expert → `router.py` (Stage 2)
**Location**: `kubrikai/core/router.py`

**Functionality**:
- Performance-based database selection
- Query routing optimization
- Historical metrics tracking
- Execution time optimization
- Resource utilization consideration

**Key Classes**:
- `DatabaseRouter`: Main routing orchestrator (contains optimization logic)
- `RoutingResult`: Routing decision with performance metrics
- `QueryExecutionResult`: Execution metrics

## 🔄 How the "Agents" Work Together

The current implementation uses a **unified orchestration model** rather than discrete agents:

```
User Query
    ↓
DatabaseRouter (router.py)
    ↓
┌─────────────────────────────────────┐
│  Stage 1: Schema Similarity         │
│  → SchemaEngine.embed_query()       │ (Schema Expert)
│  → SchemaEngine.get_similar_dbs()   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Stage 2: Classification            │
│  → Domain scoring                    │ (Schema Expert)
│  → Freshness scoring                 │ (Optimization Expert)
│  → Performance scoring               │ (Optimization Expert)
│  → Policy scoring                    │ (Policy Expert)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  SQL Execution                       │
│  → SQLValidator.validate()           │ (Syntax Expert)
│  → PolicyEngine.validate_query()     │ (Policy Expert)
│  → PolicyEngine.enforce_limits()     │ (Policy Expert)
│  → Execute on selected database      │
└─────────────────────────────────────┘
```

## 📝 Code Examples

### Accessing the "Schema Expert"
```python
from kubrikai.core.schema_engine import SchemaEngine, DatabaseSchema

# Initialize schema engine
schema_engine = SchemaEngine()

# Register a database schema (Schema Expert's knowledge base)
schema = DatabaseSchema(
    db_id="ecommerce_db",
    tables=[...],
    relationships=[...]
)
schema_engine.register_database(schema)

# Analyze query intent (Schema Expert's reasoning)
query_analysis = schema_engine.analyze_query_intent("Show top customers")
print(query_analysis.domains)  # e.g., ["ecommerce"]
```

### Accessing the "Policy Expert"
```python
from kubrikai.core.policy_engine import PolicyEngine, DatabasePolicy, QueryContext

# Initialize policy engine
policy_engine = PolicyEngine()

# Configure policy (Policy Expert's rules)
policy = DatabasePolicy(
    db_id="production_db",
    read_only=True,
    max_rows=1000
)
policy_engine.register_database_policy(policy)

# Validate query (Policy Expert's enforcement)
context = QueryContext(user_id="analyst")
violations = policy_engine.validate_sql_query(sql, "production_db", context)
```

### Accessing the "Syntax Expert"
```python
from kubrikai.core.sql_validator import SQLValidator

# Initialize validator
validator = SQLValidator()

# Validate SQL (Syntax Expert's validation)
result = validator.validate_sql("SELECT * FROM users", "postgres")
print(f"Valid: {result.is_valid}")
print(f"Warnings: {result.warnings}")
```

### Using the Complete System (All Experts Together)
```python
from kubrikai.core.router import DatabaseRouter, QueryContext

# Initialize router (coordinates all experts)
router = DatabaseRouter()

# Register databases...
# router.register_database(db_info, schema)

# Route query (all experts collaborate)
context = QueryContext(user_id="analyst")
result = await router.route_query("Show top customers", context)

print(f"Selected DB: {result.selected_db_id}")
print(f"Confidence: {result.confidence_score}")
print(f"Explanation: {result.explanation}")

# Execute query (with all expert validations)
exec_result = await router.execute_query(sql, result.selected_db_id, context)
```

## 🚀 Future Evolution

### Migration to Discrete Agents
To align with the documented MoE architecture, the codebase could evolve to:

1. **Create `kubrikai/experts/` directory** with discrete agent modules:
   ```
   kubrikai/experts/
   ├── base_expert.py          # Base expert interface
   ├── schema_expert.py        # Refactor from schema_engine.py
   ├── optimization_expert.py  # Extract from router.py
   ├── policy_expert.py        # Refactor from policy_engine.py
   └── syntax_expert.py        # Refactor from sql_validator.py + connectors
   ```

2. **Implement Expert Gating Mechanism**:
   - Dynamic expert selection based on query type
   - Weighted ensemble of expert predictions
   - Cascaded expert consultation

3. **Add Sub-Agent Architecture**:
   - Each expert could have specialized sub-agents
   - Example: Schema Expert → Table Expert, Relationship Expert, Constraint Expert
   - Example: Policy Expert → Access Control Agent, Audit Agent, Limit Enforcement Agent

## 📊 Why the Current Architecture Works

The current implementation provides all the **functionality** of the documented MoE system, just organized differently:

✅ **Schema Understanding**: SchemaEngine provides semantic understanding  
✅ **Optimization**: DatabaseRouter includes performance-based selection  
✅ **Policy Enforcement**: PolicyEngine ensures governance  
✅ **Syntax Validation**: SQLValidator handles multi-dialect support  

The modular design allows for easy refactoring into discrete agents if needed.

## 🔍 Quick Reference

| Function | Current Location | Future Agent |
|----------|-----------------|--------------|
| Schema embedding & similarity | `core/schema_engine.py` | Schema Expert |
| Database selection optimization | `core/router.py` (Stage 2) | Optimization Expert |
| Policy validation & enforcement | `core/policy_engine.py` | Policy Expert |
| SQL validation & dialect support | `core/sql_validator.py` + `connectors/` | Syntax Expert |
| Query routing orchestration | `core/router.py` | MoE Gating Network |

## 📚 Related Documentation

- **Implementation Details**: See `IMPLEMENTATION.md` for detailed technical specifications
- **API Usage**: See `README.md` for usage examples and configuration
- **Testing**: See `tests/test_routing.py` for integration examples

---

**Summary**: The "agents" (experts) currently exist as integrated modules within the core package rather than as discrete agent files. All the functionality described in the README is present, just organized in a monolithic architecture rather than a microservices-style agent architecture.
