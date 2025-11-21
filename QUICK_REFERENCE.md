# Quick Reference: Where is the Agent Code?

## 🎯 Quick Answer

The "agents" (experts) in KubrikAI are currently implemented as integrated modules in the `kubrikai/core/` directory rather than as discrete agent files.

## 📍 Code Location Map

| Expert/Agent Type | File Location | Key Classes |
|------------------|---------------|-------------|
| **Schema Expert** | `kubrikai/core/schema_engine.py` | `SchemaEngine`, `DatabaseSchema` |
| **Policy Expert** | `kubrikai/core/policy_engine.py` | `PolicyEngine`, `DatabasePolicy` |
| **Syntax Expert** | `kubrikai/core/sql_validator.py` | `SQLValidator`, `ValidationResult` |
| **Optimization Expert** | `kubrikai/core/router.py` | `DatabaseRouter` (Stage 2) |
| **Main Orchestrator** | `kubrikai/core/router.py` | `DatabaseRouter` |

## 🔍 Directory Structure

```
kubrikai/
├── core/
│   ├── router.py           ← Main orchestrator + Optimization Expert
│   ├── schema_engine.py    ← Schema Expert
│   ├── policy_engine.py    ← Policy Expert
│   ├── sql_validator.py    ← Syntax Expert
│   └── mock_embeddings.py  ← Embedding utilities
├── connectors/             ← Database-specific implementations
│   ├── base.py
│   └── postgresql.py
└── cli/
    └── main.py
```

## 💡 Why No `experts/` Directory?

The README.md describes a future **Mixture of Experts (MoE)** architecture with a planned `kubrikai/experts/` directory, but the current implementation uses an **integrated router-based architecture** where expert functionality is embedded in core modules.

## 📚 Learn More

- **Detailed Guide**: See [AGENTS_ARCHITECTURE.md](AGENTS_ARCHITECTURE.md) for complete documentation
- **Implementation**: See [IMPLEMENTATION.md](IMPLEMENTATION.md) for technical details
- **Examples**: See `examples/routing_demo.py` for usage examples

## 🚀 Quick Start

```python
from kubrikai.core import DatabaseRouter, QueryContext

# Initialize the router (coordinates all experts)
router = DatabaseRouter()

# The router internally uses:
# - schema_engine (Schema Expert)
# - policy_engine (Policy Expert)
# - sql_validator (Syntax Expert)
# - routing logic (Optimization Expert)

# Route a query (all experts work together)
context = QueryContext(user_id="analyst")
result = await router.route_query("Show top customers", context)
```

---

**Bottom Line**: All expert functionality exists and works - it's just organized in core modules rather than separate agent files. See AGENTS_ARCHITECTURE.md for the full mapping.
