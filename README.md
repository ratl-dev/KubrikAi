# KubrikAI

KubrikAI is a Python package for intelligent database routing and SQL governance. It helps route natural-language requests to the most appropriate database, validate SQL safety, and apply access policies before query execution.

This project is built around a two-stage routing system:

- Stage 1: schema embedding similarity to shortlist likely databases
- Stage 2: rule-based scoring based on domain fit, freshness, performance, and policy compliance

It is designed for multi-database environments where safety, governance, and query routing matter.

## Why this project exists

Most data systems are still built around static database selection and ad hoc SQL execution. KubrikAI adds a lightweight decision layer that:

- understands database schema structure
- evaluates query intent against schema semantics
- enforces read-only and safety policies
- reduces accidental access to the wrong data source
- supports multi-tenant or multi-domain database setups

## Features

- Smart database routing using schema embeddings
- Query intent analysis for domain and schema matching
- Policy engine for access control and read-only enforcement
- SQL validation and safety checks
- Configurable database policies and routing thresholds
- Example demo for multi-database routing scenarios

## Project layout

```text
KubrikAi/
├── kubrikai/
│   ├── __init__.py
│   ├── cli/
│   ├── connectors/
│   └── core/
│       ├── policy_engine.py
│       ├── router.py
│       ├── schema_engine.py
│       └── sql_validator.py
├── examples/
│   └── routing_demo.py
├── tests/
├── config.yaml
├── IMPLEMENTATION.md
├── pyproject.toml
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repo and install it locally:

```bash
git clone https://github.com/ratl-dev/KubrikAi.git
cd KubrikAi
pip install -e .
```

## Quick start

Run the included demo to see intelligent routing in action:

```bash
python examples/routing_demo.py
```

## Basic usage

```python
from kubrikai.core import DatabaseRouter, DatabaseSchema, TableSchema, DatabasePolicy, QueryContext
from kubrikai.core.router import DatabaseInfo
from datetime import datetime

router = DatabaseRouter()

# Example: register database metadata + schema
# router.register_database(db_info, schema)

# Example: policies
# router.policy_engine.register_database_policy(DatabasePolicy(db_id="analytics_warehouse", read_only=True, max_rows=1000))

# Example: route a query
# context = QueryContext(
#     user_id="analyst_1",
#     session_id="session_1",
#     query="Show top customers by total order value",
#     db_id="",
#     timestamp=datetime.now(),
#     user_roles=["analyst"],
#     request_metadata={"domain": "business"}
# )
# result = await router.route_query("Show top customers by total order value", context)
# print(result.selected_db_id, result.confidence_score)
```

## Architecture

### 1. Schema engine

The schema engine converts database metadata into structured text and uses embeddings to quantify similarity between a query and available database schemas.

### 2. Routing engine

The router performs a two-pass selection process:

- shortlist likely databases using embedding similarity
- rank candidates using domain fit, freshness, policy safety, and performance metrics

### 3. Policy engine

The policy layer enforces:

- read-only access requirements
- maximum row counts
- maximum execution time
- blocked tables/functions
- domain restrictions and freshness checks

## Example domains supported by the demo

The repository includes demo databases for:

- ecommerce
- analytics
- hr

This makes it easy to model realistic routing logic across different business domains.

## Notes

This repo is a strong foundation for a database routing and SQL safety layer, but it is intentionally lightweight and demo-oriented. The implementation focuses on architecture, validation, and policy enforcement rather than a production database connector layer.

## License

This project is released under the MIT license.

## Contributing

Contributions are welcome. If you are working on this project:

```bash
git checkout -b feature/my-change
pytest
```

For more detailed implementation notes, see `IMPLEMENTATION.md`.
