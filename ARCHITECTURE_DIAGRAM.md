# KubrikAI Architecture Diagram

## 🎯 Question: Where is the Code for Agents and Sub-Agents?

**Answer**: The agents (experts) are integrated modules in `kubrikai/core/`, not separate agent files.

## 🏗️ Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Natural Language Query                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               DatabaseRouter (router.py)                         │
│                   Main Orchestrator                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Schema Expert  │ │Policy Expert │ │ Syntax Expert    │
│                 │ │              │ │                  │
│ schema_engine.py│ │policy_engine │ │sql_validator.py  │
│                 │ │      .py     │ │+ connectors/     │
└────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
         │                 │                   │
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Optimization Expert    │
              │  (within router.py)     │
              │  Stage 2 Classification │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Selected Database     │
              │   Query Execution       │
              └────────────────────────┘
```

## 📊 Component Breakdown

### Stage 1: Query Analysis (Schema Expert)
```
User Query → SchemaEngine
    ↓
- Generate query embedding
- Find similar database schemas
- Analyze query intent
- Detect domain (ecommerce, analytics, etc.)
    ↓
Shortlist of candidate databases
```

### Stage 2: Classification (All Experts Collaborate)
```
Candidate Databases → DatabaseRouter Stage 2
    ↓
┌─────────────────────────────────────────┐
│ Schema Expert: Domain matching          │
│ Optimization Expert: Performance scoring│
│ Policy Expert: Compliance checking      │
│ Schema Expert: Compatibility checking   │
└─────────────────────────────────────────┘
    ↓
Best database selected with confidence score
```

### Stage 3: Validation & Execution
```
SQL Query → Validation Pipeline
    ↓
┌──────────────────────────────────┐
│ Syntax Expert: SQL validation    │
│ Policy Expert: Access control    │
│ Policy Expert: Resource limits   │
└──────────────────────────────────┘
    ↓
Execute on selected database
```

## 🗂️ File-to-Expert Mapping

```
kubrikai/core/
│
├── router.py ─────────────┐
│   ├── DatabaseRouter     │ Main Orchestrator
│   ├── Stage 1 Logic      │ (coordinates all experts)
│   └── Stage 2 Logic      └─ Optimization Expert
│
├── schema_engine.py ──────┐
│   ├── SchemaEngine       │ Schema Expert
│   ├── DatabaseSchema     │ (schema understanding,
│   └── QueryAnalysis      │  similarity, domain detection)
│
├── policy_engine.py ──────┐
│   ├── PolicyEngine       │ Policy Expert
│   ├── DatabasePolicy     │ (access control, governance,
│   └── QueryContext       │  resource limits, audit)
│
└── sql_validator.py ──────┐
    ├── SQLValidator       │ Syntax Expert
    └── ValidationResult   │ (SQL validation, security,
                           │  multi-dialect support)
                           
connectors/ ───────────────┐
    ├── base.py            │ Syntax Expert
    └── postgresql.py      │ (database-specific
                           │  implementations)
```

## 🔄 Data Flow Example

**Query**: "Show top customers by revenue"

```
1. User Input
   └─> "Show top customers by revenue"

2. DatabaseRouter.route_query()
   └─> SchemaEngine.embed_query()
       └─> Query embedding generated
   
3. Stage 1: Schema Similarity
   └─> SchemaEngine.get_similar_databases()
       └─> Candidates: [ecommerce_db, sales_db, analytics_db]

4. Stage 2: Classification
   ├─> SchemaEngine.analyze_query_intent()
   │   └─> Domain: "ecommerce", Entities: ["customers", "revenue"]
   ├─> SchemaEngine.get_schema_compatibility()
   │   └─> ecommerce_db has "customers" and "orders" tables ✓
   ├─> DatabaseRouter._calculate_performance_score()
   │   └─> ecommerce_db: high success rate, fast response ✓
   └─> PolicyEngine.evaluate_database_access()
       └─> User has read access to ecommerce_db ✓

5. Selection
   └─> Best match: ecommerce_db (confidence: 0.89)

6. SQL Generation & Validation
   ├─> Generate SQL query
   ├─> SQLValidator.validate_sql()
   │   └─> Syntax valid ✓, No security issues ✓
   └─> PolicyEngine.validate_sql_query()
       └─> No policy violations ✓

7. Execution
   └─> Execute on ecommerce_db
       └─> Return results to user
```

## 💡 Why This Architecture?

**Current (Integrated)**:
- ✅ Simpler to understand and maintain
- ✅ Faster inter-component communication
- ✅ All expert functionality present
- ✅ Easier testing and debugging

**Future (Discrete Agents)**:
- 🔮 More modular and extensible
- 🔮 Independent expert evolution
- 🔮 Dynamic expert selection
- 🔮 Support for sub-agents

## 📚 Learn More

- **Quick Lookup**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Detailed Guide**: [AGENTS_ARCHITECTURE.md](AGENTS_ARCHITECTURE.md)
- **Implementation**: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Examples**: `examples/routing_demo.py`

---

**Key Takeaway**: The "agents" exist and work - they're just integrated modules rather than separate files. All the expert intelligence described in the README is fully functional.
