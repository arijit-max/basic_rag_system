                RAG
                 │
       ┌─────────┴─────────┐
       │                   │
   RETRIEVAL           GENERATION
       │                   │
       ↓                   ↓
retriever.invoke()     llm.invoke()
       │                   │
       ↓                   ↓
Relevant docs          Final answer