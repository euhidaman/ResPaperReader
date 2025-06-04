<!-- 
theme: default
paginate: true
size: 16:9
marp: true
-->

# System Workflows

## Search Pipeline
1. **Intent Analysis**: Determine search type & keywords
2. **Query Processing**: Clean and expand search terms
3. **Source Selection**: Choose local DB, arXiv, or both
4. **Local Search**: Query SQLite and vector search
5. **External Search**: Query arXiv API if needed
6. **Results Merging**: Combine and deduplicate results
7. **Relevance Ranking**: Score and sort by relevance
8. **Response Formatting**: Present with source attribution

## RAG Chat Pipeline
1. **Paper Selection**: Identify target paper
2. **Context Retrieval**: Vector search for relevant chunks
3. **Relevance Filtering**: Score and filter best matches
4. **Prompt Construction**: Build prompt with context
5. **LLM Processing**: Generate answer with Gemini
6. **Source Attribution**: Identify supporting passages
7. **Response Formatting**: Structure with citations
8. **Chat History Update**: Save to conversation history

## Comparison Pipeline

```
User Comparison Request
        │
        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Query Parsing   │ -> │ Paper Resolution│ -> │  Paper Retrieval│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Content Extract │ -> │  LLM Processing │ -> │ Report Generate │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Format Response │ -> │  Session Store  │ -> │  User Display   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```