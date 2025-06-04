<!-- 
theme: default
paginate: true
size: 16:9
marp: true
-->

# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research Paper Assistant                     │
├─────────────────────────────────────────────────────────────────┤
│                     Streamlit Frontend                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Chat Interface│  Upload Interface│    Search Interface         │
├─────────────────┼─────────────────┼─────────────────────────────┤
│                 │                 │                             │
│    Natural Language Processing Layer                            │
│                 │                 │                             │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                Research Assistant Core                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Gemini Agent  │  PDF Processor  │    Vector Store             │
├─────────────────┼─────────────────┼─────────────────────────────┤
│     LangChain   │   pdfplumber    │     ChromaDB                │
├─────────────────┼─────────────────┼─────────────────────────────┤
│                Database Layer & External APIs                   │
├─────────────────┴─────────────────┴─────────────────────────────┤
│  SQLite DB      │     arXiv API   │   File System               │
└─────────────────────────────────────────────────────────────────┘
```

## Paper Upload Workflow

1. **File Upload**: User uploads PDF through interface
2. **PDF Processing**: Extract text using PyPDFLoader/pdfplumber
3. **Metadata Extraction**: Parse title, authors, abstract
4. **Full Text Processing**: Extract and clean complete content
5. **Vector Embedding**: Create embeddings for semantic search
6. **Database Storage**: Store metadata in SQLite, vectors in ChromaDB
7. **Confirmation**: Return success with paper details