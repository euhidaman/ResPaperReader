# Research Paper Reader

A lightweight LLM-based Research Assistant Agent that enables users to perform tasks through natural language commands.

## Authors

- Euhid Aman - M11315803
- Alexander Morinvil - m11352802

## Overview

This project implements a natural language-driven research assistant that helps users manage, search, and analyze research papers. The system uses a combination of Retrieval-Augmented Generation (RAG) and ReAct (Reasoning + Acting) to dynamically understand user intent and execute appropriate actions.

## Core Features

- **Natural Language Interface**: All interactions are through conversational commands
- **Internal Paper Database**:
  - Search your personal collection using keywords or semantic queries
  - Vector-based similarity search for better results
  - Automated metadata extraction from PDFs
- **Paper Upload System**:
  - Direct PDF uploads with automatic metadata extraction
  - Title and abstract parsing
  - Storage in both SQL database and vector store
- **External Paper Search**:
  - Integration with arXiv and Semantic Scholar APIs
  - Conference-specific paper searches
  - Recent paper discovery
- **Comparative Analysis**:
  - Structured comparison reports between papers
  - Analysis of research goals, methods, and contributions
  - Identification of strengths and weaknesses

## System Architecture

### Components
- **LLM Agent**: Powers natural language understanding and response generation
- **Vector Store**: ChromaDB for semantic search capabilities
- **SQL Database**: Stores paper metadata and relationships
- **PDF Processing**: Extracts and processes paper content
- **API Integration**: Connects to external research paper repositories

### Technical Stack
- **Database**: SQLite + ChromaDB for vector storage
- **PDF Processing**: pdfplumber
- **APIs**: arXiv and Semantic Scholar
- **Embeddings**: Sentence Transformers
- **Backend**: Python with LangChain integration

## Requirements

- Python 3.8+
- Google Generative AI (Gemini) API key
- Internet connection for external searches
- Minimum 4GB RAM (8GB recommended)
- Storage space for paper database

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/euhidaman/ResPaperReader.git
   cd ResPaperReader
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Create a .env file with:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   - Or provide via command line:
     ```bash
     python run.py --gemini-key=your_api_key_here
     ```

## Usage Guide

### Starting the Application
```bash
python run.py [--port=8501]
```

### Example Commands

1. **Searching Papers**
   ```
   "Find papers about contrastive learning for vision models"
   "Search for recent ICLR papers on diffusion models"
   ```

2. **Uploading Papers**
   ```
   "I want to upload a new research paper"
   "Add this PDF to my library"
   ```

3. **Comparing Papers**
   ```
   "Compare the paper I just uploaded with the most recent one about diffusion models"
   "What are the main differences between these two papers?"
   ```

4. **Managing Library**
   ```
   "Show me all papers in my library about transformers"
   "List papers I uploaded this week"
   ```

## Data Management

### Database Schema
```sql
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Storage Locations
- PDF files: `data/uploads/`
- Database: `data/papers.db`
- Vector store: `data/chroma_db/`

## Troubleshooting

Common Issues:
1. **PDF Parsing Errors**: Some PDFs may not parse correctly due to formatting
   - Solution: Try re-saving the PDF with a different tool
   
2. **Memory Usage**: High RAM usage during vector operations
   - Solution: Reduce batch size in settings or process fewer papers simultaneously

3. **API Limits**: Rate limiting from external APIs
   - Solution: Implement exponential backoff or upgrade API tier

## Self-Evaluation Table

| Objective                  | Implementation Status | Performance Rating (1-5) | Notes                                           |
| -------------------------- | --------------------- | ------------------------ | ----------------------------------------------- |
| Natural Language Interface | Complete              | 5                        | Successfully interprets various query types     |
| Internal Paper Search      | Complete              | 4                        | RAG implementation works well                   |
| PDF Upload & Processing    | Complete              | 4                        | Handles most PDF formats                        |
| External API Integration   | Complete              | 4                        | Successfully queries arXiv and Semantic Scholar |
| Paper Comparison           | Complete              | 4                        | Generates structured reports                    |
| Memory Management          | Complete              | 3                        | Maintains context during conversations          |
| Tool Selection             | Complete              | 5                        | Dynamic tool invocation works reliably          |
| Database Performance       | Complete              | 4                        | Efficient querying and storage                  |
| Vector Search              | Complete              | 4                        | Good semantic matching                          |
| Error Handling             | Complete              | 3                        | Handles most edge cases                         |

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## License

[MIT License](LICENSE)