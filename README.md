# 📚 Research Paper Assistant

A sophisticated LLM-powered Research Assistant that enables natural language interactions for managing, searching, analyzing, and comparing research papers. Built with Streamlit, LangChain, and Google's Gemini AI.

## 👥 Authors

- **Euhid Aman** - M11315803
- **Alexander Morinvil** - M11352802

## 🎯 Overview

This project implements an intelligent research assistant that combines Retrieval-Augmented Generation (RAG), vector embeddings, and natural language processing to help researchers efficiently manage their paper collections. The system understands conversational commands and dynamically executes appropriate actions without requiring users to learn specific syntax.

## ✨ Core Features

### 🗣️ Natural Language Interface
- **Conversational Commands**: All interactions through natural language
- **Context-Aware Responses**: Maintains conversation history and context
- **Multi-Modal Interactions**: Text input with file upload capabilities
- **Dynamic Intent Recognition**: Automatically determines user intent from queries

### 🗄️ Internal Paper Database
- **Semantic Search**: Vector-based similarity search using ChromaDB
- **Keyword Search**: Traditional SQL-based search capabilities
- **Metadata Storage**: Automated extraction and storage of paper metadata
- **Full-Text Processing**: PDF content extraction and indexing
- **Paper Management**: Add, delete, and organize papers

### 📤 Paper Upload System
- **PDF Processing**: Automatic metadata extraction from PDFs
- **Multi-Format Support**: Handles various PDF formats and structures
- **Vector Indexing**: Creates embeddings for semantic search
- **Duplicate Detection**: Prevents duplicate paper entries
- **Progress Tracking**: Real-time upload and processing status

### 🌐 External Paper Search
- **arXiv Integration**: Direct access to arXiv repository with advanced search strategies
- **Conference Search**: Search by specific conferences (ICLR, NeurIPS, ICML, etc.)
- **Unified Search**: Combines local and external results intelligently
- **Paper Storage**: Option to save external papers to local library

### 🔄 Comparative Analysis
- **Intelligent Comparison**: AI-powered structured comparison reports
- **Multi-Source Comparison**: Compare papers from different sources
- **Flexible Paper Selection**: Reference papers by title, index, or description
- **Detailed Reports**: Covers methodology, findings, strengths, and limitations

### 💬 Paper-Specific Chat (RAG)
- **Document Q&A**: Ask questions about specific papers
- **Context-Aware Responses**: Answers based on paper content
- **Source Attribution**: Shows which parts of the paper informed the answer
- **Multi-Paper Sessions**: Switch between different paper conversations

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research Paper Assistant                     │
├─────────────────────────────────────────────────────────────────┤
│                     Streamlit Frontend                         │
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
│                 │                 │                             │
│                Database Layer & External APIs                   │
│                 │                 │                             │
├─────────────────┴─────────────────┴─────────────────────────────┤
│  SQLite DB      │     arXiv API   │   File System               │
└─────────────────────────────────────────────────────────────────┘
```

### 🧩 Components

#### **Core Engine**
- **ResearchAssistant**: Main orchestrator handling all user interactions
- **GeminiAgent**: LLM integration for natural language understanding
- **PDF Processor**: Document parsing and text extraction
- **Vector Store**: Semantic embedding storage and retrieval

#### **Data Layer**
- **SQLite Database**: Paper metadata and relationships
- **ChromaDB**: Vector embeddings for semantic search
- **File System**: PDF storage and management

#### **External Integrations**
- **arXiv API**: Academic paper repository access
- **Google Gemini**: Large language model for AI capabilities

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 1GB+ for paper database
- **Internet**: Required for external searches and AI

### API Requirements
- **Google Gemini API Key**: Required for AI functionality
- **Internet Connection**: For arXiv searches

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/euhidaman/ResPaperReader.git
cd ResPaperReader
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
# Option 1: Environment Variable
export GEMINI_API_KEY="your_api_key_here"

# Option 2: Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Option 3: Enter via UI (after starting the app)
```

### 4. Start Application
```bash
python run.py
# or specify port
python run.py --port=8502
```

## 🎮 Usage Guide

### 🚀 Starting the Application
```bash
python run.py
# Navigate to http://localhost:8501 in your browser
```

## 💬 Natural Language Commands

### 📖 Paper Search Commands

#### **Basic Search**
```
"Find papers about transformer architectures"
"Search for machine learning papers"
"Look for recent papers on diffusion models"
"Show me papers about computer vision"
```

#### **Advanced Search**
```
"Find papers on TinyML from 2023"
"Search for ICLR papers about reinforcement learning"
"Look for papers by Geoffrey Hinton"
"Find recent papers on knowledge distillation"
```

#### **Conference-Specific Search**
```
"Search ICLR 2023 papers"
"Find NeurIPS papers on attention mechanisms"
"Show me ICML papers about GANs"
```

### 📤 Upload Commands

#### **Basic Upload**
```
"Upload a paper"
"I want to add a research paper"
"Upload this PDF to my library"
"Add a new paper"
```

#### **Upload with Context**
```
"Upload the paper I just downloaded"
"Add this transformer paper to my collection"
"I have a new paper on diffusion models to upload"
```

### 🔍 Paper Analysis Commands

#### **Paper Information**
```
"What is this paper about?"
"Tell me about the paper [title]"
"Summarize the paper I just uploaded"
"Explain the methodology in [paper title]"
```

#### **Specific Questions**
```
"What datasets were used in [paper title]?"
"What are the main contributions of this paper?"
"How does this paper compare to previous work?"
"What are the limitations mentioned?"
```

### 🔄 Comparison Commands

#### **Simple Comparison**
```
"Compare these two papers"
"What's the difference between paper 1 and paper 2?"
"Compare [paper title 1] with [paper title 2]"
```

#### **Advanced Comparison**
```
"Compare my uploaded paper with the first paper on transformers"
"Compare the latest diffusion model paper with my paper"
"What are the differences between the second paper you found and my paper?"
```

### 📚 Library Management Commands

#### **Library Overview**
```
"What papers do I have?"
"Show me my library"
"List all papers in my database"
"How many papers do I have?"
```

#### **Specific Library Queries**
```
"Show me papers about deep learning"
"List papers from 2023"
"What papers did I upload this week?"
"Find papers by [author name] in my library"
```

### 💾 Storage Commands

#### **Save External Papers**
```
"Store paper 1"
"Save the third paper to my library"
"Download and store paper 2"
"Add the first search result to my collection"
```

### 💬 Chat Commands

#### **Start Paper Chat**
```
"Chat with [paper title]"
"I want to discuss the transformer paper"
"Ask questions about [paper title]"
"Start a conversation with my latest paper"
```

#### **Exit Paper Chat**
```
"Exit chat"
"Return to main chat"
"Stop paper discussion"
"Back to general chat"
```

## 🔄 System Pipelines

### 📤 Paper Upload Pipeline

```
User Upload Request
        │
        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │ -> │  PDF Processing │ -> │ Metadata Extract│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Full Text Extract│ -> │ Vector Embedding│ -> │  Database Store │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Session Update │ -> │   Index Update  │ -> │ Success Response│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Steps Explained:**
1. **File Upload**: User uploads PDF through Streamlit interface
2. **PDF Processing**: Extract text content using pdfplumber
3. **Metadata Extract**: Parse title, authors, abstract from content
4. **Full Text Extract**: Process complete document text
5. **Vector Embedding**: Create semantic embeddings using sentence transformers
6. **Database Store**: Save metadata to SQLite, embeddings to ChromaDB
7. **Session Update**: Update in-memory session state
8. **Index Update**: Refresh search indices
9. **Success Response**: Confirm upload and display paper details

### 🔍 Search Pipeline

```
Natural Language Query
        │
        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Intent Analysis │ -> │ Query Processing│ -> │  Source Decision│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Local DB Search │ -> │ Vector Search   │ -> │ arXiv Search    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Results Merge   │ -> │ Relevance Score │ -> │ Format Response │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Search Strategy:**
1. **Intent Analysis**: Determine search type and extract keywords
2. **Query Processing**: Clean and expand search terms
3. **Source Decision**: Choose local DB, arXiv, or both
4. **Local DB Search**: SQL and vector search in local database
5. **Vector Search**: Semantic similarity search using embeddings
6. **arXiv Search**: Query arXiv API with multiple search strategies
7. **Results Merge**: Combine and deduplicate results
8. **Relevance Score**: Rank results by relevance
9. **Format Response**: Present results with source attribution

### 🔄 Comparison Pipeline

```
Comparison Request
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

**Comparison Process:**
1. **Query Parsing**: Extract paper references from natural language
2. **Paper Resolution**: Resolve references to specific papers
3. **Paper Retrieval**: Fetch full paper content and metadata
4. **Content Extract**: Prepare abstracts and full text for comparison
5. **LLM Processing**: Generate structured comparison using Gemini
6. **Report Generate**: Create detailed comparison report
7. **Format Response**: Structure output with clear sections
8. **Session Store**: Save comparison for future reference
9. **User Display**: Present formatted comparison to user

### 💬 RAG Chat Pipeline

```
User Question about Paper
        │
        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Paper Selection │ -> │ Context Retrieval│ -> │ Relevance Filter│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Prompt Building │ -> │  LLM Generation │ -> │ Source Citation │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Response Format │ -> │  Chat Update    │ -> │  User Display   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**RAG Process:**
1. **Paper Selection**: Identify target paper for conversation
2. **Context Retrieval**: Find relevant chunks using vector search
3. **Relevance Filter**: Score and filter most relevant sections
4. **Prompt Building**: Construct prompt with context and question
5. **LLM Generation**: Generate answer using Gemini with context
6. **Source Citation**: Identify which parts of paper informed answer
7. **Response Format**: Structure response with proper formatting
8. **Chat Update**: Update conversation history
9. **User Display**: Show answer with expandable source references

## 🎛️ User Interface Guide

### 📱 Main Navigation

The application features five main sections accessible via the sidebar:

#### 1. 💬 **Chat Assistant**
- **Primary Interface**: Main conversational interface
- **Natural Language**: Type commands and questions naturally
- **Context Awareness**: Maintains conversation history
- **File Upload**: Direct PDF upload from chat
- **Real-time Processing**: Live response generation

#### 2. 📤 **Upload Papers**
- **Drag & Drop**: Easy file upload interface
- **Progress Tracking**: Real-time upload and processing status
- **Paper Preview**: Immediate display of extracted metadata
- **Error Handling**: Clear feedback for upload issues

#### 3. 🔍 **Search Papers**

**Internal Search Tab:**
- Search your personal paper collection
- Semantic and keyword search capabilities
- Instant results with relevance scoring

**Web Search Tab:**
- Query arXiv database
- Advanced search strategies for better results
- Intelligent result merging with local papers

**Conference Search Tab:**
- Search by specific conferences
- Year-based filtering
- Venue-specific paper discovery

#### 4. 📚 **My Library**
- **Complete Collection**: View all your papers
- **Management Tools**: Delete, analyze, and organize papers
- **Search Integration**: Find papers in your collection
- **Metadata Display**: Rich paper information display

#### 5. 💬 **Chat with Papers**
- **Paper Selection**: Choose specific papers for conversation
- **RAG Interface**: Ask detailed questions about paper content
- **Source Attribution**: See which parts of the paper informed answers
- **Conversation History**: Maintain separate chat history per paper

### 🎨 Chat Interface Features

#### **Visual Distinction**
- **USER Badge**: Blue background for your messages
- **AI Badge**: Orange background for assistant responses
- **Larger Font**: 19px font size for better readability
- **Source Expandables**: Collapsible sections for detailed information

#### **Interactive Elements**
- **Live File Upload**: Upload papers directly in chat
- **Expandable Results**: Click to see detailed search results
- **Source Citations**: View paper excerpts that informed answers
- **Progress Indicators**: Real-time processing status

## 🔧 Configuration

### 🔑 API Configuration

#### **Gemini API Setup**
1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Set Environment Variable**:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. **Or use .env file**:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```
4. **Or configure via UI**: Enter key in the sidebar API Settings

### ⚙️ Advanced Configuration

#### **Database Settings**
- **Location**: `data/papers.db` (SQLite)
- **Vector Store**: `data/chroma_db/` (ChromaDB)
- **Uploads**: `data/uploads/` (PDF files)

#### **Performance Tuning**
```python
# Modify in research_assistant.py
VECTOR_SEARCH_K = 5  # Number of similar chunks to retrieve
MAX_TEXT_LENGTH = 30000  # Maximum text length for processing
SEARCH_RESULTS_LIMIT = 10  # Default search result limit
```

## 📊 Data Management

### 🗃️ Database Schema

#### **Papers Table (SQLite)**
```sql
CREATE TABLE papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,  -- JSON array of author names
    source TEXT,   -- 'upload', 'arxiv'
    file_path TEXT,
    full_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **Vector Store (ChromaDB)**
- **Documents**: Text chunks from papers
- **Embeddings**: Semantic vector representations
- **Metadata**: Paper ID, page numbers, chunk indices
- **Filters**: Enable paper-specific search

### 💾 Storage Organization

```
data/
├── papers.db              # Main SQLite database
├── chroma_db/            # Vector embeddings
│   ├── chroma.sqlite3    # ChromaDB index
│   └── ...
└── uploads/              # PDF files
    ├── 20250526_210113_paper1.pdf
    ├── 20250526_215117_paper2.pdf
    └── ...
```

### 🔄 Data Flow

1. **PDF Upload** → **Text Extraction** → **Metadata Parsing**
2. **Database Storage** (SQLite) + **Vector Indexing** (ChromaDB)
3. **Search Query** → **Vector Similarity** + **SQL Search**
4. **Result Ranking** → **User Display**

## 🛠️ Troubleshooting

### ❌ Common Issues

#### **1. PDF Parsing Errors**
**Problem**: Some PDFs fail to parse correctly
**Solutions**:
- Ensure PDF is not password-protected
- Try re-saving PDF with different tools
- Check for corrupted or scanned-only PDFs
- Verify PDF has extractable text

#### **2. API Key Issues**
**Problem**: Gemini API not working
**Solutions**:
```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Test API access
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$GEMINI_API_KEY"
```

#### **3. Memory Issues**
**Problem**: High RAM usage during processing
**Solutions**:
- Process fewer papers simultaneously
- Reduce `MAX_TEXT_LENGTH` in configuration
- Restart application periodically
- Upgrade system RAM

#### **4. Search Results Issues**
**Problem**: Poor search results or no results
**Solutions**:
- Try different search terms
- Use more specific keywords
- Check if papers are properly indexed
- Rebuild vector index if necessary

#### **5. Upload Failures**
**Problem**: Papers fail to upload
**Solutions**:
- Check file permissions in `data/uploads/`
- Ensure sufficient disk space
- Verify PDF file integrity
- Check logs for specific error messages

### 🔍 Debugging

#### **Enable Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### **Check Database**
```bash
sqlite3 data/papers.db
.tables
SELECT COUNT(*) FROM papers;
.quit
```

#### **Vector Store Status**
```python
from models.vector_store import VectorStore
vs = VectorStore()
print(f"Vector count: {vs.db._collection.count()}")
```

## 📈 Performance Optimization

### ⚡ Speed Improvements

#### **Vector Search Optimization**
- **Batch Processing**: Process multiple papers together
- **Index Optimization**: Regular vector index maintenance
- **Caching**: Cache frequent search results
- **Parallel Processing**: Multi-threaded PDF processing

#### **Database Optimization**
```sql
-- Add indices for faster search
CREATE INDEX idx_papers_title ON papers(title);
CREATE INDEX idx_papers_source ON papers(source);
CREATE INDEX idx_papers_created_at ON papers(created_at);
```

#### **Memory Management**
- **Lazy Loading**: Load papers on-demand
- **Text Chunking**: Process large papers in smaller chunks
- **Session Cleanup**: Regular cleanup of conversation history

## 🧪 Advanced Features

### 🤖 Custom Prompts

Modify system prompts in `models/gemini_agent.py`:

```python
def get_system_prompt(self):
    return """You are a specialized research assistant for [your domain].
    Focus on [specific research areas].
    Provide detailed analysis of [specific aspects].
    """
```

### 🔧 Plugin Development

Create custom search plugins:

```python
# models/custom_source.py
class CustomSource:
    def search(self, query: str) -> List[Dict]:
        # Implement custom search logic
        return results
```

### 📊 Analytics Integration

Add usage tracking:

```python
# Track user interactions
def log_interaction(action: str, query: str):
    # Implement analytics logging
    pass
```

## 🧪 Testing

### 🔬 Unit Tests

```bash
# Run basic tests
python -m pytest tests/

# Test specific components
python test_db.py
```

### 🎯 Integration Tests

```bash
# Test complete workflows
python tests/test_upload_pipeline.py
python tests/test_search_pipeline.py
python tests/test_comparison_pipeline.py
```

### 📝 Manual Testing Checklist

- [ ] Upload various PDF formats
- [ ] Test search with different queries
- [ ] Verify comparison functionality
- [ ] Check RAG chat responses
- [ ] Test error handling
- [ ] Validate data persistence

## 🔮 Future Enhancements

### 📋 Planned Features
- **Semantic Scholar Integration**: Add Semantic Scholar API support
- **Multi-language Support**: Support for non-English papers
- **Citation Network**: Visualize paper relationships
- **Collaborative Features**: Share libraries between users
- **Advanced Analytics**: Usage statistics and insights
- **Mobile Interface**: Responsive design for mobile devices

### 🛣️ Roadmap
1. **Q1 2025**: Semantic Scholar API integration
2. **Q2 2025**: Citation network visualization
3. **Q3 2025**: Multi-language support
4. **Q4 2025**: Collaborative features

## 🤝 Contributing

### 📝 Development Setup

```bash
# Clone repository
git clone https://github.com/euhidaman/ResPaperReader.git
cd ResPaperReader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_db.py

# Start development server
python run.py
```

### 🔄 Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 📋 Code Standards
- **Python Style**: Follow PEP 8
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features
- **Type Hints**: Use type annotations
- **Logging**: Appropriate logging levels

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: Framework for LLM application development
- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for embeddings
- **Google Gemini**: AI language model
- **arXiv**: Open access research paper repository
- **pdfplumber**: PDF text extraction library

## 📞 Support

For support, questions, or feature requests:

- **GitHub Issues**: [Repository Issues](https://github.com/your-username/ResPaperReader/issues)
- **Documentation**: [Project Wiki](https://github.com/your-username/ResPaperReader/wiki)

---

**Built with ❤️ for the research community**