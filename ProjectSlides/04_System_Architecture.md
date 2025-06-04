<!-- 
theme: default
paginate: true
size: 16:9
marp: true
-->

<style>
section {
  background: linear-gradient(to bottom, #f5f7fa, #e8eaed);
  color: #333;
  padding: 40px;
}

h1 {
  color: #1a73e8;
  border-bottom: 2px solid #1a73e8;
  padding-bottom: 10px;
}

.architecture-diagram {
  background-color: white;
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  text-align: center;
}

pre {
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
  background-color: #f0f0f0;
  border-radius: 4px;
  padding: 15px;
  margin: 15px 0;
  overflow: auto;
  text-align: left;
}

.workflow-box {
  background-color: white;
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.workflow-box h2 {
  color: #1a73e8;
  font-size: 1.5rem;
  margin-top: 0;
  margin-bottom: 15px;
}

.workflow-steps {
  text-align: left;
  font-size: 1.1rem;
}

.workflow-steps ol {
  padding-left: 25px;
}
</style>

# System Architecture

<div class="architecture-diagram">
  <h2>High-Level Architecture</h2>

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
</div>

<div class="workflow-box">
  <h2>Paper Upload Workflow</h2>
  <div class="workflow-steps">
    <ol>
      <li><strong>File Upload</strong>: User uploads PDF through interface</li>
      <li><strong>PDF Processing</strong>: Extract text using PyPDFLoader/pdfplumber</li>
      <li><strong>Metadata Extraction</strong>: Parse title, authors, abstract</li>
      <li><strong>Full Text Processing</strong>: Extract and clean complete content</li>
      <li><strong>Vector Embedding</strong>: Create embeddings for semantic search</li>
      <li><strong>Database Storage</strong>: Store metadata in SQLite, vectors in ChromaDB</li>
      <li><strong>Confirmation</strong>: Return success with paper details</li>
    </ol>
  </div>
</div>