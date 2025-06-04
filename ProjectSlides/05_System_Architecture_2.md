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

.workflow-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
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
  font-size: 0.85em;
  background-color: #f0f0f0;
  border-radius: 4px;
  padding: 10px;
  margin: 10px 0;
  overflow: auto;
  text-align: left;
}
</style>

# System Workflows

<div class="workflow-container">
  <div class="workflow-box">
    <h2>Search Pipeline</h2>
    <div class="workflow-steps">
      <ol>
        <li><strong>Intent Analysis</strong>: Determine search type & keywords</li>
        <li><strong>Query Processing</strong>: Clean and expand search terms</li>
        <li><strong>Source Selection</strong>: Choose local DB, arXiv, or both</li>
        <li><strong>Local Search</strong>: Query SQLite and vector search</li>
        <li><strong>External Search</strong>: Query arXiv API if needed</li>
        <li><strong>Results Merging</strong>: Combine and deduplicate results</li>
        <li><strong>Relevance Ranking</strong>: Score and sort by relevance</li>
        <li><strong>Response Formatting</strong>: Present with source attribution</li>
      </ol>
    </div>
  </div>

  <div class="workflow-box">
    <h2>RAG Chat Pipeline</h2>
    <div class="workflow-steps">
      <ol>
        <li><strong>Paper Selection</strong>: Identify target paper</li>
        <li><strong>Context Retrieval</strong>: Vector search for relevant chunks</li>
        <li><strong>Relevance Filtering</strong>: Score and filter best matches</li>
        <li><strong>Prompt Construction</strong>: Build prompt with context</li>
        <li><strong>LLM Processing</strong>: Generate answer with Gemini</li>
        <li><strong>Source Attribution</strong>: Identify supporting passages</li>
        <li><strong>Response Formatting</strong>: Structure with citations</li>
        <li><strong>Chat History Update</strong>: Save to conversation history</li>
      </ol>
    </div>
  </div>
</div>

<div class="architecture-diagram">
  <h2>Comparison Pipeline</h2>

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
</div>