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

ul {
  margin-top: 20px;
  font-size: 1.3rem;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-top: 20px;
}

.tool-box {
  background-color: white;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.tool-box h3 {
  color: #1a73e8;
  margin-top: 0;
}
</style>

# Project Specifications

## Overview

Research Paper Assistant is a sophisticated tool designed to help researchers efficiently manage, search, analyze, and compare research papers using natural language interactions.

## Key Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 1GB+ for paper database
- **API**: Google Gemini API Key
- **Internet**: Required for external searches and AI functionality

<div class="grid">
  <div class="tool-box">
    <h3>Core Technologies</h3>
    <ul>
      <li>Streamlit (Frontend)</li>
      <li>LangChain (NLP Framework)</li>
      <li>Google Gemini (LLM)</li>
      <li>ChromaDB (Vector Database)</li>
      <li>SQLite (Metadata Storage)</li>
    </ul>
  </div>
  
  <div class="tool-box">
    <h3>Libraries & Tools</h3>
    <ul>
      <li>PyPDFLoader (PDF Processing)</li>
      <li>pdfplumber (Text Extraction)</li>
      <li>arXiv API (Paper Search)</li>
      <li>Sentence Transformers</li>
      <li>Python Standard Library</li>
    </ul>
  </div>
</div>