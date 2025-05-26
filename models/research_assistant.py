import os
import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple

from models.database import DatabaseManager
from models.vector_store import VectorStore
from models.pdf_processor import PDFProcessor
from models.api_service import APIService
from models.gemini_agent import GeminiAgent

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document


class ResearchAssistant:
    def __init__(self, gemini_api_key=None):
        """Initialize the research assistant with all necessary components using LangChain."""
        self.db = DatabaseManager()
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.api_service = APIService()
        self.agent = GeminiAgent(api_key=gemini_api_key)
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

        # Flag to indicate if response is being generated
        self.is_generating = False

        # Initialize the LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.2, google_api_key=self.api_key)
        except Exception as e:
            logging.error(f"Error initializing LangChain LLM: {e}")
            self.llm = None

        # Session memory for conversation context
        self.session_memory = {
            "uploaded_papers": [],
            "search_results": [],
            "last_comparison": None,
            "active_paper_chat": None,
            "pending_upload": False
        }

        # Initialize LangChain tools and agent
        self._initialize_tools()
        self._initialize_agent()

    def _initialize_tools(self):
        """Initialize LangChain tools."""
        self.tools = [
            Tool(
                name="internal_search",
                func=self.search_internal_papers,
                description="Search for papers in the internal database using keywords or semantic search"
            ),
            Tool(
                name="web_search",
                func=self.search_web_papers,
                description="Search for papers using arXiv API"
            ),
            Tool(
                name="conference_search",
                func=self.search_conference_papers,
                description="Search for papers from a specific conference"
            ),
            Tool(
                name="compare_papers",
                func=self.generate_paper_comparison,
                description="Compare two research papers and generate a structured report"
            ),
            Tool(
                name="chat_with_specific_paper",
                func=self.initiate_paper_chat,
                description="Start a conversation with a specific paper in your library"
            ),
            Tool(
                name="request_paper_upload",
                func=self.request_paper_upload,
                description="Request to upload a new research paper"
            ),
            Tool(
                name="store_paper",
                func=self.store_paper_from_search,
                description="Store a paper from search results into the database by providing the paper index"
            )
        ]

    def _initialize_agent(self):
        """Initialize LangChain agent with tools."""
        if not self.llm:
            self.agent_executor = None
            return

        try:
            # Create system prompt that emphasizes direct responses
            system_prompt = """You are a Research Paper Assistant that helps users find, analyze, and compare research papers.
            You can understand natural language commands and use tools to find relevant information.
            Always respond directly and concisely. Never expose your reasoning process or tool usage.
            Provide clean, helpful responses without mentioning tools or internal processes.
            """

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

            # Create agent without verbose parameter (removed deprecated parameter)
            agent = create_react_agent(
                self.llm, self.tools, prompt)

            # Create agent executor with all verbose settings disabled
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=False,
                return_intermediate_steps=False,
                max_iterations=3,
                early_stopping_method="generate"
            )

            logging.info("Successfully initialized LangChain agent")
        except Exception as e:
            logging.error(f"Error initializing LangChain agent: {e}")
            self.agent_executor = None

    def upload_paper(self, pdf_file):
        """
        Upload and process a research paper with LangChain.

        Args:
            pdf_file: File object from Streamlit

        Returns:
            Dict with paper information and success status
        """
        try:
            # Reset pending upload flag
            self.session_memory["pending_upload"] = False

            # Save the file
            file_path = self.pdf_processor.save_pdf(pdf_file)
            if not file_path:
                logging.error("Failed to save PDF file")
                return {"success": False, "message": "Failed to save PDF file"}

            logging.info(f"PDF saved to {file_path}")

            # Extract metadata
            metadata = self.pdf_processor.extract_metadata(file_path)
            logging.info(f"Extracted metadata: {metadata}")

            # Extract full text
            full_text = self.pdf_processor.extract_full_text(file_path)
            logging.info(
                f"Extracted full text (length: {len(full_text) if full_text else 0})")

            # Store in database with full text
            paper_id = self.db.add_paper(
                title=metadata["title"],
                abstract=metadata["abstract"],
                authors=metadata["authors"],
                source="internal_upload",
                file_path=file_path,
                full_text=full_text
            )

            logging.info(f"Added paper to database with ID: {paper_id}")

            # Extract documents using LangChain
            documents = self.pdf_processor.extract_documents(file_path)
            logging.info(f"Extracted {len(documents)} document chunks")

            # Set document metadata
            for doc in documents:
                doc.metadata.update({
                    "doc_id": paper_id,
                    "title": metadata["title"],
                    "abstract": metadata["abstract"]
                })

            # Add to vector store using LangChain documents
            if documents:
                ids = self.vector_store.add_documents(documents)
                logging.info(f"Added {len(ids)} documents to vector store")
            else:
                logging.warning(
                    "No document chunks extracted, skipping vector store")

            # Update session memory
            self.session_memory["uploaded_papers"].append({
                "id": paper_id,
                "title": metadata["title"],
                "abstract": metadata["abstract"],
                "path": file_path,
                "has_full_text": bool(full_text)
            })

            return {
                "success": True,
                "paper_id": paper_id,
                "title": metadata["title"],
                "abstract": metadata["abstract"],
                "authors": metadata["authors"],
                "full_text_length": len(full_text) if full_text else 0
            }

        except Exception as e:
            logging.error(f"Error uploading paper: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Upload failed: {str(e)}"}

    def search_internal_papers(self, query: str) -> List[Dict]:
        """
        Search for papers in the internal database using LangChain and improved fuzzy matching.

        Args:
            query: Search query

        Returns:
            List of matching papers
        """
        # First try fuzzy title matching for better exact paper finding
        fuzzy_results = self.db.find_paper_by_title_fuzzy(query, threshold=0.6)

        if fuzzy_results:
            # Update session memory and return fuzzy results
            self.session_memory["search_results"] = fuzzy_results
            return fuzzy_results

        # Try semantic search next
        semantic_results = self.vector_store.search(query, k=5)

        # If semantic results found, get full paper details
        if semantic_results:
            results = []
            for item in semantic_results:
                if item.get('doc_id'):
                    paper = self.db.get_paper(item['doc_id'])
                    if paper:
                        paper['score'] = item.get('score', 0)
                        results.append(paper)

            if results:
                # Update session memory
                self.session_memory["search_results"] = results
                return results

        # Fall back to keyword search with case-insensitive matching
        keyword_results = self.db.search_papers(query)

        # Update session memory
        self.session_memory["search_results"] = keyword_results
        return keyword_results

    def search_web_papers(self, query: str, source: str = None) -> List[Dict]:
        """
        Search for papers using arXiv API.

        Args:
            query: Search query
            source: Optional source (only arxiv supported now)

        Returns:
            List of matching papers
        """
        results = self.api_service.search_arxiv(query, max_results=5)

        # Update session memory
        self.session_memory["search_results"] = results
        return results

    def search_conference_papers(self, conference: str, year: str = None) -> List[Dict]:
        """
        Search for papers from a specific conference.

        Args:
            conference: Conference name
            year: Optional conference year

        Returns:
            List of matching papers
        """
        results = self.api_service.search_papers_by_conference(
            conference, year)

        # Update session memory
        self.session_memory["search_results"] = results
        return results

    def generate_paper_comparison(self, paper_id_1: str, paper_id_2: str) -> str:
        """
        Compare two papers using LangChain.

        Args:
            paper_id_1: ID or index of first paper
            paper_id_2: ID or index of second paper

        Returns:
            Comparison report
        """
        # Helper to get paper by ID or index from memory
        def get_paper(paper_id):
            try:
                # Try as direct DB ID
                paper = self.db.get_paper(int(paper_id))
                if paper:
                    return paper
            except ValueError:
                # Not a valid integer ID
                pass

            # Try as index in recent results
            try:
                idx = int(paper_id)
                if 0 <= idx < len(self.session_memory["search_results"]):
                    return self.session_memory["search_results"][idx]
            except (ValueError, IndexError):
                # Not a valid index
                pass

            return None

        # Get papers to compare
        paper1 = get_paper(paper_id_1)
        paper2 = get_paper(paper_id_2)

        if not paper1 or not paper2:
            return "Could not find one or both papers to compare. Please provide valid paper IDs or indices."

        # Generate comparison using LangChain
        comparison = self.agent.compare_papers(paper1, paper2)

        # Update session memory
        self.session_memory["last_comparison"] = {
            "papers": [paper1, paper2],
            "report": comparison
        }

        return comparison

    def create_retrieval_chain(self, paper_id: str):
        """
        Create a LangChain retrieval chain for a specific paper.

        Args:
            paper_id: ID of the paper to create a retrieval chain for

        Returns:
            A ConversationalRetrievalChain for the paper
        """
        try:
            paper = self.db.get_paper(paper_id)
            if not paper or not paper.get('file_path'):
                return None

            # Extract documents from the paper
            documents = self.pdf_processor.extract_documents(
                paper['file_path'])

            # Create a retriever
            retriever = self.vector_store.db.as_retriever(
                search_kwargs={"k": 5, "filter": {"doc_id": paper_id}}
            )

            # Create the chain with verbose=False
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=False
            )

            return chain
        except Exception as e:
            logging.error(f"Error creating retrieval chain: {e}")
            return None

    def process_natural_language_query(self, query: str) -> Dict:
        """
        Process a natural language query from the user using direct processing.

        Args:
            query: User's query

        Returns:
            Response from the assistant
        """
        try:
            query_lower = query.lower()

            # Handle direct search requests
            search_keywords = ["find", "search", "look for",
                               "get me", "show me", "summary of", "summarize"]
            paper_keywords = ["paper", "research", "article", "study"]

            # Handle direct paper questions (e.g., "What is the paper X about?")
            paper_question_patterns = [
                r"what is (?:the )?paper ['\"]?([^\"']+)['\"]? about",
                r"what's (?:the )?paper ['\"]?([^\"']+)['\"]? about",
                r"(?:tell me|explain) about (?:the )?paper ['\"]?([^\"']+)['\"]",
                r"(?:summarize|summarise) (?:the )?paper ['\"]?([^\"']+)['\"]",
                r"(?:can you tell me|tell me) (?:a bit )?more about (?:the )?paper ['\"]?([^\"']+)['\"]",
                r"(?:what|tell me) (?:about|more about) ['\"]?([^\"']+)['\"]",
                # Add patterns for general paper questions without specific title
                r"what is this paper about",
                r"what's this paper about",
                r"tell me about this paper",
                r"what is the paper about",
                r"what's the paper about",
                r"summarize this paper",
                r"what does this paper discuss",
                r"what is this about"
            ]

            # First check if this is a direct question about a specific paper
            paper_title = None
            is_general_paper_question = False

            for pattern in paper_question_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    if match.groups():  # Pattern has a capture group for title
                        paper_title = match.group(1)
                    else:  # General paper question without specific title
                        is_general_paper_question = True
                    break

            # If we found a potential paper title, try to find it
            if paper_title:
                # Search for the paper in the database
                papers = self.search_internal_papers(paper_title)
                if papers:
                    # Use the first (most relevant) match
                    paper = papers[0]
                    # Generate analysis of the paper directly
                    analysis_result = self.analyze_paper(paper["id"])
                    if analysis_result.get("analysis"):
                        return {
                            "action": "response",
                            "message": analysis_result["analysis"]
                        }
                    else:
                        # Fallback to basic information
                        response = f"**{paper.get('title', 'Unknown Title')}**\n\n"
                        if paper.get('abstract'):
                            response += f"**Abstract:** {paper['abstract']}\n\n"
                        if paper.get('authors'):
                            authors = paper['authors']
                            if isinstance(authors, list):
                                authors_str = ', '.join(authors)
                            else:
                                authors_str = str(authors)
                            response += f"**Authors:** {authors_str}\n\n"

                        return {
                            "action": "response",
                            "message": response if response.strip() != f"**{paper.get('title', 'Unknown Title')}**" else "I found the paper but don't have detailed information about it."
                        }
                else:
                    return {
                        "action": "response",
                        "message": f"I couldn't find a paper titled '{paper_title}' in your library. You might need to upload it first or try a different search term."
                    }

            # Handle general paper questions (like "what is this paper about")
            elif is_general_paper_question:
                # Check if there's a recently uploaded paper
                if self.session_memory["uploaded_papers"]:
                    # Use the most recently uploaded paper
                    recent_paper = self.session_memory["uploaded_papers"][-1]
                    analysis_result = self.analyze_paper(recent_paper["id"])
                    if analysis_result.get("analysis"):
                        return {
                            "action": "response",
                            "message": analysis_result["analysis"]
                        }
                    else:
                        # Fallback to basic information about the recent paper
                        response = f"**{recent_paper.get('title', 'Unknown Title')}**\n\n"
                        if recent_paper.get('abstract'):
                            response += f"**Abstract:** {recent_paper['abstract']}\n\n"

                        # Get full paper details from DB if needed
                        full_paper = self.db.get_paper(recent_paper["id"])
                        if full_paper and full_paper.get('authors'):
                            authors = full_paper['authors']
                            if isinstance(authors, list):
                                authors_str = ', '.join(authors)
                            else:
                                authors_str = str(authors)
                            response += f"**Authors:** {authors_str}\n\n"

                        return {
                            "action": "response",
                            "message": response if response.strip() != f"**{recent_paper.get('title', 'Unknown Title')}**" else "I found the paper but don't have detailed information about it."
                        }
                else:
                    return {
                        "action": "response",
                        "message": "I don't see any papers in your library yet. Please upload a paper first, and then I can tell you about it."
                    }

            is_search_query = any(
                keyword in query_lower for keyword in search_keywords)
            mentions_paper = any(
                keyword in query_lower for keyword in paper_keywords)

            # Handle store paper requests
            store_keywords = ["store", "save",
                              "add to library", "download and store"]
            if any(keyword in query_lower for keyword in store_keywords):
                numbers = re.findall(r'\b(\d+)\b', query)
                if numbers:
                    paper_idx = numbers[0]
                    result = self.store_paper_from_search(paper_idx)
                    return {
                        "action": "store_result",
                        "result": result,
                        "message": result.get("message", "Operation completed.")
                    }
                else:
                    return {
                        "action": "response",
                        "message": "Please specify which paper to store by providing its index number (e.g., 'store paper 0' or 'save paper 1')."
                    }

            # Handle search queries directly
            if is_search_query and mentions_paper:
                # Extract the search term
                search_term = self._extract_search_term(query, search_keywords)

                if search_term:
                    results = self.unified_search(search_term)
                    return {
                        "action": "search_results",
                        "results": results,
                        "message": self._format_search_results(results)
                    }

            # Handle paper comparison requests
            if any(phrase in query.lower() for phrase in ["compare", "difference between", "versus", "vs"]):
                # Check for complex comparison patterns
                complex_comparison_patterns = [
                    r"compare (?:the )?(?:paper )?(?:i uploaded|my (?:uploaded )?paper|my paper) (?:with|to|and) (?:the )?(?:(\w+) paper|paper (\d+))",
                    r"compare (?:the )?(?:(\w+) paper|paper (\d+)) (?:with|to|and) (?:the )?(?:paper )?(?:i uploaded|my (?:uploaded )?paper|my paper)",
                    r"compare (?:the )?(?:paper )?(?:i uploaded|my (?:uploaded )?paper) (?:with|to|and) (?:the )?(?:(\w+) paper you found|(\d+)(?:st|nd|rd|th)? paper)",
                    r"compare (?:my|the) (?:uploaded )?paper (?:with|to|and) (?:the )?(?:(\w+) paper|paper (\d+))"
                ]

                comparison_match = None
                for pattern in complex_comparison_patterns:
                    match = re.search(pattern, query.lower())
                    if match:
                        comparison_match = match
                        break

                if comparison_match:
                    # Extract paper references
                    groups = [g for g in comparison_match.groups()
                              if g is not None]

                    # Get the most recent uploaded paper
                    uploaded_paper = None
                    if self.session_memory["uploaded_papers"]:
                        uploaded_paper = self.session_memory["uploaded_papers"][-1]

                    # Get the referenced paper (either by ordinal number or topic)
                    referenced_paper = None
                    if groups:
                        ref = groups[0]
                        if ref.isdigit():
                            # It's a number - get from search results
                            idx = int(ref) - 1  # Convert to 0-based index
                            if 0 <= idx < len(self.session_memory["search_results"]):
                                referenced_paper = self.session_memory["search_results"][idx]
                        else:
                            # It's a topic - search for it
                            search_results = self.unified_search(ref)
                            if search_results:
                                referenced_paper = search_results[0]
                                # Update search results for future reference
                                self.session_memory["search_results"] = search_results

                    if uploaded_paper and referenced_paper:
                        # Perform the comparison
                        comparison = self.agent.compare_papers(
                            self.db.get_paper(
                                uploaded_paper["id"]) or uploaded_paper,
                            referenced_paper
                        )

                        return {
                            "action": "comparison_result",
                            "comparison": comparison,
                            "papers": [uploaded_paper, referenced_paper],
                            "message": comparison
                        }
                    elif not uploaded_paper:
                        return {
                            "action": "response",
                            "message": "I don't see any uploaded papers in your library. Please upload a paper first before comparing."
                        }
                    else:
                        return {
                            "action": "response",
                            "message": f"I couldn't find the referenced paper. If you meant a specific paper from search results, please make sure you've searched for papers first."
                        }

                # Fall back to original comparison logic
                papers = self._extract_paper_queries(query)
                if papers:
                    result = self.enhanced_paper_comparison(
                        papers[0], papers[1])
                    return {
                        "action": "comparison_result",
                        "result": result
                    }

            # Handle upload requests
            upload_phrases = ["upload a paper", "upload paper",
                              "add a paper", "add paper", "upload research", "upload pdf"]
            if any(phrase in query.lower() for phrase in upload_phrases) or query.lower().startswith("upload"):
                self.session_memory["pending_upload"] = True
                return {
                    "action": "upload_prompt",
                    "message": "I'd be happy to help you upload a paper to your library. Please upload a PDF file to continue."
                }

            # Handle pending upload request
            if self.session_memory.get("pending_upload"):
                return {
                    "action": "upload_prompt",
                    "message": "Please upload a PDF file to continue."
                }

            # Check for active paper chat
            active_paper = self.session_memory.get("active_paper_chat")
            if active_paper:
                paper_id = active_paper.get("id")
                if paper_id:
                    # Check for exit commands
                    if any(exit_phrase in query.lower() for exit_phrase in ["exit chat", "leave chat", "stop chat", "back to main", "return to main"]):
                        self.session_memory["active_paper_chat"] = None
                        return {
                            "action": "response",
                            "message": "Exited paper chat. You're now back in the main chat interface."
                        }

                    # Continue with paper-specific chat
                    response = self.chat_with_paper(paper_id, query)
                    return {
                        "action": "paper_chat_response",
                        "message": response.get("response", "Error processing your question."),
                        "sources": response.get("sources", []),
                        "paper": active_paper
                    }

            # Handle specific paper chat initiation
            chat_keywords = ["chat with", "talk to", "discuss", "ask about"]
            if any(keyword in query_lower for keyword in chat_keywords):
                # Extract paper name/query
                paper_query = query
                for keyword in chat_keywords:
                    if keyword in query_lower:
                        parts = query_lower.split(keyword, 1)
                        if len(parts) > 1:
                            paper_query = parts[1].strip()
                            break

                if paper_query:
                    return self.initiate_paper_chat(paper_query)

            # Handle general queries using Gemini agent (without LangChain)
            context = {
                "uploaded_papers": [p.get("title") for p in self.session_memory["uploaded_papers"]],
                "recent_searches": [p.get("title") for p in self.session_memory["search_results"]]
            }

            # Check if query is related to research papers, science, or the project
            if self._is_relevant_query(query):
                # Handle specific database queries
                if any(phrase in query_lower for phrase in ["papers in database", "papers in library", "what papers", "list papers", "show papers", "papers in your database", "papers in your library"]):
                    papers = self.db.get_all_papers(limit=20)
                    if not papers:
                        return {
                            "action": "response",
                            "message": "Your database is currently empty. You can upload papers using the 'Upload Papers' section or search for papers online and store them in your library."
                        }

                    response = f"You have {len(papers)} papers in your library:\n\n"
                    for i, paper in enumerate(papers[:10], 1):
                        title = paper.get('title', 'Unknown Title')
                        authors = paper.get('authors', [])
                        if isinstance(authors, list):
                            authors_str = ', '.join(authors[:2])
                            if len(authors) > 2:
                                authors_str += f" and {len(authors) - 2} others"
                        else:
                            authors_str = str(
                                authors) if authors else 'Unknown'

                        response += f"**{i}.** {title}\n"
                        response += f"   **Authors:** {authors_str}\n"
                        if paper.get('abstract'):
                            abstract_preview = paper['abstract'][:150] + "..." if len(
                                paper['abstract']) > 150 else paper['abstract']
                            response += f"   **Abstract:** {abstract_preview}\n"
                        response += "\n"

                    if len(papers) > 10:
                        response += f"... and {len(papers) - 10} more papers."

                    return {
                        "action": "response",
                        "message": response
                    }

                # Use the Gemini agent for other research-related queries
                response = self.agent.process_query(query, context)

                # Clean the response from any tool markup and reasoning
                cleaned_response = self._clean_response(response)

                return {
                    "action": "response",
                    "message": cleaned_response
                }
            else:
                # Handle out-of-context queries
                return {
                    "action": "response",
                    "message": "I'm a research paper assistant focused on helping with academic papers, research, and scientific literature. I can help you upload papers, search for research, analyze papers, and compare studies. Please ask me something related to research papers or academic work."
                }

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "action": "response",
                "message": f"I encountered an error: {str(e)}"
            }

    def _extract_search_term(self, query: str, search_keywords: list) -> str:
        """Extract the search term from a query with improved logic."""
        query_lower = query.lower()
        search_term = query

        # Find the search keyword and extract what comes after it
        for keyword in search_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    search_term = parts[1].strip()

                    # Check if this looks like a conference query (preserve conference names and years)
                    conference_patterns = [
                        r'\b(acl|emnlp|naacl|eacl|coling|conll|neurips|nips|icml|iclr|cvpr|iccv|eccv|aaai|ijcai|aamas|icse|fse|sigmod|vldb|icde|ccs|usenix|ndss|stoc|focs|soda|siggraph|icra|iros|rss)\b',
                        r'\b\d{4}\b',  # Years
                        r'conference|workshop|symposium|proceedings'
                    ]

                    # Check if this is a technical term query (preserve technical terms)
                    technical_patterns = [
                        r'\b(tinyml|federated learning|transformer|bert|gpt|lstm|cnn|neural|machine learning|deep learning|nlp|computer vision|robotics|ai|artificial intelligence)\b',
                        # Mixed case acronyms like TinyML
                        r'\b[A-Z]{2,}[a-z]*\b',
                        r'\b[A-Z]+\b'  # All caps acronyms
                    ]

                    is_conference_query = any(re.search(
                        pattern, search_term, re.IGNORECASE) for pattern in conference_patterns)
                    is_technical_query = any(re.search(
                        pattern, search_term, re.IGNORECASE) for pattern in technical_patterns)

                    if is_conference_query or is_technical_query:
                        # For conference/technical queries, only remove very basic words
                        search_term = re.sub(
                            r'\b(the|a|an|for|me|about|from)\b', '', search_term, flags=re.IGNORECASE).strip()
                    else:
                        # For regular queries, be more selective about what to remove
                        search_term = re.sub(
                            r'\b(the|a|an|for|me|about)\b', '', search_term, flags=re.IGNORECASE).strip()

                        # Remove "papers" or "research" only if they're at the end
                        search_term = re.sub(
                            r'\b(papers?|research|articles?|studies)\s*$', '', search_term, flags=re.IGNORECASE).strip()

                    # Remove quotes if present
                    if (search_term.startswith("'") and search_term.endswith("'")) or \
                       (search_term.startswith('"') and search_term.endswith('"')):
                        search_term = search_term[1:-1]

                    break

        # If we extracted something meaningful, use it; otherwise use original query
        if search_term and len(search_term.strip()) > 2:
            return search_term.strip()
        else:
            # Fall back to original query with minimal cleaning
            cleaned = re.sub(r'\b(find|search|look for|get me|show me)\b',
                             '', query, flags=re.IGNORECASE).strip()
            return cleaned if cleaned else query

    def _clean_response(self, response: str) -> str:
        """
        Clean the response from any tool markup or internal reasoning.
        Ensures only direct results are shown to the user.
        """
        # Remove tool markup
        tool_pattern = r"<tool>.*?</tool>"
        response = re.sub(tool_pattern, "", response, flags=re.DOTALL)

        # Remove all possible reasoning indicators
        reasoning_patterns = [
            # Thinking about what the user wants
            r"(?:The user|They|He|She) (?:wants|is asking|is looking for|requested|asked|needs).*?(?:[\.\n]|$)",
            r"(?:The user|They|He|She) (?:seems to|appears to|might|may|could|would like to).*?(?:[\.\n]|$)",
            r"This (?:query|question|request) (?:is about|is asking|wants|requires).*?(?:[\.\n]|$)",
            r"(?:I understand|I see that|I notice|I can tell) (?:the user|you).*?(?:[\.\n]|$)",

            # Planning statements
            r"(?:I should|I need to|I will|I'll|Let me|I can|I must|I'm going to).*?(?:[\.\n]|$)",
            r"(?:First|Let|Now|Next|Then|After that|Finally),? (?:I'?ll|I will|let'?s|we'?ll|we will|we can).*?(?:[\.\n]|$)",
            r"(?:For this|To do this|To answer this|To respond|To handle this),? (?:I'?ll|I will|I need to|I should).*?(?:[\.\n]|$)",
            r"(?:Let me|I'?ll|I will) (?:check|search|look for|find|analyze|examine|investigate).*?(?:[\.\n]|$)",
            r"(?:My approach|My strategy|The best way) (?:is|would be|will be).*?(?:[\.\n]|$)",

            # Self-referential statements
            r"I (?:think|believe|see|notice|observe|know|understand|found|discovered|can see|can tell).*?(?:[\.\n]|$)",
            r"(?:Since|Because|As|Given that) .*?, (?:I'?ll|I will|I should|I need to|I can).*?(?:[\.\n]|$)",
            r"(?:Since|Because|As|Given that) .*?, (?:let'?s|we can|we should|we need to).*?(?:[\.\n]|$)",
            r"(?:Based on|According to|From) .*?, (?:I'?ll|I will|I should|I can).*?(?:[\.\n]|$)",

            # Queries requiring multi-step processing
            r"To (?:answer|handle|process|respond to|address) this,? (?:I'?ll|I will|I need to|I should).*?(?:[\.\n]|$)",
            r"This (?:appears|seems) to be (?:a request|an inquiry|a question) about.*?(?:[\.\n]|$)",
            r"I'll (?:now|first) (?:provide|give|show|present|display) the (?:results|information|data|findings|papers).*?(?:[\.\n]|$)",

            # Paper-specific reasoning patterns
            r"Since the paper is listed as .*?(?:[\.\n]|$)",
            r"If (?:the paper|it) is.*?(?:[\.\n]|$)",
            r"If that fails.*?(?:[\.\n]|$)",
            r"To (?:summarize|analyze) this paper.*?(?:[\.\n]|$)",
            r"This appears to be.*?(?:[\.\n]|$)",
            r"The user is asking about a paper.*?(?:[\.\n]|$)",
            r"I'll analyze this paper.*?(?:[\.\n]|$)",
            r"I need to find information about.*?(?:[\.\n]|$)"
        ]

        for pattern in reasoning_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)

        # Remove introductory phrases
        intro_phrases = [
            r"^(?:Based on|According to|From) (?:this|the|your|my).*?,?\s*",
            r"^(?:Therefore|Thus|Hence|So|In summary|To summarize|To answer your question|To respond to your query|Looking at the|In response),?\s*",
            r"^(?:Here's|Here are|I found|I've found|I have found|The following|These are|This is) (?:the|some|my|what I|what you|your|our).*?(?:[\.\n]|$)",
            r"^Let me (?:answer|provide|give you|show you|present|explain).*?(?:[\.\n]|$)",
            r"^(?:To answer|In response to|Regarding|About|Concerning|On the topic of) your (?:question|query|request).*?(?:[\.\n]|$)",
            r"^(?:Sure|Okay|Alright|Right|Yes|Of course|Certainly|Absolutely|I'd be happy to|I can).*?(?:[\.\n]|$)"
        ]

        for phrase in intro_phrases:
            response = re.sub(phrase, "", response, flags=re.IGNORECASE)

        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()

        # If the response is empty after cleaning, provide a fallback
        if not response.strip():
            return "I couldn't find specific information about that."

        return response

    def analyze_paper(self, paper_id: int) -> Dict:
        """
        Analyze a paper using LangChain.

        Args:
            paper_id: ID of the paper to analyze

        Returns:
            Analysis results
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {"error": "Paper not found"}

        # Use full text if available, otherwise fall back to abstract
        if paper.get("full_text"):
            return self.agent.analyze_paper(paper["title"], paper["abstract"], paper["full_text"])
        else:
            return self.agent.analyze_paper(paper["title"], paper["abstract"])

    def initiate_paper_chat(self, paper_query: str) -> Dict:
        """
        Start a conversation with a specific paper based on query.

        Args:
            paper_query: Query to identify a paper (title, keywords)

        Returns:
            Dict with paper chat information
        """
        try:
            # Search for papers matching the query
            papers = self.search_internal_papers(paper_query)

            if not papers:
                return {
                    "action": "response",
                    "message": f"I couldn't find any papers matching '{paper_query}' in your library. Try uploading the paper first or refine your search."
                }

            # Use the top result
            paper = papers[0]

            # Set as active paper chat
            self.session_memory["active_paper_chat"] = paper

            return {
                "action": "paper_chat_start",
                "paper": paper,
                "message": f"I've started a chat session with '{paper.get('title')}'. You can now ask questions specifically about this paper. Say 'exit chat' when you want to return to the main chat."
            }
        except Exception as e:
            logging.error(f"Error initiating paper chat: {e}")
            return {
                "action": "response",
                "message": f"Sorry, I couldn't start a chat with that paper. Error: {str(e)}"
            }

    def request_paper_upload(self, request: str = None) -> Dict:
        """
        Handle a request to upload a paper.

        Args:
            request: User's upload request message

        Returns:
            Dict with upload prompt
        """
        # Set pending upload flag
        self.session_memory["pending_upload"] = True

        return {
            "action": "upload_prompt",
            "message": "I'd be happy to help you add a paper to your database. Please upload a PDF file to continue."
        }

    def check_generation_status(self) -> bool:
        """
        Check if a response is currently being generated.

        Returns:
            Boolean indicating if generation is in progress
        """
        return self.is_generating

    def chat_with_paper(self, paper_id: int, query: str) -> Dict:
        """
        Have a RAG-based conversation with a specific paper.

        Args:
            paper_id: ID of the paper to chat with
            query: User's question about the paper

        Returns:
            Dict with response and relevant source contexts
        """
        if not self.llm:
            return {"error": "LLM not initialized", "response": "Language model not available. Please check your API key."}

        try:
            # Get paper info
            paper = self.db.get_paper(paper_id)
            if not paper:
                return {"error": "Paper not found", "response": "Could not find the paper in the database."}

            # Create a retriever for this specific paper
            retriever = self.vector_store.db.as_retriever(
                search_kwargs={"k": 3, "filter": {"doc_id": paper_id}}
            )

            # Get relevant chunks
            docs = retriever.get_relevant_documents(query)

            # If no chunks are found, try using stored full text if available
            if not docs and paper.get('full_text'):
                # Create a prompt that uses the stored full text
                template = """You are an AI research assistant helping with questions about academic papers.
                Answer the question based on the paper information provided below. If the information
                needed to answer the question is not contained in the text, say "I don't have enough specific 
                information about that in this paper."
                
                Paper Title: {title}
                Paper Abstract: {abstract}
                
                Paper Full Text (excerpt):
                {full_text}
                
                Question: {question}
                """

                # Truncate full text if it's too long
                max_text_length = 30000
                full_text = paper.get('full_text', '')
                if len(full_text) > max_text_length:
                    full_text = full_text[:max_text_length] + \
                        "... [text truncated]"

                prompt = ChatPromptTemplate.from_template(template)

                chain = (
                    {"title": lambda _: paper.get('title', "Unknown Paper"),
                     "abstract": lambda _: paper.get('abstract', ""),
                     "full_text": lambda _: full_text,
                     "question": lambda x: x}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )

                response = chain.invoke(query)

                return {
                    "response": response,
                    "sources": [{"text": "Based on the full text of the paper", "metadata": {"note": "Using stored full text"}}]
                }

            # If no chunks found and no full text, fall back to using just metadata
            elif not docs:
                # Get the paper's full text if available but not stored in DB
                full_text = ""
                if paper.get('file_path') and os.path.exists(paper.get('file_path')):
                    try:
                        full_text = self.pdf_processor.extract_full_text(
                            paper.get('file_path'))
                    except Exception as e:
                        logging.error(f"Error extracting full text: {e}")

                # If we have some text to work with
                if full_text:
                    # Create a prompt that uses the paper title and abstract at minimum
                    template = """You are an AI research assistant helping with questions about academic papers.
                    
                    Paper Title: {title}
                    Paper Abstract: {abstract}
                    
                    You've been asked about this paper, but could only access limited information.
                    Do your best to answer based on the title and abstract, and explain what might be needed for a more complete answer.
                    
                    Question: {question}
                    """

                    prompt = ChatPromptTemplate.from_template(template)

                    chain = (
                        {"title": lambda _: paper.get('title', "Unknown Paper"),
                         "abstract": lambda _: paper.get('abstract', ""),
                         "question": lambda x: x}
                        | prompt
                        | self.llm
                        | StrOutputParser()
                    )

                    response = chain.invoke(query)

                    return {
                        "response": response,
                        "sources": [{"text": paper.get('abstract', ""), "metadata": {"note": "Only abstract available"}}]
                    }
                else:
                    # Minimal information case
                    return {
                        "response": f"I don't have enough information from '{paper.get('title')}' to answer your question. The paper may not have been fully processed or indexed correctly. Try uploading the paper again or reformulating your question.",
                        "sources": []
                    }

            # If chunks found, create prompt for RAG with the retrieved documents
            template = """You are an AI research assistant helping with questions about academic papers.
            Answer the question based ONLY on the context provided below. If you don't know or the answer
            is not in the context, say "I don't have enough information about that in this paper."
            
            Paper Title: {title}
            
            Context from the paper:
            {context}
            
            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            # Combine context from documents
            contexts = [doc.page_content for doc in docs]
            combined_context = "\n\n---\n\n".join(contexts)

            # Create and execute RAG chain
            chain = (
                {"context": lambda _: combined_context,
                 "question": lambda x: x,
                 "title": lambda _: paper.get('title', "Unknown Paper")}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response = chain.invoke(query)

            return {
                "response": response,
                "sources": [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
            }

        except Exception as e:
            logging.error(f"Error in chat_with_paper: {e}")
            return {"error": str(e), "response": f"Error processing your question: {str(e)}"}

    def delete_paper(self, paper_id: int) -> Dict:
        """
        Delete a paper from the database and clean up associated resources.

        Args:
            paper_id: ID of the paper to delete

        Returns:
            Dict with success status and message
        """
        try:
            # Delete from database and get file path
            success, result = self.db.delete_paper(paper_id)

            if not success:
                return {"success": False, "message": result}

            file_path = result

            # Delete the associated PDF file if it exists
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.warning(f"Could not delete file {file_path}: {e}")

            # Remove from vector store if present
            try:
                # Filter by doc_id to find and delete vector store entries for this paper
                if self.vector_store and self.vector_store.db:
                    # Get all documents and filter for the ones with this paper's ID
                    docs = self.vector_store.db.get()
                    if docs and "metadatas" in docs:
                        ids_to_delete = []
                        for i, metadata in enumerate(docs["metadatas"]):
                            if metadata.get("doc_id") == paper_id:
                                ids_to_delete.append(docs["ids"][i])

                        # Delete the documents from vector store if any found
                        if ids_to_delete:
                            self.vector_store.db.delete(ids_to_delete)
                            self.vector_store.db.persist()
            except Exception as e:
                logging.warning(f"Error cleaning up vector store entries: {e}")

            # Update session memory to remove the paper if present
            for i, paper in enumerate(self.session_memory["uploaded_papers"]):
                if paper.get("id") == paper_id:
                    self.session_memory["uploaded_papers"].pop(i)
                    break

            return {"success": True, "message": "Paper successfully deleted"}
        except Exception as e:
            logging.error(f"Error deleting paper: {e}")
            return {"success": False, "message": str(e)}

    def unified_search(self, query: str, source: str = None) -> List[Dict]:
        """
        Perform a unified search with proper priority: local database first, then arXiv if needed.

        Args:
            query: Search query
            source: Optional source to search (local, arxiv, or None for intelligent search)

        Returns:
            List of papers with their sources
        """
        results = []

        # Step 1: Always search local database first (unless specifically requesting arXiv only)
        if source != "arxiv":
            local_results = self.search_internal_papers(query)
            for paper in local_results:
                paper['search_source'] = 'local_database'
                results.append(paper)

            # If we found relevant papers locally, check if they're sufficient
            if local_results:
                print(f"Found {len(local_results)} papers in local database")

                # If we have good local results, return them
                if len(local_results) >= 3:  # Sufficient local results
                    self.session_memory["search_results"] = results
                    return results

                # If we have some but not many local results, we'll supplement with arXiv below

        # Step 2: Search arXiv only if:
        # - No local results found, OR
        # - Explicitly requested arXiv, OR
        # - Local results are insufficient (< 3 papers)
        should_search_arxiv = (
            len(results) == 0 or  # No local results
            source == "arxiv" or  # Explicitly requested
            (source is None and len(results) < 3)  # Insufficient local results
        )

        if should_search_arxiv:
            print(
                f"Searching arXiv because: local_results={len(results)}, source={source}")
            arxiv_results = self.api_service.search_arxiv(
                query, max_results=10)

            # Add arXiv results, avoiding duplicates
            for paper in arxiv_results:
                if not self._is_duplicate_paper(paper, results):
                    paper['search_source'] = 'arxiv'
                    results.append(paper)

        # Update session memory
        self.session_memory["search_results"] = results
        return results

    def _is_duplicate_paper(self, paper: Dict, existing_papers: List[Dict]) -> bool:
        """Check if a paper is already in the results based on title similarity."""
        if not paper.get('title'):
            return False

        title = paper['title'].lower()
        for existing in existing_papers:
            if not existing.get('title'):
                continue
            existing_title = existing['title'].lower()
            # Simple string similarity check
            if (title in existing_title or existing_title in title or
                    self._string_similarity(title, existing_title) > 0.8):
                return True
        return False

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity ratio."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()

    def enhanced_paper_comparison(self, paper1_query: str, paper2_query: str) -> Dict:
        """
        Enhanced paper comparison that can fetch papers from any source.

        Args:
            paper1_query: Query or identifier for first paper
            paper2_query: Query or identifier for second paper

        Returns:
            Dict with comparison results and paper info
        """
        papers = []

        # Helper function to get a paper with improved search
        def get_paper(query):
            # Try multiple search strategies for better paper matching

            # Strategy 1: Try fuzzy title matching with lower threshold
            fuzzy_results = self.db.find_paper_by_title_fuzzy(
                query, threshold=0.4)
            if fuzzy_results:
                return fuzzy_results[0]

            # Strategy 2: Try direct database search for exact title matches
            all_papers = self.db.get_all_papers()
            query_lower = query.lower().strip()

            # Check for exact title matches (case insensitive)
            for paper in all_papers:
                if paper.get('title'):
                    title_lower = paper['title'].lower().strip()
                    if title_lower == query_lower:
                        return paper

            # Strategy 3: Check for substring matches in titles
            for paper in all_papers:
                if paper.get('title'):
                    title_lower = paper['title'].lower().strip()
                    # Check if query is contained in title or vice versa
                    if query_lower in title_lower or title_lower in query_lower:
                        # Calculate similarity to ensure it's a good match
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(
                            None, query_lower, title_lower).ratio()
                        if similarity > 0.3:  # Lower threshold for partial matches
                            return paper

            # Strategy 4: Try semantic search in internal papers
            semantic_results = self.search_internal_papers(query)
            if semantic_results:
                return semantic_results[0]

            # Strategy 5: Try ArXiv as fallback
            arxiv_results = self.api_service.search_arxiv(query, max_results=3)
            if arxiv_results:
                return arxiv_results[0]

            return None

        # Get both papers
        paper1 = get_paper(paper1_query)
        paper2 = get_paper(paper2_query)

        if not paper1:
            return {
                "success": False,
                "message": f"Could not find the first paper: '{paper1_query}'. Please check the title or try uploading it first."
            }

        if not paper2:
            return {
                "success": False,
                "message": f"Could not find the second paper: '{paper2_query}'. Please check the title or try uploading it first."
            }

        # Generate comparison
        try:
            comparison = self.agent.compare_papers(paper1, paper2)
            return {
                "success": True,
                "comparison": comparison,
                "papers": [paper1, paper2]
            }
        except Exception as e:
            logging.error(f"Error generating comparison: {e}")
            return {
                "success": False,
                "message": f"Error generating comparison: {str(e)}"
            }

    def _extract_paper_queries(self, query: str) -> List[str]:
        """Extract paper queries from a comparison request."""
        # Common comparison phrases
        patterns = [
            r"compare\s+(.*?)\s+(?:and|with|to)\s+(.*?)(?:\.|$)",
            r"difference\s+between\s+(.*?)\s+and\s+(.*?)(?:\.|$)",
            r"(.*?)\s+(?:vs\.?|versus)\s+(.*?)(?:\.|$)"
        ]

        for pattern in patterns:
            matches = re.search(pattern, query, re.IGNORECASE)
            if matches:
                return [matches.group(1).strip(), matches.group(2).strip()]

        return []

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results into a readable message with clear source indication."""
        if not results:
            return "No papers found in your local library. Searching arXiv for relevant papers...\n\n" + \
                   "No relevant papers found on arXiv either. Try different search terms."

        # Separate local and external results
        local_results = [p for p in results if p.get(
            'search_source') == 'local_database']
        external_results = [p for p in results if p.get(
            'search_source') != 'local_database']

        message = ""

        # Handle local results first
        if local_results:
            message += f"📚 **Found {len(local_results)} papers in your local library:**\n\n"

            for i, paper in enumerate(local_results[:10]):
                title = paper.get('title', 'Unknown Title')
                authors = paper.get('authors', [])

                # Handle different author formats
                if isinstance(authors, list):
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += f" and {len(authors) - 3} others"
                else:
                    authors_str = str(authors) if authors else 'Unknown'

                # Display with 1-based indexing for user friendliness
                message += f"**{i+1}.** {title}\n"
                message += f"   **Authors:** {authors_str}\n"
                message += f"   **Source:** Local Library\n"

                # Add abstract preview if available
                abstract = paper.get('abstract', '')
                if abstract:
                    abstract = abstract.replace('\n', ' ').strip()
                    preview = abstract[:200] + \
                        "..." if len(abstract) > 200 else abstract
                    message += f"   **Abstract:** {preview}\n"

                message += "\n"

            if len(local_results) > 10:
                message += f"... and {len(local_results) - 10} more papers in your library.\n\n"
            else:
                message += "\n"

        # Handle external results
        if external_results:
            if local_results:
                if len(local_results) < 3:
                    message += f"🌐 **Also found {len(external_results)} additional papers from arXiv:**\n\n"
                else:
                    message += f"🌐 **Additional papers from arXiv (since you asked for more):**\n\n"
            else:
                message += f"🌐 **No papers found in local library. Found {len(external_results)} papers from arXiv:**\n\n"

            # Continue numbering from where local results left off
            start_index = len(local_results)

            for i, paper in enumerate(external_results[:10], start=start_index + 1):
                title = paper.get('title', 'Unknown Title')
                authors = paper.get('authors', [])

                if isinstance(authors, list):
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += f" and {len(authors) - 3} others"
                else:
                    authors_str = str(authors) if authors else 'Unknown'

                source = paper.get(
                    'search_source', paper.get('source', 'external'))

                message += f"**{i}.** {title}\n"
                message += f"   **Authors:** {authors_str}\n"
                message += f"   **Source:** {source.capitalize()}\n"

                # Add abstract preview if available
                abstract = paper.get('abstract', '')
                if abstract:
                    abstract = abstract.replace('\n', ' ').strip()
                    preview = abstract[:200] + \
                        "..." if len(abstract) > 200 else abstract
                    message += f"   **Abstract:** {preview}\n"

                # Add ArXiv ID if available
                if paper.get('arxiv_id'):
                    message += f"   **ArXiv ID:** {paper.get('arxiv_id')}\n"

                # Add paper URL if available
                if paper.get('url'):
                    message += f"   **Link:** {paper.get('url')}\n"

                message += "\n"

            if len(external_results) > 10:
                message += f"... and {len(external_results) - 10} more papers from external sources.\n\n"

        # Add action message
        if external_results:
            message += "💾 **To store any external papers in your library, say 'store paper X' where X is the paper number.**"

        return message

    def store_paper_from_search(self, paper_index: str) -> Dict:
        """
        Store a paper from search results directly into the database.

        Args:
            paper_index: Index of paper in search results (1-based for user convenience)

        Returns:
            Dict with success status and message
        """
        try:
            # Convert to 0-based indexing for internal use
            paper_idx = int(paper_index) - 1

            if paper_idx < 0 or paper_idx >= len(self.session_memory["search_results"]):
                return {
                    "success": False,
                    "message": f"Invalid paper index. Please choose a number between 1 and {len(self.session_memory['search_results'])}."
                }

            paper = self.session_memory["search_results"][paper_idx]

            # Check if paper is from arXiv and has a PDF URL
            if paper.get('source') == 'arxiv' and paper.get('pdf_url'):
                # Download the PDF
                import tempfile
                from datetime import datetime

                # Create filename based on arxiv ID or title
                arxiv_id = paper.get('arxiv_id', '').replace('/', '_')
                safe_title = re.sub(
                    r'[^\w\s-]', '', paper.get('title', 'paper')).strip()
                safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{arxiv_id or safe_title}.pdf"

                # Ensure uploads directory exists
                uploads_dir = os.path.join(os.path.dirname(
                    __file__), '..', 'data', 'uploads')
                os.makedirs(uploads_dir, exist_ok=True)

                file_path = os.path.join(uploads_dir, filename)

                # Download the PDF
                success = self.api_service.download_paper_pdf(
                    paper['pdf_url'], file_path)

                if not success:
                    return {
                        "success": False,
                        "message": "Failed to download the PDF from arXiv."
                    }

                # Extract full text from downloaded PDF
                full_text = self.pdf_processor.extract_full_text(file_path)

                # Store in database
                paper_id = self.db.add_paper(
                    title=paper.get('title', 'Unknown Title'),
                    abstract=paper.get('abstract', ''),
                    authors=paper.get('authors', []),
                    source=f"arxiv_{paper.get('arxiv_id', '')}",
                    file_path=file_path,
                    full_text=full_text
                )

                # Extract documents and add to vector store
                documents = self.pdf_processor.extract_documents(file_path)
                for doc in documents:
                    doc.metadata.update({
                        "doc_id": paper_id,
                        "title": paper.get('title', 'Unknown Title'),
                        "abstract": paper.get('abstract', '')
                    })

                if documents:
                    self.vector_store.add_documents(documents)

                # Update session memory
                self.session_memory["uploaded_papers"].append({
                    "id": paper_id,
                    "title": paper.get('title', 'Unknown Title'),
                    "abstract": paper.get('abstract', ''),
                    "path": file_path,
                    "has_full_text": bool(full_text)
                })

                return {
                    "success": True,
                    "message": f"Successfully stored '{paper.get('title')}' in your library!",
                    "paper_id": paper_id
                }

            else:
                # For non-arXiv papers or papers without PDF, store metadata only
                paper_id = self.db.add_paper(
                    title=paper.get('title', 'Unknown Title'),
                    abstract=paper.get('abstract', ''),
                    authors=paper.get('authors', []),
                    source=paper.get('source', 'external'),
                    file_path=None,
                    full_text=None
                )

                # Update session memory
                self.session_memory["uploaded_papers"].append({
                    "id": paper_id,
                    "title": paper.get('title', 'Unknown Title'),
                    "abstract": paper.get('abstract', ''),
                    "path": None,
                    "has_full_text": False
                })

                return {
                    "success": True,
                    "message": f"Successfully stored metadata for '{paper.get('title')}' in your library! (PDF not available for download)",
                    "paper_id": paper_id
                }

        except ValueError:
            return {
                "success": False,
                "message": "Please provide a valid paper index number."
            }
        except Exception as e:
            logging.error(f"Error storing paper from search: {e}")
            return {
                "success": False,
                "message": f"Failed to store paper: {str(e)}"
            }

    def _is_relevant_query(self, query: str) -> bool:
        """
        Check if a query is relevant to research papers, science, or the project.

        Args:
            query: User's query

        Returns:
            Boolean indicating if the query is relevant
        """
        query_lower = query.lower()

        # Research and academic keywords
        research_keywords = [
            "paper", "research", "study", "article", "publication", "journal",
            "conference", "arxiv", "academic", "scholar", "citation", "thesis",
            "dissertation", "manuscript", "preprint", "peer review"
        ]

        # Science and technology keywords
        science_keywords = [
            "science", "scientific", "experiment", "analysis", "method", "methodology",
            "data", "results", "findings", "hypothesis", "theory", "model",
            "algorithm", "machine learning", "ai", "artificial intelligence",
            "deep learning", "neural network", "computer science", "engineering",
            "mathematics", "physics", "chemistry", "biology", "medicine"
        ]

        # Project-specific keywords
        project_keywords = [
            "database", "library", "upload", "search", "compare", "analysis",
            "pdf", "document", "abstract", "title", "author", "citation"
        ]

        # Check if query contains any relevant keywords
        all_keywords = research_keywords + science_keywords + project_keywords

        # Check for direct matches
        for keyword in all_keywords:
            if keyword in query_lower:
                return True

        # Check for question patterns that might be research-related
        research_patterns = [
            r"what.*(?:paper|research|study)",
            r"how.*(?:work|method|approach)",
            r"find.*(?:paper|research|article)",
            r"compare.*(?:paper|study|method)",
            r"analyze.*(?:data|paper|research)",
            r"explain.*(?:method|approach|algorithm)",
            r"summarize.*(?:paper|research|study)"
        ]

        for pattern in research_patterns:
            if re.search(pattern, query_lower):
                return True

        # If none of the above match, it's likely not relevant
        return False

    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """Search papers by topic with improved strategies."""
        try:
            print(f"Searching for topic: {topic}")

            # Enhanced search strategies for technical terms
            search_strategies = [
                topic,  # Exact term
                topic.lower(),  # Lowercase
                topic.upper(),  # Uppercase for acronyms
                # Remove spaces (e.g., "Tiny ML" -> "TinyML")
                topic.replace(" ", ""),
                topic.replace("ml", "machine learning"),  # Expand ML
                topic.replace("ai", "artificial intelligence"),  # Expand AI
                topic.replace("dl", "deep learning"),  # Expand DL
                f"{topic} learning",  # Add "learning" for ML terms
                f"{topic} systems",  # Add "systems" for technical terms
                f"{topic} models",  # Add "models"
                f"{topic} algorithms",  # Add "algorithms"
            ]

            # For TinyML specifically, add domain-specific variations
            if "tinyml" in topic.lower():
                search_strategies.extend([
                    "tiny machine learning",
                    "embedded machine learning",
                    "edge machine learning",
                    "mobile machine learning",
                    "iot machine learning",
                    "microcontroller machine learning",
                    "resource constrained machine learning",
                    "edge ai",
                    "embedded ai",
                    "on-device machine learning",
                    "efficient neural networks",
                    "model compression",
                    "neural network quantization"
                ])

            # Remove duplicates while preserving order
            search_strategies = list(dict.fromkeys(search_strategies))

            all_results = []
            seen_ids = set()

            for strategy in search_strategies:
                print(f"Trying search strategy: {strategy}")

                # Try vector search first
                try:
                    vector_results = self.vector_store.search(
                        strategy, k=limit*2)
                    print(
                        f"Vector search returned {len(vector_results)} results")

                    for result in vector_results:
                        paper_id = result.get('doc_id')
                        if paper_id and paper_id not in seen_ids:
                            # Get full paper details from database
                            paper = self.db.get_paper(paper_id)
                            if paper:
                                # Enhanced relevance scoring
                                relevance_score = self._calculate_topic_relevance(
                                    paper, topic)
                                paper['relevance_score'] = relevance_score

                                if relevance_score > 0.1:  # Lower threshold for technical terms
                                    all_results.append(paper)
                                    seen_ids.add(paper_id)

                except Exception as e:
                    print(f"Vector search failed for '{strategy}': {e}")

                # Try keyword search in database
                try:
                    keyword_results = self.db.search_papers(strategy)
                    print(
                        f"Keyword search returned {len(keyword_results)} results")

                    for result in keyword_results:
                        paper_id = result.get('id')
                        if paper_id and paper_id not in seen_ids:
                            relevance_score = self._calculate_topic_relevance(
                                result, topic)
                            result['relevance_score'] = relevance_score

                            if relevance_score > 0.1:
                                all_results.append(result)
                                seen_ids.add(paper_id)

                except Exception as e:
                    print(f"Keyword search failed for '{strategy}': {e}")

                # If we have enough high-quality results, we can stop early
                high_quality_results = [
                    r for r in all_results if r.get('relevance_score', 0) > 0.5]
                if len(high_quality_results) >= limit:
                    break

            # Sort by relevance score and return top results
            all_results.sort(key=lambda x: x.get(
                'relevance_score', 0), reverse=True)
            final_results = all_results[:limit]

            print(
                f"Topic search completed. Found {len(final_results)} relevant papers.")
            for i, result in enumerate(final_results[:3]):
                print(
                    f"  {i+1}. {result.get('title', 'No title')} (score: {result.get('relevance_score', 0):.3f})")

            return final_results

        except Exception as e:
            print(f"Error in topic search: {e}")
            return []

    def _calculate_topic_relevance(self, result: Dict, topic: str) -> float:
        """
        Calculate relevance score for topic search with improved scoring.

        Args:
            result: The paper data
            topic: The topic string

        Returns:
            Relevance score between 0 and 1
        """
        try:
            score = 0.0
            topic_lower = topic.lower()

            # Get text fields
            title = result.get('title', '').lower()
            abstract = result.get('abstract', '').lower()
            content = result.get('content', '').lower()

            # Exact match bonuses (higher for technical terms)
            if topic_lower in title:
                score += 1.0
            if topic_lower in abstract:
                score += 0.8
            if topic_lower in content:
                score += 0.5

            # Partial match bonuses
            topic_words = topic_lower.split()
            for word in topic_words:
                if len(word) > 2:  # Skip short words
                    if word in title:
                        score += 0.3
                    if word in abstract:
                        score += 0.2
                    if word in content:
                        score += 0.1

            # Technical term specific bonuses
            if "tinyml" in topic_lower or "tiny ml" in topic_lower:
                technical_terms = [
                    "embedded", "edge", "mobile", "iot", "microcontroller",
                    "resource constrained", "efficient", "compression",
                    "quantization", "pruning", "distillation", "optimization",
                    "low power", "energy efficient", "on-device"
                ]

                for term in technical_terms:
                    if term in title:
                        score += 0.4
                    elif term in abstract:
                        score += 0.3
                    elif term in content:
                        score += 0.1

            # Conference/venue bonus for academic papers
            venue = result.get('venue', '').lower()
            if venue:
                if any(conf in venue for conf in ['icml', 'nips', 'iclr', 'aaai', 'ijcai']):
                    score += 0.2

            # Year recency bonus (prefer recent papers)
            year = result.get('year')
            if year and isinstance(year, (int, str)):
                try:
                    year_int = int(year)
                    if year_int >= 2020:
                        score += 0.1
                    if year_int >= 2022:
                        score += 0.1
                except:
                    pass

            return min(score, 2.0)  # Cap at 2.0

        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def _get_search_strategies(self, query: str) -> List[str]:
        """Generate multiple search strategies for better coverage."""
        strategies = []
        query_lower = query.lower()

        # Original query
        strategies.append(query)

        # Technical term expansions
        if "tinyml" in query_lower:
            strategies.extend([
                "tiny machine learning",
                "edge machine learning",
                "mobile ML",
                "embedded AI",
                "on-device ML",
                "resource-constrained ML",
                "efficient neural networks",
                "model compression",
                "quantized neural networks"
            ])

        if "nlp" in query_lower:
            strategies.extend([
                "natural language processing",
                "language models",
                "text processing",
                "computational linguistics"
            ])

        if "cv" in query_lower or "computer vision" in query_lower:
            strategies.extend([
                "image processing",
                "visual recognition",
                "deep learning vision",
                "convolutional neural networks"
            ])

        # Add keyword variations
        words = query.split()
        if len(words) > 1:
            # Try individual important words
            for word in words:
                if len(word) > 3:
                    strategies.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy.lower() not in seen:
                seen.add(strategy.lower())
                unique_strategies.append(strategy)

        return unique_strategies[:8]  # Limit to 8 strategies
