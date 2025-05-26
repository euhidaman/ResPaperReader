import os
import json
import logging
from typing import List, Dict, Any

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import Tool, create_structured_chat_agent
from langchain.agents.agent import AgentExecutor


class GeminiAgent:
    def __init__(self, api_key=None):
        """Initialize the Gemini agent with API key using LangChain."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.configure()
        self.conversation_history = []
        self.available_tools = self._define_tools()

    def configure(self):
        """Configure the LangChain Gemini integration."""
        try:
            os.environ["GOOGLE_API_KEY"] = self.api_key
            # Using LangChain's wrapper for Gemini
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.2)
            self.output_parser = StrOutputParser()
            logging.info(
                "Successfully configured Gemini API via LangChain with model: gemini-2.0-flash")
        except Exception as e:
            logging.error(f"Error configuring Gemini API via LangChain: {e}")
            self.llm = None

    def _define_tools(self):
        """Define the available tools for the agent."""
        return {
            "internal_search": {
                "name": "internal_search",
                "description": "Search for papers in the internal database using keywords or semantic search",
                "parameters": {
                    "query": "The search query"
                }
            },
            "web_search": {
                "name": "web_search",
                "description": "Search for papers using external APIs like arXiv and Semantic Scholar",
                "parameters": {
                    "query": "The search query",
                    "source": "Optional source to search (arxiv or semantic_scholar)"
                }
            },
            "conference_search": {
                "name": "conference_search",
                "description": "Search for papers from a specific conference",
                "parameters": {
                    "conference": "Conference name (e.g., ICLR, NeurIPS, ACL)",
                    "year": "Optional year of the conference"
                }
            },
            "compare_papers": {
                "name": "compare_papers",
                "description": "Compare two research papers and generate a structured report",
                "parameters": {
                    "paper_id_1": "ID of the first paper to compare",
                    "paper_id_2": "ID of the second paper to compare"
                }
            }
        }

    def get_system_prompt(self):
        """Get the system prompt for the agent."""
        return f"""You are a Research Paper Assistant that helps users find, analyze, and compare research papers.
You can understand natural language commands and determine which tools to use.

Available tools:
{json.dumps(self.available_tools, indent=2)}

For each user query, analyze what they're asking for and choose the appropriate tool.
Format your responses in a clear, informative way suitable for academic research.
If you need to use a tool, output it in the format: <tool>tool_name(parameters)</tool>
"""

    def analyze_paper(self, title, abstract, full_text=None):
        """
        Analyze a research paper and extract key information using LangChain.

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Optional full text of the paper

        Returns:
            Dict with analysis results
        """
        if not self.llm or not self.api_key:
            return {"error": "Gemini API not configured"}

        # Include full text in analysis if available
        content = f"Title: {title}\n\nAbstract: {abstract}"

        if full_text:
            # If we have full text, provide a comprehensive section of it to the model
            # Limit the length to avoid token limits
            max_text_length = 30000  # Adjust based on model's context window
            truncated_text = full_text[:max_text_length]
            if len(full_text) > max_text_length:
                truncated_text += "... [text truncated]"

            content += f"\n\nFull Text Excerpt:\n{truncated_text}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="You are a research paper analysis assistant that extracts key information from papers. Your responses should be clear, direct, and free of meta-commentary. Do not include phrases like 'Based on the paper', 'From the information provided', or 'Since the paper is about'. Just provide the analysis directly."),
            HumanMessage(content=f"""Analyze the following research paper and extract key information:
{content}

Please provide:
1. Research problem/question
2. Key methods
3. Main contributions
4. Findings and conclusions
5. 3-5 keywords
6. Implications for the field

Format your response as a structured report with clear headings.
""")
        ])

        try:
            chain = prompt | self.llm | self.output_parser
            analysis = chain.invoke({})

            # Clean out any potential reasoning patterns
            analysis = self._clean_response(analysis)

            return {
                "analysis": analysis,
                "title": title
            }
        except Exception as e:
            logging.error(f"Error generating paper analysis: {e}")
            return {"error": str(e)}

    def _clean_response(self, response):
        """Clean up the response to remove internal reasoning patterns."""
        import re

        # Remove common reasoning patterns
        patterns_to_remove = [
            r"(?:Based on|According to|From) the (?:paper|text|information provided|abstract|title).*?(?:[\.\n]|$)",
            r"Since the paper is (?:about|on|related to).*?(?:[\.\n]|$)",
            r"This paper (?:appears to|seems to|is about).*?(?:[\.\n]|$)",
            r"From what I can (?:see|understand|gather).*?(?:[\.\n]|$)",
            r"The paper discusses.*?(?:[\.\n]|$)",
            r"From the (?:given|provided) (?:information|text|abstract|excerpt).*?(?:[\.\n]|$)",
            r"If that fails.*?(?:[\.\n]|$)",
        ]

        for pattern in patterns_to_remove:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)

        # Clean up whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()

        return response

    def compare_papers(self, paper1, paper2):
        """
        Compare two research papers and generate a structured report using LangChain.

        Args:
            paper1: Dict containing first paper metadata
            paper2: Dict containing second paper metadata

        Returns:
            Comparison report
        """
        if not self.llm or not self.api_key:
            return "Gemini API not configured. Please add your API key."

        # Prepare content for paper 1
        paper1_content = f"Title: {paper1.get('title', 'Unknown')}\nAbstract: {paper1.get('abstract', 'Not available')}"
        if paper1.get('full_text'):
            # Limit the length to prevent token overflow
            max_text_length = 15000  # Reduced size to accommodate both papers
            truncated_text = paper1.get('full_text')[:max_text_length]
            if len(paper1.get('full_text', '')) > max_text_length:
                truncated_text += "... [text truncated]"
            paper1_content += f"\n\nFull Text Excerpt:\n{truncated_text}"

        # Prepare content for paper 2
        paper2_content = f"Title: {paper2.get('title', 'Unknown')}\nAbstract: {paper2.get('abstract', 'Not available')}"
        if paper2.get('full_text'):
            # Limit the length to prevent token overflow
            max_text_length = 15000  # Reduced size to accommodate both papers
            truncated_text = paper2.get('full_text')[:max_text_length]
            if len(paper2.get('full_text', '')) > max_text_length:
                truncated_text += "... [text truncated]"
            paper2_content += f"\n\nFull Text Excerpt:\n{truncated_text}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="You are a research paper comparison assistant with expertise in analyzing academic papers."),
            HumanMessage(content=f"""Compare the following two research papers based on all provided information:

Paper 1:
{paper1_content}

Paper 2:
{paper2_content}

Please provide a structured comparison including:
1. Research goals and objectives
2. Methodologies and approaches
3. Key contributions and innovations
4. Main findings and results
5. Strengths and limitations of each paper
6. Significant similarities between the papers
7. Important differences and contrasting aspects
8. Recommendations for which paper might be more relevant for different research contexts

Format your response as a structured report with clear headings and bullet points.
""")
        ])

        try:
            chain = prompt | self.llm | self.output_parser
            return chain.invoke({})
        except Exception as e:
            logging.error(f"Error generating paper comparison: {e}")
            return f"Failed to generate comparison: {str(e)}"

    def process_query(self, query, context=None):
        """
        Process a user query using LangChain and provide direct answers.

        Args:
            query: User's natural language query
            context: Additional context for the query

        Returns:
            Clean, direct response from the agent
        """
        if not self.llm or not self.api_key:
            return "Gemini API not configured. Please add your API key."

        # Prepare context information
        context_info = ""
        if context:
            uploaded_papers = context.get("uploaded_papers", [])
            recent_searches = context.get("recent_searches", [])

            if uploaded_papers:
                context_info += f"Available papers in library: {', '.join(uploaded_papers[:5])}\n"
            if recent_searches:
                context_info += f"Recent search results: {', '.join(recent_searches[:5])}\n"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Research Paper Assistant that provides direct, helpful answers about research papers and academic topics.

IMPORTANT INSTRUCTIONS:
- Provide direct answers without exposing your reasoning process
- Never say things like "Since the paper is..." or "Based on the information provided..."
- Never mention checking databases or searching - just provide the answer
- If you don't have specific information, say so directly
- For paper summaries, provide the actual summary content
- Be concise and informative
- Focus on answering what the user asked, not explaining how you'll do it

Your responses should be clean, professional, and focused on the actual information requested."""),
            HumanMessage(
                content=f"""User query: {query}

{context_info}

Provide a direct, helpful answer to this query. Do not explain your reasoning or mention any internal processes.""")
        ])

        try:
            chain = prompt | self.llm | self.output_parser
            response = chain.invoke({})

            # Apply thorough cleaning to remove any remaining reasoning patterns
            cleaned_response = self._clean_response_thoroughly(response)

            # Add to conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": cleaned_response
            })

            return cleaned_response
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"I encountered an error: {str(e)}"

    def _clean_response_thoroughly(self, response):
        """Thoroughly clean up the response to remove all internal reasoning patterns."""
        import re

        # Comprehensive list of reasoning patterns to remove
        reasoning_patterns = [
            # Direct mentions of reasoning
            r"(?:Since|Because|As|Given that) (?:the|this) paper (?:is|appears|seems|was|has).*?(?:[\.\n]|$)",
            r"(?:Based on|According to|From) (?:the|this|what|my) (?:paper|information|context|analysis|understanding).*?(?:[\.\n]|$)",
            r"(?:The paper|This paper|It) (?:appears|seems|looks like|is likely).*?(?:[\.\n]|$)",
            r"(?:From|Based on) (?:what|the|my|this) (?:I can see|we know|is provided|information).*?(?:[\.\n]|$)",

            # Process-related statements
            r"(?:I need to|I should|I will|I'll|Let me) (?:check|search|look|find|analyze|examine).*?(?:[\.\n]|$)",
            r"(?:I'm going to|I can|I'll|Let me) (?:provide|give|show|present|tell you).*?(?:[\.\n]|$)",
            r"(?:Let me|I'll) (?:help|assist) (?:you|with|by).*?(?:[\.\n]|$)",

            # Uncertainty indicators that expose reasoning
            r"(?:It's likely|It seems|It appears|It looks like|If I recall|If memory serves).*?(?:[\.\n]|$)",
            r"(?:I believe|I think|I suspect|I assume|In my opinion).*?(?:[\.\n]|$)",

            # Database/system references
            r"(?:in the|from the|checking the) (?:database|library|uploaded_papers|search results).*?(?:[\.\n]|$)",
            r"(?:Since|Because) (?:it's|they're|the paper is) (?:in the|listed in|stored in).*?(?:[\.\n]|$)",

            # Conditional statements that expose process
            r"If (?:the paper|it|that) (?:is|exists|can be found).*?(?:[\.\n]|$)",
            r"If (?:I|we) (?:can|could|find|locate).*?(?:[\.\n]|$)",

            # Tool/method references
            r"(?:Using|Through|Via) (?:the|my|our) (?:analysis|search|database|tools).*?(?:[\.\n]|$)",
            r"(?:After|Upon) (?:checking|searching|analyzing|reviewing).*?(?:[\.\n]|$)"
        ]

        for pattern in reasoning_patterns:
            response = re.sub(pattern, "", response,
                              flags=re.IGNORECASE | re.MULTILINE)

        # Remove introductory phrases that indicate process
        intro_patterns = [
            r"^(?:Here's what I found|Here's the information|Based on my search|After searching).*?(?:[\:\.\n]|$)",
            r"^(?:I found that|I can tell you that|I can see that|I discovered that).*?(?:[\:\.\n]|$)",
            r"^(?:The search reveals|My analysis shows|The results indicate).*?(?:[\:\.\n]|$)",
            r"^(?:Looking at|Reviewing|Examining|Checking) (?:the|this|your).*?(?:[\:\.\n]|$)"
        ]

        for pattern in intro_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)

        # Clean up excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = re.sub(r'^\s*\n', '', response)
        response = response.strip()

        # If response is empty after cleaning, provide a fallback
        if not response or response.isspace():
            return "I don't have specific information about that topic."

        return response
