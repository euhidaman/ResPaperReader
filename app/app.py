from models.research_assistant import ResearchAssistant
import streamlit as st
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'assistant' not in st.session_state:
    # Get API key from secrets or environment
    api_key = os.environ.get("GEMINI_API_KEY", "")

    # Initialize the research assistant
    st.session_state.assistant = ResearchAssistant(gemini_api_key=api_key)
    st.session_state.chat_history = []
    st.session_state.current_papers = []
    st.session_state.api_key_set = bool(api_key)
    st.session_state.is_generating = False
    st.session_state.uploading_in_chat = False

# Helper functions


def display_paper(paper, index=None, allow_delete=False):
    """Display a paper in the UI."""
    with st.container():
        col1, col2, col3 = st.columns([8, 1, 1])

        title = paper.get('title', 'Unknown Title')

        with col1:
            st.markdown(f"### {index+1 if index is not None else ''}) {title}")

        with col2:
            if st.button("Analyze", key=f"analyze_{hash(title)}"):
                if paper.get('id'):
                    with st.spinner("Analyzing paper..."):
                        analysis = st.session_state.assistant.analyze_paper(
                            paper['id'])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"## Analysis of '{title}'\n\n{analysis.get('analysis', 'Analysis failed')}"
                    })

        with col3:
            if allow_delete and paper.get('id'):
                if st.button("Delete", key=f"delete_{hash(title)}", type="primary", help="Delete this paper permanently"):
                    with st.spinner("Deleting paper..."):
                        result = st.session_state.assistant.delete_paper(
                            paper['id'])

                    if result["success"]:
                        st.success(f"Paper '{title}' deleted successfully")
                        # Add a rerun call to refresh the page
                        st.rerun()
                    else:
                        st.error(
                            f"Failed to delete paper: {result['message']}")

        # Show authors if available
        authors = paper.get('authors', [])
        if authors:
            if isinstance(authors, list):
                st.markdown(f"**Authors:** {', '.join(authors)}")
            else:
                st.markdown(f"**Authors:** {authors}")

        # Show abstract with a "Read more" expander if it's long
        abstract = paper.get('abstract', '')
        if len(abstract) > 200:
            st.markdown(f"**Abstract:** {abstract[:200]}...")
            with st.expander("Read more"):
                st.markdown(abstract)
        else:
            st.markdown(f"**Abstract:** {abstract}")

        # Show source info
        source = paper.get('source', 'Unknown')
        if source == 'arxiv':
            st.markdown(
                f"**Source:** arXiv | [Paper Link]({paper.get('url', '#')}) | [PDF]({paper.get('pdf_url', '#')})")
        elif source == 'semantic_scholar':
            st.markdown(
                f"**Source:** Semantic Scholar | [Paper Link]({paper.get('url', '#')})")
        else:
            st.markdown(f"**Source:** {source}")

        st.markdown("---")


def display_comparison_selector():
    """Display UI for selecting papers to compare."""
    st.subheader("Compare Papers")

    # Get papers from memory and database
    papers = st.session_state.current_papers

    if not papers or len(papers) < 2:
        st.warning("You need at least 2 papers in your results to compare.")
        return

    col1, col2 = st.columns(2)

    with col1:
        paper1_idx = st.selectbox(
            "Select first paper:",
            options=range(len(papers)),
            format_func=lambda i: papers[i].get('title', f"Paper {i+1}"),
            key="compare_paper1"
        )

    with col2:
        paper2_idx = st.selectbox(
            "Select second paper:",
            options=range(len(papers)),
            format_func=lambda i: papers[i].get('title', f"Paper {i+1}"),
            key="compare_paper2"
        )

    if st.button("Generate Comparison"):
        if paper1_idx == paper2_idx:
            st.error("Please select two different papers to compare.")
            return

        with st.spinner("Generating comparison report..."):
            # Use the actual paper IDs if available, otherwise use session indices
            paper1_id = papers[paper1_idx].get('id', str(paper1_idx))
            paper2_id = papers[paper2_idx].get('id', str(paper2_idx))

            comparison = st.session_state.assistant.generate_paper_comparison(
                paper_id_1=paper1_id,
                paper_id_2=paper2_id
            )

            # Add to chat history
            title1 = papers[paper1_idx].get('title', f"Paper {paper1_idx+1}")
            title2 = papers[paper2_idx].get('title', f"Paper {paper2_idx+1}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"## Comparison: '{title1}' vs '{title2}'\n\n{comparison}"
            })


# Sidebar
with st.sidebar:
    st.title("Research Paper Assistant")

    # API Key Input
    with st.expander("API Settings", expanded=not st.session_state.api_key_set):
        api_key_input = st.text_input(
            "Enter Gemini API Key:",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password"
        )

        if st.button("Save API Key"):
            os.environ["GEMINI_API_KEY"] = api_key_input
            st.session_state.assistant = ResearchAssistant(
                gemini_api_key=api_key_input)
            st.session_state.api_key_set = bool(api_key_input)
            st.success("API key saved!")

    # Navigation
    st.header("Navigation")
    page = st.radio("Go to:", ["Chat Assistant", "Upload Papers",
                    "Search Papers", "My Library", "Chat with Papers"])

# Main area
if page == "Chat Assistant":
    st.header("Research Paper Chat Assistant")

    # Check for active paper chat
    active_paper_chat = st.session_state.assistant.session_memory.get(
        "active_paper_chat")
    if active_paper_chat:
        st.info(
            f"🔍 You're currently in a paper-specific chat with: **{active_paper_chat.get('title', 'Unknown Paper')}**. Type 'exit chat' to return to the main chat.")

    # Handle file upload in chat if requested
    if st.session_state.uploading_in_chat:
        col1, col2 = st.columns([5, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a PDF research paper", type="pdf", key="chat_upload")
        with col2:
            if st.button("Cancel Upload"):
                st.session_state.uploading_in_chat = False
                st.rerun()

        if uploaded_file:
            with st.spinner("Processing paper..."):
                result = st.session_state.assistant.upload_paper(uploaded_file)

                if result["success"]:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Successfully uploaded and processed: **{result['title']}**\n\n"
                        f"**Authors:** {', '.join(result['authors']) if result['authors'] else 'Unknown'}\n\n"
                        f"**Abstract:** {result['abstract'][:300]}..."
                    })
                    st.session_state.uploading_in_chat = False
                    st.rerun()
                else:
                    error_msg = result.get('message', 'Unknown error')
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Failed to upload paper: {error_msg}"
                    })
                    st.session_state.uploading_in_chat = False
                    st.rerun()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                # Handle different types of assistant responses
                if message.get("type") == "paper_chat":
                    with st.container():
                        st.markdown(f"**Assistant (Paper Chat):** {content}")
                        if "sources" in message:
                            with st.expander("View sources from paper"):
                                for source in message["sources"]:
                                    st.markdown(f"```\n{source['text']}\n```")
                                    if "metadata" in source:
                                        st.markdown(
                                            f"*Page {source['metadata'].get('page', 'unknown')}*")

                elif message.get("action") == "search_results":
                    st.markdown(f"**Assistant:** {content}")
                    if message.get("results"):
                        with st.expander("View search results"):
                            for i, paper in enumerate(message["results"], 1):
                                st.markdown(f"**{i}. {paper.get('title')}**")
                                st.markdown(
                                    f"Source: {paper.get('search_source', 'Unknown')}")
                                if paper.get('abstract'):
                                    st.markdown(
                                        f"Abstract: {paper.get('abstract')[:200]}...")
                                st.markdown("---")

                elif message.get("action") == "comparison_result":
                    st.markdown(f"**Assistant:** {content}")
                    result = message.get("result", {})
                    if result.get("success"):
                        with st.expander("View detailed comparison"):
                            st.markdown(result["comparison"])
                    else:
                        st.error(result.get(
                            "message", "Failed to generate comparison"))

                else:
                    st.markdown(f"**Assistant:** {content}")

    # Show loading indicator while generating
    if st.session_state.is_generating:
        with st.container():
            st.markdown("**Assistant:** _Generating response..._")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Check if generation is complete
            if not st.session_state.assistant.check_generation_status():
                st.session_state.is_generating = False
                st.rerun()

    # Chat input area
    input_container = st.container()
    with input_container:
        cols = st.columns([6, 1])

        with cols[0]:
            user_input = st.chat_input(
                "Ask a question, search papers, or request a comparison...",
                key="chat_input"
            )

        with cols[1]:
            upload_button = st.button(
                "📄 Upload", help="Upload a research paper")
            if upload_button:
                st.session_state.uploading_in_chat = True
                st.rerun()

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Process query - do this before setting generating flag and rerunning
            response = st.session_state.assistant.process_natural_language_query(
                user_input)

            # Handle different response types
            if response["action"] == "upload_prompt":
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["message"]}
                )
                st.session_state.uploading_in_chat = True

            elif response["action"] == "search_results":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "action": "search_results",
                    "results": response["results"]
                })

            elif response["action"] == "comparison_result":
                # Handle different response structures for comparison results
                if "result" in response:
                    # Original comparison structure
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Here's the comparison you requested:",
                        "action": "comparison_result",
                        "result": response["result"]
                    })
                else:
                    # New comparison structure with direct comparison content
                    comparison_content = response.get(
                        "comparison", response.get("message", "Comparison completed"))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": comparison_content,
                        "action": "comparison_result",
                        "result": {"success": True, "comparison": comparison_content}
                    })

            elif response["action"] == "paper_chat_start":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "paper": response["paper"]
                })

            elif response["action"] == "paper_chat_response":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "sources": response.get("sources", []),
                    "paper": response["paper"],
                    "type": "paper_chat"
                })

            elif response["action"] == "tool_results":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"]
                })

                for result in response.get("results", []):
                    if isinstance(result.get("result"), list):
                        st.session_state.current_papers = result["result"]

            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"]
                })

            # Now set the generating flag to False and rerun
            st.session_state.is_generating = False
            st.rerun()
elif page == "Upload Papers":
    st.header("Upload Research Papers")

    uploaded_file = st.file_uploader("Upload a PDF research paper", type="pdf")

    if uploaded_file:
        with st.spinner("Processing paper..."):
            try:
                # Get file info for debugging
                file_info = f"File: {uploaded_file.name}, Size: {uploaded_file.size} bytes, Type: {uploaded_file.type}"
                logging.info(f"Processing uploaded file: {file_info}")

                # Ensure upload directory exists
                os.makedirs(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "data", "uploads"), exist_ok=True)

                # Process the paper
                result = st.session_state.assistant.upload_paper(uploaded_file)

                if result["success"]:
                    st.success(f"Successfully uploaded: {result['title']}")

                    # Display paper details
                    st.subheader("Paper Details")
                    st.markdown(f"**Title:** {result['title']}")
                    st.markdown(
                        f"**Authors:** {', '.join(result['authors']) if result['authors'] else 'Unknown'}")
                    st.markdown(f"**Abstract:** {result['abstract']}")

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Paper '{result['title']}' has been uploaded and processed."
                    })
                else:
                    error_msg = result.get('message', 'Unknown error')
                    st.error(f"Failed to upload paper: {error_msg}")
                    logging.error(f"Upload failed: {error_msg}")
            except Exception as e:
                st.error(f"An error occurred during upload: {str(e)}")
                logging.exception("Unexpected error during paper upload:")

    with st.expander("View Uploaded Papers", expanded=True):
        st.subheader("Your Uploaded Papers")

        # Get uploaded papers from the assistant's database
        uploaded_papers = st.session_state.assistant.db.get_all_papers()

        if not uploaded_papers:
            st.info(
                "No papers uploaded yet. Use the section above to upload a paper.")
        else:
            for i, paper in enumerate(uploaded_papers):
                display_paper(paper, i, allow_delete=True)

elif page == "Search Papers":
    st.header("Search for Research Papers")

    # Tabs for different search types
    tab1, tab2, tab3 = st.tabs(
        ["Internal Search", "Web Search", "Conference Search"])

    with tab1:
        st.subheader("Search Internal Library")
        query = st.text_input("Enter search terms:", key="internal_search")

        if st.button("Search Internal Library"):
            with st.spinner("Searching internal database..."):
                results = st.session_state.assistant.search_internal_papers(
                    query)

            st.session_state.current_papers = results

            st.subheader(f"Search Results ({len(results)})")
            if results:
                for i, paper in enumerate(results):
                    display_paper(paper, i)
            else:
                st.info("No papers found. Try uploading papers first.")

    with tab2:
        st.subheader("Search External Sources")
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input("Enter search terms:", key="web_search")

        with col2:
            source = st.selectbox(
                "Source:",
                options=[None, "arxiv", "semantic_scholar"],
                format_func=lambda x: "Both" if x is None else x.capitalize()
            )

        if st.button("Search External Sources"):
            with st.spinner("Searching external sources..."):
                results = st.session_state.assistant.search_web_papers(
                    query, source)

            st.session_state.current_papers = results

            st.subheader(f"Search Results ({len(results)})")
            if results:
                for i, paper in enumerate(results):
                    display_paper(paper, i)
            else:
                st.info("No papers found. Try different search terms.")

    with tab3:
        st.subheader("Search by Conference")
        col1, col2 = st.columns(2)

        with col1:
            conference = st.text_input(
                "Conference name (e.g., ICLR, NeurIPS, ACL):")

        with col2:
            year = st.text_input("Year (optional):")

        if st.button("Search Conference Papers"):
            if not conference:
                st.warning("Please enter a conference name.")
            else:
                with st.spinner(f"Searching for papers from {conference}..."):
                    results = st.session_state.assistant.search_conference_papers(
                        conference, year)

                st.session_state.current_papers = results

                st.subheader(f"Search Results ({len(results)})")
                if results:
                    for i, paper in enumerate(results):
                        display_paper(paper, i)
                else:
                    st.info("No papers found. Try a different conference or year.")

    # Paper comparison tool (appears at the bottom of search page)
    st.markdown("---")
    display_comparison_selector()

elif page == "My Library":
    st.header("My Paper Library")

    with st.spinner("Loading your library..."):
        papers = st.session_state.assistant.db.get_all_papers()

    st.session_state.current_papers = papers

    st.subheader(f"Your Papers ({len(papers)})")
    if papers:
        for i, paper in enumerate(papers):
            display_paper(paper, i, allow_delete=True)
    else:
        st.info("No papers in your library yet. Try uploading or searching for papers.")

    # Paper comparison tool
    st.markdown("---")
    display_comparison_selector()

elif page == "Chat with Papers":
    st.header("Chat with Papers")

    # Get all papers from the database
    papers = st.session_state.assistant.db.get_all_papers()

    if not papers:
        st.info("Your library is empty. Please upload some papers first.")
    else:
        # Paper selector
        selected_paper = st.selectbox(
            "Select a paper to chat with:",
            options=papers,
            format_func=lambda p: p.get('title', 'Unknown Title')
        )

        if selected_paper:
            st.write("### Selected Paper")
            display_paper(selected_paper, allow_delete=True)

            # Chat interface
            st.write("### Chat")

            # Initialize paper-specific chat history in session state if not exists
            paper_chat_key = f"paper_chat_{selected_paper['id']}"
            if paper_chat_key not in st.session_state:
                st.session_state[paper_chat_key] = []

            # Display paper-specific chat history
            for message in st.session_state[paper_chat_key]:
                role = message["role"]
                content = message["content"]

                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    with st.container():
                        st.markdown(f"**Assistant:** {content}")
                        # If the message has sources, display them in an expander
                        if "sources" in message:
                            with st.expander("View sources from paper"):
                                for source in message["sources"]:
                                    st.markdown(f"```\n{source['text']}\n```")
                                    if "metadata" in source:
                                        st.markdown(
                                            f"*Page {source['metadata'].get('page', 'unknown')}*")

            # Chat input
            question = st.chat_input(
                f"Ask a question about '{selected_paper['title']}'...")

            if question:
                # Add user question to history
                st.session_state[paper_chat_key].append({
                    "role": "user",
                    "content": question
                })

                # Get response using RAG
                with st.spinner("Searching paper and generating response..."):
                    response = st.session_state.assistant.chat_with_paper(
                        selected_paper['id'],
                        question
                    )

                if "error" in response:
                    st.error(response["error"])
                else:
                    # Add assistant response with sources to history
                    st.session_state[paper_chat_key].append({
                        "role": "assistant",
                        "content": response["response"],
                        "sources": response.get("sources", [])
                    })

                # Rerun to update the chat display
                st.rerun()
