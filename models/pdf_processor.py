import os
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
import os.path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Keep pdfplumber for backward compatibility
import pdfplumber


class PDFProcessor:
    def __init__(self, upload_dir=None):
        """Initialize PDF processor with upload directory."""
        if upload_dir is None:
            # Use absolute path for consistency
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            self.upload_dir = os.path.join(base_dir, "data", "uploads")
        else:
            self.upload_dir = upload_dir

        logging.info(f"PDF uploads directory set to: {self.upload_dir}")
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_pdf(self, pdf_file):
        """
        Save an uploaded PDF file.

        Args:
            pdf_file: File object from Streamlit

        Returns:
            Path to the saved file
        """
        try:
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{pdf_file.name}"
            file_path = os.path.join(self.upload_dir, filename)

            # Save the file
            with open(file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            return file_path
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            return None

    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract title, abstract, and authors from a PDF file using LangChain.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing title, abstract, and authors
        """
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Get text from the first 3 pages which typically contain metadata
            text = ""
            for doc in documents[:3]:
                text += doc.page_content + "\n"

            # Extract title with improved logic
            title = self._extract_title(text)

            # Look for abstract section
            abstract = self._extract_abstract(text)

            # Extract author information
            authors = self._extract_authors(text)

            return {
                "title": title,
                "abstract": abstract,
                "authors": authors
            }
        except Exception as e:
            logging.error(f"Error extracting metadata from PDF: {e}")
            return {
                "title": os.path.basename(file_path).replace('.pdf', ''),
                "abstract": "",
                "authors": []
            }

    def _extract_title(self, text: str) -> str:
        """Extract title from PDF text with improved logic."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            return "Unknown Title"

        # Filter out common non-title patterns
        filtered_lines = []
        for line in lines[:20]:  # Check first 20 lines
            # Skip lines that are clearly not titles
            if (len(line) < 5 or
                line.lower().startswith(('arxiv:', 'submitted', 'published', 'copyright', 'preprint', 'under review')) or
                '@' in line or  # Email addresses
                line.isupper() and len(line) > 50 or  # All caps long lines (likely headers)
                re.match(r'^[\d\s\.\-]+$', line) or  # Just numbers/dates
                    line.lower() in ['abstract', 'introduction', 'conclusion', 'references']):
                continue
            filtered_lines.append(line)

        if not filtered_lines:
            return lines[0] if lines else "Unknown Title"

        # Look for the longest meaningful line in the first few filtered lines
        # Title is often the longest line among the first few lines
        candidates = filtered_lines[:5]

        # Score each candidate
        best_candidate = candidates[0]
        best_score = 0

        for candidate in candidates:
            score = 0
            # Longer titles get higher scores (up to a reasonable limit)
            score += min(len(candidate), 100) * 0.1
            # Titles with proper capitalization get bonus points
            if candidate.istitle() or (candidate[0].isupper() and not candidate.isupper()):
                score += 20
            # Penalize lines with lots of numbers or special characters
            if len(re.findall(r'[^\w\s]', candidate)) > len(candidate) * 0.3:
                score -= 10
            # Bonus for common title words
            title_words = ['attention', 'learning', 'neural',
                           'deep', 'machine', 'analysis', 'study', 'approach']
            if any(word in candidate.lower() for word in title_words):
                score += 10

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract with improved logic."""
        # Multiple patterns to catch different abstract formats
        patterns = [
            r'(?:Abstract|ABSTRACT)[\s\.\:\-]*\n*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:[A-Z][a-z]|Keywords|Introduction|1\s+Introduction))',
            r'(?:Abstract|ABSTRACT)[\s\.\:\-]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\nKeywords|\n1\.|\nIntroduction)',
            r'(?:Abstract|ABSTRACT)[\s\.\:\-]*([^\.]+(?:\.[^\.]+)*?)(?:\n\n|\nKeywords|\n1\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                # Replace newlines with spaces
                abstract = re.sub(r'\n+', ' ', abstract)
                # Normalize whitespace
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Only return if it's substantial
                    return abstract

        return ""

    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors with improved logic."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Look for author patterns in the first 10 lines
        for i, line in enumerate(lines[:10]):
            # Skip title-like lines (too long or all caps)
            if len(line) > 100 or (line.isupper() and len(line) > 20):
                continue

            # Look for lines with names (containing proper nouns)
            if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', line):
                # Check if this looks like an author line
                # Authors often have commas, "and", or multiple capitalized words
                if (',' in line or ' and ' in line.lower() or
                        len(re.findall(r'[A-Z][a-z]+', line)) >= 2):

                    # Clean and split authors
                    author_line = line
                    # Remove affiliations in parentheses or with superscripts
                    author_line = re.sub(r'\([^)]*\)', '', author_line)
                    author_line = re.sub(
                        r'[¹²³⁴⁵⁶⁷⁸⁹⁰\*\†\‡]', '', author_line)

                    # Split by common separators
                    if ' and ' in author_line.lower():
                        authors = re.split(
                            r'\s+and\s+', author_line, flags=re.IGNORECASE)
                    elif ',' in author_line:
                        authors = [a.strip() for a in author_line.split(',')]
                    else:
                        authors = [author_line]

                    # Filter valid author names
                    valid_authors = []
                    for author in authors:
                        author = author.strip()
                        # Check if it looks like a person's name
                        if (author and len(author) > 3 and
                            re.search(r'[A-Z][a-z]+', author) and
                                not author.lower().startswith(('university', 'department', 'institute'))):
                            valid_authors.append(author)

                    if valid_authors:
                        return valid_authors

        return []

    def extract_full_text(self, file_path: str) -> str:
        """
        Extract full text from a PDF using LangChain.

        Args:
            file_path: Path to the PDF file

        Returns:
            Full text content of the PDF
        """
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Combine text from all documents/pages
            text = "\n\n".join([doc.page_content for doc in documents])
            return text
        except Exception as e:
            logging.error(f"Error extracting full text: {e}")
            return ""

    def extract_documents(self, file_path: str) -> List[Document]:
        """
        Extract pages as LangChain Documents from a PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of LangChain Document objects
        """
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logging.error(
                f"Error loading document as LangChain Documents: {e}")
            return []
