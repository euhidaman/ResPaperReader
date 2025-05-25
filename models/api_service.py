import arxiv
import logging
from datetime import datetime
import time
import requests
import os


class APIService:
    def __init__(self):
        """Initialize the API service for arXiv paper searches."""
        self.rate_limit_delay = 1  # seconds between requests

    def search_arxiv(self, query, max_results=10):
        """
        Search for papers on arXiv with improved relevance sorting.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of papers with metadata
        """
        try:
            # Try multiple search strategies for better results
            search_strategies = [
                f'ti:"{query}"',  # Exact title match
                f'ti:{query}',    # Title search
                f'all:{query}',   # All fields search
            ]

            all_papers = []
            seen_titles = set()

            for search_query in search_strategies:
                try:
                    search = arxiv.Search(
                        query=search_query,
                        max_results=max_results,
                        sort_by=arxiv.SortCriterion.Relevance
                    )

                    for result in search.results():
                        title_lower = result.title.lower()
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            all_papers.append({
                                'title': result.title,
                                'abstract': result.summary,
                                'authors': [author.name for author in result.authors],
                                'url': result.entry_id,
                                'pdf_url': result.pdf_url,
                                'source': 'arxiv',
                                'published': result.published.strftime('%Y-%m-%d') if result.published else None,
                                'arxiv_id': result.entry_id.split('/')[-1]
                            })

                    # If we found exact matches in title search, prioritize those
                    if search_query.startswith('ti:') and all_papers:
                        break

                except Exception as e:
                    logging.warning(
                        f"Search strategy '{search_query}' failed: {e}")
                    continue

            return all_papers[:max_results]

        except Exception as e:
            logging.error(f"Error searching arXiv: {e}")
            return []

    def download_paper_pdf(self, pdf_url, save_path):
        """
        Download a paper PDF from arXiv.

        Args:
            pdf_url: URL of the PDF to download
            save_path: Local path to save the PDF

        Returns:
            Boolean indicating success
        """
        try:
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True
        except Exception as e:
            logging.error(f"Error downloading PDF: {e}")
            return False

    def search_papers_by_conference(self, conf_name, year=None, max_results=10):
        """
        Search for papers from a specific conference on arXiv.

        Args:
            conf_name: Conference name (e.g., 'ICLR', 'NeurIPS', 'ACL')
            year: Publication year
            max_results: Maximum number of results to return

        Returns:
            List of papers with metadata
        """
        # If year is not specified, use current year
        if not year:
            year = datetime.now().year

        # Format query for arXiv
        query = f"{conf_name} {year}"
        return self.search_arxiv(query, max_results=max_results)
