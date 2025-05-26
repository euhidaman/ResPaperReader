import arxiv
import logging
from datetime import datetime
import time
import requests
import os
import re


class APIService:
    def __init__(self):
        """Initialize the API service for arXiv paper searches."""
        self.rate_limit_delay = 1  # seconds between requests

        # Comprehensive conference mapping with aliases and full names
        self.conference_mapping = {
            # Machine Learning conferences
            'neurips': ['NeurIPS', 'NIPS', 'Neural Information Processing Systems', 'Advances in Neural Information Processing Systems'],
            'nips': ['NeurIPS', 'NIPS', 'Neural Information Processing Systems', 'Advances in Neural Information Processing Systems'],
            'icml': ['ICML', 'International Conference on Machine Learning'],
            'iclr': ['ICLR', 'International Conference on Learning Representations'],
            'aistats': ['AISTATS', 'International Conference on Artificial Intelligence and Statistics'],
            'uai': ['UAI', 'Conference on Uncertainty in Artificial Intelligence'],

            # Natural Language Processing conferences
            'acl': ['ACL', 'Association for Computational Linguistics', 'Annual Meeting of the Association for Computational Linguistics'],
            'emnlp': ['EMNLP', 'Empirical Methods in Natural Language Processing', 'Conference on Empirical Methods in Natural Language Processing'],
            'naacl': ['NAACL', 'North American Chapter of the Association for Computational Linguistics'],
            'eacl': ['EACL', 'European Chapter of the Association for Computational Linguistics'],
            'coling': ['COLING', 'International Conference on Computational Linguistics'],
            'conll': ['CoNLL', 'Conference on Computational Natural Language Learning'],

            # Computer Vision conferences
            'cvpr': ['CVPR', 'Computer Vision and Pattern Recognition', 'IEEE Conference on Computer Vision and Pattern Recognition'],
            'iccv': ['ICCV', 'International Conference on Computer Vision'],
            'eccv': ['ECCV', 'European Conference on Computer Vision'],
            'bmvc': ['BMVC', 'British Machine Vision Conference'],
            'wacv': ['WACV', 'Winter Conference on Applications of Computer Vision'],

            # AI conferences
            'aaai': ['AAAI', 'Association for the Advancement of Artificial Intelligence', 'Conference on Artificial Intelligence'],
            'ijcai': ['IJCAI', 'International Joint Conference on Artificial Intelligence'],
            'aamas': ['AAMAS', 'International Conference on Autonomous Agents and Multiagent Systems'],

            # Systems and Software Engineering
            'icse': ['ICSE', 'International Conference on Software Engineering'],
            'fse': ['FSE', 'ACM SIGSOFT International Symposium on Foundations of Software Engineering'],
            'pldi': ['PLDI', 'Programming Language Design and Implementation'],
            'popl': ['POPL', 'Principles of Programming Languages'],

            # Database and Data Management
            'sigmod': ['SIGMOD', 'ACM SIGMOD International Conference on Management of Data'],
            'vldb': ['VLDB', 'Very Large Data Bases'],
            'icde': ['ICDE', 'International Conference on Data Engineering'],

            # Security
            'ccs': ['CCS', 'ACM Conference on Computer and Communications Security'],
            'usenix': ['USENIX Security', 'USENIX Security Symposium'],
            'ndss': ['NDSS', 'Network and Distributed System Security Symposium'],

            # Theory
            'stoc': ['STOC', 'Symposium on Theory of Computing'],
            'focs': ['FOCS', 'Symposium on Foundations of Computer Science'],
            'soda': ['SODA', 'Symposium on Discrete Algorithms'],

            # Graphics
            'siggraph': ['SIGGRAPH', 'ACM SIGGRAPH'],
            'eurographics': ['Eurographics', 'European Association for Computer Graphics'],

            # Robotics
            'icra': ['ICRA', 'International Conference on Robotics and Automation'],
            'iros': ['IROS', 'International Conference on Intelligent Robots and Systems'],
            'rss': ['RSS', 'Robotics: Science and Systems']
        }

    def search_arxiv(self, query, max_results=5):
        """
        Search for papers on arXiv with improved relevance sorting and conference handling.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default 5)

        Returns:
            List of papers with metadata
        """
        try:
            # Check if this is a conference-specific query
            conference_keywords = {
                'emnlp': ['EMNLP', 'Empirical Methods in Natural Language Processing'],
                'iclr': ['ICLR', 'International Conference on Learning Representations'],
                'neurips': ['NeurIPS', 'NIPS', 'Neural Information Processing Systems'],
                'icml': ['ICML', 'International Conference on Machine Learning'],
                'acl': ['ACL', 'Association for Computational Linguistics'],
                'aaai': ['AAAI', 'Association for the Advancement of Artificial Intelligence'],
                'ijcai': ['IJCAI', 'International Joint Conference on Artificial Intelligence'],
                'cvpr': ['CVPR', 'Computer Vision and Pattern Recognition'],
                'iccv': ['ICCV', 'International Conference on Computer Vision'],
                'eccv': ['ECCV', 'European Conference on Computer Vision'],
                'coling': ['COLING', 'International Conference on Computational Linguistics'],
                'naacl': ['NAACL', 'North American Chapter of the Association for Computational Linguistics'],
                'eacl': ['EACL', 'European Chapter of the Association for Computational Linguistics']
            }

            # Extract conference and topic from query
            query_lower = query.lower()
            conference_found = None
            conference_year = None
            topic = query  # Default to full query

            # Check for year patterns first
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                conference_year = year_match.group(1)

            # Check for conference patterns
            for conf_key, conf_names in conference_keywords.items():
                for conf_name in conf_names:
                    if conf_name.lower() in query_lower:
                        conference_found = conf_name
                        # Extract the topic by removing conference references and year
                        topic = query_lower
                        for remove_term in [conf_name.lower(), 'conference', 'from', 'latest', 'papers', 'about', str(conference_year) if conference_year else '']:
                            topic = topic.replace(remove_term, '')
                        topic = topic.strip()
                        break
                if conference_found:
                    break

            # If it's a pure conference query, use the specialized method
            if conference_found and (not topic or len(topic.strip()) < 3):
                return self.search_papers_by_conference(conference_found, conference_year, max_results)

            # Build search strategies based on query type
            search_strategies = []

            if conference_found and topic:
                # Conference + topic search
                search_strategies = [
                    f'all:"{conference_found}" AND all:"{topic}"',
                    f'ti:"{topic}" AND abs:"{conference_found}"',
                    f'all:"{topic}" AND submittedDate:[{conference_year}0101 TO {conference_year}1231]' if conference_year else f'all:"{topic}"',
                ]
            else:
                # Regular topic search with improved strategies for technical terms
                clean_query = query.strip()

                # For technical terms, use multiple search approaches
                search_strategies = [
                    f'ti:"{clean_query}"',  # Exact title match
                    f'abs:"{clean_query}"',  # Exact abstract match
                    # Title search (words can be separate)
                    f'ti:{clean_query}',
                    f'all:{clean_query}',   # All fields search
                ]

                # For compound technical terms (like TinyML), also search for components
                if len(clean_query.split()) == 1 and len(clean_query) > 4:
                    # Try breaking compound words for better matching
                    # Handle camelCase or compound terms
                    word_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', clean_query)
                    if len(word_parts) > 1:
                        search_strategies.append(
                            f'all:"{" ".join(word_parts)}"')
                        search_strategies.append(
                            f'ti:"{" ".join(word_parts)}"')

            all_papers = []
            seen_titles = set()

            for i, search_query in enumerate(search_strategies):
                try:
                    logging.info(
                        f"Trying search strategy {i+1}: {search_query}")

                    search = arxiv.Search(
                        query=search_query,
                        max_results=max_results * 3,  # Get more to filter and rank
                        sort_by=arxiv.SortCriterion.Relevance  # Use relevance for better topic matching
                    )

                    strategy_papers = []
                    for result in search.results():
                        title_lower = result.title.lower()
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)

                            # Calculate relevance score for ranking
                            relevance_score = self._calculate_relevance(
                                result, query, conference_found, topic if conference_found else query)

                            paper = {
                                'title': result.title,
                                'abstract': result.summary,
                                'authors': [author.name for author in result.authors],
                                'url': result.entry_id,
                                'pdf_url': result.pdf_url,
                                'source': 'arxiv',
                                'published': result.published.strftime('%Y-%m-%d') if result.published else None,
                                'arxiv_id': result.entry_id.split('/')[-1],
                                'relevance_score': relevance_score,
                                'search_strategy': i + 1
                            }
                            strategy_papers.append(paper)

                    # Add papers from this strategy
                    all_papers.extend(strategy_papers)

                    # If we found good matches with exact searches, prioritize them
                    # First two strategies are exact matches
                    if i < 2 and len(strategy_papers) > 0:
                        high_relevance_papers = [
                            p for p in strategy_papers if p.get('relevance_score', 0) > 80]
                        if len(high_relevance_papers) >= max_results:
                            break

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    logging.warning(
                        f"Search strategy '{search_query}' failed: {e}")
                    continue

            # Remove duplicates and sort by relevance
            unique_papers = {}
            for paper in all_papers:
                title_key = paper['title'].lower()
                if title_key not in unique_papers or paper.get('relevance_score', 0) > unique_papers[title_key].get('relevance_score', 0):
                    unique_papers[title_key] = paper

            final_papers = list(unique_papers.values())
            final_papers.sort(key=lambda x: x.get(
                'relevance_score', 0), reverse=True)

            result_papers = final_papers[:max_results]

            if result_papers:
                logging.info(
                    f"Found {len(result_papers)} papers for query: {query}")
            else:
                logging.warning(f"No relevant papers found for query: {query}")

            return result_papers

        except Exception as e:
            logging.error(f"Error searching arXiv: {e}")
            return []

    def _calculate_relevance(self, result, original_query, conference=None, topic=None):
        """Calculate relevance score for ranking search results with improved technical term matching."""
        score = 0
        title_lower = result.title.lower()
        abstract_lower = result.summary.lower()
        query_lower = original_query.lower()

        # Conference match in title gets highest score
        if conference and conference.lower() in title_lower:
            score += 100

        # Conference match in abstract
        if conference and conference.lower() in abstract_lower:
            score += 50

        # Enhanced topic matching for technical terms
        search_term = topic if topic else query_lower
        search_words = search_term.split()

        # Exact phrase match in title (highest priority)
        if search_term in title_lower:
            score += 150

        # Exact phrase match in abstract
        if search_term in abstract_lower:
            score += 100

        # Individual word matches in title
        title_word_matches = 0
        for word in search_words:
            if len(word) > 2 and word in title_lower:
                score += 40
                title_word_matches += 1

        # Bonus for multiple word matches in title
        if len(search_words) > 1 and title_word_matches == len(search_words):
            score += 50  # All words found in title

        # Individual word matches in abstract
        abstract_word_matches = 0
        for word in search_words:
            if len(word) > 2 and word in abstract_lower:
                score += 20
                abstract_word_matches += 1

        # Technical domain matching - look for related terms
        technical_domains = {
            'tinyml': ['edge', 'embedded', 'iot', 'microcontroller', 'mcu', 'energy', 'efficient', 'lightweight', 'mobile', 'device', 'hardware', 'compression', 'quantization', 'pruning'],
            'machine learning': ['neural', 'network', 'deep', 'learning', 'ai', 'artificial intelligence', 'model'],
            'nlp': ['language', 'text', 'linguistic', 'nlp', 'natural language', 'bert', 'transformer'],
            'computer vision': ['vision', 'image', 'visual', 'cnn', 'convolution', 'detection', 'recognition']
        }

        # Check if query matches any technical domain
        for domain, related_terms in technical_domains.items():
            # Check main terms
            if domain in query_lower or any(term in query_lower for term in related_terms[:3]):
                # Look for related terms in title and abstract
                for term in related_terms:
                    if term in title_lower:
                        score += 15
                    elif term in abstract_lower:
                        score += 8

        # Handle compound words and camelCase terms
        if len(search_term.split()) == 1 and len(search_term) > 4:
            # Try to split compound words
            word_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', search_term)
            if len(word_parts) > 1:
                # Check if parts appear together or separately
                parts_in_title = sum(
                    1 for part in word_parts if part.lower() in title_lower)
                parts_in_abstract = sum(
                    1 for part in word_parts if part.lower() in abstract_lower)

                if parts_in_title == len(word_parts):
                    score += 60  # All parts found in title
                elif parts_in_abstract == len(word_parts):
                    score += 30  # All parts found in abstract

        # Author authority bonus (look for prolific authors in the field)
        author_names = [author.name.lower() for author in result.authors]
        # This could be expanded with known expert authors in specific fields

        # Recency bonus (newer papers get slight preference)
        if result.published:
            days_old = (datetime.now() -
                        result.published.replace(tzinfo=None)).days
            if days_old < 365:  # Papers from last year
                score += 15
            elif days_old < 730:  # Papers from last 2 years
                score += 10
            elif days_old > 2190:  # Older than 6 years
                score -= 10

        # Citation pattern bonus (look for survey/review indicators)
        survey_indicators = ['survey', 'review', 'overview', 'comprehensive']
        for indicator in survey_indicators:
            if indicator in title_lower:
                score += 25  # Surveys are often high-value papers

        return max(score, 0)  # Ensure non-negative score

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

    def search_papers_by_conference(self, conf_name, year=None, max_results=5):
        """
        Search for papers from a specific conference on arXiv with improved targeting.

        Args:
            conf_name: Conference name (e.g., 'ICLR', 'NeurIPS', 'ACL')
            year: Publication year (optional)
            max_results: Maximum number of results to return (default 5)

        Returns:
            List of papers with metadata
        """
        try:
            # Normalize conference name and get all aliases
            conf_name_lower = conf_name.lower().strip()
            conference_aliases = []

            # Find the conference in our mapping
            if conf_name_lower in self.conference_mapping:
                conference_aliases = self.conference_mapping[conf_name_lower]
            else:
                # Check if the provided name is an alias in any of the mappings
                for key, aliases in self.conference_mapping.items():
                    if conf_name in aliases or conf_name.upper() in [alias.upper() for alias in aliases]:
                        conference_aliases = aliases
                        break

                # If not found in mapping, use the provided name
                if not conference_aliases:
                    conference_aliases = [conf_name, conf_name.upper()]

            # Build comprehensive search strategies
            search_strategies = []

            # Strategy 1: Conference name in title with year if provided
            for alias in conference_aliases:
                if year:
                    search_strategies.append(f'ti:"{alias} {year}"')
                    search_strategies.append(
                        f'ti:"{alias}" AND submittedDate:[{year}0101 TO {year}1231]')
                else:
                    search_strategies.append(f'ti:"{alias}"')

            # Strategy 2: Conference name in abstract with year constraint
            for alias in conference_aliases:
                if year:
                    search_strategies.append(
                        f'abs:"{alias}" AND submittedDate:[{year}0101 TO {year}1231]')
                    search_strategies.append(
                        f'abs:"accepted" AND abs:"{alias}" AND submittedDate:[{year}0101 TO {year}1231]')
                else:
                    search_strategies.append(f'abs:"{alias}"')
                    search_strategies.append(
                        f'abs:"accepted" AND abs:"{alias}"')

            # Strategy 3: Look for "accepted to" or "published at" patterns
            for alias in conference_aliases:
                acceptance_patterns = [
                    f"accepted to {alias}",
                    f"published at {alias}",
                    f"appears in {alias}",
                    f"presented at {alias}",
                    f"to appear in {alias}",
                    f"to appear at {alias}"
                ]

                for pattern in acceptance_patterns:
                    if year:
                        search_strategies.append(
                            f'abs:"{pattern}" AND submittedDate:[{year}0101 TO {year}1231]')
                    else:
                        search_strategies.append(f'abs:"{pattern}"')

            # Strategy 4: Broader search with conference context words
            conference_context_words = [
                "conference", "proceedings", "workshop", "symposium"]
            for alias in conference_aliases:
                for context in conference_context_words:
                    if year:
                        search_strategies.append(
                            f'all:"{alias}" AND all:"{context}" AND submittedDate:[{year}0101 TO {year}1231]')
                    else:
                        search_strategies.append(
                            f'all:"{alias}" AND all:"{context}"')

            # Strategy 5: Add flexible year matching for common off-by-one issues
            if year:
                year_int = int(year)
                prev_year = year_int - 1
                next_year = year_int + 1

                for alias in conference_aliases:
                    # Sometimes papers are submitted the year before the conference
                    search_strategies.append(
                        f'abs:"{alias} {prev_year}" OR abs:"{alias} {year}" OR abs:"{alias} {next_year}"')

            all_papers = []
            seen_titles = set()

            logging.info(
                f"Searching for {conf_name} papers with {len(search_strategies)} strategies")

            for i, search_query in enumerate(search_strategies):
                try:
                    logging.debug(f"Strategy {i+1}: {search_query}")

                    search = arxiv.Search(
                        query=search_query,
                        max_results=max_results * 3,  # Get more to filter and rank
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )

                    strategy_papers = []
                    for result in search.results():
                        title_lower = result.title.lower()
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)

                            # Calculate conference-specific relevance score
                            relevance_score = self._calculate_conference_relevance(
                                result, conf_name, conference_aliases, year)

                            # Only include papers with reasonable relevance
                            if relevance_score > 25:  # Minimum threshold
                                paper = {
                                    'title': result.title,
                                    'abstract': result.summary,
                                    'authors': [author.name for author in result.authors],
                                    'url': result.entry_id,
                                    'pdf_url': result.pdf_url,
                                    'source': 'arxiv',
                                    'published': result.published.strftime('%Y-%m-%d') if result.published else None,
                                    'arxiv_id': result.entry_id.split('/')[-1],
                                    'relevance_score': relevance_score,
                                    'conference': conf_name,
                                    'search_strategy': i + 1
                                }
                                strategy_papers.append(paper)

                    # Add papers from this strategy, prioritizing higher relevance
                    strategy_papers.sort(key=lambda x: x.get(
                        'relevance_score', 0), reverse=True)
                    all_papers.extend(strategy_papers)

                    # If we have enough high-quality results, stop searching
                    high_quality_papers = [
                        p for p in all_papers if p.get('relevance_score', 0) > 50]
                    if len(high_quality_papers) >= max_results:
                        logging.info(
                            f"Found {len(high_quality_papers)} high-quality papers, stopping search")
                        break

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    logging.warning(
                        f"Conference search strategy '{search_query}' failed: {e}")
                    continue

            # Remove duplicates and sort by relevance
            unique_papers = {}
            for paper in all_papers:
                title_key = paper['title'].lower()
                if title_key not in unique_papers or paper.get('relevance_score', 0) > unique_papers[title_key].get('relevance_score', 0):
                    unique_papers[title_key] = paper

            final_papers = list(unique_papers.values())
            final_papers.sort(key=lambda x: x.get(
                'relevance_score', 0), reverse=True)

            result_papers = final_papers[:max_results]
            logging.info(
                f"Returning {len(result_papers)} papers for conference {conf_name}")

            if not result_papers:
                logging.warning(
                    f"No papers found for conference {conf_name}" + (f" in {year}" if year else ""))

            return result_papers

        except Exception as e:
            logging.error(f"Error searching papers by conference: {e}")
            return []

    def _calculate_conference_relevance(self, result, conf_name, conference_aliases, year=None):
        """Calculate relevance score specifically for conference paper searches."""
        score = 0
        title_lower = result.title.lower()
        abstract_lower = result.summary.lower()

        # Very high scores for explicit conference mentions
        for alias in conference_aliases:
            alias_lower = alias.lower()

            # Conference name in title gets very high score
            if alias_lower in title_lower:
                score += 200
                # Bonus if it's an exact match or very prominent
                if f" {alias_lower} " in f" {title_lower} " or title_lower.startswith(alias_lower):
                    score += 100

            # Conference acceptance phrases in abstract
            acceptance_phrases = [
                f"accepted to {alias_lower}",
                f"accepted at {alias_lower}",
                f"published at {alias_lower}",
                f"published in {alias_lower}",
                f"appears in {alias_lower}",
                f"presented at {alias_lower}",
                f"to appear in {alias_lower}",
                f"to appear at {alias_lower}"
            ]

            for phrase in acceptance_phrases:
                if phrase in abstract_lower:
                    score += 150

            # Conference name in abstract (general mention)
            if alias_lower in abstract_lower:
                score += 75
                # Bonus for context words around conference name
                context_words = ["conference", "proceedings",
                                 "workshop", "accepted", "published", "presented"]
                for context in context_words:
                    if context in abstract_lower and abs(abstract_lower.find(context) - abstract_lower.find(alias_lower)) < 50:
                        score += 25

        # Year matching bonus
        if year and result.published:
            paper_year = result.published.year
            if paper_year == int(year):
                score += 100
            elif abs(paper_year - int(year)) <= 1:  # Within 1 year
                score += 50

        # Recency bonus if no specific year requested
        if not year and result.published:
            days_old = (datetime.now() -
                        result.published.replace(tzinfo=None)).days
            if days_old < 365:  # Last year
                score += 30
            elif days_old < 730:  # Last 2 years
                score += 15

        # Penalty for very old papers when no year specified
        if not year and result.published:
            days_old = (datetime.now() -
                        result.published.replace(tzinfo=None)).days
            if days_old > 2190:  # Older than 6 years
                score -= 20

        return score
