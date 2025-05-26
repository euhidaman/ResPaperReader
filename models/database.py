import sqlite3
import os
import json
import threading
import logging
from datetime import datetime


class DatabaseManager:
    def __init__(self, config=None):
        """Initialize the database connection with SQLite."""
        # Use SQLite database file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_dir, "data", "papers.db")

        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        logging.info(f"Database path set to: {self.db_path}")
        self._local = threading.local()
        self.create_tables()

    def _get_connection(self):
        """Get the thread-local connection, creating it if needed."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row  # For dictionary-like access
        return self._local.conn

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create papers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,
            source VARCHAR(255),
            file_path TEXT,
            embedding_id VARCHAR(255),
            full_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        cursor.close()

    def add_paper(self, title, abstract, authors=None, source="internal_upload", file_path=None, embedding_id=None, full_text=None):
        """Add a paper to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        authors_json = json.dumps(authors) if authors else None

        cursor.execute('''
        INSERT INTO papers (title, abstract, authors, source, file_path, embedding_id, full_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (title, abstract, authors_json, source, file_path, embedding_id, full_text))

        conn.commit()
        last_id = cursor.lastrowid
        cursor.close()
        return last_id

    def get_paper(self, paper_id):
        """Get a paper by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE id = ?', (paper_id,))
        row = cursor.fetchone()
        cursor.close()

        if row:
            paper = dict(row)
            if paper.get('authors'):
                try:
                    paper['authors'] = json.loads(paper['authors'])
                except (json.JSONDecodeError, TypeError):
                    paper['authors'] = []
            return paper
        return None

    def search_papers(self, keyword, limit=5):
        """Search papers by keyword in title, abstract, or full text with case-insensitive matching."""
        conn = self._get_connection()
        cursor = conn.cursor()
        keyword_pattern = f'%{keyword}%'
        cursor.execute('''
        SELECT * FROM papers 
        WHERE LOWER(title) LIKE LOWER(?) OR LOWER(abstract) LIKE LOWER(?) OR LOWER(full_text) LIKE LOWER(?)
        ORDER BY created_at DESC LIMIT ?
        ''', (keyword_pattern, keyword_pattern, keyword_pattern, limit))

        rows = cursor.fetchall()
        cursor.close()

        results = []
        for row in rows:
            paper = dict(row)
            if paper.get('authors'):
                try:
                    paper['authors'] = json.loads(paper['authors'])
                except (json.JSONDecodeError, TypeError):
                    paper['authors'] = []
            results.append(paper)

        return results

    def find_paper_by_title_fuzzy(self, title_query, threshold=0.7):
        """Find papers by title using fuzzy matching for better results."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM papers')
        rows = cursor.fetchall()
        cursor.close()

        # Parse authors field and calculate similarity scores
        from difflib import SequenceMatcher
        matches = []
        title_query_lower = title_query.lower().strip()

        for row in rows:
            paper = dict(row)
            if paper.get('authors'):
                try:
                    paper['authors'] = json.loads(paper['authors'])
                except (json.JSONDecodeError, TypeError):
                    paper['authors'] = []

            paper_title_lower = paper['title'].lower().strip()

            # Calculate similarity score
            similarity = SequenceMatcher(
                None, title_query_lower, paper_title_lower).ratio()

            # Also check if the query is contained within the title (substring match)
            contains_match = title_query_lower in paper_title_lower or paper_title_lower in title_query_lower

            if similarity >= threshold or contains_match:
                paper['similarity_score'] = similarity
                matches.append(paper)

        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        return matches

    def get_all_papers(self, limit=10):
        """Get all papers with optional limit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM papers ORDER BY created_at DESC LIMIT ?', (limit,))

        rows = cursor.fetchall()
        cursor.close()

        results = []
        for row in rows:
            paper = dict(row)
            if paper.get('authors'):
                try:
                    paper['authors'] = json.loads(paper['authors'])
                except (json.JSONDecodeError, TypeError):
                    paper['authors'] = []
            results.append(paper)

        return results

    def delete_paper(self, paper_id):
        """Delete a paper from the database.

        Args:
            paper_id: ID of the paper to delete

        Returns:
            Tuple of (success_boolean, file_path_or_error_message)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # First, get the paper to retrieve file path before deletion
            cursor.execute(
                'SELECT file_path FROM papers WHERE id = ?', (paper_id,))
            result = cursor.fetchone()

            if not result:
                cursor.close()
                return False, "Paper not found"

            file_path = result[0]

            # Delete the paper from the database
            cursor.execute('DELETE FROM papers WHERE id = ?', (paper_id,))
            conn.commit()
            cursor.close()

            return True, file_path
        except Exception as e:
            logging.error(f"Error deleting paper: {e}")
            return False, str(e)

    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
