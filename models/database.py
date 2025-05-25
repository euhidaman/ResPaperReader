import mysql.connector
import os
import json
import threading
import logging
from datetime import datetime
from mysql.connector import Error
from config.database_config import MYSQL_CONFIG


class DatabaseManager:
    def __init__(self, config=None):
        """Initialize the database connection with MySQL."""
        # Start with default configuration from config file
        self.config = MYSQL_CONFIG.copy()

        # Override with environment variables if set
        if os.environ.get('MYSQL_HOST'):
            self.config['host'] = os.environ.get('MYSQL_HOST')
        if os.environ.get('MYSQL_DATABASE'):
            self.config['database'] = os.environ.get('MYSQL_DATABASE')
        if os.environ.get('MYSQL_USER'):
            self.config['user'] = os.environ.get('MYSQL_USER')
        if os.environ.get('MYSQL_PASSWORD'):
            self.config['password'] = os.environ.get('MYSQL_PASSWORD')
        if os.environ.get('MYSQL_PORT'):
            self.config['port'] = int(os.environ.get('MYSQL_PORT'))

        # Override with provided config if any
        if config:
            self.config.update(config)

        logging.info(
            f"Database connection set to: {self.config['host']}/{self.config['database']}")
        self._local = threading.local()
        self._initialize_connection()
        self.create_tables()

    def _initialize_connection(self):
        """Initialize a thread-local connection to MySQL."""
        if not hasattr(self._local, 'conn'):
            try:
                self._local.conn = mysql.connector.connect(
                    host=self.config['host'],
                    database=self.config['database'],
                    user=self.config['user'],
                    password=self.config['password'],
                    port=self.config['port']
                )
                logging.info("MySQL database connection successful")
            except Error as e:
                logging.error(f"Error connecting to MySQL database: {e}")
                # Try to create database if it doesn't exist
                self._create_database_if_not_exists()

    def _create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        try:
            conn = mysql.connector.connect(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                port=self.config['port']
            )
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.config['database']}")
            cursor.close()
            conn.close()
            logging.info(
                f"Database {self.config['database']} created successfully")

            # Try to reconnect after creating the database
            self._local.conn = mysql.connector.connect(
                host=self.config['host'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                port=self.config['port']
            )
        except Error as e:
            logging.error(f"Error creating database: {e}")
            raise

    def _get_connection(self):
        """Get the thread-local connection, creating it if needed."""
        if not hasattr(self._local, 'conn') or self._local.conn is None or not self._local.conn.is_connected():
            self._initialize_connection()
        return self._local.conn

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create papers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,
            source VARCHAR(255),
            file_path TEXT,
            embedding_id VARCHAR(255),
            full_text LONGTEXT,
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
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (title, abstract, authors_json, source, file_path, embedding_id, full_text))

        conn.commit()
        last_id = cursor.lastrowid
        cursor.close()
        return last_id

    def get_paper(self, paper_id):
        """Get a paper by ID."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM papers WHERE id = %s', (paper_id,))
        paper = cursor.fetchone()
        cursor.close()

        if paper:
            if paper.get('authors'):
                paper['authors'] = json.loads(paper['authors'])
            return paper
        return None

    def search_papers(self, keyword, limit=5):
        """Search papers by keyword in title, abstract, or full text."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        keyword_pattern = f'%{keyword}%'
        cursor.execute('''
        SELECT * FROM papers 
        WHERE title LIKE %s OR abstract LIKE %s OR full_text LIKE %s
        ORDER BY created_at DESC LIMIT %s
        ''', (keyword_pattern, keyword_pattern, keyword_pattern, limit))

        results = cursor.fetchall()
        cursor.close()

        # Parse authors field
        for paper in results:
            if paper.get('authors'):
                paper['authors'] = json.loads(paper['authors'])

        return results

    def get_all_papers(self, limit=10):
        """Get all papers with optional limit."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT * FROM papers ORDER BY created_at DESC LIMIT %s', (limit,))

        results = cursor.fetchall()
        cursor.close()

        # Parse authors field
        for paper in results:
            if paper.get('authors'):
                paper['authors'] = json.loads(paper['authors'])

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
                'SELECT file_path FROM papers WHERE id = %s', (paper_id,))
            result = cursor.fetchone()

            if not result:
                cursor.close()
                return False, "Paper not found"

            file_path = result[0]

            # Delete the paper from the database
            cursor.execute('DELETE FROM papers WHERE id = %s', (paper_id,))
            conn.commit()
            cursor.close()

            return True, file_path
        except Error as e:
            logging.error(f"Error deleting paper: {e}")
            return False, str(e)

    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            if self._local.conn.is_connected():
                self._local.conn.close()
            self._local.conn = None

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
