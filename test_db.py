import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

# Get connection parameters from environment variables
host = os.getenv("MYSQL_HOST", "localhost")
database = os.getenv("MYSQL_DATABASE", "papers_db")
user = os.getenv("MYSQL_USER", "paperuser")
password = os.getenv("MYSQL_PASSWORD", "paperpass")

try:
    # Try to connect
    connection = mysql.connector.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )

    if connection.is_connected():
        db_info = connection.get_server_info()
        print(f"Connected to MySQL/MariaDB server version {db_info}")

        # Check if our tables exist
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(f"- {table[0]}")

        cursor.close()
        connection.close()
        print("Connection closed.")

except mysql.connector.Error as e:
    print(f"Error connecting to MySQL/MariaDB: {e}")
