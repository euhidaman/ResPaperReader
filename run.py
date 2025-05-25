#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import dotenv


def main():
    """Main entry point for the Research Paper Reader application."""
    # Load environment variables from .env file if it exists
    dotenv.load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Research Paper Reader and Assistant")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port to run the Streamlit app on")
    parser.add_argument("--gemini-key", type=str, help="Google Gemini API key")

    # MySQL database arguments
    parser.add_argument("--mysql-host", type=str, help="MySQL database host")
    parser.add_argument("--mysql-port", type=int, help="MySQL database port")
    parser.add_argument("--mysql-user", type=str, help="MySQL database user")
    parser.add_argument("--mysql-password", type=str,
                        help="MySQL database password")
    parser.add_argument("--mysql-database", type=str,
                        help="MySQL database name")

    args = parser.parse_args()

    # Set API key if provided
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    # Set MySQL environment variables if provided
    if args.mysql_host:
        os.environ["MYSQL_HOST"] = args.mysql_host
    if args.mysql_port:
        os.environ["MYSQL_PORT"] = str(args.mysql_port)
    if args.mysql_user:
        os.environ["MYSQL_USER"] = args.mysql_user
    if args.mysql_password:
        os.environ["MYSQL_PASSWORD"] = args.mysql_password
    if args.mysql_database:
        os.environ["MYSQL_DATABASE"] = args.mysql_database

    # Add environment variable to fix Torch errors in Streamlit's watcher
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    # Ensure data directories exist
    os.makedirs("data/uploads", exist_ok=True)

    # Run the Streamlit app
    cmd = [
        "streamlit", "run",
        "app/app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down Research Paper Reader...")
        sys.exit(0)


if __name__ == "__main__":
    main()
