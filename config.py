import os
from dotenv import load_dotenv

load_dotenv(override=True)

# General project configuration
PROJECT_NAME = "HR Assistant"
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
# k closest chunks
K_CLOSEST = 3

CV_ROOT = "cv_base/*"
DB_NAME = "cv_db"

# Scraper settings
USER_AGENT = "Mozilla/5.0"

# Parallel processing
MAX_WORKERS = 10  # for ThreadPoolExecutor

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
