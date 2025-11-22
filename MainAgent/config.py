"""Configuration for MainAgent system."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Basic config
CONFIG = {
    "project_name": "your-agentic-system",
    "groq_api_key": os.getenv("GROQ_API_KEY"),
    "groq_model": os.getenv("GROQ_MODEL"),
}
