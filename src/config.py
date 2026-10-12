"""
Configuration and settings module for the AI Resume Screening & Ranking System.
Zero-external-dependency environment loading and logging setup.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict

def load_env_file(filepath: str = ".env"):
    """Load key-value pairs from .env file into os.environ without requiring python-dotenv."""
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val

load_env_file(".env")

# Configure Logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ResumeScreener")

@dataclass
class ScoringWeights:
    ai_project_depth: float = 40.0
    python_backend: float = 30.0
    cloud_fullstack: float = 15.0
    github: float = 10.0
    engineering_depth: float = 5.0

@dataclass
class AppConfig:
    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # GitHub Settings
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_api_base: str = "https://api.github.com"
    github_timeout_seconds: float = 5.0
    
    # Concurrency and Performance
    concurrency_limit: int = int(os.getenv("CONCURRENCY_LIMIT", "5"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    
    # Weights
    weights: ScoringWeights = field(default_factory=ScoringWeights)

config = AppConfig()
