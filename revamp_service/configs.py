from asyncio import Queue, Semaphore
import asyncio
from dataclasses import dataclass, field
import os
import threading
from typing import Dict
from dotenv import load_dotenv
from .logger import *
logging = get_logger(__name__)
load_dotenv()

@dataclass
class taskManagerConfig:
    MAX_CONCURRENT_TASKS : int = 2
    semaphore = Semaphore(MAX_CONCURRENT_TASKS)
    active_tasks: Dict[str, dict] = field(default_factory=dict)
    task_queue: Queue = Queue()
    processing_lock = asyncio.Lock()
    is_processing: bool = False


@dataclass
class Config:

    BASE_URL: str 
    MODEL: str 
    RAG_CHUNK_SIZE: int 
    RAG_CHUNK_OVERLAP: int
    MAX_PROCESSING_ROWS: int 
    MAX_WORKERS: int 
    TEMPERATURE: float 
    NUM_CTX: int
    NUM_THREAD: int


@dataclass
class OllamaConfig(Config):

    def __init__(self):
        super().__init__(
            BASE_URL=os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
            MODEL=os.environ.get('OLLAMA_MODEL', 'llama3.2:1b'),
            RAG_CHUNK_SIZE=int(os.environ.get('RAG_CHUNK_SIZE', 300)),
            RAG_CHUNK_OVERLAP=int(os.environ.get('RAG_CHUNK_OVERLAP', 30)),
            MAX_PROCESSING_ROWS=int(os.environ.get('MAX_PROCESSING_ROWS', 100)),
            MAX_WORKERS=int(os.environ.get('MAX_WORKERS', 4)),
            TEMPERATURE=float(os.environ.get('TEMPERATURE', 0.1)),
            NUM_CTX=int(os.environ.get('NUM_CTX', 2048)),
            NUM_THREAD=int(os.environ.get('NUM_THREAD', min(4, os.cpu_count())))
        )
    # Email Configuration
@dataclass
class Mailconfig():
    SMTP_SERVER= os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT= int(os.environ.get('SMTP_PORT', '587'))
    EMAIL_USER= os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
    FROM_EMAIL = os.environ.get('FROM_EMAIL', 'tester7760775061@gmail.com')
@dataclass
class NumericColumnAnalyzerAgentConfig():
    BASE_URL: str 
    MODEL: str 
    TEMPERATURE: float 

@dataclass
class OllamaNumericColumnAnalyzerAgentConfig(NumericColumnAnalyzerAgentConfig):
    BASE_URL: str = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    MODEL: str = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
    TEMPERATURE: float =0.1,
@dataclass
class CachingConfig:
    CACHE_TTL = 3600
    MAX_CACHE_SIZE = 100
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3.2:3b"
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    REDIS_URL = "redis://localhost:6380"
    BASE_URL='http://localhost:11434'
    def __init__(self):
        # Enhanced caching (optional)
        try:
            from cachetools import TTLCache
            import redis.asyncio as redis
            self.REDIS_AVAILABLE = True
            self.TTLCache = TTLCache
            self.redis = redis
        except ImportError:
            self.REDIS_AVAILABLE = False
            logging.warning("Redis not available. Using local cache only.")
@dataclass
class GroqChatRag:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    TEMPERATURE=0.7
    MAX_TOKEN=2000
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    GROQ_API_KEY=os.environ.get('groq_api_key')
            
        

