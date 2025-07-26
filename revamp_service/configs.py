from asyncio import Queue, Semaphore
from dataclasses import dataclass, field
import os
import threading
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

@dataclass
class taskManagerConfig:
    max_concurrent_tasks : int = 2
    semaphore = Semaphore(max_concurrent_tasks)
    active_tasks: Dict[str, dict] = field(default_factory=dict)
    task_queue: Queue = Queue()
    processing_lock = threading.Lock()
    is_processing: bool = False


@dataclass
class Config:

    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
    RAG_CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', '300'))
    RAG_CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', '30'))
    MAX_PROCESSING_ROWS = int(os.environ.get('MAX_PROCESSING_ROWS', '100'))
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '4'))
    
    # Email Configuration
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    EMAIL_USER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
    FROM_EMAIL = os.environ.get('FROM_EMAIL', 'tester7760775061@gmail.com')