from asyncio import Queue, Semaphore
import asyncio
from dataclasses import dataclass, field
import os
import threading
from typing import Dict
from dotenv import load_dotenv
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
