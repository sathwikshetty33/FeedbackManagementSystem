import asyncio
import logging
from typing import Dict, List, Any
from .models import *
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.schema import Document
from .logger import *
from .prompts import *
from .ChatbotSessionManager import *
from .Simpleneo4jKB import *
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

logging = get_logger(__name__)


class BaseRAGProcessor(ABC):
    def __init__(self):
        self.session_manager = EnhancedSessionManager()  
        self.graph_kb : BaseGraph
        self.embedding_model = None
        self.Config = CachingConfig()

    @abstractmethod    
    async def initialize(self):
       pass
    @abstractmethod       
    def dataframe_to_text_rows(self, df: pd.DataFrame) -> List[str]:
        """Convert dataframe to text rows"""
        pass
    
    @abstractmethod    
    def create_chunks(self, text: str) -> List[str]:
        """Create text chunks with logging"""
        pass
            
    @abstractmethod    
    def create_simple_qa_system(self, chunks: List[str], description: str):
        """Create simple QA system"""
        pass
    
    @abstractmethod    
    async def process_and_store_data(self, chunks: List[str], session_id: str, description: str):
        """Process data and create both vector and graph stores"""
        pass
    
    @abstractmethod    
    async def answer_question(self, question: str, session_data: Dict[str, Any], use_hybrid: bool = True) -> Dict[str, Any]:
        """Enhanced question answering with detailed logging"""
        pass
    @abstractmethod    
    async def debug_session(self, session_id: str) -> Dict[str, Any]:
        """Debug method to inspect session state"""
        pass
    
    @abstractmethod    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics including cache performance"""
        pass