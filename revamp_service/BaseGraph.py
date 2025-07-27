from abc import ABC, abstractmethod
from typing import List
from .models import *
from langchain_community.llms import Ollama
from .logger import *
from .prompts import *
from .ChatbotSessionManager import *
from .configs import *

class BaseGraph(ABC):
    def __init__(self):
        self.driver = None
        self.nlp = None
        self.enabled = False
        self.Config = None
    @abstractmethod
    async def extract_entities(self, text: str) -> List[str]:
       pass
    @abstractmethod   
    async def create_simple_graph(self, chunks: List[str], session_id: str):
        """Enhanced graph creation with better error handling and logging"""
        pass
    @abstractmethod    
    async def verify_graph_creation(self, session_id: str):
        """Verify that the graph was created successfully"""
        pass
    @abstractmethod
    async def graph_search(self, query: str, session_id: str) -> List[str]:
        """Enhanced graph-based search with fallback strategies"""
        pass
    @abstractmethod
    async def fuzzy_entity_search(self, query_entities: List[str], session_id: str) -> List[str]:
        """Fuzzy search for similar entities"""
        pass
    @abstractmethod    
    async def keyword_search(self, query: str, session_id: str) -> List[str]:
        """Fallback keyword search in chunk text"""
        pass
    
    @abstractmethod
    async def debug_graph_contents(self, session_id: str) -> Dict[str, Any]:
        """Debug method to inspect graph contents"""
        pass