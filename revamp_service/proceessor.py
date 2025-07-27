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
from .graphClass import *
logging = get_logger(__name__)



# Main RAG processor with enhanced session management
class SimpleRAGProcessor:
    def __init__(self):
        self.session_manager = EnhancedSessionManager()  
        self.graph_kb = SimpleGraphKB()
        self.embedding_model = None
        self.Config = CachingConfig()
        
    async def initialize(self):
        """Initialize components"""
        await self.session_manager.init_redis()
        await self.graph_kb.init_neo4j()
        self.graph_kb.init_nlp()
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.Config.EMBEDDING_MODEL
        )
        logging.info("RAG Processor initialized")
    
    def dataframe_to_text_rows(self, df: pd.DataFrame) -> List[str]:
        """Convert dataframe to text rows"""
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            rows.append(row_text)
        return rows
    
    def create_chunks(self, text: str) -> List[str]:
        """Create text chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.Config.CHUNK_SIZE,
            chunk_overlap=self.Config.CHUNK_OVERLAP
        )
        return splitter.split_text(text)
    
    def create_simple_qa_system(self, chunks: List[str], description: str):
        """Create simple QA system"""
        # Create documents
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, self.embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": min(len(chunks), 10)})
        # Create LLM chain
        llm = Ollama(
            base_url=self.Config.BASE_URL,
            model=self.Config.LLM_MODEL,
        )
        evaluation_chain = chatbot_prompt | llm 
        
        return {
            "retriever": retriever,
            "qa_chain": evaluation_chain,
            "vectorstore": vectorstore,
            "chunks": chunks
        }
    
    async def answer_question(self, question: str, session_data: Dict[str, Any], use_hybrid: bool = True) -> Dict[str, Any]:
        """Answer question using retrieval and LLM"""
        retriever = session_data["qa_components"]["retriever"]
        qa_chain = session_data["qa_components"]["qa_chain"]
        
        # Get relevant documents
        docs = await asyncio.get_event_loop().run_in_executor(
            None, retriever.get_relevant_documents, question
        )
        
        # Try hybrid search if enabled
        graph_results = []
        # if use_hybrid and session_data.get("use_graph", False):
        graph_results = await self.graph_kb.graph_search(question, session_data["session_id"])
        
        # Combine contexts
        vector_context = "\n\n".join([doc.page_content for doc in docs])
        if graph_results:
            graph_context = "\n\n".join(graph_results)
            context = f"Vector Search Results:\n{vector_context}\n\nGraph Search Results:\n{graph_context}"
        else:
            context = vector_context
        
        # Generate answer
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            qa_chain.invoke,
            {
                "description": session_data["description"],
                "context": context,
                "question": question
            }
        )
        
        return {
            "answer": response,
            "source_count": len(docs),
            "graph_results_count": len(graph_results),
            "search_type": "hybrid" if graph_results else "vector_only"
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics including cache performance"""
        cache_stats = await self.session_manager.get_cache_stats()
        return {
            "cache_stats": cache_stats,
            "graph_enabled": self.graph_kb.enabled,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL
        }