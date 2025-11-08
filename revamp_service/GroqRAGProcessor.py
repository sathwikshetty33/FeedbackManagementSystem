import asyncio
import logging
from typing import Dict, List, Any
from .models import *
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from .logger import *
from .prompts import *
from .ChatbotSessionManager import *
from .Simpleneo4jKB import *
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from .graph import build_tools_graph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqRAGProcessor:
    def __init__(self):
        self.session_manager = EnhancedSessionManager()  
        self.graph_kb = Simpleneo4jKB()
        self.embedding_model = None
        self.Config = GroqChatRag()
        self.tools_graph = build_tools_graph()
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
        """Create text chunks with logging"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.Config.CHUNK_SIZE,
            chunk_overlap=self.Config.CHUNK_OVERLAP
        )
        chunks = splitter.split_text(text)
        logging.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        
        # Log first few chunks for debugging
        for i, chunk in enumerate(chunks[:3]):
            logging.debug(f"Chunk {i} preview: {chunk[:100]}...")
        
        return chunks
    
    def create_simple_qa_system(self, chunks: List[str], description: str):
        """Create simple QA system"""
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        vectorstore = FAISS.from_documents(documents, self.embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": min(len(chunks), 10)})
        
        llm = ChatGroq(
            model=self.Config.LLM_MODEL,
            temperature=self.Config.TEMPERATURE,
            max_tokens=self.Config.MAX_TOKEN,
            groq_api_key=self.Config.GROQ_API_KEY
        )
        output_parser = StrOutputParser()
        evaluation_chain = chatbot_prompt | llm | output_parser
        
        logging.info(f"QA system created with {len(chunks)} chunks")
        return {
            "retriever": retriever,
            "qa_chain": evaluation_chain,
            "vectorstore": vectorstore,
            "chunks": chunks
        }
    
    async def process_and_store_data(self, chunks: List[str], session_id: str, description: str):
        """Process data and create both vector and graph stores"""
        # Create QA system
        qa_components = self.create_simple_qa_system(chunks, description)
        
        # Create graph knowledge base
        if self.graph_kb.enabled:
            logging.info(f"Creating graph for session {session_id} with {len(chunks)} chunks")
            await self.graph_kb.create_simple_graph(chunks, session_id)
            
            # Debug: Check what was actually stored
            debug_info = await self.graph_kb.debug_graph_contents(session_id)
            logging.info(f"Graph debug info: {debug_info}")
        else:
            logging.warning("Graph knowledge base is disabled")
        
        return {
            "qa_components": qa_components,
            "session_id": session_id,
            "description": description,
            "use_graph": self.graph_kb.enabled,
            "chunk_count": len(chunks)
        }
    
    async def answer_question(self, question: str, session_data: Dict[str, Any], use_hybrid: bool = True) -> Dict[str, Any]:
        """Enhanced question answering with detailed logging"""
        retriever = session_data["qa_components"]["retriever"]
        qa_chain = session_data["qa_components"]["qa_chain"]
        
        logging.info(f"Answering question: '{question}' for session: {session_data['session_id']}")
        
        # Get relevant documents from vector search
        docs = await asyncio.get_event_loop().run_in_executor(
            None, retriever.invoke, question
        )
        
        logging.info(f"Vector search returned {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):  # Log first 2 docs
            logging.debug(f"Vector doc {i}: {doc.page_content[:100]}...")
        
        # Try graph search if enabled
        graph_results = []
        if use_hybrid and self.graph_kb.enabled:
            logging.info("Attempting graph search...")
            graph_results = await self.graph_kb.graph_search(question, session_data["session_id"])
            logging.info(f"Graph search returned {len(graph_results)} results")
            
            for i, result in enumerate(graph_results[:2]):  # Log first 2 results
                logging.debug(f"Graph result {i}: {result[:100]}...")
        else:
            logging.info("Graph search skipped (disabled or not requested)")
        
        # Combine contexts
        vector_context = "\n\n".join([doc.page_content for doc in docs])
        if graph_results:
            graph_context = "\n\n".join(graph_results)
            context = f"Vector Search Results:\n{vector_context}\n\nGraph Search Results:\n{graph_context}"
            logging.info("Using hybrid context (vector + graph)")
        else:
            context = vector_context
            logging.info("Using vector-only context")
        
        # Generate answer
        logging.debug(f"Context length: {len(context)} characters")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            qa_chain.invoke,
            {
                "description": session_data["description"],
                "context": context,
                "question": question
            }
        )
        
        result = {
            "answer": response.strip(),
            "source_count": len(docs),
            "graph_results_count": len(graph_results),
            "search_type": "hybrid" if graph_results else "vector_only",
            "context_length": len(context)
        }
        
        logging.info(f"Answer generated: {len(result['answer'])} characters, "
                    f"search_type: {result['search_type']}")
        
        return result
    
    async def debug_session(self, session_id: str) -> Dict[str, Any]:
        """Debug method to inspect session state"""
        debug_info = {
            "session_id": session_id,
            "graph_enabled": self.graph_kb.enabled,
            "embedding_model": self.Config.EMBEDDING_MODEL if hasattr(self.Config, 'EMBEDDING_MODEL') else "Unknown"
        }
        
        if self.graph_kb.enabled:
            graph_debug = await self.graph_kb.debug_graph_contents(session_id)
            debug_info["graph_contents"] = graph_debug
        
        return debug_info
    async def answer_question_with_tools(self, question: str, session_data: Dict[str, Any], use_hybrid: bool = True) -> Dict[str, Any]:
        """Answer a question using LangGraph tools workflow"""
        
        logger.info("=" * 80)
        logger.info("🚀 STARTING QUERY PROCESSING")
        logger.info("=" * 80)
        logger.info(f"📝 Question: {question}")
        
        try:
            retriever = session_data["qa_components"]["retriever"]
            
            # Get relevant documents
            logger.info("🔍 Retrieving relevant documents...")
            docs = await asyncio.get_event_loop().run_in_executor(
                None, 
                retriever.invoke,
                question
            )
            logger.info(f"✅ Retrieved {len(docs)} documents")
            
            context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"📄 Total context length: {len(context)} characters")

            # Prepare LangGraph state
            state = {
                "question": question,
                "context": context,
                "dataset_description": session_data["metadata"]["columns"],
                "sheet_url": session_data["metadata"]["sheet_url"],
                "analysis": {},
                "answer": "",
                "decision": ""
            }
            
            logger.info(f"🌐 Sheet URL: {state['sheet_url'][:50]}...")
            logger.info(f"📊 Available columns: {[c.get('name', c) for c in state['dataset_description']]}")

            # Run the graph
            logger.info("🔄 Invoking LangGraph workflow...")
            try:
                if hasattr(self.tools_graph, 'ainvoke'):
                    result_state = await self.tools_graph.ainvoke(state)
                else:
                    result_state = await asyncio.get_event_loop().run_in_executor(
                        None, self.tools_graph.invoke, state
                    )
                
                logger.info("✅ LangGraph workflow completed")
                logger.info(f"🎯 Final decision was: {result_state.get('decision', 'unknown')}")
                logger.info(f"📊 Tool used: {bool(result_state.get('analysis'))}")
                
            except Exception as graph_error:
                logger.error(f"❌ LangGraph execution failed: {str(graph_error)}")
                raise

            # Return final result
            tool_used = "MCP Analysis (Complete Dataset)" if result_state.get("analysis") and not result_state["analysis"].get("error") else "Context Only (Sample Data)"
            
            result = {
                "answer": result_state.get("answer", "Unable to generate answer"),
                "used_tool": tool_used,
                "context_length": len(context),
                "sources": [doc.metadata for doc in docs] if docs else []
            }
            
            logger.info("=" * 80)
            logger.info("✅ QUERY PROCESSING COMPLETED")
            logger.info(f"🔧 Tool used: {result['used_tool']}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in answer_question_with_tools: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing question: {str(e)}",
                "used_tool": "Error",
                "context_length": 0,
                "error": str(e)
            }

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics including cache performance"""
        cache_stats = await self.session_manager.get_cache_stats()
        return {
            "cache_stats": cache_stats,
            "graph_enabled": self.graph_kb.enabled,
            "embedding_model": getattr(self.Config, 'EMBEDDING_MODEL', 'Unknown'),
            "llm_model": getattr(self.Config, 'LLM_MODEL', 'Unknown')
        }