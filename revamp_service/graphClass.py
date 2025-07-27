import logging
from typing import List
from .models import *
from langchain_community.llms import Ollama
from .logger import *
from .prompts import *
from .ChatbotSessionManager import *
from .configs import *
logging = get_logger(__name__)

try:
    import networkx as nx
    from neo4j import AsyncGraphDatabase
    import spacy
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning("Graph dependencies not available. Install neo4j, networkx, spacy for full functionality.")

# Enhanced caching (optional)
try:
    from cachetools import TTLCache
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Using local cache only.")
class SimpleGraphKB:
    def __init__(self):
        self.driver = None
        self.nlp = None
        self.enabled = GRAPH_AVAILABLE
        self.Config = CachingConfig()
        
    async def init_neo4j(self):
        if not self.enabled:
            return
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.Config.NEO4J_URI,
                auth=(self.Config.NEO4J_USER, self.Config.NEO4J_PASSWORD)
            )
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            logging.info("Neo4j connected successfully")
        except Exception as e:
            logging.error(f"Neo4j connection failed: {e}")
            self.enabled = False
            
    def init_nlp(self):
        if not self.enabled:
            return
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("SpaCy model not found")
            self.enabled = False
    
    async def extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        if not self.enabled or not self.nlp:
            return []
        
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "MONEY", "PERCENT"]]
    
    async def create_simple_graph(self, chunks: List[str], session_id: str):
        """Create simple knowledge graph"""
        if not self.enabled or not self.driver:
            return
        
        try:
            async with self.driver.session() as session:
                # Create session node
                await session.run(
                    "MERGE (s:Session {id: $session_id, created_at: datetime()})",
                    session_id=session_id
                )
                
                for i, chunk in enumerate(chunks):
                    entities = await self.extract_entities(chunk)
                    chunk_id = f"{session_id}_{i}"
                    
                    # Create chunk node
                    await session.run(
                        """
                        MERGE (c:Chunk {id: $chunk_id, session_id: $session_id})
                        SET c.text = $text
                        WITH c
                        MATCH (s:Session {id: $session_id})
                        MERGE (s)-[:CONTAINS]->(c)
                        """,
                        chunk_id=chunk_id,
                        text=chunk[:500],
                        session_id=session_id
                    )
                    
                    # Create entity relationships
                    for entity in entities:
                        await session.run(
                            """
                            MERGE (e:Entity {name: $entity})
                            WITH e
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (c)-[:MENTIONS]->(e)
                            """,
                            entity=entity,
                            chunk_id=chunk_id
                        )
        except Exception as e:
            logging.error(f"Graph creation failed: {e}")
    
    async def graph_search(self, query: str, session_id: str) -> List[str]:
        """Simple graph-based search"""
        if not self.enabled or not self.driver:
            return []
        
        try:
            query_entities = await self.extract_entities(query)
            if not query_entities:
                return []
            
            async with self.driver.session() as session_db:
                result = await session_db.run(
                    """
                    MATCH (s:Session {id: $session_id})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $entities
                    RETURN DISTINCT c.text as text
                    LIMIT 5
                    """,
                    session_id=session_id,
                    entities=query_entities
                )
                
                return [record["text"] async for record in result]
        except Exception as e:
            logging.error(f"Graph search failed: {e}")
            return []
