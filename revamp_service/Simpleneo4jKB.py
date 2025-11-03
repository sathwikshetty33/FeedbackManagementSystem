import logging
from typing import Any, List
from .models import *
from langchain_community.llms import Ollama
from .logger import *
from .prompts import *
from .ChatbotSessionManager import *
from .configs import *
from .BaseGraph import *
logging = get_logger(__name__)

try:
    import networkx as nx
    from neo4j import AsyncGraphDatabase
    import spacy
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning("Graph dependencies not available. Install neo4j, networkx, spacy for full functionality.")

class Simpleneo4jKB(BaseGraph):
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
            logging.info("SpaCy model loaded successfully")
        except OSError:
            logging.warning("SpaCy model not found")
            self.enabled = False
    
    async def extract_entities(self, text: str) -> List[str]:
        """Enhanced entity extraction with debugging"""
        if not self.enabled or not self.nlp:
            logging.debug("Entity extraction disabled - NLP not available")
            return []
        
        doc = self.nlp(text)
        entities = [ent.text.lower().strip() for ent in doc.ents 
                   if ent.label_ in ["PERSON", "ORG", "GPE", "MONEY", "PERCENT", "PRODUCT", "EVENT"]]
        
        # Also extract important keywords (nouns, proper nouns)
        keywords = [token.text.lower().strip() for token in doc 
                   if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2 
                   and not token.is_stop and token.is_alpha]
        
        # Combine and deduplicate
        all_entities = list(set(entities + keywords))
        return all_entities
    
    async def create_simple_graph(self, chunks: List[str], session_id: str):
        """Enhanced graph creation with better error handling and logging"""
        if not self.enabled or not self.driver:
            logging.warning("Graph creation skipped - Neo4j not available")
            return
        
        try:
            async with self.driver.session() as session:
                # Create session node
                await session.run(
                    "MERGE (s:Session {id: $session_id}) SET s.created_at = datetime()",
                    session_id=session_id
                )
                logging.info(f"Created session node for: {session_id}")
                
                total_entities = 0
                for i, chunk in enumerate(chunks):
                    entities = await self.extract_entities(chunk)
                    chunk_id = f"{session_id}_{i}"
                    
                    # Create chunk node with more metadata
                    await session.run(
                        """
                        MERGE (c:Chunk {id: $chunk_id, session_id: $session_id})
                        SET c.text = $text, c.full_text = $full_text, c.chunk_index = $index
                        WITH c
                        MATCH (s:Session {id: $session_id})
                        MERGE (s)-[:CONTAINS]->(c)
                        """,
                        chunk_id=chunk_id,
                        text=chunk[:500],  # Truncated for display
                        full_text=chunk,   # Full text for search
                        index=i,
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
                    
                    total_entities += len(entities)
                
                logging.info(f"Graph created: {len(chunks)} chunks, {total_entities} total entities")
                
                # Verify graph creation
                await self.verify_graph_creation(session_id)
                
        except Exception as e:
            logging.error(f"Graph creation failed: {e}", exc_info=True)
    
    async def verify_graph_creation(self, session_id: str):
        """Verify that the graph was created successfully"""
        try:
            async with self.driver.session() as session:
                # Count chunks
                result = await session.run(
                    "MATCH (s:Session {id: $session_id})-[:CONTAINS]->(c:Chunk) RETURN count(c) as chunk_count",
                    session_id=session_id
                )
                chunk_count = (await result.single())["chunk_count"]
                
                # Count entities
                result = await session.run(
                    "MATCH (s:Session {id: $session_id})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity) RETURN count(DISTINCT e) as entity_count",
                    session_id=session_id
                )
                entity_count = (await result.single())["entity_count"]
                
                logging.info(f"Graph verification - Session: {session_id}, Chunks: {chunk_count}, Entities: {entity_count}")
                
        except Exception as e:
            logging.error(f"Graph verification failed: {e}")
    
    async def graph_search(self, query: str, session_id: str) -> List[str]:
        """Enhanced graph-based search with fallback strategies"""
        if not self.enabled or not self.driver:
            logging.debug("Graph search skipped - Neo4j not available")
            return []
        
        try:
            query_entities = await self.extract_entities(query)
            logging.debug(f"Query: '{query}' -> Entities: {query_entities}")
            
            if not query_entities:
                logging.debug("No entities found in query, trying keyword search")
                return await self.keyword_search(query, session_id)
            
            async with self.driver.session() as session_db:
                # Primary search: exact entity match
                result = await session_db.run(
                    """
                    MATCH (s:Session {id: $session_id})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $entities
                    RETURN DISTINCT c.full_text as text, count(e) as entity_matches
                    ORDER BY entity_matches DESC
                    LIMIT 5
                    """,
                    session_id=session_id,
                    entities=query_entities
                )
                
                results = [record["text"] async for record in result]
                
                if results:
                    logging.info(f"Graph search found {len(results)} results via entity matching")
                    return results
                
                # Fallback: fuzzy entity search
                logging.debug("No exact matches, trying fuzzy search")
                return await self.fuzzy_entity_search(query_entities, session_id)
                
        except Exception as e:
            logging.error(f"Graph search failed: {e}", exc_info=True)
            return []
    
    async def fuzzy_entity_search(self, query_entities: List[str], session_id: str) -> List[str]:
        """Fuzzy search for similar entities"""
        try:
            async with self.driver.session() as session_db:
                # Search for entities that contain any of the query terms
                fuzzy_conditions = []
                for entity in query_entities:
                    fuzzy_conditions.append(f"e.name CONTAINS '{entity}'")
                
                if not fuzzy_conditions:
                    return []
                
                fuzzy_query = f"""
                MATCH (s:Session {{id: $session_id}})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE {' OR '.join(fuzzy_conditions)}
                RETURN DISTINCT c.full_text as text
                LIMIT 3
                """
                
                result = await session_db.run(fuzzy_query, session_id=session_id)
                results = [record["text"] async for record in result]
                
                if results:
                    logging.info(f"Fuzzy search found {len(results)} results")
                
                return results
                
        except Exception as e:
            logging.error(f"Fuzzy search failed: {e}")
            return []
    
    async def keyword_search(self, query: str, session_id: str) -> List[str]:
        """Fallback keyword search in chunk text"""
        try:
            # Extract meaningful words from query
            keywords = [word.lower().strip() for word in query.split() 
                       if len(word) > 2 and word.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
            
            if not keywords:
                return []
            
            async with self.driver.session() as session_db:
                # Search in chunk text
                keyword_conditions = []
                for keyword in keywords:
                    keyword_conditions.append(f"toLower(c.full_text) CONTAINS '{keyword}'")
                
                keyword_query = f"""
                MATCH (s:Session {{id: $session_id}})-[:CONTAINS]->(c:Chunk)
                WHERE {' OR '.join(keyword_conditions)}
                RETURN c.full_text as text
                LIMIT 3
                """
                
                result = await session_db.run(keyword_query, session_id=session_id)
                results = [record["text"] async for record in result]
                
                if results:
                    logging.info(f"Keyword search found {len(results)} results")
                
                return results
                
        except Exception as e:
            logging.error(f"Keyword search failed: {e}")
            return []
    
    async def debug_graph_contents(self, session_id: str) -> Dict[str, Any]:
        """Debug method to inspect graph contents"""
        if not self.enabled or not self.driver:
            return {"error": "Graph not available"}
        
        try:
            async with self.driver.session() as session:
                # Get all entities for this session
                result = await session.run(
                    """
                    MATCH (s:Session {id: $session_id})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                    RETURN e.name as entity, count(c) as chunk_count
                    ORDER BY chunk_count DESC
                    LIMIT 20
                    """,
                    session_id=session_id
                )
                
                entities = [(record["entity"], record["chunk_count"]) async for record in result]
                
                # Get sample chunks
                result = await session.run(
                    """
                    MATCH (s:Session {id: $session_id})-[:CONTAINS]->(c:Chunk)
                    RETURN c.text as text
                    LIMIT 3
                    """,
                    session_id=session_id
                )
                
                sample_chunks = [record["text"] async for record in result]
                
                return {
                    "session_id": session_id,
                    "entities": entities,
                    "sample_chunks": sample_chunks,
                    "total_entities": len(entities)
                }
                
        except Exception as e:
            logging.error(f"Debug query failed: {e}")
            return {"error": str(e)}