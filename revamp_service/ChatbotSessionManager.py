import asyncio
import logging
from typing import Dict, Optional, Any
import json
from datetime import datetime, date
from .models import *
import pandas as pd
from .logger import *
from .prompts import *
from .configs import *
logging = get_logger(__name__)
from cachetools import TTLCache
import redis.asyncio as redis

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return {
                '__datetime__': True,
                'value': obj.isoformat()
            }
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return {
                '__custom_object__': True,
                'type': obj.__class__.__name__,
                'data': obj.__dict__ if hasattr(obj, '__dict__') else str(obj)
            }
        return super().default(obj)

def datetime_decoder(dct):
    """Custom JSON decoder that reconstructs datetime objects"""
    if '__datetime__' in dct:
        return datetime.fromisoformat(dct['value'])
    elif '__custom_object__' in dct:
        # For custom objects, return a simplified representation
        return f"<{dct['type']} object>"
    return dct

class EnhancedSessionManager:
    def __init__(self):
        self.Config = CachingConfig()
        # Local cache as fallback
        if self.Config.REDIS_AVAILABLE:
            self.local_cache = TTLCache(maxsize=self.Config.MAX_CACHE_SIZE, ttl=self.Config.CACHE_TTL)
        else:
            self.local_cache = {}
        
        self.redis_client = None
        self.redis_available = False
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.Config.REDIS_AVAILABLE:
            logging.warning("Redis library not available, using local cache only")
            return
            
        try:
            self.redis_client = redis.from_url(self.Config.REDIS_URL)
            await self.redis_client.ping()
            self.redis_available = True
            logging.info("Redis connected successfully - using as primary cache")
        except Exception as e:
            logging.warning(f"Redis connection failed: {e}. Falling back to local cache only")
            self.redis_available = False
    
    def _make_serializable(self, obj):
        """Recursively make objects JSON serializable"""
        if isinstance(obj, (datetime, date)):
            return {
                '__datetime__': True,
                'value': obj.isoformat()
            }
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            # Handle complex objects
            try:
                # Try to convert to dict if it has __dict__
                return {
                    '__custom_object__': True,
                    'type': obj.__class__.__name__,
                    'data': self._make_serializable(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)
                }
            except Exception:
                return str(obj)
        else:
            return obj
    
    async def _serialize_data(self, data: Dict[str, Any]) -> str:
        """Serialize data for Redis storage with proper datetime handling"""
        try:
            # Handle complex objects that can't be JSON serialized
            serializable_data = {}
            for key, value in data.items():
                if key == "qa_components":
                    # Don't store complex LangChain objects in cache
                    serializable_data[key] = "COMPLEX_OBJECT_PLACEHOLDER"
                else:
                    # Make the value serializable
                    serializable_data[key] = self._make_serializable(value)
            
            return json.dumps(serializable_data, cls=DateTimeEncoder, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Data serialization failed: {e}")
            # Log more details about what failed
            logging.error(f"Data keys: {list(data.keys())}")
            for key, value in data.items():
                try:
                    json.dumps({key: value}, cls=DateTimeEncoder)
                except Exception as inner_e:
                    logging.error(f"Failed to serialize key '{key}' with value type {type(value)}: {inner_e}")
            return None
    
    async def _deserialize_data(self, data_str: str) -> Optional[Dict[str, Any]]:
        """Deserialize data from Redis storage with datetime reconstruction"""
        try:
            return json.loads(data_str, object_hook=datetime_decoder)
        except Exception as e:
            logging.error(f"Data deserialization failed: {e}")
            return None
    
    async def set_session(self, session_id: str, data: Dict[str, Any]):
        """Set session data with Redis as primary, local as fallback"""
        # Always store in local cache (original data with complex objects)
        self.local_cache[session_id] = data
        
        # Try to store in Redis if available
        if self.redis_available and self.redis_client:
            try:
                serialized_data = await self._serialize_data(data)
                if serialized_data:
                    await self.redis_client.setex(
                        f"session:{session_id}",
                        self.Config.CACHE_TTL,
                        serialized_data
                    )
                    logging.info(f"Session {session_id} stored in Redis (primary) and local cache (fallback)")
                else:
                    logging.warning(f"Failed to serialize session {session_id}, stored in local cache only")
            except Exception as e:
                logging.error(f"Redis set failed for session {session_id}: {e}. Data stored in local cache only")
        else:
            logging.info(f"Session {session_id} stored in local cache only (Redis unavailable)")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Redis first, then local cache"""
        
        # Try Redis first if available
        if self.redis_available and self.redis_client:
            try:
                redis_data = await self.redis_client.get(f"session:{session_id}")
                if redis_data:
                    deserialized_data = await self._deserialize_data(redis_data.decode('utf-8'))
                    if deserialized_data:
                        logging.info(f"Session {session_id} retrieved from Redis (primary cache)")
                        # Also update local cache for faster subsequent access
                        if session_id in self.local_cache:
                            # Merge with local cache to get complex objects back
                            local_data = self.local_cache[session_id]
                            for key, value in local_data.items():
                                if key not in deserialized_data or deserialized_data[key] == "COMPLEX_OBJECT_PLACEHOLDER":
                                    deserialized_data[key] = value
                        return deserialized_data
            except Exception as e:
                logging.error(f"Redis get failed for session {session_id}: {e}. Trying local cache")
        
        # Fallback to local cache
        local_data = self.local_cache.get(session_id)
        if local_data:
            logging.info(f"Session {session_id} retrieved from local cache (fallback)")
            return local_data
        
        logging.warning(f"Session {session_id} not found in either Redis or local cache")
        return None
    
    async def delete_session(self, session_id: str):
        """Delete session from both Redis and local cache"""
        # Delete from Redis
        if self.redis_available and self.redis_client:
            try:
                await self.redis_client.delete(f"session:{session_id}")
                logging.info(f"Session {session_id} deleted from Redis")
            except Exception as e:
                logging.error(f"Redis delete failed for session {session_id}: {e}")
        
        # Delete from local cache
        if session_id in self.local_cache:
            del self.local_cache[session_id]
            logging.info(f"Session {session_id} deleted from local cache")
    
    async def clear_expired_sessions(self):
        """Clear expired sessions (Redis handles TTL automatically, this is for local cache)"""
        if self.Config.REDIS_AVAILABLE:
            # TTLCache handles expiration automatically
            expired_count = len(self.local_cache) - self.Config.MAX_CACHE_SIZE
            if expired_count > 0:
                logging.info(f"Local cache auto-cleaned {expired_count} entries")
        else:
            # Manual cleanup for simple dict
            current_time = asyncio.get_event_loop().time()
            expired_keys = []
            for key, (data, timestamp) in self.local_cache.items():
                if current_time - timestamp > self.Config.CACHE_TTL:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.local_cache[key]
            
            if expired_keys:
                logging.info(f"Cleaned {len(expired_keys)} expired sessions from local cache")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "local_cache_size": len(self.local_cache),
            "redis_available": self.redis_available,
            "redis_connected": self.redis_client is not None
        }
        
        if self.redis_available and self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get("used_memory_human", "N/A"),
                    "redis_connected_clients": info.get("connected_clients", "N/A"),
                    "redis_keyspace_hits": info.get("keyspace_hits", "N/A"),
                    "redis_keyspace_misses": info.get("keyspace_misses", "N/A")
                })
            except Exception as e:
                logging.error(f"Failed to get Redis stats: {e}")
                stats["redis_stats_error"] = str(e)
        
        return stats