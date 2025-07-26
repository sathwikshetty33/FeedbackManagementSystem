import asyncio
from cachetools import TTLCache
import pandas as pd
from .prompts import *
import asyncio
from asyncio import Semaphore, Queue
from datetime import datetime
from typing import Dict
import threading
from .utils import *
import asyncio
from .logger import logging
from .models import *
from .configs import *
cache = TTLCache(maxsize=100, ttl=1800)  # 30 min
class TaskManager:
    def __init__(self):
        self.config = taskManagerConfig()
        
    async def add_task(self, task_id: str, request: AnalysisRequest):
        """Add task to queue with status tracking"""
        task_info = {
            'task_id': task_id,
            'request': request,
            'status': 'queued',
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None
        }
        
        with self.processing_lock:
            self.active_tasks[task_id] = task_info
        
        await self.task_queue.put((task_id, request))
        
        # Start processing if not already running
        asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process tasks from queue with concurrency control"""
        while not self.task_queue.empty():
            async with self.semaphore:  # Limit concurrent tasks
                try:
                    task_id, request = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                    
                    # Update task status
                    with self.processing_lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['status'] = 'processing'
                            self.active_tasks[task_id]['started_at'] = datetime.now()
                    
                    # Process the task
                    await self._execute_task(task_id, request)
                    
                except asyncio.TimeoutError:
                    break  # No more tasks in queue
                except Exception as e:
                    logging.error(f"Error processing task queue: {e}")
    
    async def _execute_task(self, task_id: str, request: AnalysisRequest):
        """Execute individual analysis task"""
        try:
            await process_analysis_task(request, task_id)
            
            # Update task status
            with self.processing_lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['status'] = 'completed'
                    self.active_tasks[task_id]['completed_at'] = datetime.now()
                    
        except Exception as e:
            logging.error(f"Task {task_id} failed: {e}")
            with self.processing_lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['status'] = 'failed'
                    self.active_tasks[task_id]['error'] = str(e)
                    self.active_tasks[task_id]['completed_at'] = datetime.now()
    
    def get_task_status(self, task_id: str) -> dict:
        """Get status of a specific task"""
        with self.processing_lock:
            return self.active_tasks.get(task_id, {'status': 'not_found'})
    
    def get_queue_info(self) -> dict:
        """Get overall queue information"""
        with self.processing_lock:
            active_count = sum(1 for task in self.active_tasks.values() 
                             if task['status'] == 'processing')
            queued_count = sum(1 for task in self.active_tasks.values() 
                             if task['status'] == 'queued')
            
            return {
                'active_tasks': active_count,
                'queued_tasks': queued_count,
                'total_tasks': len(self.active_tasks),
                'max_concurrent': self.max_concurrent_tasks
            }