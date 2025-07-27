import asyncio
from revamp_service.prompts import *
import asyncio
from datetime import datetime
from revamp_service.utils import *
import asyncio
from revamp_service.logger import *
from revamp_service.models import *
from revamp_service.configs import *
logging = get_logger(__name__)

class TaskManager:
    def __init__(self):
        self.config = taskManagerConfig()

    async def add_task(self, task_id: str, request: AnalysisRequest):
        """Add task to queue with status tracking"""
        task_info: TaskInfo = {
            'task_id': task_id,
            'request': request,
            'status': 'queued',
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
        }

        async with self.config.processing_lock:
            self.config.active_tasks[task_id] = task_info

        logging.debug(f"Task: {task_id} added to queue")

        await self.config.task_queue.put((task_id, request))

        async with self.config.processing_lock:
            if not self.config.is_processing:
                asyncio.create_task(self._process_queue())
                self.config.is_processing = True

    
    async def _process_queue(self):
        """Process tasks from queue with concurrency control"""
        while True:
            async with self.config.semaphore:  # Limit concurrent tasks
                try:
                    task_id, request = await asyncio.wait_for(
                        self.config.task_queue.get(), timeout=2.0
                    )
                    
                    # Update task status
                    async with self.config.processing_lock:
                        if task_id in self.config.active_tasks:
                            self.config.active_tasks[task_id]['status'] = 'processing'
                            self.config.active_tasks[task_id]['started_at'] = datetime.now()
                            logging.debug("Processing called for Task: %s at time : %s", task_id, datetime.now())
                    
                    # Process the task
                    await self._execute_task(task_id, request)
                    
                except asyncio.TimeoutError:
                    async with self.config.processing_lock:
                        self.config.is_processing = False
                    break  # No more tasks in queue
                except Exception as e:
                    logging.error(f"Error processing task queue: {e}")
    
    async def _execute_task(self, task_id: str, request: AnalysisRequest):
        """Execute individual analysis task"""
        try:
            await process_analysis_task(request, task_id)
            
            # Update task status
            async with self.config.processing_lock:
                if task_id in self.config.active_tasks:
                    self.config.active_tasks[task_id]['status'] = 'completed'
                    self.config.active_tasks[task_id]['completed_at'] = datetime.now()
                    
        except Exception as e:
            logging.error(f"Task {task_id} failed: {e}")
            async with self.config.processing_lock:
                if task_id in self.config.active_tasks:
                    self.config.active_tasks[task_id]['status'] = 'failed'
                    self.config.active_tasks[task_id]['error'] = str(e)
                    self.config.active_tasks[task_id]['completed_at'] = datetime.now()
    
    async def get_task_status(self, task_id: str) -> dict:
        """Get status of a specific task"""
        async with self.config.processing_lock:
            return self.config.active_tasks.get(task_id, {'status': 'not_found'})
    
    async def get_queue_info(self) -> dict:
        """Get overall queue information"""
        async with self.config.processing_lock:
            active_count = sum(1 for task in self.config.active_tasks.values() 
                             if task['status'] == 'processing')
            queued_count = sum(1 for task in self.config.active_tasks.values() 
                             if task['status'] == 'queued')
            
            return {
                'active_tasks': active_count,
                'queued_tasks': queued_count,
                'total_tasks': len(self.config.active_tasks),
                'max_concurrent': self.config.MAX_CONCURRENT_TASKS
            }