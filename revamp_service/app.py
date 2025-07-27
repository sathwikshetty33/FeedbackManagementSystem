from langchain_community.embeddings import HuggingFaceEmbeddings
from revamp_service.prompts import *
from fastapi import FastAPI, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from revamp_service.utils import *
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from .logger import *
from cachetools import TTLCache
# Load environment variables from .env file
load_dotenv()
from revamp_service.models import *
from revamp_service.taskManager import *
from .proceessor import *
logging = get_logger(__name__)
cache = TTLCache(maxsize=100, ttl=1800)  # 30 min



app = FastAPI(title="Feedback Analysis Service")
task_manager = TaskManager()  
processor = SimpleRAGProcessor()
# FastAPI Endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest):
    """Start feedback analysis task with queue management"""
    import uuid
    task_id = str(uuid.uuid4())
    
    queue_info = await task_manager.get_queue_info()
    try:
        await task_manager.add_task(task_id, request)
    except Exception as e:
        logging.error(f"Error adding task to queue: {e}")
    estimated_wait = queue_info['queued_tasks'] * 5  
    return AnalysisResponse(
        status="accepted",
        message=f"Analysis queued. Current position: {queue_info['queued_tasks'] + 1}. Estimated wait: {estimated_wait} minutes. You will receive an email when complete.",
        task_id=task_id
    )
@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific analysis task"""
    status = task_manager.get_task_status(task_id)
    return status

# Add endpoint to check overall queue status
@app.get("/queue/status")
async def get_queue_status():
    """Get overall queue status"""
    return task_manager.get_queue_info()
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "feedback-analysis"}

# @app.post("/start_session")
# async def start_session(data: StartSession):
#     print(f"Received start_session for {data.session_id}")
#     # Step 1: Download the CSV
#     try:
#         df = await fetch_worksheet_data(data.sheet_url)
#     except Exception as e:
#         return {"error": f"Failed to load sheet: {str(e)}"}

#     # Step 2: Format rows with headers
#     rows = dataframe_to_text_rows(df)
#     text = "\n".join(rows)
#     print("Loaded and combined text.")

#     # Step 3: Chunk and embed
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_text(text)
#     print(f"Split into {len(chunks)} chunks.")

#     try:
#         embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

#         vectorstore = FAISS.from_texts(
#     texts=chunks,
#     embedding=embedding_model
# )

#         print("Vectorstore created.")
#     except Exception as e:
#         return {"error": f"Error creating vectorstore: {str(e)}"}

#     retriever = vectorstore.as_retriever()
#     retriever.search_kwargs["k"] = len(chunks)


#     # Custom prompt
#     refine_prompt = refine_prompt

# # This is used to refine the answer as additional chunks are processed

# #     question_prompt = question_prompt
# #     prompt = PromptTemplate(
# #     template=system_template,
# #     input_variables=["description", "context", "question"]
# # )


#     qa_chain = RetrievalQA.from_chain_type(
#     llm=Ollama(model="llama3.2:1b"),
#     retriever=retriever,
#     chain_type="refine",
#     chain_type_kwargs={
#         "question_prompt": question_prompt,
#         "refine_prompt": refine_prompt
#     }
# )
#     print(f"Session created with ID: {data.session_id}")
#     cache[data.session_id] = {
#         "qa_chain": qa_chain,
#         "description": data.description
#     }

#     return {"message": "Session created."}



# @app.post("/query")
# async def query(q: QueryRequest):
#     print(f"Received query for session {q.session_id}: {q.question}")
#     session = cache.get(q.session_id)
#     if not session:
#         return {"error": "Session expired or not found"}

#     qa_chain = session["qa_chain"]
#     description = session["description"]

#     response = qa_chain(
#         {
#             "description": description,
#             "query": q.question
#         }
#     )

#     return {"answer": response}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your Django domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_analysis:app", 
        host="0.0.0.0", 
        port=8001, 
        workers=1,  # Use 1 worker for the main app, parallel processing handled internally
        reload=True
    )


@app.post("/start_session")
async def start_session(data: StartSession):
    """Simple but robust session initialization"""
    logging.info(f"Starting session {data.session_id}")
    
    try:
        # Check existing session
        existing_session = await processor.session_manager.get_session(data.session_id)
        if existing_session:
            return {"message": "Session loaded from cache", "cached": True}
        
        # Load data
        df = await fetch_worksheet_data(data.sheet_url)
        rows = processor.dataframe_to_text_rows(df)
        text = "\n".join(rows)
        
        # Create chunks
        chunks = processor.create_chunks(text)
        logging.info(f"Created {len(chunks)} chunks")
        
        # Create QA system
        qa_components = processor.create_simple_qa_system(chunks, data.description)
        
        # Create knowledge graph if enabled
        if data.use_graph:
            await processor.graph_kb.create_simple_graph(chunks, data.session_id)
        
        # Store session
        session_data = {
            "session_id": data.session_id,
            "description": data.description,
            "qa_components": qa_components,
            "use_graph": data.use_graph,
            "created_at": datetime.now(),
            "metadata": {
                "columns": list(df.columns),
                "shape": df.shape,
                "chunk_count": len(chunks)
            }
        }
        
        await processor.session_manager.set_session(data.session_id, session_data)
        
        logging.info(f"Session {data.session_id} created successfully")
        return {
            "message": "Session created successfully",
            "chunks_created": len(chunks),
            "graph_enabled": data.use_graph,
            "metadata": session_data["metadata"]
        }
        
    except Exception as e:
        logging.error(f"Session creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.post("/query")
async def query_session(data: QueryRequest):
    """Simple query handling"""
    logging.info(f"Processing query for session {data.session_id}")
    
    try:
        session_data = await processor.session_manager.get_session(data.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Answer question
        result = await processor.answer_question(
            data.question, 
            session_data, 
            use_hybrid=data.use_hybrid_search
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    await processor.initialize()