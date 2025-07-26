from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from revamp_service.prompts import *
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from revamp_service.utils import *
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from .logger import logging
# Load environment variables from .env file
load_dotenv()
from revamp_service.models import *
from revamp_service.taskManager import *



app = FastAPI(title="Feedback Analysis Service")
task_manager = TaskManager()  
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

@app.post("/start_session")
async def start_session(data: StartSession):
    print(f"Received start_session for {data.session_id}")
    # Step 1: Download the CSV
    try:
        df = await fetch_worksheet_data(data.sheet_url)
    except Exception as e:
        return {"error": f"Failed to load sheet: {str(e)}"}

    # Step 2: Format rows with headers
    rows = dataframe_to_text_rows(df)
    text = "\n".join(rows)
    print("Loaded and combined text.")

    # Step 3: Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks.")

    try:
        embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

        vectorstore = FAISS.from_texts(
    texts=chunks,
    embedding=embedding_model
)

        print("Vectorstore created.")
    except Exception as e:
        return {"error": f"Error creating vectorstore: {str(e)}"}

    retriever = vectorstore.as_retriever()
    retriever.search_kwargs["k"] = len(chunks)


    # Custom prompt
    refine_prompt = refine_prompt

# This is used to refine the answer as additional chunks are processed

    question_prompt = question_prompt


#     prompt = PromptTemplate(
#     template=system_template,
#     input_variables=["description", "context", "question"]
# )


    qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3.2:1b"),
    retriever=retriever,
    chain_type="refine",
    chain_type_kwargs={
        "question_prompt": question_prompt,
        "refine_prompt": refine_prompt
    }
)





    # print(f"Session created with ID: {data.session_id}")
    # cache[data.session_id] = {
    #     "qa_chain": qa_chain,
    #     "description": data.description
    # }

    return {"message": "Session created."}



# @app.post("/query")
# async def query(q: QueryRequest):
#     print(f"Received query for session {q.session_id}: {q.question}")
#     # session = cache.get(q.session_id)
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