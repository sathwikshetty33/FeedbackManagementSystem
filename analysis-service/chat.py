from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests as httpx
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from cachetools import TTLCache




class StartSession(BaseModel):
    session_id: str
    sheet_url: str
    description: str

class QueryRequest(BaseModel):
    session_id: str
    question: str
