from dataclasses import dataclass
import os
from dotenv import load_dotenv
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    event_name: str
    worksheet_url: str
    recipient_email: str = "sathwikshetty9876@gmail.com"
    
    class Config:
        extra = "allow"
        schema_extra = {
            "example": {
                "event_name": "Tech Conference 2024",
                "worksheet_url": "https://docs.google.com/spreadsheets/d/abc123/edit",
                "recipient_email": "user@example.com"
            }
        }

class AnalysisResponse(BaseModel):
    status: str
    message: str
    task_id: str
    


class StartSession(BaseModel):
    session_id: str
    sheet_url: str
    description: str

class QueryRequest(BaseModel):
    session_id: str
    question: str