from datetime import datetime
from typing import Literal, Optional, Sequence, TypedDict, Union
from pydantic import BaseModel, Field


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
    use_graph: bool = True
    cache_enabled: bool = True

class QueryRequest(BaseModel):
    session_id: str
    question: str
    use_hybrid_search: bool = True

class TaskInfo(TypedDict):
    """ Used to enforce type safety for task info """
    task_id: str  # Unique identifier for the task
    request: AnalysisRequest  # Original analysis request payload for the task
    status: Literal["queued", "running", "completed", "failed"]  # Task status
    created_at: datetime  # When the task was created
    started_at: Optional[datetime]  # When the task started
    completed_at: Optional[datetime]  # When the task completed

class EvaluationResponse(BaseModel):
    feedback: str = Field(description="feedback about the distribution of values")
class QuartileDict(TypedDict):
    Q1: float
    Q3: float
class NumericAnalysis(TypedDict):
    """ Used to enforce type safety for numeric analysis """
    column_heading : str
    type: Literal["numerical"]
    total_responses: int
    mean: float
    median: float
    std_dev: float
    min_value: Union[int, float]
    max_value: Union[int, float]
    quartiles: QuartileDict
    feedback: Optional[str]

class CategoricalAnalysis(TypedDict):
    column_heading : str
    type: Literal['categorical']
    total_responses: int
    unique_categories: int
    distribution: dict
    percentages: float
    most_common: str
    least_common: str
    all_values : Sequence[str]
    feedback: Optional[str]
