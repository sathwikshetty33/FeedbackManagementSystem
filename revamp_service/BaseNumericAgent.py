from abc import ABC, abstractmethod
from typing import Any
from .configs import *
from .models import *
class BaseAgent(ABC):
    def __init__(self,prompt=None,output_parser=None,llm=None):
        self.config : Any
        self.llm : Any
        self.output_parser : Any
        self.prompt : Any
    @abstractmethod
    def evaluate(self, req: NumericAnalysis)-> EvaluationResponse:
        pass
