from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple, List
from langchain_core.documents import Document
from .configs import *
class Analyzer(ABC):
    def __init__(self):
        self.config : Config
        self.embeddings : Any
        self.llm :Any
    @abstractmethod
    def preprocess_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        pass

    @abstractmethod
    def _categorize_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        pass

    @abstractmethod
    def analyze_numerical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def analyze_categorical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def analyze_text_column(self, df: pd.DataFrame, column: str) -> Tuple[Dict[str, Any], str]:
        pass

    @abstractmethod
    def generate_column_insights(self, column_name: str, analysis_data: Dict[str, Any], all_text: str = None) -> str:
        pass

    @abstractmethod
    def _fallback_analysis(self, column_name: str, analysis_data: Dict[str, Any], documents: List[Document]) -> str:
        pass

    @abstractmethod
    def _generate_basic_insights(self, column_name: str, analysis_data: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def analyze_all_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
        pass
