from typing import Any
from .configs import *
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from .models import *
from .prompts import *
from .BaseNumericAgent import *
from langchain_core.output_parsers import JsonOutputParser
from .logger import *
logging = get_logger(__name__)

class OllamaNumeriCAnalyzer(BaseAgent):
    def __init__(self,prompt=None,output_parser=None,llm=None):
        self.config = OllamaNumericColumnAnalyzerAgentConfig()
        self.llm = Ollama(
            base_url=self.config.BASE_URL,
            model=self.config.MODEL,
        )
        self.output_parser = JsonOutputParser(pydantic_object = EvaluationResponse) 
        self.prompt = numeric_analysis_prompt_v2
    def evaluate(self, req: NumericAnalysis)-> EvaluationResponse:
        evaluation_chain = self.prompt | self.llm | self.output_parser
        print(req['column_heading'])
        result = evaluation_chain.invoke({
    "column_heading": req['column_heading'],
    "total_responses": req['total_responses'],
    "mean": req['mean'],
    "median": req['median'],
    "std_dev": req['std_dev'],
    "min_value": req['min_value'],
    "max_value": req['max_value'],
    "q1": req['quartiles']['Q1'],
    "q3": req['quartiles']['Q3'],
    "format_instructions": self.output_parser.get_format_instructions()
})
        return EvaluationResponse(
            feedback=result["feedback"]
        )
