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

class OllamaCategoricalCAnalyzer(BaseAgent):
    def __init__(self,prompt=None,output_parser=None,llm=None):
        self.config = OllamaNumericColumnAnalyzerAgentConfig()
        self.llm = Ollama(
            base_url=self.config.BASE_URL,
            model=self.config.MODEL,
        )
        self.output_parser = JsonOutputParser(pydantic_object = EvaluationResponse) 
        self.prompt = categorical_analysis_prompt
    def evaluate(self, req: CategoricalAnalysis)-> EvaluationResponse:
        evaluation_chain = self.prompt | self.llm | self.output_parser
        print(req['column_heading'])
        result = evaluation_chain.invoke({
    'type': 'categorical',
            'column_heading': req['column_heading'],
            'total_responses': req['total_responses'],
            'unique_categories': req['unique_categories'],
            'distribution': req['distribution'],
            'percentages': req['percentages'],
            'most_common': req['most_common'],
            'least_common': req['least_common'],
    "format_instructions": self.output_parser.get_format_instructions()
})
        return EvaluationResponse(
            feedback=result["feedback"]
        )
