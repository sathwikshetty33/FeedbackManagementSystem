import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

app = FastAPI(title="Feedback Analysis Service")
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from email.mime.base import MIMEBase
from email import encoders
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import requests
from io import StringIO
import re
from collections import Counter
import statistics
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import logging

app = FastAPI(title="Feedback Analysis Service")

# Add validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"❌ Validation Error: {exc}")
    print(f"📋 Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "body": str(await request.body())
        }
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.ERROR)
# Request/Response Models
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

# Configuration
class Config:
    load_dotenv()
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
    RAG_CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', '300'))
    RAG_CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', '30'))
    MAX_PROCESSING_ROWS = int(os.environ.get('MAX_PROCESSING_ROWS', '100'))
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '4'))
    
    # Email Configuration
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    EMAIL_USER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
    FROM_EMAIL = os.environ.get('FROM_EMAIL', 'tester7760775061@gmail.com')

config = Config()

# Copy your existing functions here:
def print_terminal_separator(title):
    print("=" * 80)
    print(f"    {title}")
    print("=" * 80)


class FeedbackRAGAnalyzer:
    def __init__(self, ollama_base_url, model_name, chunk_size=300, chunk_overlap=30):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=model_name,
            show_progress=False
        )
        
        self.llm = Ollama(
            base_url=ollama_base_url,
            model=model_name,
            temperature=0.1,
            num_ctx=2048,
            num_thread=min(4, os.cpu_count()),
            verbose=False
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def preprocess_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        print("🔄 Preprocessing columns...")
        
        # Enhanced irrelevant patterns - more comprehensive filtering
        irrelevant_patterns = [
            # Identity/ID patterns
            r'^usn$', r'^roll.*no$', r'^student.*id$',r'^student.*usn$', r'^id$', r'^entry.*id$', r'^serial$', r'^index$',
            
            # Contact information patterns  
            r'^email.*address$', r'^phone$', r'^contact$', r'^mobile$', r'^address$',
            
            # Personal identification patterns
            r'^name$', r'^participant.*name$', r'^student.*name$', r'^user.*name$', 
            r'^full.*name$', r'^first.*name$', r'^last.*name$',
            
            # Event/Organization identification patterns (NEW)
            r'^event.*name$', r'^hackathon.*name$', r'^competition.*name$', r'^course.*name$',
            r'^workshop.*name$', r'^seminar.*name$', r'^conference.*name$',
            r'^organization$', r'^company$', r'^institution$', r'^university$', r'^college$',
            r'^department$', r'^branch$', r'^stream$', r'^batch$', r'^section$',
            
            # Date/Time patterns
            r'^timestamp$', r'^date$', r'^time$', r'^created$', r'^updated$', r'^submitted$',
            
            # Administrative patterns
            r'^status$', r'^approved$', r'^verified$', r'^processed$',
            
            # Location patterns (NEW)
            r'^location$', r'^venue$', r'^city$', r'^state$', r'^country$',
            
            # Registration patterns (NEW)
            r'^registration.*id$', r'^participant.*id$', r'^team.*name$', r'^team.*id$'
        ]
        
        original_columns = df.columns.tolist()
        relevant_columns = []
        removed_columns = []
        
        for col in df.columns:
            col_lower = col.lower().strip().replace(' ', '.*')  # Handle spaces in column names
            is_irrelevant = any(re.match(pattern, col_lower) for pattern in irrelevant_patterns)
            
            # Additional contextual checks for better filtering
            if not is_irrelevant:
                # Check if column contains only identification data (low variance)
                try:
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    # If almost every value is unique, it's likely an identifier
                    if unique_ratio > 0.9 and df[col].nunique() > 10:
                        # Check if it looks like names or IDs
                        sample_values = df[col].dropna().astype(str).head(10).tolist()
                        if any(any(indicator in str(val).lower() for indicator in 
                                ['hackathon', 'event', 'workshop', 'participant', 'student', 'user']) 
                            for val in sample_values):
                            is_irrelevant = True
                            print(f"🔍 Auto-detected irrelevant column based on content: {col}")
                except:
                    pass
            
            if not is_irrelevant:
                relevant_columns.append(col)
            else:
                removed_columns.append(col)
        
        print(f"📊 Original columns: {len(original_columns)}")
        print(f"✅ Relevant columns: {len(relevant_columns)}")
        print(f"❌ Removed columns: {removed_columns}")
        
        processed_df = df[relevant_columns].copy()
        column_types = self._categorize_columns(processed_df)
        
        return processed_df, column_types

    def _categorize_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        column_types = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if col_data.empty:
                column_types[col] = 'empty'
                continue
            
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if not numeric_data.isna().all():
                    unique_vals = sorted(numeric_data.dropna().unique())
                    if len(unique_vals) <= 10 and all(isinstance(x, (int, float)) for x in unique_vals):
                        column_types[col] = 'rating'
                    else:
                        column_types[col] = 'numerical'
                    continue
            except:
                pass
            
            unique_count = col_data.nunique()
            total_count = len(col_data)
            
            if unique_count <= 20 and unique_count / total_count < 0.5:
                sample_values = col_data.str.lower().unique()[:10] if hasattr(col_data, 'str') else []
                categorical_keywords = ['excellent', 'good', 'poor', 'bad', 'average', 'satisfied', 'dissatisfied', 'yes', 'no']
                
                if any(any(keyword in str(val) for keyword in categorical_keywords) for val in sample_values):
                    column_types[col] = 'categorical'
                elif unique_count <= 10:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'
            else:
                column_types[col] = 'text'
        
        print(f"📋 Column categorization: {column_types}")
        return column_types

    def analyze_numerical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        data = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if data.empty:
            return {"error": "No valid numerical data"}
        
        analysis = {
            'type': 'numerical',
            'total_responses': len(data),
            'mean': round(data.mean(), 2),
            'median': round(data.median(), 2),
            'std_dev': round(data.std(), 2),
            'min_value': data.min(),
            'max_value': data.max(),
            'quartiles': {
                'Q1': round(data.quantile(0.25), 2),
                'Q3': round(data.quantile(0.75), 2)
            }
        }
        
        unique_vals = sorted(data.unique())
        if len(unique_vals) <= 10:
            analysis['rating_distribution'] = data.value_counts().sort_index().to_dict()
            analysis['mode'] = data.mode().iloc[0] if not data.mode().empty else None
        
        return analysis

    def analyze_categorical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        data = df[column].dropna().astype(str)
        
        if data.empty:
            return {"error": "No valid categorical data"}
        
        value_counts = data.value_counts()
        total = len(data)
        
        analysis = {
            'type': 'categorical',
            'total_responses': total,
            'unique_categories': len(value_counts),
            'distribution': value_counts.to_dict(),
            'percentages': (value_counts / total * 100).round(2).to_dict(),
            'most_common': value_counts.index[0],
            'least_common': value_counts.index[-1]
        }
        
        return analysis

    def analyze_text_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        data = df[column].dropna().astype(str)
        data = data[data.str.len() > 0]
        
        if data.empty:
            return {"error": "No valid text data"}
        
        lengths = data.str.len()
        word_counts = data.str.split().str.len()
        all_text = ' '.join(data.values)
        
        analysis = {
            'type': 'text',
            'total_responses': len(data),
            'avg_length': round(lengths.mean(), 2),
            'avg_word_count': round(word_counts.mean(), 2),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'sample_responses': data.head(3).tolist()
        }
        
        return analysis, all_text

    def generate_column_insights(self, column_name: str, analysis_data: Dict[str, Any], 
                       all_text: str = None) -> str:
        print(f"🧠 Generating insights for column: {column_name}")
        
        documents = []
        
        if analysis_data.get('type') == 'numerical' or analysis_data.get('type') == 'rating':
            doc_content = f"""
            Column: {column_name}
            Type: {analysis_data['type']}
            Total Responses: {analysis_data['total_responses']}
            Mean: {analysis_data['mean']}
            Median: {analysis_data['median']}
            Standard Deviation: {analysis_data['std_dev']}
            Range: {analysis_data['min_value']} to {analysis_data['max_value']}
            """
            
            if 'rating_distribution' in analysis_data:
                doc_content += f"\nRating Distribution: {analysis_data['rating_distribution']}"
            
            documents.append(Document(page_content=doc_content, metadata={"column": column_name, "type": "numerical"}))
            
        elif analysis_data.get('type') == 'categorical':
            doc_content = f"""
            Column: {column_name}
            Type: Categorical
            Total Responses: {analysis_data['total_responses']}
            Categories: {analysis_data['unique_categories']}
            Distribution: {analysis_data['distribution']}
            Most Common: {analysis_data['most_common']}
            Least Common: {analysis_data['least_common']}
            """
            
            documents.append(Document(page_content=doc_content, metadata={"column": column_name, "type": "categorical"}))
            
        elif analysis_data.get('type') == 'text' and all_text:
            text_chunks = self.text_splitter.split_text(all_text)
            for i, chunk in enumerate(text_chunks):
                documents.append(Document(
                    page_content=chunk, 
                    metadata={"column": column_name, "type": "text", "chunk": i}
                ))
            
            summary_doc = f"""
            Column: {column_name}
            Type: Text Analysis
            Total Responses: {analysis_data['total_responses']}
            Average Length: {analysis_data['avg_length']} characters
            Average Word Count: {analysis_data['avg_word_count']} words
            """
            documents.append(Document(page_content=summary_doc, metadata={"column": column_name, "type": "text_summary"}))
        
        if not documents:
            return f"Unable to generate insights for {column_name} - insufficient data"
        
        try:
            vectorstore = FAISS.from_documents(
                documents, 
                self.embeddings,
                distance_strategy="COSINE"
            )
            
            if analysis_data.get('type') in ['numerical', 'rating']:
                mean_val = analysis_data.get('mean', 0)
                total_responses = analysis_data.get('total_responses', 0)
                
                prompt_template = f"""Based on the following EXACT data for column '{column_name}':
                - Total responses: {total_responses}
                - Average score: {mean_val}
                - Score range: {analysis_data.get('min_value', 'N/A')} to {analysis_data.get('max_value', 'N/A')}
                
                Context from data: {{context}}
                
                Provide ONLY factual analysis based on this specific data:
                1. What this average score of {mean_val} indicates for performance
                2. How the {total_responses} responses distribute across the scale
                3. One specific, actionable recommendation based on this score
                
                Do not add general advice. Focus strictly on what this data shows."""
                
            elif analysis_data.get('type') == 'categorical':
                most_common = analysis_data.get('most_common', 'N/A')
                total_responses = analysis_data.get('total_responses', 0)
                
                prompt_template = f"""Based on the following EXACT data for column '{column_name}':
                - Total responses: {total_responses}
                - Most selected option: {most_common}
                - Number of different options: {analysis_data.get('unique_categories', 0)}
                
                Context from data: {{context}}
                
                Provide ONLY factual analysis based on this specific data:
                1. What the selection of '{most_common}' as the top choice indicates
                2. How responses are distributed across the available options
                3. One specific insight based on this distribution pattern
                
                Do not add general recommendations. Focus on what this data reveals."""
                
            else:
                total_responses = analysis_data.get('total_responses', 0)
                avg_length = analysis_data.get('avg_length', 0)
                
                prompt_template = f"""Based on the following EXACT data for column '{column_name}':
                - Total text responses: {total_responses}
                - Average response length: {avg_length} characters
                
                Context from actual responses: {{context}}
                
                Analyze ONLY the provided text content:
                1. Main themes that appear in the actual responses
                2. Common patterns or sentiments expressed
                3. Specific points mentioned by respondents
                
                Base analysis strictly on the provided text. Do not add general suggestions."""
            
            custom_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )
            
            llm_chain = LLMChain(
                llm=self.llm, 
                prompt=custom_prompt,
                verbose=False
            )
            
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context",
                verbose=False
            )
            
            qa_chain = RetrievalQA(
                combine_documents_chain=stuff_chain,
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2}
                ),
                return_source_documents=False,
                verbose=False
            )
            
            query = f"Analyze the specific data for {column_name} based on the provided metrics"
            try:
                result = qa_chain.invoke({"query": query})
                insights = result.get('result', '') if isinstance(result, dict) else str(result)
                return insights
            except Exception as chain_error:
                logger.error(f"Chain invoke error for {column_name}: {str(chain_error)}")
                return self._fallback_analysis(column_name, analysis_data, documents)
            
        except Exception as e:
            logger.error(f"Error generating insights for {column_name}: {str(e)}")
            return self._fallback_analysis(column_name, analysis_data, documents)

    def _fallback_analysis(self, column_name: str, analysis_data: Dict[str, Any], documents: List[Document]) -> str:
        try:
            if analysis_data.get('type') in ['numerical', 'rating']:
                mean_val = analysis_data.get('mean', 0)
                total = analysis_data.get('total_responses', 0)
                
                if analysis_data.get('type') == 'rating':
                    if mean_val >= 4.0:
                        return f"The {column_name} shows strong performance with an average rating of {mean_val} from {total} responses. This indicates high satisfaction levels."
                    elif mean_val >= 3.0:
                        return f"The {column_name} has a moderate rating of {mean_val} from {total} responses. Performance is acceptable but has room for improvement."
                    else:
                        return f"The {column_name} rating of {mean_val} from {total} responses indicates areas that need attention and improvement."
                else:
                    return f"The {column_name} shows an average value of {mean_val} across {total} responses, ranging from {analysis_data.get('min_value')} to {analysis_data.get('max_value')}."
                    
            elif analysis_data.get('type') == 'categorical':
                most_common = analysis_data.get('most_common', '')
                total = analysis_data.get('total_responses', 0)
                categories = analysis_data.get('unique_categories', 0)
                
                return f"For {column_name}, '{most_common}' was selected most frequently among {total} responses across {categories} available options."
                    
            elif analysis_data.get('type') == 'text':
                total = analysis_data.get('total_responses', 0)
                avg_length = analysis_data.get('avg_length', 0)
                
                return f"The {column_name} received {total} text responses with an average length of {avg_length:.0f} characters, indicating {'detailed' if avg_length > 50 else 'brief'} participant feedback."
            
        except Exception as fallback_error:
            logger.error(f"Fallback analysis failed for {column_name}: {str(fallback_error)}")
            return self._generate_basic_insights(column_name, analysis_data)

    
    def _generate_basic_insights(self, column_name: str, analysis_data: Dict[str, Any]) -> str:
        insights = []
        
        if analysis_data.get('type') in ['numerical', 'rating']:
            mean_val = analysis_data.get('mean', 0)
            total = analysis_data.get('total_responses', 0)
            
            if analysis_data.get('type') == 'rating':
                if mean_val >= 4.0:
                    insights.append(f"• Excellent performance with {mean_val:.1f} average rating from {total} participants")
                    insights.append("• High satisfaction levels indicate this aspect is working well")
                elif mean_val >= 3.0:
                    insights.append(f"• Good performance with {mean_val:.1f} average rating from {total} participants")
                    insights.append("• Room for improvement to reach higher satisfaction levels")
                else:
                    insights.append(f"• Needs attention with {mean_val:.1f} average rating from {total} participants")
                    insights.append("• Priority area for improvement to address participant concerns")
                    
                if 'rating_distribution' in analysis_data:
                    mode = analysis_data.get('mode')
                    insights.append(f"• Most participants rated this aspect as {mode}")
            else:
                insights.append(f"• Average value of {mean_val:.1f} from {total} responses")
                insights.append(f"• Values range from {analysis_data.get('min_value')} to {analysis_data.get('max_value')}")
                
        elif analysis_data.get('type') == 'categorical':
            most_common = analysis_data.get('most_common', '')
            total = analysis_data.get('total_responses', 0)
            categories = analysis_data.get('unique_categories', 0)
            
            insights.append(f"• {total} participants responded across {categories} available options")
            insights.append(f"• '{most_common}' was the most selected choice")
            
            if 'distribution' in analysis_data:
                dist = analysis_data['distribution']
                if most_common and most_common in dist:
                    percentage = (dist[most_common] / total) * 100
                    insights.append(f"• {percentage:.1f}% of participants selected '{most_common}'")
                    
        elif analysis_data.get('type') == 'text':
            total = analysis_data.get('total_responses', 0)
            avg_length = analysis_data.get('avg_length', 0)
            
            insights.append(f"• Received {total} detailed text responses")
            insights.append(f"• Average response length of {avg_length:.0f} characters")
            
            if avg_length > 50:
                insights.append("• Participants provided detailed feedback showing high engagement")
            elif avg_length > 20:
                insights.append("• Responses show moderate detail in participant feedback")
            else:
                insights.append("• Brief responses indicate participants provided concise feedback")
        
        return "\n".join(insights)
    def analyze_all_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
        results = {}
        
        for column, col_type in column_types.items():
            print(f"📊 Analyzing column: {column} (type: {col_type})")
            
            try:
                if col_type in ['numerical', 'rating']:
                    analysis = self.analyze_numerical_column(df, column)
                    insights = self.generate_column_insights(column, analysis)
                    results[column] = {
                        'analysis': analysis,
                        'insights': insights,
                        'type': col_type
                    }
                    
                elif col_type == 'categorical':
                    analysis = self.analyze_categorical_column(df, column)
                    insights = self.generate_column_insights(column, analysis)
                    results[column] = {
                        'analysis': analysis,
                        'insights': insights,
                        'type': col_type
                    }
                    
                elif col_type == 'text':
                    analysis, all_text = self.analyze_text_column(df, column)
                    insights = self.generate_column_insights(column, analysis, all_text)
                    results[column] = {
                        'analysis': analysis,
                        'insights': insights,
                        'type': col_type
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing column {column}: {str(e)}")
                results[column] = {
                    'error': str(e),
                    'type': col_type
                }
        
        return results
async def fetch_worksheet_data(worksheet_url: str) -> pd.DataFrame:
    """Async version of fetch_worksheet_data"""
    loop = asyncio.get_event_loop()
    
    def _fetch():
        try:
            if 'docs.google.com/spreadsheets' in worksheet_url:
                sheet_id = worksheet_url.split('/d/')[1].split('/')[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            else:
                csv_url = worksheet_url
            
            print(f"📥 Fetching data from: {csv_url}")
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching worksheet data: {str(e)}")
            raise Exception(f"Failed to fetch worksheet data: {str(e)}")
    
    return await loop.run_in_executor(None, _fetch)
def generate_pdf_report(results: Dict[str, Any], event_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceBefore=20,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceBefore=15,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=8
    )
    
    story = []
    
    story.append(Paragraph(f"📊 FEEDBACK ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Event: {event_name}", heading_style))
    story.append(Spacer(1, 20))
    
    type_counts = {}
    for col, data in results.items():
        col_type = data.get('type', 'unknown')
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    story.append(Paragraph("📋 OVERVIEW", heading_style))
    story.append(Paragraph(f"Total Columns Analyzed: {len(results)}", body_style))
    
    for col_type, count in type_counts.items():
        story.append(Paragraph(f"{col_type.title()} Columns: {count}", body_style))
    
    story.append(Spacer(1, 20))
    
    for column, data in results.items():
        if 'error' in data:
            continue
            
        story.append(Paragraph(f"{column.upper()}", heading_style))
        story.append(Paragraph(f"Type: {data['type'].title()}", subheading_style))
        
        if data['type'] in ['numerical', 'rating']:
            analysis = data['analysis']
            stats_data = [
                ['Total Responses', str(analysis['total_responses'])],
                ['Average Score', str(analysis['mean'])],
                ['Median', str(analysis['median'])],
                ['Standard Deviation', str(analysis['std_dev'])]
            ]
            
            if 'rating_distribution' in analysis and analysis.get('mode'):
                stats_data.append(['Most Common Rating', str(analysis.get('mode', 'N/A'))])
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e1e8f0'))
            ]))
            story.append(stats_table)
        
        elif data['type'] == 'categorical':
            analysis = data['analysis']
            stats_data = [
                ['Total Responses', str(analysis['total_responses'])],
                ['Most Selected', str(analysis['most_common'])],
                ['Total Options', str(analysis['unique_categories'])]
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e1e8f0'))
            ]))
            story.append(stats_table)
        
        elif data['type'] == 'text':
            analysis = data['analysis']
            stats_data = [
                ['Total Responses', str(analysis['total_responses'])],
                ['Average Length', f"{analysis['avg_length']:.0f} characters"],
                ['Average Word Count', f"{analysis['avg_word_count']:.0f} words"]
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e1e8f0'))
            ]))
            story.append(stats_table)
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("🔍 Key Insights", subheading_style))
        
        insights_text = data['insights'].replace('•', '• ').replace('\n', '<br/>')
        story.append(Paragraph(insights_text, body_style))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_summary_report(results: Dict[str, Any]) -> str:
    html_parts = []
    
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f7fa; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
            .header h1 { margin: 0; font-size: 2.5em; font-weight: 700; }
            .header p { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }
            .overview { background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%); padding: 25px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #e1e8f0; }
            .overview h2 { color: #667eea; margin-top: 0; font-size: 1.8em; font-weight: 600; }
            .overview p { margin: 8px 0; font-size: 1.1em; }
            .column-section { background: white; margin-bottom: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); overflow: hidden; border: 1px solid #e8eef5; }
            .column-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin: 0; font-size: 1.4em; font-weight: 600; }
            .column-content { padding: 25px; }
            .stats { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
            .stat-item { background: linear-gradient(135deg, #f0f4ff 0%, #e8f2ff 100%); padding: 15px 20px; border-radius: 10px; flex: 1; min-width: 140px; border: 1px solid #d6e3f0; }
            .stat-label { font-size: 13px; color: #5a6c7d; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
            .stat-value { font-size: 24px; font-weight: 700; color: #667eea; margin-top: 5px; }
            .insights { background: #f9fafb; padding: 20px; border-radius: 10px; margin-top: 20px; border-left: 4px solid #667eea; }
            .insights h4 { color: #667eea; margin-top: 0; font-size: 1.2em; font-weight: 600; }
            .insights p { margin: 10px 0 0 0; font-size: 1em; line-height: 1.7; }
            .insights ul { margin: 10px 0; padding-left: 20px; }
            .insights li { margin: 8px 0; font-size: 1em; line-height: 1.6; }
            .type-badge { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); color: white; padding: 6px 16px; border-radius: 25px; font-size: 12px; display: inline-block; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
            .highlight { font-weight: 600; color: #667eea; }
            .metric-highlight { font-weight: 700; color: #2c3e50; }
        </style>
    </head>
    <body>
    """)
    
    html_parts.append("""
        <div class="header">
            <h1>📊 FEEDBACK ANALYSIS REPORT</h1>
            <p>Comprehensive analysis of participant feedback and responses</p>
        </div>
    """)
    
    type_counts = {}
    for col, data in results.items():
        col_type = data.get('type', 'unknown')
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    html_parts.append(f"""
        <div class="overview">
            <h2>📋 OVERVIEW</h2>
            <p><span class="highlight">Total Columns Analyzed:</span> <span class="metric-highlight">{len(results)}</span></p>
    """)
    
    for col_type, count in type_counts.items():
        html_parts.append(f"<p><span class=\"highlight\">{col_type.title()} Columns:</span> <span class=\"metric-highlight\">{count}</span></p>")
    
    html_parts.append("</div>")
    
    for column, data in results.items():
        if 'error' in data:
            continue
            
        html_parts.append(f"""
            <div class="column-section">
                <h2 class="column-header">{column.upper()}</h2>
                <div class="column-content">
                    <span class="type-badge">{data['type'].title()}</span>
                    <div class="stats">
        """)
        
        if data['type'] in ['numerical', 'rating']:
            analysis = data['analysis']
            html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Total Responses</div>
                            <div class="stat-value">{analysis['total_responses']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Average Score</div>
                            <div class="stat-value">{analysis['mean']}</div>
                        </div>
            """)
            if 'rating_distribution' in analysis and analysis.get('mode'):
                html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Most Common Rating</div>
                            <div class="stat-value">{analysis.get('mode', 'N/A')}</div>
                        </div>
                """)
        
        elif data['type'] == 'categorical':
            analysis = data['analysis']
            html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Total Responses</div>
                            <div class="stat-value">{analysis['total_responses']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Most Selected</div>
                            <div class="stat-value">{analysis['most_common']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Total Options</div>
                            <div class="stat-value">{analysis['unique_categories']}</div>
                        </div>
            """)
        
        elif data['type'] == 'text':
            analysis = data['analysis']
            html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Total Responses</div>
                            <div class="stat-value">{analysis['total_responses']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Average Length</div>
                            <div class="stat-value">{analysis['avg_length']:.0f}</div>
                        </div>
            """)
        
        formatted_insights = data['insights'].replace('\n', '<br>')
        formatted_insights = formatted_insights.replace('•', '<li>').replace('<li>', '</li><li>').replace('</li><li>', '<li>', 1)
        if formatted_insights.startswith('<li>'):
            formatted_insights = '<ul>' + formatted_insights + '</ul>'
        
        html_parts.append(f"""
                    </div>
                    <div class="insights">
                        <h4>🔍 Key Insights</h4>
                        <div>{formatted_insights}</div>
                    </div>
                </div>
            </div>
        """)
    
    html_parts.append("""
        </body>
        </html>
    """)
    
    return "".join(html_parts)

async def send_analysis_email(recipient_email: str, report: str, event_name: str, results: Dict[str, Any]):
    try:
        if not config.EMAIL_USER or not config.EMAIL_PASSWORD:
            raise Exception("EMAIL_USER and EMAIL_PASSWORD must be set in environment variables")
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Feedback Analysis Report - {event_name}"
        msg['From'] = config.FROM_EMAIL
        msg['To'] = recipient_email
        
        html_part = MIMEText(report, 'html')
        msg.attach(html_part)
        
        pdf_data = generate_pdf_report(results, event_name)
        
        pdf_attachment = MIMEBase('application', 'pdf')
        pdf_attachment.set_payload(pdf_data)
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename="feedback_analysis_{event_name.replace(" ", "_")}.pdf"'
        )
        msg.attach(pdf_attachment)
        
        loop = asyncio.get_event_loop()
        
        def _send_email():
            try:
                print(f"🔄 Connecting to {config.SMTP_SERVER}:{config.SMTP_PORT}")
                
                with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                    server.timeout = 30
                    
                    print("🔐 Starting TLS...")
                    server.starttls()
                    
                    print(f"🔑 Logging in as: {config.EMAIL_USER}")
                    server.login(str(config.EMAIL_USER), str(config.EMAIL_PASSWORD))
                    
                    print(f"📤 Sending to: {recipient_email}")
                    server.send_message(msg)
                    
                    print("✅ Email sent successfully!")
                    
            except smtplib.SMTPAuthenticationError as auth_error:
                print(f"❌ Authentication failed: {auth_error}")
                print("💡 If using Gmail, make sure you're using an App Password!")
                raise Exception(f"SMTP Authentication failed: {auth_error}")
                
            except smtplib.SMTPException as smtp_error:
                print(f"❌ SMTP error: {smtp_error}")
                raise Exception(f"SMTP error: {smtp_error}")
                
            except Exception as general_error:
                print(f"❌ General error: {general_error}")
                raise Exception(f"Email sending failed: {general_error}")
        
        await loop.run_in_executor(None, _send_email)
        print(f"✅ Analysis report sent to {recipient_email}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        print(f"❌ Failed to send email: {str(e)}")
        return False
    
async def process_analysis_task(request: AnalysisRequest, task_id: str):
    try:
        print_terminal_separator("🎯 RAG FEEDBACK ANALYSIS STARTED")
        logger.info(f"Starting analysis task {task_id}")
        
        analyzer = FeedbackRAGAnalyzer(
            ollama_base_url=config.OLLAMA_BASE_URL,
            model_name=config.OLLAMA_MODEL,
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP
        )
        
        df = await fetch_worksheet_data(request.worksheet_url)
        
        if len(df) > config.MAX_PROCESSING_ROWS:
            print(f"⚠️ Limiting analysis to {config.MAX_PROCESSING_ROWS} rows")
            df = df.head(config.MAX_PROCESSING_ROWS)
        
        processed_df, column_types = analyzer.preprocess_columns(df)
        
        if processed_df.empty:
            raise Exception("No relevant columns found for analysis")
        
        results = await analyze_columns_parallel(analyzer, processed_df, column_types)
        
        summary_report = generate_summary_report(results)
        
        email_sent = await send_analysis_email(
            request.recipient_email, 
            summary_report, 
            request.event_name,
            results
        )
        
        print_terminal_separator("✅ ANALYSIS COMPLETE")
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        await send_error_email(request.recipient_email, str(e), request.event_name)

async def analyze_columns_parallel(analyzer: FeedbackRAGAnalyzer, 
                                 df: pd.DataFrame, 
                                 column_types: Dict[str, str]) -> Dict[str, Any]:
    """Analyze columns in parallel using ThreadPoolExecutor"""
    
    def analyze_single_column(column_data):
        column, col_type = column_data
        print(f"📊 Analyzing column: {column} (type: {col_type})")
        
        try:
            if col_type in ['numerical', 'rating']:
                analysis = analyzer.analyze_numerical_column(df, column)
                insights = analyzer.generate_column_insights(column, analysis)
                return column, {
                    'analysis': analysis,
                    'insights': insights,
                    'type': col_type
                }
                
            elif col_type == 'categorical':
                analysis = analyzer.analyze_categorical_column(df, column)
                insights = analyzer.generate_column_insights(column, analysis)
                return column, {
                    'analysis': analysis,
                    'insights': insights,
                    'type': col_type
                }
                
            elif col_type == 'text':
                analysis, all_text = analyzer.analyze_text_column(df, column)
                insights = analyzer.generate_column_insights(column, analysis, all_text)
                return column, {
                    'analysis': analysis,
                    'insights': insights,
                    'type': col_type
                }
                
        except Exception as e:
            logger.error(f"Error analyzing column {column}: {str(e)}")
            return column, {
                'error': str(e),
                'type': col_type
            }
    
    # Process columns in parallel
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        column_items = list(column_types.items())
        
        # Submit all tasks
        tasks = [
            loop.run_in_executor(executor, analyze_single_column, item)
            for item in column_items
        ]
        
        # Wait for all results
        results_list = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        results = dict(results_list)
        
    return results



# FastAPI Endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start feedback analysis task"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Add the analysis task to background tasks
    background_tasks.add_task(process_analysis_task, request, task_id)
    
    return AnalysisResponse(
        status="accepted",
        message="Analysis started. You will receive an email when complete.",
        task_id=task_id
    )
async def send_error_email(recipient_email: str, error_msg: str, event_name: str):
    try:
        subject = f"❌ Feedback Analysis Failed - {event_name}"
        body = f"""
        <html>
        <body>
            <h2>Analysis Failed</h2>
            <p>The feedback analysis for {event_name} encountered an error:</p>
            <p><strong>Error:</strong> {error_msg}</p>
            <p>Please contact support for assistance.</p>
        </body>
        </html>
        """
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.FROM_EMAIL
        msg['To'] = recipient_email
        
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        loop = asyncio.get_event_loop()
        
        def _send_email():
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls()
                server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
                server.send_message(msg)
        
        await loop.run_in_executor(None, _send_email)
        
    except Exception as e:
        logger.error(f"Failed to send error email: {str(e)}")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "feedback-analysis"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_analysis:app", 
        host="0.0.0.0", 
        port=8001, 
        workers=1,  # Use 1 worker for the main app, parallel processing handled internally
        reload=True
    )