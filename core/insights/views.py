from dotenv import load_dotenv

load_dotenv()
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from home.models import Event

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

import re
from functools import lru_cache
import hashlib
from io import StringIO
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feedback_insights_langchain.log')
    ]
)
logger = logging.getLogger(__name__)

def print_terminal_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted separator in terminal"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")

def print_insights_section(title: str, content: Any, max_length: int = 1000):
    """Print insights section with formatting"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    if isinstance(content, dict):
        print(json.dumps(content, indent=2, ensure_ascii=False)[:max_length])
        if len(str(content)) > max_length:
            print(f"\n... (truncated, full length: {len(str(content))} characters)")
    elif isinstance(content, list):
        for i, item in enumerate(content[:5]):  # Show first 5 items
            print(f"\n[{i+1}] {str(item)[:200]}")
            if len(str(item)) > 200:
                print("    ... (truncated)")
        if len(content) > 5:
            print(f"\n... and {len(content) - 5} more items")
    else:
        content_str = str(content)[:max_length]
        print(content_str)
        if len(str(content)) > max_length:
            print(f"\n... (truncated, full length: {len(str(content))} characters)")

class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LangChain to track token usage and timing"""
    
    def __init__(self):
        self.start_time = None
        self.tokens_used = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        import time
        self.start_time = time.time()
        logger.info(f"LLM request started with {len(prompts)} prompts")
        print(f"🤖 LLM Request Started - Processing {len(prompts)} prompt(s)")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        import time
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"LLM request completed in {duration:.2f}s")
            print(f"✅ LLM Request Completed in {duration:.2f}s")

class LangChainRAGInsightsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configuration from environment
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model_name = os.environ.get('OLLAMA_MODEL', 'gemma:2b')
        self.chunk_size = int(os.environ.get('RAG_CHUNK_SIZE', '500'))
        self.chunk_overlap = int(os.environ.get('RAG_CHUNK_OVERLAP', '50'))
        self.max_tokens = int(os.environ.get('MAX_TOKENS', '512'))
        self.temperature = float(os.environ.get('TEMPERATURE', '0.3'))
        self.request_timeout = int(os.environ.get('OLLAMA_TIMEOUT', '30'))
        
        # Initialize LangChain components
        self._initialize_langchain_components()
        
        # Cache for results
        self.cache = {}
        
        logger.info(f"Initialized LangChain RAG with model: {self.model_name}")
        print_terminal_separator("🚀 LANGCHAIN RAG INITIALIZED")
        print(f"Model: {self.model_name}")
        print(f"Base URL: {self.ollama_base_url}")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Temperature: {self.temperature}")
    
    def _initialize_langchain_components(self):
        """Initialize LangChain components"""
        try:
            print("\n🔧 Initializing LangChain Components...")
            
            # Initialize callback handler
            self.callback_handler = CustomCallbackHandler()
            callback_manager = CallbackManager([self.callback_handler])
            
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                callback_manager=callback_manager,
                verbose=False
            )
            print("✅ LLM initialized")
            
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-mpnet-base-v2",  
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embeddings model initialized")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
            print("✅ Text splitter initialized")
            
            # Initialize prompt templates
            self._initialize_prompt_templates()
            print("✅ Prompt templates initialized")
            
            logger.info("LangChain components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain components: {str(e)}")
            print(f"❌ Error initializing components: {str(e)}")
            raise
    
    def _initialize_prompt_templates(self):
        """Initialize enhanced prompt templates with better context understanding"""
        
        # Enhanced template for chunk analysis with context awareness
        self.chunk_analysis_template = PromptTemplate(
            input_variables=["feedback_data", "chunk_info"],
            template="""You are analyzing student feedback data. Each response contains structured question-answer pairs.

    FEEDBACK DATA TO ANALYZE:
    {feedback_data}

    ANALYSIS INFO: {chunk_info}

    Please analyze this feedback data and extract insights focusing on:

    1. RESPONSE PATTERNS: What are the common response patterns for each question/criteria?
    2. SATISFACTION LEVELS: Based on ratings (Excellent, Very Good, Good, etc.), what's the overall satisfaction?
    3. SPECIFIC STRENGTHS: What aspects are performing well based on positive responses?
    4. AREAS FOR IMPROVEMENT: What aspects show lower satisfaction or negative feedback?
    5. QUESTION-SPECIFIC INSIGHTS: Analyze each question/criteria individually
    6. CORRELATION PATTERNS: Are there patterns between different responses from the same respondent?

    IMPORTANT: 
    - Pay attention to what each response is rating (the question/criteria)
    - Consider "Excellent" and "Very Good" as positive, "Good" as neutral, "Fair" and "Poor" as negative
    - Look for specific themes in text responses
    - Note any recurring issues or praise

    Respond in structured JSON format with these sections:
    {{
    "overall_satisfaction": "summary of general satisfaction level",
    "response_distribution": "breakdown of rating distributions",
    "strengths": ["list of identified strengths"],
    "improvement_areas": ["list of areas needing improvement"],
    "question_specific_insights": {{
        "question_name": "insight for that specific question"
    }},
    "key_patterns": ["list of notable patterns found"],
    "sentiment_summary": "overall sentiment analysis"
    }}"""
        )
        
        # Enhanced synthesis template
        self.synthesis_template = PromptTemplate(
            input_variables=["chunk_insights", "total_chunks"],
            template="""Synthesize comprehensive feedback analysis from {total_chunks} data chunks:

    CHUNK INSIGHTS:
    {chunk_insights}

    Create a comprehensive final analysis that combines all chunks with:

    1. OVERALL SATISFACTION METRICS: Combined satisfaction levels across all feedback
    2. TOP PERFORMING AREAS: What questions/criteria received the highest ratings
    3. PRIORITY IMPROVEMENT AREAS: What questions/criteria need the most attention
    4. CONSISTENT PATTERNS: Patterns that appear across multiple chunks
    5. DETAILED RECOMMENDATIONS: Specific, actionable recommendations
    6. RESPONSE DISTRIBUTION: Overall distribution of ratings across all criteria

    Format as comprehensive JSON:
    {{
    "executive_summary": "brief overview of key findings",
    "overall_satisfaction_score": "calculated satisfaction level",
    "top_performing_areas": [
        {{
        "area": "criteria name",
        "performance": "description",
        "rating_summary": "rating distribution"
        }}
    ],
    "priority_improvements": [
        {{
        "area": "criteria name", 
        "issue": "description of problem",
        "recommendation": "specific action to take"
        }}
    ],
    "response_patterns": {{
        "positive_feedback_themes": ["themes in positive responses"],
        "negative_feedback_themes": ["themes in negative responses"],
        "rating_distribution": "overall rating breakdown"
    }},
    "actionable_recommendations": [
        "specific recommendation 1",
        "specific recommendation 2"
    ]
    }}"""
        )
        
        # Enhanced retrieval template
        self.retrieval_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on this structured feedback context:
    {context}

    Answer this specific question: {question}

    IMPORTANT GUIDELINES:
    - Reference specific questions/criteria from the feedback forms
    - Quote actual responses when relevant
    - Distinguish between different types of feedback (ratings vs. text responses)
    - Provide specific examples from the data
    - Consider the context of what each response is rating

    Provide detailed insights based only on the feedback data provided, with specific references to the questions and response patterns."""
        )
    
    def extract_data_from_sheet(self, worksheet_url: str) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]]]:
        """Extract data from Google Sheets with improved error handling"""
        try:
            print_terminal_separator("📊 DATA EXTRACTION PHASE")
            print(f"🔗 Extracting data from: {worksheet_url}")
            logger.info(f"Extracting data from: {worksheet_url}")
            
            if "spreadsheets/d/" in worksheet_url:
                sheet_id = worksheet_url.split("spreadsheets/d/")[1].split("/")[0]
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
                response = requests.get(export_url, timeout=10)
                if response.status_code != 200:
                    print(f"❌ Failed to fetch Google Sheet: HTTP {response.status_code}")
                    logger.error(f"Failed to fetch Google Sheet: HTTP {response.status_code}")
                    return None, None
                
                df = pd.read_csv(StringIO(response.text))
                
                if df.empty:
                    print("⚠️ Sheet contains no data")
                    logger.warning("Sheet contains no data")
                    return None, None
                    
                data = df.to_dict(orient='records')
                
                print(f"✅ Successfully extracted {len(data)} records")
                print(f"📋 Columns found: {list(df.columns)[:10]}")  # Show first 10 columns
                print(f"📏 Data shape: {df.shape}")
                
                # Show sample data
                print("\n📋 Sample Data (first 2 rows):")
                for i, row in df.head(2).iterrows():
                    print(f"Row {i+1}: {dict(list(row.items())[:5])}")  # Show first 5 columns
                
                logger.info(f"Successfully extracted {len(data)} records from sheet")
                return df, data
            else:
                print(f"❌ Invalid Google Sheets URL format")
                logger.error(f"Invalid Google Sheets URL format: {worksheet_url}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error extracting data: {str(e)}")
            logger.error(f"Error extracting data from sheet: {str(e)}")
            return None, None
    
    def preprocess_feedback_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess feedback data and identify relevant columns"""
        try:
            print_terminal_separator("🔧 DATA PREPROCESSING PHASE")
            print(f"📊 Starting preprocessing for {len(df)} rows, {len(df.columns)} columns")
            logger.info("Starting feedback data preprocessing")
            
            processed_df = df.copy()
            
            # Common PII/irrelevant columns to exclude
            exclude_patterns = [
                r'name', r'fullname', r'first.*name', r'last.*name',
                r'usn', r'registration', r'student.*id', r'email', r'phone',
                r'mobile', r'address', r'timestamp', r'submitted.*at',
                r'ip.*address', r'age', r'roll.*no', r'dob', r'date.*birth'
            ]
            
            # Create combined pattern
            exclude_pattern = '|'.join(exclude_patterns)
            columns_to_drop = [
                col for col in processed_df.columns 
                if re.search(exclude_pattern, col, re.IGNORECASE)
            ]
            
            if columns_to_drop:
                print(f"🗑️ Dropping {len(columns_to_drop)} irrelevant columns:")
                for col in columns_to_drop:
                    print(f"   - {col}")
                logger.info(f"Dropping {len(columns_to_drop)} irrelevant columns")
                processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')
            
            # Fill missing values
            print("\n🔧 Filling missing values...")
            missing_before = processed_df.isnull().sum().sum()
            
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    processed_df[col] = processed_df[col].fillna('No response')
                else:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            
            missing_after = processed_df.isnull().sum().sum()
            print(f"✅ Filled {missing_before - missing_after} missing values")
            
            print(f"📊 Final processed data: {processed_df.shape}")
            print(f"📋 Remaining columns: {list(processed_df.columns)}")
            
            return processed_df
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            logger.error(f"Error preprocessing feedback data: {str(e)}")
            return df
    
    def create_documents_from_dataframe(self, df: pd.DataFrame) -> List[Document]:
        try:
            print_terminal_separator("📄 DOCUMENT CREATION PHASE")
            
            documents = []
            
            # Store column headers for context
            column_headers = df.columns.tolist()
            
            # Identify feedback columns (text columns with substantial content)
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Filter out PII columns from feedback analysis
            exclude_patterns = [
                r'timestamp', r'email', r'name', r'usn', r'student.*id', 
                r'roll.*no', r'phone', r'mobile', r'address'
            ]
            exclude_pattern = '|'.join(exclude_patterns)
            
            feedback_cols = [
                col for col in text_cols 
                if not re.search(exclude_pattern, col, re.IGNORECASE) and
                df[col].str.len().mean() > 3  # Lowered threshold for rating scales
            ]
            
            if not feedback_cols:
                feedback_cols = [col for col in text_cols if not re.search(exclude_pattern, col, re.IGNORECASE)][:5]
            
            print(f"📝 Using {len(feedback_cols)} columns for document creation:")
            for col in feedback_cols:
                avg_length = df[col].str.len().mean()
                unique_values = df[col].nunique()
                print(f"   - {col} (avg length: {avg_length:.1f}, unique values: {unique_values})")
            
            logger.info(f"Using {len(feedback_cols)} columns for document creation: {feedback_cols}")
            
            # Create documents from each row with enhanced context
            for idx, row in df.iterrows():
                # Create structured feedback with question-answer pairs
                feedback_sections = []
                metadata = {
                    "row_index": idx, 
                    "feedback_columns": feedback_cols,
                    "total_columns": len(column_headers),
                    "column_headers": column_headers
                }
                
                # Add column headers as context at the beginning
                context_header = "=== FEEDBACK FORM STRUCTURE ===\n"
                context_header += "This feedback contains responses to the following questions/criteria:\n"
                for i, col in enumerate(feedback_cols, 1):
                    context_header += f"{i}. {col}\n"
                context_header += "\n=== INDIVIDUAL RESPONSES ===\n"
                
                # Process each feedback column with full context
                for col in feedback_cols:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        # Create detailed question-answer format
                        question_text = col
                        response_text = str(row[col]).strip()
                        
                        # Enhanced formatting for better understanding
                        feedback_entry = f"Question: {question_text}\n"
                        feedback_entry += f"Response: {response_text}\n"
                        
                        # Add interpretation hints for common rating scales
                        if response_text.lower() in ['excellent', 'very good', 'good', 'fair', 'poor']:
                            feedback_entry += f"(Rating scale response indicating satisfaction level)\n"
                        elif response_text.lower() in ['strongly agree', 'agree', 'neutral', 'disagree', 'strongly disagree']:
                            feedback_entry += f"(Agreement scale response)\n"
                        elif response_text.isdigit() and 1 <= int(response_text) <= 10:
                            feedback_entry += f"(Numeric rating on scale)\n"
                        
                        feedback_sections.append(feedback_entry)
                            
                # Add non-feedback columns as metadata
                for col in df.columns:
                    if col not in feedback_cols and pd.notna(row[col]):
                        metadata[f"meta_{col}"] = str(row[col])
                
                if feedback_sections:
                    # Combine context header with individual responses
                    combined_text = context_header + "\n".join(feedback_sections)
                    
                    doc = Document(
                        page_content=combined_text,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            print(f"✅ Created {len(documents)} documents from DataFrame")
            
            # Show enhanced sample document
            if documents:
                print("\n📄 Sample Document with Context:")
                sample_doc = documents[0]
                print(f"Content preview: {sample_doc.page_content[:500]}...")
                print(f"Metadata keys: {list(sample_doc.metadata.keys())}")
            
            logger.info(f"Created {len(documents)} documents with enhanced context")
            return documents
            
        except Exception as e:
            print(f"❌ Error creating documents: {str(e)}")
            logger.error(f"Error creating documents from DataFrame: {str(e)}")
    
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """Create FAISS vector store from documents"""
        try:
            print_terminal_separator("🔍 VECTOR STORE CREATION PHASE")
            
            if not documents:
                print("❌ No documents provided for vector store creation")
                logger.error("No documents provided for vector store creation")
                return None
            
            print(f"🔧 Creating vector store from {len(documents)} documents")
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            # Split documents into chunks
            print("✂️ Splitting documents into chunks...")
            split_docs = self.text_splitter.split_documents(documents)
            print(f"✅ Split into {len(split_docs)} chunks")
            
            # Show chunk statistics
            chunk_lengths = [len(doc.page_content) for doc in split_docs]
            print(f"📊 Chunk statistics:")
            print(f"   - Average length: {np.mean(chunk_lengths):.1f}")
            print(f"   - Min/Max length: {min(chunk_lengths)}/{max(chunk_lengths)}")
            
            # Sample chunk
            if split_docs:
                print(f"\n📄 Sample chunk:")
                print(f"{split_docs[0].page_content[:200]}...")
            
            logger.info(f"Split into {len(split_docs)} chunks")
            
            # Create vector store
            print("🧮 Creating embeddings and building FAISS index...")
            vector_store = FAISS.from_documents(split_docs, self.embeddings)
            
            print("✅ Vector store created successfully")
            logger.info("Vector store created successfully")
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def analyze_chunk_with_llm(self, chunk_text: str, chunk_info: str) -> Dict[str, Any]:
        """Analyze a single chunk using LLM"""
        try:
            print(f"\n🤖 Analyzing {chunk_info}...")
            print(f"📝 Chunk preview: {chunk_text[:150]}...")
            
            # Create LLM chain
            chain = LLMChain(
                llm=self.llm,
                prompt=self.chunk_analysis_template,
                verbose=False
            )
            
            # Generate response
            response = chain.run(
                feedback_data=chunk_text,
                chunk_info=chunk_info
            )
            
            print(f"📤 Raw LLM Response for {chunk_info}:")
            print(f"{response[:500]}...")
            
            # Try to parse JSON response
            json_content = self._extract_json_from_response(response)
            if json_content:
                print(f"✅ Successfully parsed JSON insights for {chunk_info}")
                print_insights_section(f"Parsed Insights - {chunk_info}", json_content, 800)
                return json_content
            else:
                print(f"⚠️ Could not parse JSON, using raw response for {chunk_info}")
                return {"raw_insights": response[:1000]}
                
        except Exception as e:
            print(f"❌ Error analyzing {chunk_info}: {str(e)}")
            logger.error(f"Error analyzing chunk with LLM: {str(e)}")
            return {"error": str(e)}
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        try:
            # Method 1: Look for JSON code blocks
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            json_matches = re.findall(json_pattern, response)
            
            for json_text in json_matches:
                try:
                    return json.loads(json_text.strip())
                except json.JSONDecodeError:
                    continue
            
            # Method 2: Find JSON object in text
            start = response.find("{")
            if start >= 0:
                stack = 1
                for i in range(start + 1, len(response)):
                    if response[i] == "{":
                        stack += 1
                    elif response[i] == "}":
                        stack -= 1
                        if stack == 0:
                            try:
                                return json.loads(response[start:i+1])
                            except json.JSONDecodeError:
                                break
            
            # Method 3: Try to clean and parse
            cleaned = re.sub(r'(\w+):', r'"\1":', response)  # Add quotes to keys
            cleaned = cleaned.replace("'", '"')  # Replace single quotes
            
            json_match = re.search(r'\{[^{}]*\}', cleaned)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {str(e)}")
            return None
    
    def process_feedback_with_rag(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main RAG processing pipeline"""
        try:
            print_terminal_separator("🚀 MAIN RAG PROCESSING PIPELINE")
            print(f"🎯 Processing {len(df)} rows of feedback data")
            logger.info(f"Starting RAG processing for {len(df)} rows")
            
            # 1. Preprocess data
            print("\n📊 Step 1: Data Preprocessing")
            processed_df = self.preprocess_feedback_data(df)
            
            # 2. Create documents
            print("\n📄 Step 2: Document Creation")
            documents = self.create_documents_from_dataframe(processed_df)
            if not documents:
                print("❌ Failed to create documents")
                return {"error": "Failed to create documents from data"}
            
            # 3. Create vector store
            print("\n🔍 Step 3: Vector Store Creation")
            vector_store = self.create_vector_store(documents)
            if not vector_store:
                print("❌ Failed to create vector store")
                return {"error": "Failed to create vector store"}
            
            # 4. Analyze chunks in parallel
            print("\n🤖 Step 4: Chunk Analysis")
            chunk_insights = self._analyze_chunks_parallel(documents)
            
            if not chunk_insights:
                print("❌ Failed to generate insights from any chunks")
                return {"error": "Failed to generate insights from any chunks"}
            
            print_insights_section("ALL CHUNK INSIGHTS", chunk_insights)
            
            # 5. Synthesize insights
            print("\n🔄 Step 5: Insight Synthesis")
            if len(chunk_insights) == 1:
                print("ℹ️ Single chunk processed, returning direct insights")
                logger.info("Single chunk processed, returning direct insights")
                final_insights = chunk_insights[0]
            else:
                final_insights = self._synthesize_insights(chunk_insights)
                print_insights_section("SYNTHESIZED INSIGHTS", final_insights)
            
            # 6. Add RAG-specific enhancements
            print("\n🔍 Step 6: RAG Enhancement")
            enhanced_insights = self._enhance_with_rag_queries(
                vector_store, final_insights
            )
            
            print_insights_section("FINAL ENHANCED INSIGHTS", enhanced_insights)
            
            return enhanced_insights
            
        except Exception as e:
            print(f"❌ Error in RAG processing: {str(e)}")
            logger.error(f"Error in RAG processing: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_chunks_parallel(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Enhanced parallel chunk analysis with better context preservation"""
        try:
            print_terminal_separator("⚡ PARALLEL CHUNK ANALYSIS WITH CONTEXT")
            
            # Ensure we don't lose context by making chunks too small
            # For feedback data, we want to maintain question-answer relationships
            min_chunk_size = max(5, len(documents) // 6)  # Larger chunks to preserve context
            processing_chunks = [
                documents[i:i + min_chunk_size] 
                for i in range(0, len(documents), min_chunk_size)
            ]
            
            print(f"📦 Processing {len(processing_chunks)} chunks in parallel")
            print(f"📊 Chunk sizes: {[len(chunk) for chunk in processing_chunks]}")
            logger.info(f"Processing {len(processing_chunks)} chunks with enhanced context")
            
            insights = []
            max_workers = min(3, len(processing_chunks))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for i, chunk_docs in enumerate(processing_chunks):
                    # Preserve the context structure when combining documents
                    combined_sections = []
                    
                    # Extract column headers from first document for context
                    if chunk_docs and 'column_headers' in chunk_docs[0].metadata:
                        headers = chunk_docs[0].metadata['column_headers']
                        context_intro = f"=== FEEDBACK ANALYSIS CONTEXT ===\n"
                        context_intro += f"This chunk contains feedback responses to {len(headers)} questions/criteria.\n"
                        context_intro += f"Total responses in this chunk: {len(chunk_docs)}\n\n"
                        combined_sections.append(context_intro)
                    
                    # Combine documents while preserving individual response structure
                    for j, doc in enumerate(chunk_docs):
                        response_header = f"--- RESPONSE {j+1} ---\n"
                        combined_sections.append(response_header + doc.page_content)
                    
                    combined_text = "\n\n".join(combined_sections)
                    chunk_info = f"Chunk {i+1}/{len(processing_chunks)} ({len(chunk_docs)} responses)"
                    
                    print(f"🚀 Submitting {chunk_info} for contextual analysis")
                    print(f"📝 Combined text length: {len(combined_text)} characters")
                    
                    future = executor.submit(
                        self.analyze_chunk_with_llm, 
                        combined_text, 
                        chunk_info
                    )
                    futures.append(future)
                
                # Collect results with better error handling
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result(timeout=90)  # Increased timeout for better analysis
                        if "error" not in result:
                            insights.append(result)
                            print(f"✅ Successfully processed contextual chunk {i+1}")
                            
                            # Show preview of insights
                            if isinstance(result, dict) and 'overall_satisfaction' in result:
                                print(f"   📊 Satisfaction: {result.get('overall_satisfaction', 'N/A')}")
                            
                            logger.info(f"Successfully processed contextual chunk {i+1}")
                        else:
                            print(f"⚠️ Error in chunk {i+1}: {result.get('error')}")
                            logger.warning(f"Error in chunk {i+1}: {result.get('error')}")
                    except Exception as e:
                        print(f"❌ Exception processing chunk {i+1}: {str(e)}")
                        logger.error(f"Exception processing chunk {i+1}: {str(e)}")
            
            print(f"📊 Collected {len(insights)} successful contextual chunk analyses")
            return insights
            
        except Exception as e:
            print(f"❌ Error in parallel chunk analysis: {str(e)}")
            logger.error(f"Error in parallel chunk analysis: {str(e)}")
            return []
    
    def _synthesize_insights(self, chunk_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize insights from multiple chunks"""
        try:
            print_terminal_separator("🔄 INSIGHT SYNTHESIS")
            print(f"🧠 Synthesizing insights from {len(chunk_insights)} chunks")
            logger.info(f"Synthesizing insights from {len(chunk_insights)} chunks")
            
            # Create synthesis chain
            chain = LLMChain(
                llm=self.llm,
                prompt=self.synthesis_template,
                verbose=False
            )
            
            # Prepare insights for synthesis
            insights_text = json.dumps(chunk_insights, ensure_ascii=False, indent=2)
            print(f"📝 Input for synthesis (length: {len(insights_text)} chars)")
            
            # Generate synthesis
            print("🤖 Generating synthesis...")
            response = chain.run(
                chunk_insights=insights_text,
                total_chunks=len(chunk_insights)
            )
            
            print(f"📤 Raw synthesis response:")
            print(f"{response[:800]}...")
            
            # Parse response
            json_content = self._extract_json_from_response(response)
            if json_content:
                print("✅ Successfully parsed synthesis JSON")
                return json_content
            else:
                print("⚠️ Could not parse synthesis JSON, using raw response")
                return {"raw_synthesis": response[:2000]}
                
        except Exception as e:
            print(f"❌ Error synthesizing insights: {str(e)}")
            logger.error(f"Error synthesizing insights: {str(e)}")
            return {"error": str(e)}
    
    def _enhance_with_rag_queries(self, vector_store: FAISS, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced RAG queries with context-aware questions"""
        try:
            print_terminal_separator("🔍 CONTEXT-AWARE RAG ENHANCEMENT")
            print("🚀 Enhancing insights with context-aware RAG queries")
            logger.info("Enhancing insights with context-aware RAG queries")
            
            # Enhanced context-aware queries
            enhancement_queries = [
                "Which specific questions or criteria received the highest ratings and what do students appreciate most?",
                "Which specific questions or criteria received the lowest ratings and what are the main concerns?",
                "What are the most frequently mentioned improvement suggestions in the text responses?",
                "Are there any patterns between different criteria - do students who rate one area highly also rate others highly?",
                "What specific aspects of the course/program do students find most valuable based on their ratings?",
                "What are the main pain points or challenges mentioned across different feedback criteria?"
            ]
            
            enhanced_insights = insights.copy()
            rag_enhancements = {}
            
            # Create retrieval chain with enhanced prompt
            chain = LLMChain(
                llm=self.llm,
                prompt=self.retrieval_template,
                verbose=False
            )
            
            for i, query in enumerate(enhancement_queries, 1):
                try:
                    print(f"\n🔍 Context-Aware Query {i}/{len(enhancement_queries)}:")
                    print(f"   {query}")
                    
                    # Retrieve relevant documents with higher k for better context
                    relevant_docs = vector_store.similarity_search(query, k=5)
                    
                    if relevant_docs:
                        print(f"📄 Found {len(relevant_docs)} relevant documents")
                        
                        # Show retrieved content preview with context
                        for j, doc in enumerate(relevant_docs):
                            # Extract question context if available
                            content_preview = doc.page_content[:150].replace('\n', ' ')
                            print(f"   Doc {j+1}: {content_preview}...")
                        
                        # Combine retrieved content while preserving question-answer structure
                        context_sections = []
                        for doc in relevant_docs:
                            if "Question:" in doc.page_content:
                                # This document has structured Q&A format
                                context_sections.append(doc.page_content)
                            else:
                                # Add structure to unstructured content
                                context_sections.append(f"Feedback Content:\n{doc.page_content}")
                        
                        context = "\n\n---\n\n".join(context_sections)
                        
                        # Generate enhanced response with context awareness
                        print(f"🤖 Generating context-aware enhancement response...")
                        response = chain.run(context=context, question=query)
                        
                        print(f"📤 Enhancement response preview: {response[:200]}...")
                        
                        # Store enhancement with better key naming
                        query_key = f"insight_{i}_{query.split()[1:4]}"  # Use first few words
                        query_key = re.sub(r'[^\w]', '_', query_key.lower())
                        
                        rag_enhancements[query_key] = {
                            "query": query,
                            "response": response[:800],  # Increased length for detailed insights
                            "relevant_docs_count": len(relevant_docs)
                        }
                        
                        print(f"✅ Successfully processed context-aware query {i}")
                        
                    else:
                        print(f"⚠️ No relevant documents found for query {i}")
                        
                except Exception as e:
                    print(f"❌ Error processing enhancement query {i}: {str(e)}")
                    logger.warning(f"Error processing enhancement query '{query}': {str(e)}")
                    continue
            
            if rag_enhancements:
                enhanced_insights["contextual_rag_analysis"] = rag_enhancements
                print(f"✅ Added {len(rag_enhancements)} context-aware RAG enhancements")
                print_insights_section("CONTEXT-AWARE RAG ENHANCEMENTS", rag_enhancements)
            else:
                print("⚠️ No context-aware RAG enhancements were generated")
            
            return enhanced_insights
            
        except Exception as e:
            print(f"❌ Error enhancing with context-aware RAG queries: {str(e)}")
            logger.error(f"Error enhancing with context-aware RAG queries: {str(e)}")
            return insights
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama service is available"""
        try:
            print("🏥 Checking Ollama health...")
            health_url = f"{self.ollama_base_url}/api/version"
            response = requests.get(health_url, timeout=5)
            is_healthy = response.status_code == 200
            if is_healthy:
                print("✅ Ollama service is healthy")
            else:
                print(f"❌ Ollama service unhealthy: HTTP {response.status_code}")
            return is_healthy
        except Exception as e:
            print(f"❌ Ollama health check failed: {str(e)}")
            return False
    
    def post(self, request, *args, **kwargs):
        """Main API endpoint"""
        print_terminal_separator("🎯 RAG FEEDBACK ANALYSIS REQUEST")
        logger.info("Received LangChain RAG feedback analysis request")
        
        event_id = request.data.get('event_id')
        print(f"📋 Event ID: {event_id}")
        
        if not event_id:
            print("❌ Request missing required 'event_id' parameter")
            logger.error("Request missing required 'event_id' parameter")
            return Response({"error": "Event ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Check Ollama health
            if not self._check_ollama_health():
                print("❌ Ollama service is unavailable")
                return Response({
                    "error": "Ollama service unavailable",
                    "details": f"Cannot connect to {self.ollama_base_url}"
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Get event
            print(f"🔍 Fetching event with ID: {event_id}")
            event = Event.objects.get(id=event_id)
            print(f"✅ Found event: {event.name}")
            
            # Permission check
            if not request.user.is_staff:
                print("❌ Permission denied - user is not staff")
                return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
            print("✅ Permission check passed")
            
            # Check worksheet URL
            if not event.worksheet_url:
                print("❌ Event has no associated worksheet URL")
                return Response({"error": "Event has no associated worksheet URL"}, 
                              status=status.HTTP_400_BAD_REQUEST)
            print(f"✅ Worksheet URL found: {event.worksheet_url}")
            
            # Extract data
            df, data = self.extract_data_from_sheet(event.worksheet_url)
            
            if df is None or data is None:
                print("❌ Could not extract data from Google Sheet")
                return Response({
                    "error": "Could not extract data from Google Sheet",
                    "details": "Please ensure the sheet is publicly accessible"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Limit dataset size for Qwen 0.5B
            original_size = len(df)
            if len(df) > 50:
                print(f"⚠️ Large dataset detected ({len(df)} rows). Sampling to 50 rows.")
                logger.warning(f"Large dataset detected ({len(df)} rows). Sampling to 50 rows.")
                df = df.sample(n=50, random_state=42)
                print(f"📊 Dataset size reduced from {original_size} to {len(df)} rows")
            
            # Process with timeout
            print_terminal_separator("⏱️ PROCESSING WITH TIMEOUT")
            print("🚀 Starting processing thread...")
            
            result_queue = queue.Queue()
            
            def process_with_timeout():
                try:
                    print("🔄 Processing thread started")
                    insights = self.process_feedback_with_rag(df)
                    print("✅ Processing thread completed successfully")
                    result_queue.put(("success", insights))
                except Exception as e:
                    print(f"❌ Processing thread failed: {str(e)}")
                    result_queue.put(("error", str(e)))
            
            # Start processing thread
            process_thread = threading.Thread(target=process_with_timeout)
            process_thread.daemon = True
            process_thread.start()
            
            # Wait for results with timeout
            timeout = 120  # 2 minutes for Qwen 0.5B
            print(f"⏳ Waiting for results (timeout: {timeout}s)...")
            process_thread.join(timeout)
            
            if process_thread.is_alive():
                print(f"❌ Processing timed out after {timeout} seconds")
                logger.error(f"Processing timed out after {timeout} seconds")
                return Response({
                    "error": "Processing timed out",
                    "details": "Try with a smaller dataset"
                }, status=status.HTTP_408_REQUEST_TIMEOUT)
            
            # Get results
            if not result_queue.empty():
                status_code, insights = result_queue.get()
                
                if status_code == "error":
                    print(f"❌ Error generating insights: {insights}")
                    return Response({
                        "error": "Error generating insights",
                        "details": insights
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
                print_terminal_separator("🎉 SUCCESS - FINAL RESULTS")
                print(f"✅ Successfully generated LangChain RAG insights for event {event_id}")
                print(f"📊 Event: {event.name}")
                print(f"🤖 Model: {self.model_name}")
                print(f"📝 Processed rows: {len(df)}")
                
                print_insights_section("COMPLETE FINAL INSIGHTS", insights, 2000)
                
                logger.info(f"Successfully generated LangChain RAG insights for event {event_id}")
                
                response_data = {
                    "event_name": event.name,
                    "insights": insights,
                    "method": "LangChain RAG",
                    "model": self.model_name,
                    "processed_rows": len(df)
                }
                
                print("\n📤 Returning response to client")
                return Response(response_data)
            else:
                print("❌ No results generated")
                return Response({
                    "error": "No results generated"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Event.DoesNotExist:
            print(f"❌ Event not found: {event_id}")
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)