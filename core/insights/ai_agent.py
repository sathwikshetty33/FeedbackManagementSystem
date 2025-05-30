import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import re
from datetime import datetime
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
import json

logger = logging.getLogger(__name__)

class FeedbackColumnPreprocessor:
    """Handles preprocessing and categorization of feedback columns"""
    
    def __init__(self):
        # Define patterns for irrelevant columns
        self.irrelevant_patterns = [
            r'.*\b(usn|id|user.*id|student.*id|emp.*id|employee.*id)\b.*',
            r'.*\b(name|first.*name|last.*name|full.*name)\b.*',
            r'.*\b(email|mail|e-mail)\b.*',
            r'.*\b(phone|mobile|contact|tel)\b.*',
            r'.*\b(date|time|timestamp|created|updated)\b.*',
            r'.*\b(ip|address|location)\b.*',
            r'.*\b(session|token|key|hash)\b.*',
        ]
        
        # Define patterns for relevant feedback columns
        self.rating_patterns = [
            r'.*\b(rating|rate|score|stars?)\b.*',
            r'.*\b(satisfaction|quality|performance)\b.*',
            r'.*\b(scale|level)\b.*'
        ]
        
        self.categorical_patterns = [
            r'.*\b(experience|feedback|opinion|choice)\b.*',
            r'.*\b(recommend|suggestion|preference)\b.*',
            r'.*\b(category|type|kind)\b.*'
        ]
        
        self.text_patterns = [
            r'.*\b(comment|review|feedback|suggestion|remark)\b.*',
            r'.*\b(describe|explain|elaborate|detail)\b.*',
            r'.*\b(improvement|complaint|praise)\b.*'
        ]
    
    def is_irrelevant_column(self, column_name: str) -> bool:
        """Check if a column is irrelevant for feedback analysis"""
        column_lower = column_name.lower().strip()
        return any(re.match(pattern, column_lower, re.IGNORECASE) 
                  for pattern in self.irrelevant_patterns)
    
    def categorize_column(self, column_name: str, column_data: pd.Series) -> str:
        """Categorize column into numerical, categorical, or textual"""
        column_lower = column_name.lower().strip()
        
        # Check if it's a rating/numerical column
        if any(re.match(pattern, column_lower, re.IGNORECASE) 
               for pattern in self.rating_patterns):
            return 'numerical'
        
        # Check data type and content
        if pd.api.types.is_numeric_dtype(column_data):
            unique_values = column_data.nunique()
            if unique_values <= 10:  # Likely a rating scale
                return 'numerical'
            elif unique_values > len(column_data) * 0.8:  # Too many unique values
                return 'irrelevant'
        
        # Check for categorical patterns
        if any(re.match(pattern, column_lower, re.IGNORECASE) 
               for pattern in self.categorical_patterns):
            return 'categorical'
        
        # Check for text patterns
        if any(re.match(pattern, column_lower, re.IGNORECASE) 
               for pattern in self.text_patterns):
            return 'textual'
        
        # Analyze content
        non_null_data = column_data.dropna()
        if len(non_null_data) == 0:
            return 'irrelevant'
        
        # Check average text length
        avg_length = non_null_data.astype(str).str.len().mean()
        unique_ratio = non_null_data.nunique() / len(non_null_data)
        
        if avg_length > 50:  # Long text responses
            return 'textual'
        elif unique_ratio < 0.3:  # Limited unique values
            return 'categorical'
        elif pd.api.types.is_numeric_dtype(column_data) or self._is_numeric_like(non_null_data):
            return 'numerical'
        else:
            return 'categorical'
    
    def _is_numeric_like(self, data: pd.Series) -> bool:
        """Check if data contains numeric-like values"""
        try:
            pd.to_numeric(data.astype(str))
            return True
        except (ValueError, TypeError):
            return False
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess dataframe and categorize columns"""
        result = {
            'relevant_columns': {},
            'irrelevant_columns': [],
            'column_categories': {},
            'preprocessing_summary': {}
        }
        
        for column in df.columns:
            if self.is_irrelevant_column(column):
                result['irrelevant_columns'].append(column)
                continue
            
            category = self.categorize_column(column, df[column])
            
            if category == 'irrelevant':
                result['irrelevant_columns'].append(column)
                continue
            
            result['relevant_columns'][column] = category
            result['column_categories'][category] = result['column_categories'].get(category, [])
            result['column_categories'][category].append(column)
        
        # Generate preprocessing summary
        result['preprocessing_summary'] = {
            'total_columns': len(df.columns),
            'relevant_columns': len(result['relevant_columns']),
            'irrelevant_columns': len(result['irrelevant_columns']),
            'numerical_columns': len(result['column_categories'].get('numerical', [])),
            'categorical_columns': len(result['column_categories'].get('categorical', [])),
            'textual_columns': len(result['column_categories'].get('textual', []))
        }
        
        return result


class ColumnWiseRAGAnalyzer:
    """Performs column-wise RAG analysis for feedback data"""
    
    def __init__(self, ollama_base_url: str, model_name: str, chunk_size: int = 300, 
                 chunk_overlap: int = 30, max_tokens: int = 256, temperature: float = 0.1):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize LangChain components
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=ollama_base_url
        )
        
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def analyze_numerical_column(self, column_name: str, data: pd.Series) -> Dict[str, Any]:
        """Analyze numerical/rating columns"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {"error": "No valid data for analysis"}
        
        # Statistical analysis
        stats = {
            'count': len(clean_data),
            'mean': float(clean_data.mean()),
            'median': float(clean_data.median()),
            'std': float(clean_data.std()),
            'min': float(clean_data.min()),
            'max': float(clean_data.max()),
            'distribution': dict(clean_data.value_counts().sort_index())
        }
        
        # Create context for RAG
        context = f"""
        Numerical Analysis for {column_name}:
        - Total responses: {stats['count']}
        - Average score: {stats['mean']:.2f}
        - Median score: {stats['median']:.2f}
        - Standard deviation: {stats['std']:.2f}
        - Score range: {stats['min']:.1f} to {stats['max']:.1f}
        - Score distribution: {json.dumps(stats['distribution'], indent=2)}
        
        This appears to be feedback data where higher scores typically indicate better ratings.
        """
        
        # Generate insights using RAG
        prompt = f"""
        Based on the numerical feedback data for '{column_name}', provide detailed insights including:
        1. Overall performance assessment
        2. Distribution analysis and patterns
        3. Areas of concern (if any)
        4. Recommendations for improvement
        5. Key takeaways for stakeholders
        
        Context: {context}
        """
        
        insights = self._generate_insights(prompt, context)
        
        return {
            'column_name': column_name,
            'column_type': 'numerical',
            'statistics': stats,
            'insights': insights,
            'analysis_type': 'statistical_and_rag'
        }
    
    def analyze_categorical_column(self, column_name: str, data: pd.Series) -> Dict[str, Any]:
        """Analyze categorical columns"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {"error": "No valid data for analysis"}
        
        # Categorical analysis
        value_counts = clean_data.value_counts()
        percentages = (value_counts / len(clean_data) * 100).round(2)
        
        analysis = {
            'count': len(clean_data),
            'unique_values': clean_data.nunique(),
            'value_distribution': dict(value_counts),
            'percentage_distribution': dict(percentages),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None
        }
        
        # Create context for RAG
        context = f"""
        Categorical Analysis for {column_name}:
        - Total responses: {analysis['count']}
        - Unique categories: {analysis['unique_values']}
        - Distribution: {json.dumps(analysis['value_distribution'], indent=2)}
        - Percentages: {json.dumps(analysis['percentage_distribution'], indent=2)}
        - Most common response: {analysis['most_common']}
        - Least common response: {analysis['least_common']}
        
        This represents categorical feedback where each category represents a different user choice or opinion.
        """
        
        # Generate insights using RAG
        prompt = f"""
        Based on the categorical feedback data for '{column_name}', provide detailed insights including:
        1. Dominant patterns and trends
        2. Distribution analysis
        3. What the categories reveal about user sentiment
        4. Potential areas for investigation
        5. Strategic recommendations based on the distribution
        
        Context: {context}
        """
        
        insights = self._generate_insights(prompt, context)
        
        return {
            'column_name': column_name,
            'column_type': 'categorical',
            'analysis': analysis,
            'insights': insights,
            'analysis_type': 'distribution_and_rag'
        }
    
    def analyze_textual_column(self, column_name: str, data: pd.Series) -> Dict[str, Any]:
        """Analyze textual columns using RAG"""
        clean_data = data.dropna().astype(str)
        clean_data = clean_data[clean_data.str.strip() != '']
        
        if len(clean_data) == 0:
            return {"error": "No valid text data for analysis"}
        
        # Basic text statistics
        text_stats = {
            'count': len(clean_data),
            'avg_length': clean_data.str.len().mean(),
            'total_words': clean_data.str.split().str.len().sum(),
            'avg_words_per_response': clean_data.str.split().str.len().mean()
        }
        
        # Prepare documents for RAG
        documents = []
        for idx, text in enumerate(clean_data):
            if len(str(text).strip()) > 10:  # Only include substantial text
                doc = Document(
                    page_content=str(text),
                    metadata={'response_id': idx, 'column': column_name}
                )
                documents.append(doc)
        
        if not documents:
            return {"error": "No substantial text content for analysis"}
        
        # Create vector store
        try:
            # Split documents if they're too long
            split_docs = self.text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            
            # Generate comprehensive analysis
            analysis_prompt = f"""
            Analyze all the textual feedback for '{column_name}'. Provide insights on:
            1. Common themes and topics mentioned
            2. Overall sentiment (positive, negative, neutral)
            3. Frequently mentioned issues or concerns
            4. Suggestions and recommendations from users
            5. Areas that need immediate attention
            6. Positive aspects that should be maintained
            7. Key actionable insights for improvement
            
            Focus on patterns across all responses rather than individual comments.
            """
            
            insights = qa_chain.run(analysis_prompt)
            
            return {
                'column_name': column_name,
                'column_type': 'textual',
                'text_statistics': text_stats,
                'insights': insights,
                'analysis_type': 'rag_based_text_analysis',
                'documents_processed': len(split_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in textual analysis for {column_name}: {str(e)}")
            return {
                'column_name': column_name,
                'column_type': 'textual',
                'text_statistics': text_stats,
                'error': f"RAG analysis failed: {str(e)}",
                'analysis_type': 'basic_statistics_only'
            }
    
    def _generate_insights(self, prompt: str, context: str) -> str:
        """Generate insights using the LLM"""
        try:
            full_prompt = f"{context}\n\nQuestion: {prompt}\n\nAnswer:"
            response = self.llm(full_prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return f"Unable to generate detailed insights due to: {str(e)}"
    
    def analyze_all_columns(self, df: pd.DataFrame, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform column-wise analysis for all relevant columns"""
        results = {
            'preprocessing_info': column_info,
            'column_analyses': {},
            'summary': {
                'total_analyzed': 0,
                'successful_analyses': 0,
                'failed_analyses': 0
            }
        }
        
        relevant_columns = column_info['relevant_columns']
        
        for column_name, column_type in relevant_columns.items():
            try:
                print(f"🔍 Analyzing {column_type} column: {column_name}")
                
                if column_type == 'numerical':
                    analysis = self.analyze_numerical_column(column_name, df[column_name])
                elif column_type == 'categorical':
                    analysis = self.analyze_categorical_column(column_name, df[column_name])
                elif column_type == 'textual':
                    analysis = self.analyze_textual_column(column_name, df[column_name])
                else:
                    analysis = {"error": f"Unknown column type: {column_type}"}
                
                results['column_analyses'][column_name] = analysis
                results['summary']['total_analyzed'] += 1
                
                if 'error' not in analysis:
                    results['summary']['successful_analyses'] += 1
                else:
                    results['summary']['failed_analyses'] += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing column {column_name}: {str(e)}")
                results['column_analyses'][column_name] = {
                    'column_name': column_name,
                    'error': str(e)
                }
                results['summary']['failed_analyses'] += 1
        
        return results


# Integration function for your Django view
def process_feedback_data(df: pd.DataFrame, ollama_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to process feedback data with column-wise RAG analysis
    
    Args:
        df: DataFrame containing feedback data
        ollama_config: Configuration dictionary with Ollama settings
    
    Returns:
        Dictionary containing complete analysis results
    """
    try:
        # Step 1: Preprocess and categorize columns
        preprocessor = FeedbackColumnPreprocessor()
        column_info = preprocessor.preprocess_dataframe(df)
        
        print(f"📊 Preprocessing Summary:")
        print(f"   Total columns: {column_info['preprocessing_summary']['total_columns']}")
        print(f"   Relevant columns: {column_info['preprocessing_summary']['relevant_columns']}")
        print(f"   Irrelevant columns: {column_info['preprocessing_summary']['irrelevant_columns']}")
        
        # Step 2: Initialize RAG analyzer
        analyzer = ColumnWiseRAGAnalyzer(
            ollama_base_url=ollama_config['base_url'],
            model_name=ollama_config['model_name'],
            chunk_size=ollama_config.get('chunk_size', 300),
            chunk_overlap=ollama_config.get('chunk_overlap', 30),
            max_tokens=ollama_config.get('max_tokens', 256),
            temperature=ollama_config.get('temperature', 0.1)
        )
        
        # Step 3: Perform column-wise analysis
        analysis_results = analyzer.analyze_all_columns(df, column_info)
        
        return {
            'status': 'success',
            'results': analysis_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in feedback processing: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }