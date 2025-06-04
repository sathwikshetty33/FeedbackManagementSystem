
# # import os
# # import pandas as pd
# # import numpy as np
# # from typing import Dict, List, Any, Tuple
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # import requests
# # from io import StringIO
# # import re
# # from collections import Counter
# # import statistics

# # from django.core.mail import send_mail
# # from django.conf import settings
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status
# # from rest_framework.authentication import TokenAuthentication
# # from rest_framework.permissions import IsAuthenticated

# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import OllamaEmbeddings
# # from langchain_community.llms import Ollama
# # from langchain.chains import RetrievalQA
# # from langchain.schema import Document
# # from langchain.prompts import PromptTemplate
# # from langchain.chains.llm import LLMChain
# # from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# # import logging
# # logger = logging.getLogger(__name__)

# # # Configure logging to reduce FAISS GPU warnings
# # logging.getLogger("faiss").setLevel(logging.ERROR)

# # def print_terminal_separator(title):
# #     print("=" * 80)
# #     print(f"    {title}")
# #     print("=" * 80)

# # class FeedbackRAGAnalyzer:
# #     def __init__(self, ollama_base_url, model_name, chunk_size=300, chunk_overlap=30):
# #         self.ollama_base_url = ollama_base_url
# #         self.model_name = model_name
# #         self.chunk_size = chunk_size
# #         self.chunk_overlap = chunk_overlap
        
# #         # Initialize LangChain components with CPU optimization
# #         self.embeddings = OllamaEmbeddings(
# #             base_url=ollama_base_url,
# #             model=model_name,
# #             show_progress=False  # Reduce verbose output
# #         )
        
# #         self.llm = Ollama(
# #             base_url=ollama_base_url,
# #             model=model_name,
# #             temperature=0.1,
# #             num_ctx=2048,  # Context window for CPU efficiency
# #             num_thread=min(4, os.cpu_count()),  # Limit threads for CPU
# #             verbose=False  # Reduce logging
# #         )
        
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=chunk_size,
# #             chunk_overlap=chunk_overlap,
# #             length_function=len,
# #         )

# #     def preprocess_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
# #         """
# #         Preprocess DataFrame by removing irrelevant columns and categorizing remaining ones
# #         """
# #         print("🔄 Preprocessing columns...")
        
# #         # Define irrelevant column patterns (case-insensitive)
# #         irrelevant_patterns = [
# #             r'usn', r'roll.*no', r'student.*id', r'id', r'name', r'email', 
# #             r'phone', r'contact', r'timestamp', r'date', r'time', r'created',
# #             r'updated', r'serial', r'index', r'entry.*id'
# #         ]
        
# #         # Remove irrelevant columns
# #         original_columns = df.columns.tolist()
# #         relevant_columns = []
# #         removed_columns = []
        
# #         for col in df.columns:
# #             is_irrelevant = any(re.search(pattern, col.lower()) for pattern in irrelevant_patterns)
# #             if not is_irrelevant:
# #                 relevant_columns.append(col)
# #             else:
# #                 removed_columns.append(col)
        
# #         print(f"📊 Original columns: {len(original_columns)}")
# #         print(f"✅ Relevant columns: {len(relevant_columns)}")
# #         print(f"❌ Removed columns: {removed_columns}")
        
# #         # Filter DataFrame
# #         processed_df = df[relevant_columns].copy()
        
# #         # Categorize columns by type
# #         column_types = self._categorize_columns(processed_df)
        
# #         return processed_df, column_types

# #     def _categorize_columns(self, df: pd.DataFrame) -> Dict[str, str]:
# #         """
# #         Categorize columns into numerical, categorical, and text types
# #         """
# #         column_types = {}
        
# #         for col in df.columns:
# #             col_data = df[col].dropna()
            
# #             if col_data.empty:
# #                 column_types[col] = 'empty'
# #                 continue
            
# #             # Check if numerical (including ratings)
# #             try:
# #                 numeric_data = pd.to_numeric(col_data, errors='coerce')
# #                 if not numeric_data.isna().all():
# #                     # Check if it's a rating scale (1-5, 1-10, etc.)
# #                     unique_vals = sorted(numeric_data.dropna().unique())
# #                     if len(unique_vals) <= 10 and all(isinstance(x, (int, float)) for x in unique_vals):
# #                         column_types[col] = 'rating'
# #                     else:
# #                         column_types[col] = 'numerical'
# #                     continue
# #             except:
# #                 pass
            
# #             # Check if categorical (limited unique values)
# #             unique_count = col_data.nunique()
# #             total_count = len(col_data)
            
# #             if unique_count <= 20 and unique_count / total_count < 0.5:
# #                 # Check for common categorical patterns
# #                 sample_values = col_data.str.lower().unique()[:10] if hasattr(col_data, 'str') else []
# #                 categorical_keywords = ['excellent', 'good', 'poor', 'bad', 'average', 'satisfied', 'dissatisfied', 'yes', 'no']
                
# #                 if any(any(keyword in str(val) for keyword in categorical_keywords) for val in sample_values):
# #                     column_types[col] = 'categorical'
# #                 elif unique_count <= 10:
# #                     column_types[col] = 'categorical'
# #                 else:
# #                     column_types[col] = 'text'
# #             else:
# #                 column_types[col] = 'text'
        
# #         print(f"📋 Column categorization: {column_types}")
# #         return column_types

# #     def analyze_numerical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
# #         """
# #         Analyze numerical/rating columns
# #         """
# #         data = pd.to_numeric(df[column], errors='coerce').dropna()
        
# #         if data.empty:
# #             return {"error": "No valid numerical data"}
        
# #         analysis = {
# #             'type': 'numerical',
# #             'total_responses': len(data),
# #             'mean': round(data.mean(), 2),
# #             'median': round(data.median(), 2),
# #             'std_dev': round(data.std(), 2),
# #             'min_value': data.min(),
# #             'max_value': data.max(),
# #             'quartiles': {
# #                 'Q1': round(data.quantile(0.25), 2),
# #                 'Q3': round(data.quantile(0.75), 2)
# #             }
# #         }
        
# #         # Rating-specific analysis
# #         unique_vals = sorted(data.unique())
# #         if len(unique_vals) <= 10:
# #             analysis['rating_distribution'] = data.value_counts().sort_index().to_dict()
# #             analysis['mode'] = data.mode().iloc[0] if not data.mode().empty else None
        
# #         return analysis

# #     def analyze_categorical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
# #         """
# #         Analyze categorical columns
# #         """
# #         data = df[column].dropna().astype(str)
        
# #         if data.empty:
# #             return {"error": "No valid categorical data"}
        
# #         value_counts = data.value_counts()
# #         total = len(data)
        
# #         analysis = {
# #             'type': 'categorical',
# #             'total_responses': total,
# #             'unique_categories': len(value_counts),
# #             'distribution': value_counts.to_dict(),
# #             'percentages': (value_counts / total * 100).round(2).to_dict(),
# #             'most_common': value_counts.index[0],
# #             'least_common': value_counts.index[-1]
# #         }
        
# #         return analysis

# #     def analyze_text_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
# #         """
# #         Analyze text columns (reviews, suggestions, comments)
# #         """
# #         data = df[column].dropna().astype(str)
# #         data = data[data.str.len() > 0]  # Remove empty strings
        
# #         if data.empty:
# #             return {"error": "No valid text data"}
        
# #         # Basic text statistics
# #         lengths = data.str.len()
# #         word_counts = data.str.split().str.len()
        
# #         # Combine all text for further analysis
# #         all_text = ' '.join(data.values)
        
# #         analysis = {
# #             'type': 'text',
# #             'total_responses': len(data),
# #             'avg_length': round(lengths.mean(), 2),
# #             'avg_word_count': round(word_counts.mean(), 2),
# #             'min_length': lengths.min(),
# #             'max_length': lengths.max(),
# #             'sample_responses': data.head(3).tolist()
# #         }
        
# #         return analysis, all_text

# #     def generate_column_insights(self, column_name: str, analysis_data: Dict[str, Any], 
# #                                all_text: str = None) -> str:
# #         """
# #         Generate insights for a specific column using RAG - FIXED VERSION
# #         """
# #         print(f"🧠 Generating insights for column: {column_name}")
        
# #         # Create documents for RAG
# #         documents = []
        
# #         if analysis_data.get('type') == 'numerical' or analysis_data.get('type') == 'rating':
# #             # Create document from numerical analysis
# #             doc_content = f"""
# #             Column: {column_name}
# #             Type: {analysis_data['type']}
# #             Total Responses: {analysis_data['total_responses']}
# #             Mean: {analysis_data['mean']}
# #             Median: {analysis_data['median']}
# #             Standard Deviation: {analysis_data['std_dev']}
# #             Range: {analysis_data['min_value']} to {analysis_data['max_value']}
# #             """
            
# #             if 'rating_distribution' in analysis_data:
# #                 doc_content += f"\nRating Distribution: {analysis_data['rating_distribution']}"
            
# #             documents.append(Document(page_content=doc_content, metadata={"column": column_name, "type": "numerical"}))
            
# #         elif analysis_data.get('type') == 'categorical':
# #             # Create document from categorical analysis
# #             doc_content = f"""
# #             Column: {column_name}
# #             Type: Categorical
# #             Total Responses: {analysis_data['total_responses']}
# #             Categories: {analysis_data['unique_categories']}
# #             Distribution: {analysis_data['distribution']}
# #             Most Common: {analysis_data['most_common']}
# #             Least Common: {analysis_data['least_common']}
# #             """
            
# #             documents.append(Document(page_content=doc_content, metadata={"column": column_name, "type": "categorical"}))
            
# #         elif analysis_data.get('type') == 'text' and all_text:
# #             # Split text into chunks for RAG
# #             text_chunks = self.text_splitter.split_text(all_text)
# #             for i, chunk in enumerate(text_chunks):
# #                 documents.append(Document(
# #                     page_content=chunk, 
# #                     metadata={"column": column_name, "type": "text", "chunk": i}
# #                 ))
            
# #             # Add summary document
# #             summary_doc = f"""
# #             Column: {column_name}
# #             Type: Text Analysis
# #             Total Responses: {analysis_data['total_responses']}
# #             Average Length: {analysis_data['avg_length']} characters
# #             Average Word Count: {analysis_data['avg_word_count']} words
# #             """
# #             documents.append(Document(page_content=summary_doc, metadata={"column": column_name, "type": "text_summary"}))
        
# #         if not documents:
# #             return f"Unable to generate insights for {column_name} - insufficient data"
        
# #         # Create vector store with CPU-optimized settings
# #         try:
# #             # Use CPU-optimized FAISS
# #             vectorstore = FAISS.from_documents(
# #                 documents, 
# #                 self.embeddings,
# #                 distance_strategy="COSINE"  # More efficient for CPU
# #             )
            
# #             # FIXED: Create prompt template with correct variable mapping
# #             if analysis_data.get('type') in ['numerical', 'rating']:
# #                 prompt_template = """Use the following context to analyze feedback data for the column '{column_name}'. 
# #                 Based on the numerical/rating data provided, generate comprehensive insights including:
# #                 1. Overall performance summary
# #                 2. Key trends and patterns
# #                 3. Areas of concern (if any)
# #                 4. Recommendations for improvement
# #                 5. Statistical significance of the findings
                
# #                 Context: {context}
                
# #                 Analysis:""".replace('{column_name}', column_name)
                
# #             elif analysis_data.get('type') == 'categorical':
# #                 prompt_template = """Use the following context to analyze feedback data for the column '{column_name}'.
# #                 Based on the categorical data provided, generate comprehensive insights including:
# #                 1. Distribution analysis and what it reveals
# #                 2. Dominant patterns and their implications
# #                 3. Minority responses and their significance
# #                 4. Recommendations based on the categorical trends
# #                 5. Action items for stakeholders
                
# #                 Context: {context}
                
# #                 Analysis:""".replace('{column_name}', column_name)
                
# #             else:  # text
# #                 prompt_template = """Use the following context to analyze textual feedback for the column '{column_name}'.
# #                 Based on the text responses provided, generate comprehensive insights including:
# #                 1. Common themes and sentiments
# #                 2. Positive feedback patterns
# #                 3. Areas of concern and complaints
# #                 4. Suggestions mentioned by respondents
# #                 5. Actionable recommendations for improvement
# #                 6. Priority areas based on frequency of mentions
                
# #                 Context: {context}
                
# #                 Analysis:""".replace('{column_name}', column_name)
            
# #             # FIXED: Create prompt with correct input variables
# #             custom_prompt = PromptTemplate(
# #                 template=prompt_template,
# #                 input_variables=["context"]  # Only context is needed for stuff chain
# #             )
            
# #             # FIXED: Create LLM chain first, then document chain
# #             llm_chain = LLMChain(
# #                 llm=self.llm, 
# #                 prompt=custom_prompt,
# #                 verbose=False
# #             )
            
# #             # FIXED: Create stuff documents chain with proper configuration
# #             stuff_chain = StuffDocumentsChain(
# #                 llm_chain=llm_chain,
# #                 document_variable_name="context",  # This should match the prompt variable
# #                 verbose=False
# #             )
            
# #             # FIXED: Create retrieval QA chain with proper configuration
# #             qa_chain = RetrievalQA(
# #                 combine_documents_chain=stuff_chain,
# #                 retriever=vectorstore.as_retriever(
# #                     search_type="similarity",
# #                     search_kwargs={"k": 3}
# #                 ),
# #                 return_source_documents=False,
# #                 verbose=False
# #             )
            
# #             # Generate insights using invoke method
# #             query = f"Analyze the {column_name} column data and provide comprehensive insights"
# #             try:
# #                 result = qa_chain.invoke({"query": query})
# #                 insights = result.get('result', '') if isinstance(result, dict) else str(result)
# #                 return insights
# #             except Exception as chain_error:
# #                 logger.error(f"Chain invoke error for {column_name}: {str(chain_error)}")
# #                 # Fallback to direct LLM call if chain fails
# #                 return self._fallback_analysis(column_name, analysis_data, documents)
            
# #         except Exception as e:
# #             logger.error(f"Error generating insights for {column_name}: {str(e)}")
# #             # Use fallback method when vector store creation fails
# #             return self._fallback_analysis(column_name, analysis_data, documents)

# #     def _fallback_analysis(self, column_name: str, analysis_data: Dict[str, Any], documents: List[Document]) -> str:
# #         """
# #         Fallback method for direct LLM analysis when chain fails
# #         """
# #         try:
# #             # Create a simple context from documents
# #             context = "\n".join([doc.page_content for doc in documents[:3]])
            
# #             if analysis_data.get('type') in ['numerical', 'rating']:
# #                 prompt = f"""
# #                 Analyze the feedback data for column "{column_name}":
                
# #                 Data Summary:
# #                 - Total responses: {analysis_data.get('total_responses', 0)}
# #                 - Average: {analysis_data.get('mean', 'N/A')}
# #                 - Range: {analysis_data.get('min_value', 'N/A')} to {analysis_data.get('max_value', 'N/A')}
                
# #                 Context: {context}
                
# #                 Provide insights on performance trends, areas of concern, and recommendations.
# #                 """
                
# #             elif analysis_data.get('type') == 'categorical':
# #                 prompt = f"""
# #                 Analyze the feedback data for column "{column_name}":
                
# #                 Data Summary:
# #                 - Total responses: {analysis_data.get('total_responses', 0)}
# #                 - Most common: {analysis_data.get('most_common', 'N/A')}
# #                 - Categories: {analysis_data.get('unique_categories', 0)}
                
# #                 Context: {context}
                
# #                 Provide insights on distribution patterns and recommendations.
# #                 """
                
# #             else:  # text
# #                 prompt = f"""
# #                 Analyze the textual feedback for column "{column_name}":
                
# #                 Data Summary:
# #                 - Total responses: {analysis_data.get('total_responses', 0)}
# #                 - Average length: {analysis_data.get('avg_length', 0)} characters
                
# #                 Sample responses: {context}
                
# #                 Provide insights on common themes, sentiments, and actionable recommendations.
# #                 """
            
# #             # Direct LLM call
# #             response = self.llm.invoke(prompt)
# #             return response if isinstance(response, str) else str(response)
            
# #         except Exception as fallback_error:
# #             logger.error(f"Fallback analysis failed for {column_name}: {str(fallback_error)}")
# #             return self._generate_basic_insights(column_name, analysis_data)

# #     def _generate_basic_insights(self, column_name: str, analysis_data: Dict[str, Any]) -> str:
# #         """
# #         Generate basic statistical insights when AI analysis fails
# #         """
# #         insights = [f"Analysis for {column_name}:"]
        
# #         if analysis_data.get('type') in ['numerical', 'rating']:
# #             mean_val = analysis_data.get('mean', 0)
# #             total = analysis_data.get('total_responses', 0)
            
# #             if analysis_data.get('type') == 'rating':
# #                 if mean_val >= 4:
# #                     insights.append("• Overall rating is excellent, indicating high satisfaction")
# #                 elif mean_val >= 3:
# #                     insights.append("• Rating is good but has room for improvement")
# #                 else:
# #                     insights.append("• Rating indicates areas needing significant attention")
                    
# #                 insights.append(f"• {total} respondents provided ratings")
# #                 if 'rating_distribution' in analysis_data:
# #                     mode = analysis_data.get('mode')
# #                     insights.append(f"• Most common rating: {mode}")
# #             else:
# #                 insights.append(f"• Average value: {mean_val}")
# #                 insights.append(f"• Range: {analysis_data.get('min_value')} to {analysis_data.get('max_value')}")
                
# #         elif analysis_data.get('type') == 'categorical':
# #             most_common = analysis_data.get('most_common', '')
# #             total = analysis_data.get('total_responses', 0)
# #             categories = analysis_data.get('unique_categories', 0)
            
# #             insights.append(f"• {total} responses across {categories} categories")
# #             insights.append(f"• Most common response: '{most_common}'")
            
# #             if 'distribution' in analysis_data:
# #                 dist = analysis_data['distribution']
# #                 if most_common and most_common in dist:
# #                     percentage = (dist[most_common] / total) * 100
# #                     insights.append(f"• '{most_common}' represents {percentage:.1f}% of responses")
                    
# #         elif analysis_data.get('type') == 'text':
# #             total = analysis_data.get('total_responses', 0)
# #             avg_length = analysis_data.get('avg_length', 0)
            
# #             insights.append(f"• {total} text responses received")
# #             insights.append(f"• Average response length: {avg_length:.1f} characters")
            
# #             if avg_length > 50:
# #                 insights.append("• Responses are detailed, indicating engaged participants")
# #             elif avg_length > 20:
# #                 insights.append("• Responses are moderately detailed")
# #             else:
# #                 insights.append("• Responses are brief, consider encouraging more detailed feedback")
        
# #         return "\n".join(insights)

# #     def analyze_all_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
# #         """
# #         Analyze all relevant columns
# #         """
# #         results = {}
        
# #         for column, col_type in column_types.items():
# #             print(f"📊 Analyzing column: {column} (type: {col_type})")
            
# #             try:
# #                 if col_type in ['numerical', 'rating']:
# #                     analysis = self.analyze_numerical_column(df, column)
# #                     insights = self.generate_column_insights(column, analysis)
# #                     results[column] = {
# #                         'analysis': analysis,
# #                         'insights': insights,
# #                         'type': col_type
# #                     }
                    
# #                 elif col_type == 'categorical':
# #                     analysis = self.analyze_categorical_column(df, column)
# #                     insights = self.generate_column_insights(column, analysis)
# #                     results[column] = {
# #                         'analysis': analysis,
# #                         'insights': insights,
# #                         'type': col_type
# #                     }
                    
# #                 elif col_type == 'text':
# #                     analysis, all_text = self.analyze_text_column(df, column)
# #                     insights = self.generate_column_insights(column, analysis, all_text)
# #                     results[column] = {
# #                         'analysis': analysis,
# #                         'insights': insights,
# #                         'type': col_type
# #                     }
                    
# #             except Exception as e:
# #                 logger.error(f"Error analyzing column {column}: {str(e)}")
# #                 results[column] = {
# #                     'error': str(e),
# #                     'type': col_type
# #                 }
        
# #         return results

# # class LangChainRAGInsightsView(APIView):
# #     authentication_classes = [TokenAuthentication]
# #     permission_classes = [IsAuthenticated]
    
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         # Configuration from environment
# #         self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
# #         self.model_name = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
# #         self.chunk_size = int(os.environ.get('RAG_CHUNK_SIZE', '300'))
# #         self.chunk_overlap = int(os.environ.get('RAG_CHUNK_OVERLAP', '30'))
# #         self.max_tokens = int(os.environ.get('MAX_TOKENS', '256'))
# #         self.temperature = float(os.environ.get('TEMPERATURE', '0.1'))
# #         self.request_timeout = int(os.environ.get('OLLAMA_TIMEOUT', '60'))
        
# #         # Performance optimizations
# #         self.max_processing_rows = int(os.environ.get('MAX_PROCESSING_ROWS', '100'))
# #         self.enable_parallel = os.environ.get('ENABLE_PARALLEL', 'true').lower() == 'true'
# #         self.max_workers = min(2, os.cpu_count())

# #     def fetch_worksheet_data(self, worksheet_url: str) -> pd.DataFrame:
# #         """
# #         Fetch data from Google Sheets URL
# #         """
# #         try:
# #             # Convert Google Sheets URL to CSV export URL
# #             if 'docs.google.com/spreadsheets' in worksheet_url:
# #                 sheet_id = worksheet_url.split('/d/')[1].split('/')[0]
# #                 csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
# #             else:
# #                 csv_url = worksheet_url
            
# #             print(f"📥 Fetching data from: {csv_url}")
# #             response = requests.get(csv_url, timeout=30)
# #             response.raise_for_status()
            
# #             # Read CSV data
# #             df = pd.read_csv(StringIO(response.text))
# #             print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
# #             return df
            
# #         except Exception as e:
# #             logger.error(f"Error fetching worksheet data: {str(e)}")
# #             raise Exception(f"Failed to fetch worksheet data: {str(e)}")

# #     def generate_summary_report(self, results: Dict[str, Any]) -> str:
# #         """
# #         Generate an executive summary of all column analyses
# #         """
# #         summary_parts = []
# #         summary_parts.append("# FEEDBACK ANALYSIS REPORT")
# #         summary_parts.append("=" * 50)
# #         summary_parts.append("")
        
# #         # Count analysis by type
# #         type_counts = {}
# #         for col, data in results.items():
# #             col_type = data.get('type', 'unknown')
# #             type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
# #         summary_parts.append(f"## OVERVIEW")
# #         summary_parts.append(f"Total Columns Analyzed: {len(results)}")
# #         for col_type, count in type_counts.items():
# #             summary_parts.append(f"- {col_type.title()} Columns: {count}")
# #         summary_parts.append("")
        
# #         # Individual column summaries
# #         for column, data in results.items():
# #             if 'error' in data:
# #                 continue
                
# #             summary_parts.append(f"## {column.upper()}")
# #             summary_parts.append(f"**Type:** {data['type'].title()}")
            
# #             if data['type'] in ['numerical', 'rating']:
# #                 analysis = data['analysis']
# #                 summary_parts.append(f"**Responses:** {analysis['total_responses']}")
# #                 summary_parts.append(f"**Average:** {analysis['mean']}")
# #                 if 'rating_distribution' in analysis:
# #                     summary_parts.append(f"**Most Common Rating:** {analysis.get('mode', 'N/A')}")
            
# #             elif data['type'] == 'categorical':
# #                 analysis = data['analysis']
# #                 summary_parts.append(f"**Responses:** {analysis['total_responses']}")
# #                 summary_parts.append(f"**Most Common:** {analysis['most_common']}")
# #                 summary_parts.append(f"**Categories:** {analysis['unique_categories']}")
            
# #             elif data['type'] == 'text':
# #                 analysis = data['analysis']
# #                 summary_parts.append(f"**Responses:** {analysis['total_responses']}")
# #                 summary_parts.append(f"**Avg Length:** {analysis['avg_length']} characters")
            
# #             summary_parts.append("**Key Insights:**")
# #             summary_parts.append(data['insights'])
# #             summary_parts.append("")
# #             summary_parts.append("-" * 40)
# #             summary_parts.append("")
        
# #         return "\n".join(summary_parts)

# #     def send_analysis_email(self, recipient_email: str, report: str, event_id: str):
# #         """
# #         Send analysis report via email
# #         """
# #         try:
# #             subject = f"Feedback Analysis Report - Event {event_id}"
# #             message = f"""
# #             Dear User,
            
# #             Please find below the comprehensive feedback analysis report for Event {event_id}.
            
# #             {report}
            
# #             Best regards,
# #             Feedback Analysis System
# #             """
            
# #             send_mail(
# #                 subject=subject,
# #                 message=message,
# #                 from_email=settings.DEFAULT_FROM_EMAIL,
# #                 recipient_list=[recipient_email],
# #                 fail_silently=False,
# #             )
            
# #             print(f"✅ Analysis report sent to {recipient_email}")
# #             return True
            
# #         except Exception as e:
# #             logger.error(f"Error sending email: {str(e)}")
# #             return False

# #     def post(self, request, *args, **kwargs):
# #         """Main API endpoint"""
# #         print_terminal_separator("🎯 RAG FEEDBACK ANALYSIS REQUEST")
# #         logger.info("Received LangChain RAG feedback analysis request")
        
# #         event_id = request.data.get('event_id')
# #         recipient_email = request.data.get('recipient_email', 'sathwikshetty9876@gmail.com')
        
# #         print(f"📋 Event ID: {event_id}")
# #         print(f"📧 Recipient Email: {recipient_email}")
        
# #         if not event_id:
# #             print("❌ Request missing required 'event_id' parameter")
# #             logger.error("Request missing required 'event_id' parameter")
# #             return Response({"error": "Event ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        
# #         try:
# #             # Get event and worksheet URL (assuming you have an Event model)
# #             from home.models import Event  # Adjust import based on your models
# #             event = Event.objects.get(id=event_id)
# #             worksheet_url = event.worksheet_url
            
# #             if not worksheet_url:
# #                 return Response({"error": "No worksheet URL found for this event"}, 
# #                               status=status.HTTP_400_BAD_REQUEST)
            
# #             # Initialize RAG analyzer
# #             analyzer = FeedbackRAGAnalyzer(
# #                 ollama_base_url=self.ollama_base_url,
# #                 model_name=self.model_name,
# #                 chunk_size=self.chunk_size,
# #                 chunk_overlap=self.chunk_overlap
# #             )
            
# #             # Fetch and preprocess data
# #             print("📊 Starting data analysis...")
# #             df = self.fetch_worksheet_data(worksheet_url)
            
# #             # Limit rows for performance
# #             if len(df) > self.max_processing_rows:
# #                 print(f"⚠️ Limiting analysis to {self.max_processing_rows} rows for performance")
# #                 df = df.head(self.max_processing_rows)
            
# #             # Preprocess columns
# #             processed_df, column_types = analyzer.preprocess_columns(df)
            
# #             if processed_df.empty:
# #                 return Response({"error": "No relevant columns found for analysis"}, 
# #                               status=status.HTTP_400_BAD_REQUEST)
            
# #             # Analyze all columns
# #             print("🔍 Performing column-wise analysis...")
# #             results = analyzer.analyze_all_columns(processed_df, column_types)
            
# #             # Generate summary report
# #             summary_report = self.generate_summary_report(results)
            
# #             # Send email
# #             email_sent = self.send_analysis_email(recipient_email, summary_report, event_id)
            
# #             print_terminal_separator("✅ ANALYSIS COMPLETE")
            
# #             return Response({
# #                 "status": "success",
# #                 "message": "Feedback analysis completed successfully",
# #                 "event_id": event_id,
# #                 "columns_analyzed": len(results),
# #                 "email_sent": email_sent,
# #                 "summary": summary_report[:500] + "..." if len(summary_report) > 500 else summary_report
# #             }, status=status.HTTP_200_OK)
            
# #         except Event.DoesNotExist:
# #             return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
# #         except Exception as e:
# #             logger.error(f"Error in feedback analysis: {str(e)}")
# #             print(f"❌ Analysis failed: {str(e)}")
# #             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# import os
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Any, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import requests
# from io import StringIO
# import re
# from collections import Counter
# import statistics

# from django.core.mail import send_mail
# from django.conf import settings
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.authentication import TokenAuthentication
# from rest_framework.permissions import IsAuthenticated

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# import logging
# logger = logging.getLogger(__name__)

# logging.getLogger("faiss").setLevel(logging.ERROR)

# def print_terminal_separator(title):
#     print("=" * 80)
#     print(f"    {title}")
#     print("=" * 80)

# class LangChainRAGInsightsView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
#         self.model_name = os.environ.get('OLLAMA_MODEL', 'llama3.2:1b')
#         self.chunk_size = int(os.environ.get('RAG_CHUNK_SIZE', '300'))
#         self.chunk_overlap = int(os.environ.get('RAG_CHUNK_OVERLAP', '30'))
#         self.max_tokens = int(os.environ.get('MAX_TOKENS', '256'))
#         self.temperature = float(os.environ.get('TEMPERATURE', '0.1'))
#         self.request_timeout = int(os.environ.get('OLLAMA_TIMEOUT', '60'))
#         self.max_processing_rows = int(os.environ.get('MAX_PROCESSING_ROWS', '100'))
#         self.enable_parallel = os.environ.get('ENABLE_PARALLEL', 'true').lower() == 'true'
#         self.max_workers = min(2, os.cpu_count())

#     def fetch_worksheet_data(self, worksheet_url: str) -> pd.DataFrame:
#         try:
#             if 'docs.google.com/spreadsheets' in worksheet_url:
#                 sheet_id = worksheet_url.split('/d/')[1].split('/')[0]
#                 csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
#             else:
#                 csv_url = worksheet_url
            
#             print(f"📥 Fetching data from: {csv_url}")
#             response = requests.get(csv_url, timeout=30)
#             response.raise_for_status()
            
#             df = pd.read_csv(StringIO(response.text))
#             print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
#             return df
            
#         except Exception as e:
#             logger.error(f"Error fetching worksheet data: {str(e)}")
#             raise Exception(f"Failed to fetch worksheet data: {str(e)}")

    
    
#     def send_analysis_email(self, recipient_email: str, report: str, event_id: str):
#         try:
#             subject = f"📊 Feedback Analysis Report - Event {event_id}"
            
#             send_mail(
#                 subject=subject,
#                 message="Please enable HTML to view the full report.",
#                 from_email=settings.DEFAULT_FROM_EMAIL,
#                 recipient_list=[recipient_email],
#                 fail_silently=False,
#                 html_message=report
#             )
            
#             print(f"✅ Analysis report sent to {recipient_email}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error sending email: {str(e)}")
#             return False
#     def post(self, request, *args, **kwargs):
#         print_terminal_separator("🎯 RAG FEEDBACK ANALYSIS REQUEST")
#         logger.info("Received LangChain RAG feedback analysis request")
        
#         event_id = request.data.get('event_id')
#         recipient_email = request.data.get('recipient_email', 'sathwikshetty9876@gmail.com')
        
#         print(f"📋 Event ID: {event_id}")
#         print(f"📧 Recipient Email: {recipient_email}")
        
#         if not event_id:
#             print("❌ Request missing required 'event_id' parameter")
#             logger.error("Request missing required 'event_id' parameter")
#             return Response({"error": "Event ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        
#         try:
#             from home.models import Event
#             event = Event.objects.get(id=event_id)
#             worksheet_url = event.worksheet_url
            
#             if not worksheet_url:
#                 return Response({"error": "No worksheet URL found for this event"}, 
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             analyzer = FeedbackRAGAnalyzer(
#                 ollama_base_url=self.ollama_base_url,
#                 model_name=self.model_name,
#                 chunk_size=self.chunk_size,
#                 chunk_overlap=self.chunk_overlap
#             )
            
#             print("📊 Starting data analysis...")
#             df = self.fetch_worksheet_data(worksheet_url)
            
#             if len(df) > self.max_processing_rows:
#                 print(f"⚠️ Limiting analysis to {self.max_processing_rows} rows for performance")
#                 df = df.head(self.max_processing_rows)
            
#             processed_df, column_types = analyzer.preprocess_columns(df)
            
#             if processed_df.empty:
#                 return Response({"error": "No relevant columns found for analysis"}, 
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             print("🔍 Performing column-wise analysis...")
#             results = analyzer.analyze_all_columns(processed_df, column_types)
            
#             summary_report = self.generate_summary_report(results)
#             email_sent = self.send_analysis_email(recipient_email, summary_report, event_id)
            
#             print_terminal_separator("✅ ANALYSIS COMPLETE")
            
#             return Response({
#                 "status": "success",
#                 "message": "Feedback analysis completed successfully",
#                 "event_id": event_id,
#                 "columns_analyzed": len(results),
#                 "email_sent": email_sent,
#                 "summary": summary_report[:500] + "..." if len(summary_report) > 500 else summary_report
#             }, status=status.HTTP_200_OK)
            
#         except Event.DoesNotExist:
#             return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             logger.error(f"Error in feedback analysis: {str(e)}")
#             print(f"❌ Analysis failed: {str(e)}")
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
# Simplified Django View (replace your existing LangChainRAGInsightsView)
import os
import requests
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

logger = logging.getLogger(__name__)
def print_terminal_separator(title):
    print("=" * 80)
    print(f"    {title}")
    print("=" * 80)
class LangChainRAGInsightsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FastAPI service URL
        self.fastapi_url = os.environ.get('FASTAPI_ANALYSIS_URL', 'http://localhost:8001')
    
    def post(self, request, *args, **kwargs):
        print_terminal_separator("🎯 RAG FEEDBACK ANALYSIS REQUEST")
        logger.info("Received LangChain RAG feedback analysis request")
        
        event_id = request.data.get('event_id')
        recipient_email = request.user.email
        
        print(f"📋 Event ID: {event_id}")
        print(f"📧 Recipient Email: {recipient_email}")
        
        if not event_id:
            print("❌ Request missing required 'event_id' parameter")
            logger.error("Request missing required 'event_id' parameter")
            return Response({"error": "Event ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            from home.models import Event
            event = Event.objects.get(id=event_id)
            worksheet_url = event.worksheet_url
            
            if not worksheet_url:
                return Response({"error": "No worksheet URL found for this event"}, 
                              status=status.HTTP_400_BAD_REQUEST)
            
            # Call FastAPI service
            fastapi_payload = {
                "event_name": str(event.name),  # Ensure it's a string
                "worksheet_url": worksheet_url,
                "recipient_email": recipient_email
            }
            
            print(f"🚀 Sending to FastAPI: {fastapi_payload}")
            
            try:
                response = requests.post(
                    f"{self.fastapi_url}/analyze",
                    json=fastapi_payload,
                    timeout=5,  # Short timeout since we don't wait for completion
                    headers={"Content-Type": "application/json"}
                )
                
                print(f"📡 FastAPI Response Status: {response.status_code}")
                print(f"📡 FastAPI Response Body: {response.text}")
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Analysis task successfully queued")
                    
                    return Response({
                        "status": "success",
                        "message": "Feedback analysis started. You will receive an email when complete.",
                        "event_id": event_id,
                        "task_id": result.get("task_id")
                    }, status=status.HTTP_200_OK)
                else:
                    logger.error(f"FastAPI service error: {response.status_code} - {response.text}")
                    return Response({
                        "error": f"Analysis service error: {response.text}"
                    }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                    
            except requests.exceptions.Timeout:
                logger.error("FastAPI service timeout")
                return Response({
                    "error": "Analysis service is busy. Please try again later."
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                
            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to FastAPI service")
                return Response({
                    "error": "Analysis service is unavailable"
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error in feedback analysis request: {str(e)}")
            print(f"❌ Analysis request failed: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)