from .OllamaCategoricalAnalyzer import *
from .configs import *
from typing import Dict, List, Any, Tuple
from cachetools import TTLCache
import pandas as pd
from .prompts import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .utils import *
import re
from .logger import *
from .models import *
from .baseAnalyzer import *
from .BaseNumericAgent import *
from .OllamaNumericAgent import *

logging = get_logger(__name__)
cache = TTLCache(maxsize=100, ttl=1800)  # 30 min


class OllamaRAGAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.config = OllamaConfig()
        self.embeddings = OllamaEmbeddings(
            base_url=self.config.BASE_URL,
            model=self.config.MODEL,
            show_progress=False
        )
        
        self.llm = Ollama(
            base_url=self.config.BASE_URL,
            model=self.config.MODEL,
            temperature=self.config.TEMPERATURE,
            num_ctx=self.config.NUM_CTX,
            num_thread=self.config.NUM_THREAD,
            verbose=False
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.RAG_CHUNK_SIZE,
            chunk_overlap=self.config.RAG_CHUNK_OVERLAP,
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
            r'^organization$', r'^institution$', r'^university$', r'^college$',
            r'^department$', r'^branch$',r'^batch$', r'^section$',
            
            # Date/Time patterns
            r'^timestamp$', r'^date$', r'^time$', r'^created$', r'^updated$', r'^submitted$',
            
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
        logging.debug(f"{len(relevant_columns)} columns considerd and they are {relevant_columns}")
        logging.debug(f"{len(removed_columns)} columns removed and they are {removed_columns}")

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
        
        analysis : NumericAnalysis = {
             'column_heading': df.columns[0],
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
             
        return analysis

    def analyze_categorical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        data = df[column].dropna().astype(str)
        
        if data.empty:
            return {"error": "No valid categorical data"}
        
        value_counts = data.value_counts()
        total = len(data)
        
        analysis: CategoricalAnalysis = {
            'type': 'categorical',
            'column_heading': column,
            'total_responses': total,
            'unique_categories': len(value_counts),
            'distribution': value_counts.to_dict(),
            'percentages': (value_counts / total * 100).round(2).to_dict(),
            'most_common': value_counts.index[0],
            'least_common': value_counts.index[-1],
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
        analyzer : BaseAgent
        if analysis_data.get('type') == 'numerical' or analysis_data.get('type') == 'rating':
            analysis : NumericAnalysis = {
            'column_heading': analysis_data['column_heading'],
            'type': 'numerical',
            'total_responses': analysis_data['total_responses'],
            'mean': analysis_data['mean'],
            'median': analysis_data['median'],
            'std_dev': analysis_data['std_dev'],
            'min_value': analysis_data['min_value'],
            'max_value': analysis_data['max_value'],
            'quartiles': {
                'Q1': analysis_data['quartiles']['Q1'],
                'Q3': analysis_data['quartiles']['Q3']
            }
        }
            analyzer = OllamaNumeriCAnalyzer()
            analysis['feedback'] = analyzer.evaluate(analysis)  
            return analysis['feedback']
            
        elif analysis_data.get('type') == 'categorical':
            analysis: CategoricalAnalysis = {
            'type': 'categorical',
            'column_heading': analysis_data['column_heading'],
            'total_responses': analysis_data['total_responses'],
            'unique_categories': analysis_data['unique_categories'],
            'distribution': analysis_data['distribution'],
            'percentages': analysis_data['percentages'],
            'most_common': analysis_data['most_common'],
            'least_common': analysis_data['least_common'],
        }
            analyzer = OllamaCategoricalCAnalyzer()
            analysis['feedback'] = analyzer.evaluate(analysis)  
            return analysis['feedback']
            
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
            
            # Modern LangChain approach using LCEL (LangChain Expression Language)
            custom_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
            
            # Create the chain using LCEL
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            chain = (
                {"context": retriever | format_docs}
                | custom_prompt
                | self.llm
                | StrOutputParser()
            )
            
            query = f"Analyze the specific data for {column_name} based on the provided metrics"
            try:
                insights = chain.invoke(query)
                return insights
            except Exception as chain_error:
                logging.error(f"Chain invoke error for {column_name}: {str(chain_error)}")
                return self._fallback_analysis(column_name, analysis_data, documents)
            
        except Exception as e:
            logging.error(f"Error generating insights for {column_name}: {str(e)}")
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
            logging.error(f"Fallback analysis failed for {column_name}: {str(fallback_error)}")
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
                logging.error(f"Error analyzing column {column}: {str(e)}")
                results[column] = {
                    'error': str(e),
                    'type': col_type
                }
        
        return results
